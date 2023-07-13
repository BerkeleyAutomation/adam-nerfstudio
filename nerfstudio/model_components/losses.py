# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of Losses.
"""
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from torchvision import models, transforms
import torch.nn.functional as F

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import masked_reduction, normalized_depth_scale_and_shift

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7

# Sigma scale factor from Urban Radiance Fields (Rematas et al., 2022)
URF_SIGMA_SCALE_FACTOR = 3.0


class DepthLossType(Enum):
    """Types of depth losses for depth supervision."""

    DS_NERF = 1
    URF = 2


def outer(
    t0_starts: Float[Tensor, "*batch num_samples_0"],
    t0_ends: Float[Tensor, "*batch num_samples_0"],
    t1_starts: Float[Tensor, "*batch num_samples_1"],
    t1_ends: Float[Tensor, "*batch num_samples_1"],
    y1: Float[Tensor, "*batch num_samples_1"],
) -> Float[Tensor, "*batch num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: Float[Tensor, "*batch num_samples_1"],
    w: Float[Tensor, "*batch num_samples"],
    t_env: Float[Tensor, "*batch num_samples_1"],
    w_env: Float[Tensor, "*batch num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list) -> torch.Tensor:
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    assert len(ray_samples_list) > 0

    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))

    assert isinstance(loss_interlevel, Tensor)
    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


def nerfstudio_distortion_loss(
    ray_samples: RaySamples,
    densities: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
    weights: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
) -> Float[Tensor, "*bs 1"]:
    """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples: Ray samples to compute loss over
        densities: Predicted sample densities
        weights: Predicted weights from densities and sample locations
    """
    if torch.is_tensor(densities):
        assert not torch.is_tensor(weights), "Cannot use both densities and weights"
        assert densities is not None
        # Compute the weight at each sample location
        weights = ray_samples.get_weights(densities)
    if torch.is_tensor(weights):
        assert not torch.is_tensor(densities), "Cannot use both densities and weights"
    assert weights is not None

    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends

    assert starts is not None and ends is not None, "Ray samples must have spacing starts and ends"
    midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

    return loss


def orientation_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    viewdirs: Float[Tensor, "*bs 3"],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs * -1
    n_dot_v = (n * v[..., None, :]).sum(dim=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    pred_normals: Float[Tensor, "*bs num_samples 3"],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


def ds_nerf_depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    lengths: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
) -> Float[Tensor, "*batch 1"]:
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        steps: Sampling distances along rays.
        lengths: Distances between steps.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = termination_depth > 0

    loss = -torch.log(weights + EPS) * torch.exp(-((steps - termination_depth[:, None]) ** 2) / (2 * sigma)) * lengths
    loss = loss.sum(-2) * depth_mask
    return torch.mean(loss)


def urban_radiance_field_depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
) -> Float[Tensor, "*batch 1"]:
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        steps: Sampling distances along rays.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = termination_depth > 0

    # Expected depth loss
    expected_depth_loss = (termination_depth - predicted_depth) ** 2

    # Line of sight losses
    target_distribution = torch.distributions.normal.Normal(0.0, sigma / URF_SIGMA_SCALE_FACTOR)
    termination_depth = termination_depth[:, None]
    line_of_sight_loss_near_mask = torch.logical_and(
        steps <= termination_depth + sigma, steps >= termination_depth - sigma
    )
    line_of_sight_loss_near = (weights - torch.exp(target_distribution.log_prob(steps - termination_depth))) ** 2
    line_of_sight_loss_near = (line_of_sight_loss_near_mask * line_of_sight_loss_near).sum(-2)
    line_of_sight_loss_empty_mask = steps < termination_depth - sigma
    line_of_sight_loss_empty = (line_of_sight_loss_empty_mask * weights**2).sum(-2)
    line_of_sight_loss = line_of_sight_loss_near + line_of_sight_loss_empty

    loss = (expected_depth_loss + line_of_sight_loss) * depth_mask
    return torch.mean(loss)


def depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    ray_samples: RaySamples,
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    sigma: Float[Tensor, "0"],
    directions_norm: Float[Tensor, "*batch 1"],
    is_euclidean: bool,
    depth_loss_type: DepthLossType,
) -> Float[Tensor, "0"]:
    """Implementation of depth losses.

    Args:
        weights: Weights predicted for each sample.
        ray_samples: Samples along rays corresponding to weights.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        sigma: Uncertainty around depth value.
        directions_norm: Norms of ray direction vectors in the camera frame.
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
        depth_loss_type: Type of depth loss to apply.

    Returns:
        Depth loss scalar.
    """
    if not is_euclidean:
        termination_depth = termination_depth * directions_norm
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    if depth_loss_type == DepthLossType.DS_NERF:
        lengths = ray_samples.frustums.ends - ray_samples.frustums.starts
        return ds_nerf_depth_loss(weights, termination_depth, steps, lengths, sigma)

    if depth_loss_type == DepthLossType.URF:
        return urban_radiance_field_depth_loss(weights, termination_depth, predicted_depth, steps, sigma)

    raise NotImplementedError("Provided depth loss type not implemented.")


def monosdf_normal_loss(
    normal_pred: Float[Tensor, "num_samples 3"], normal_gt: Float[Tensor, "num_samples 3"]
) -> Float[Tensor, "0"]:
    """
    Normal consistency loss proposed in monosdf - https://niujinshuchong.github.io/monosdf/
    Enforces consistency between the volume rendered normal and the predicted monocular normal.
    With both angluar and L1 loss. Eq 14 https://arxiv.org/pdf/2206.00665.pdf
    Args:
        normal_pred: volume rendered normal
        normal_gt: monocular normal
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos


class MiDaSMSELoss(nn.Module):
    """
    data term from MiDaS paper
    """

    def __init__(self, reduction_type: Literal["image", "batch"] = "batch"):
        super().__init__()

        self.reduction_type: Literal["image", "batch"] = reduction_type
        # reduction here is different from the image/batch-based reduction. This is either "mean" or "sum"
        self.mse_loss = MSELoss(reduction="none")

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            mse loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        image_loss = torch.sum(self.mse_loss(prediction, target) * mask, (1, 2))
        # multiply by 2 magic number?
        image_loss = masked_reduction(image_loss, 2 * summed_mask, self.reduction_type)

        return image_loss


# losses based on https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
class GradientLoss(nn.Module):
    """
    multiscale, scale-invariant gradient matching term to the disparity space.
    This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
    More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
    """

    def __init__(self, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.reduction_type: Literal["image", "batch"] = reduction_type
        self.__scales = scales

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            gradient loss based on reduction function
        """
        assert self.__scales >= 1
        total = 0.0

        for scale in range(self.__scales):
            step = pow(2, scale)

            grad_loss = self.gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
            )
            total += grad_loss

        assert isinstance(total, Tensor)
        return total

    def gradient_loss(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        multiscale, scale-invariant gradient matching term to the disparity space.
        This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
        More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            reduction: reduction function, either reduction_batch_based or reduction_image_based
        Returns:
            gradient loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
        image_loss = masked_reduction(image_loss, summed_mask, self.reduction_type)

        return image_loss


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale and shift invariant loss as described in
    "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    https://arxiv.org/pdf/1907.01341.pdf
    """

    def __init__(self, alpha: float = 0.5, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            alpha: weight of the regularization term
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.__data_loss = MiDaSMSELoss(reduction_type=reduction_type)
        self.__regularization_loss = GradientLoss(scales=scales, reduction_type=reduction_type)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map (unnormalized)
            target: ground truth depth map (normalized)
            mask: mask of valid pixels
        Returns:
            scale and shift invariant loss
        """
        scale, shift = normalized_depth_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        """
        scale and shift invariant prediction
        from https://arxiv.org/pdf/1907.01341.pdf equation 1
        """
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


def tv_loss(grids: Float[Tensor, "grids feature_dim row column"]) -> Float[Tensor, ""]:
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    number_of_grids = grids.shape[0]
    h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
    return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


class _GradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scale gradients by a constant factor.
    """

    @staticmethod
    def forward(ctx, value, scaling):
        ctx.save_for_backward(scaling)
        return value, scaling

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (scaling,) = ctx.saved_tensors
        return output_grad * scaling, grad_scaling


def scale_gradients_by_distance_squared(
    field_outputs: Dict[FieldHeadNames, torch.Tensor],
    ray_samples: RaySamples,
) -> Dict[FieldHeadNames, torch.Tensor]:
    """
    Scale gradients by the ray distance to the pixel
    as suggested in `Radiance Field Gradient Scaling for Unbiased Near-Camera Training` paper

    Note: The scaling is applied on the interval of [0, 1] along the ray!

    Example:
        GradientLoss should be called right after obtaining the densities and colors from the field. ::
            >>> field_outputs = scale_gradient_by_distance_squared(field_outputs, ray_samples)
    """
    out = {}
    ray_dist = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    scaling = torch.square(ray_dist).clamp(0, 1)
    for key, value in field_outputs.items():
        out[key], _ = cast(Tuple[Tensor, Tensor], _GradientScaler.apply(value, scaling))
    return out

# ###### estimate linear color transform
# def match_colors_for_image_set(content_images, style_img):
#     content_sub = content_images.view(-1, 3)
#     style_sub = style_img.view(-1, 3).to(content_sub.device)

#     mu_c = content_sub.mean(0, keepdim=True)
#     mu_s = style_sub.mean(0, keepdim=True)

#     cov_c = torch.matmul((content_sub - mu_c).transpose(1, 0), content_sub - mu_c) / float(content_sub.size(0))
#     cov_s = torch.matmul((style_sub - mu_s).transpose(1, 0), style_sub - mu_s) / float(style_sub.size(0))

#     u_c, sig_c, _ = torch.svd(cov_c)
#     u_s, sig_s, _ = torch.svd(cov_s)

#     u_c_i = u_c.transpose(1, 0)
#     u_s_i = u_s.transpose(1, 0)

#     scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
#     scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

#     tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
#     tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

#     content_sub = content_sub @ tmp_mat.T + tmp_vec.view(1, 3)
#     content_sub = content_sub.contiguous().clamp_(0.0, 1.0)

#     tf = torch.eye(4).float().to(tmp_mat.device)
#     tf[:3, :3] = tmp_mat
#     tf[:3, 3:4] = tmp_vec.T
#     return content_sub, tf


# class VGG(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vgg = models.vgg16(pretrained=True).eval()
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def get_feats(self, x, layers=[], supress_assert=True):
#         # Layer indexes:
#         # Conv1_*: 1,3
#         # Conv2_*: 6,8
#         # Conv3_*: 11, 13, 15
#         # Conv4_*: 18, 20, 22
#         # Conv5_*: 25, 27, 29

#         if not supress_assert:
#             assert x.min() >= 0.0 and x.max() <= 1.0, "input is expected to be an image scaled between 0 and 1"

#         x = self.normalize(x)
#         final_ix = max(layers)
#         outputs = []

#         for ix, layer in enumerate(self.vgg.features):
#             x = layer(x)
#             if ix in layers:
#                 outputs.append(x)

#             if ix == final_ix:
#                 break

#         return outputs


# def cos_distance(a, b, center=True):
#     """a: [b, c, hw],
#     b: [b, c, h2w2]
#     """
#     # """cosine distance
#     if center:
#         a = a - a.mean(2, keepdims=True)
#         b = b - b.mean(2, keepdims=True)

#     a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
#     b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()

#     a = a / (a_norm + 1e-8)
#     b = b / (b_norm + 1e-8)

#     d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)
#     # """"

#     """
#     a_norm_sq = (a * a).sum(1).unsqueeze(2)
#     b_norm_sq = (b * b).sum(1).unsqueeze(1)

#     d_mat = a_norm_sq + b_norm_sq - 2.0 * torch.matmul(a.transpose(2, 1), b)
#     """
#     return d_mat


# def cos_loss(a, b):
#     # """cosine loss
#     a_norm = (a * a).sum(1, keepdims=True).sqrt()
#     b_norm = (b * b).sum(1, keepdims=True).sqrt()
#     a_tmp = a / (a_norm + 1e-8)
#     b_tmp = b / (b_norm + 1e-8)
#     cossim = (a_tmp * b_tmp).sum(1)
#     cos_d = 1.0 - cossim
#     return cos_d.mean()
#     # """

#     # return ((a - b) ** 2).mean()


# def feat_replace(a, b):
#     n, c, h, w = a.size()
#     n2, c, h2, w2 = b.size()

#     assert (n == 1) and (n2 == 1)

#     a_flat = a.view(n, c, -1)
#     b_flat = b.view(n2, c, -1)
#     b_ref = b_flat.clone()

#     z_new = []

#     # Loop is slow but distance matrix requires a lot of memory
#     for i in range(n):
#         z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

#         z_best = torch.argmin(z_dist, 2)
#         del z_dist

#         z_best = z_best.unsqueeze(1).repeat(1, c, 1)
#         feat = torch.gather(b_ref, 2, z_best)

#         z_new.append(feat)

#     z_new = torch.cat(z_new, 0)
#     z_new = z_new.view(n, c, h, w)
#     return z_new


# def guided_feat_replace(a, b, trgt):
#     n, c, h, w = a.size()
#     n2, c2, h2, w2 = b.size()
#     n3, c3, h3, w3 = trgt.size()
#     assert (n == 1) and (n2 == 1) and (c == c2) and (n3 == 1) and (h2 == h3) and (w2 == w3)

#     a_flat = a.view(n, c, -1)
#     b_flat = b.view(n2, c2, -1)
#     trgt = trgt.view(n3, c3, -1)

#     z_new = []

#     # Loop is slow but distance matrix requires a lot of memory
#     for i in range(n):
#         z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

#         z_best = torch.argmin(z_dist, 2)
#         del z_dist

#         z_best = z_best.unsqueeze(1).repeat(1, c3, 1)
#         feat = torch.gather(trgt, 2, z_best)

#         z_new.append(feat)

#     z_new = torch.cat(z_new, 0)
#     z_new = z_new.view(n, c3, h, w)
#     return z_new


# def nn_loss(outputs, styles, vgg, blocks=[2]):

#     blocks.sort()
#     block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
#     total_loss = 0.0

#     all_layers = []
#     for block in blocks:
#         all_layers += block_indexes[block]

#     x_feats_all = vgg.get_feats(outputs, all_layers)
#     with torch.no_grad():
#         s_feats_all = vgg.get_feats(styles, all_layers)

#     ix_map = {}
#     for a, b in enumerate(all_layers):
#         ix_map[b] = a

#     for block in blocks:
#         layers = block_indexes[block]
#         x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
#         s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

#         target_feats = feat_replace(x_feats, s_feats)
#         total_loss += cos_loss(x_feats, target_feats)

#     return total_loss


# def gram_matrix(feature_maps, center=False):
#     """
#     feature_maps: b, c, h, w
#     gram_matrix: b, c, c
#     """
#     b, c, h, w = feature_maps.size()
#     features = feature_maps.view(b, c, h * w)
#     if center:
#         features = features - features.mean(dim=-1, keepdims=True)
#     G = torch.bmm(features, torch.transpose(features, 1, 2))
#     return G


# class NNLoss(torch.nn.Module):
    # def __init__(self, device):
    #     super().__init__()
    #     self.vgg = VGG().to(device)

    # def forward(
    #     self,
    #     outputs,
    #     styles,
    #     blocks=[
    #         2,
    #     ],
    #     loss_names=["nn_loss"],  # can also include 'gram_loss', 'content_loss'
    #     contents=None,
    # ):
    #     blocks.sort()
    #     block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

    #     all_layers = []
    #     for block in blocks:
    #         all_layers += block_indexes[block]

    #     x_feats_all = self.vgg.get_feats(outputs, all_layers)
    #     with torch.no_grad():
    #         s_feats_all = self.vgg.get_feats(styles, all_layers)
    #         if "content_loss" in loss_names:
    #             content_feats_all = self.vgg.get_feats(contents, all_layers)

    #     ix_map = {}
    #     for a, b in enumerate(all_layers):
    #         ix_map[b] = a

    #     nn_loss = 0.0
    #     gram_loss = 0.0
    #     content_loss = 0.0
    #     import pdb; pdb.set_trace()
    #     for block in blocks:
    #         layers = block_indexes[block]
    #         x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
    #         s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

    #         if "nn_loss" in loss_names:
    #             target_feats = feat_replace(x_feats, s_feats)
    #             nn_loss += cos_loss(x_feats, target_feats)

    #         if "gram_loss" in loss_names:
    #             gram_loss += torch.mean(( - gram_matrix(s_feats)) ** 2)

    #         if "content_loss" in loss_names:
    #             content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
    #             content_loss += torch.mean((content_feats - x_feats) ** 2)

    #     return nn_loss, gram_loss, content_loss

    # def get_style_nn(
    #     self,
    #     outputs,
    #     styles,
    #     blocks=[
    #         2,
    #     ],
    # ):
    #     blocks.sort()
    #     block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

    #     all_layers = []
    #     for block in blocks:
    #         all_layers += block_indexes[block]

    #     x_feats_all = self.vgg.get_feats(outputs, all_layers)
    #     with torch.no_grad():
    #         s_feats_all = self.vgg.get_feats(styles, all_layers)

    #     ix_map = {}
    #     for a, b in enumerate(all_layers):
    #         ix_map[b] = a

    #     trgt_feats = []
    #     for block in blocks:
    #         layers = block_indexes[block]
    #         x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
    #         s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

    #         _, _, h, w = s_feats.size()
    #         styles_resample = F.interpolate(styles, (h, w), mode="bilinear")
    #         # ic(x_feats.shape, s_feats.shape, styles_resample.shape)

    #         feats = guided_feat_replace(x_feats, s_feats, styles_resample)
    #         trgt_feats.append(feats)

    #     return trgt_feats
    

def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best


def nn_feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G


class NNFMLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = models.vgg16(pretrained=True).eval().to(device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
    ):
        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss']

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.get_feats(contents, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a


        loss_dict = dict([(x, 0.) for x in loss_names])
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)
            # import pdb; pdb.set_trace()
            # 9, 660
            if "nnfm_loss" in loss_names:
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                import pdb; pdb.set_trace()
                loss_dict["gram_loss"] += torch.mean((gram_matrix(x_feats, center=True) - gram_matrix(s_feats, center=True)) ** 2)

            if "content_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                loss_dict["content_loss"] += torch.mean((content_feats - x_feats) ** 2)

        return loss_dict


""" VGG-16 Structure
Input image is [-1, 3, 224, 224]
-------------------------------------------------------------------------------
        Layer (type)               Output Shape         Param #     Layer index
===============================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792     
              ReLU-2         [-1, 64, 224, 224]               0               1
            Conv2d-3         [-1, 64, 224, 224]          36,928     
              ReLU-4         [-1, 64, 224, 224]               0               3
         MaxPool2d-5         [-1, 64, 112, 112]               0     
            Conv2d-6        [-1, 128, 112, 112]          73,856     
              ReLU-7        [-1, 128, 112, 112]               0               6
            Conv2d-8        [-1, 128, 112, 112]         147,584     
              ReLU-9        [-1, 128, 112, 112]               0               8
        MaxPool2d-10          [-1, 128, 56, 56]               0     
           Conv2d-11          [-1, 256, 56, 56]         295,168     
             ReLU-12          [-1, 256, 56, 56]               0              11
           Conv2d-13          [-1, 256, 56, 56]         590,080     
             ReLU-14          [-1, 256, 56, 56]               0              13
           Conv2d-15          [-1, 256, 56, 56]         590,080     
             ReLU-16          [-1, 256, 56, 56]               0              15
        MaxPool2d-17          [-1, 256, 28, 28]               0     
           Conv2d-18          [-1, 512, 28, 28]       1,180,160     
             ReLU-19          [-1, 512, 28, 28]               0              18
           Conv2d-20          [-1, 512, 28, 28]       2,359,808     
             ReLU-21          [-1, 512, 28, 28]               0              20
           Conv2d-22          [-1, 512, 28, 28]       2,359,808     
             ReLU-23          [-1, 512, 28, 28]               0              22
        MaxPool2d-24          [-1, 512, 14, 14]               0     
           Conv2d-25          [-1, 512, 14, 14]       2,359,808     
             ReLU-26          [-1, 512, 14, 14]               0              25
           Conv2d-27          [-1, 512, 14, 14]       2,359,808     
             ReLU-28          [-1, 512, 14, 14]               0              27
           Conv2d-29          [-1, 512, 14, 14]       2,359,808    
             ReLU-30          [-1, 512, 14, 14]               0              29
        MaxPool2d-31            [-1, 512, 7, 7]               0    
===============================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.39
Params size (MB): 56.13
Estimated Total Size (MB): 275.10
----------------------------------------------------------------
"""
