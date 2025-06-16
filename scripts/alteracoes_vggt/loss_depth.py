# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# DIRTY VERSION, TO BE CLEANED UP

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor

from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

import torch
import torch.nn.functional as F

def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max = 100):
    """
    Verifica e corrige valores NaN ou Inf no tensor de perda.
    """
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        print(f"[{loss_name}] contém NaN ou Inf. Substituindo por 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )
    return torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

def depth_loss(pred, conf, batch, gamma=1.0, alpha=0.2):
    """
    Função de perda para mapas de altura com incerteza, inspirada no VGGT.

    Args:
        pred (torch.Tensor): Altura predita, shape [B, 1, H, W]
        conf (torch.Tensor): Incerteza predita, shape [B, 1, H, W]
        batch (dict): Deve conter 'depths' e 'point_masks'
        gamma (float): Peso dos termos principais da perda
        alpha (float): Peso da regularização da incerteza

    Returns:
        dict: com loss total e componentes: val, grad, reg
    """
    target = batch["depths"]
    mask = batch["point_masks"]

    pred = check_and_fix_inf_nan(pred, "pred")
    target = check_and_fix_inf_nan(target, "target")

    # Termo de valor absoluto
    diff = torch.abs(pred - target)
    loss_val = conf * diff
    loss_val = loss_val[mask].mean()

    # Termo de gradiente
    grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    grad_conf_x = conf[:, :, :, :-1]

    grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    grad_conf_y = conf[:, :, :-1, :]

    grad_x = grad_conf_x * torch.abs(grad_pred_x - grad_target_x)
    grad_y = grad_conf_y * torch.abs(grad_pred_y - grad_target_y)
    loss_grad = (grad_x.mean() + grad_y.mean()) / 2

    # Regularização da incerteza
    reg_uncertainty = -alpha * torch.log(conf[mask] + 1e-6).mean()

    # Total
    total_loss = gamma * (loss_val + loss_grad) + reg_uncertainty

    return {
        "loss_conf_depth": total_loss,
        "loss_val_depth": loss_val,
        "loss_grad_depth": loss_grad,
        "loss_reg_uncertainty": reg_uncertainty
    }
