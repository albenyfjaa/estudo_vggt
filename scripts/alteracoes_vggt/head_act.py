# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract predictions and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for prediction (height/depth/xyz).
        conf_activation: Activation for confidence channel (if present)

    Returns:
        Tuple of (prediction_tensor, confidence_tensor or None)
    """
    B, C, H, W = out.shape

    if C == 1:
        # Apenas predição de altura ou profundidade
        return out, None

    elif C == 2:
        # [predição, confiança]
        pred = out[:, 0:1, :, :]
        conf = out[:, 1:2, :, :]

        if activation == "exp":
            pred = torch.exp(pred)
        elif activation == "inv_log":
            pred = inverse_log_transform(pred)
        elif activation == "relu":
            pred = F.relu(pred)
        elif activation == "linear":
            pass  # sem alteração
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if conf_activation == "expp1":
            conf = 1 + conf.exp()
        elif conf_activation == "expp0":
            conf = conf.exp()
        elif conf_activation == "sigmoid":
            conf = torch.sigmoid(conf)
        else:
            raise ValueError(f"Unknown conf_activation: {conf_activation}")

        return pred, conf

    else:
        # Casos como pontos 3D: [X,Y,Z,conf], etc. (permute necessário)
        fmap = out.permute(0, 2, 3, 1)  # B,H,W,C
        xyz = fmap[:, :, :, :-1]
        conf = fmap[:, :, :, -1]

        if activation == "norm_exp":
            d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            xyz_normed = xyz / d
            pred = xyz_normed * torch.expm1(d)
        elif activation == "norm":
            pred = xyz / xyz.norm(dim=-1, keepdim=True)
        elif activation == "exp":
            pred = torch.exp(xyz)
        elif activation == "relu":
            pred = F.relu(xyz)
        elif activation == "inv_log":
            pred = inverse_log_transform(xyz)
        elif activation == "xy_inv_log":
            xy, z = xyz.split([2, 1], dim=-1)
            z = inverse_log_transform(z)
            pred = torch.cat([xy * z, z], dim=-1)
        elif activation == "sigmoid":
            pred = torch.sigmoid(xyz)
        elif activation == "linear":
            pred = xyz
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if conf_activation == "expp1":
            conf_out = 1 + conf.exp()
        elif conf_activation == "expp0":
            conf_out = conf.exp()
        elif conf_activation == "sigmoid":
            conf_out = torch.sigmoid(conf)
        else:
            raise ValueError(f"Unknown conf_activation: {conf_activation}")

        return pred, conf_out



def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))



def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
