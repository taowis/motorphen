"""Masked autoencoder Transformer over joint-frame tokens (after Motta et al.).

Input is a batch of joint angle windows in radians, shape (B, J, T, 3), and a
boolean mask (B, J, T), True where a token is hidden from the encoder. Tokens
are ordered joint-major: index = j * T + t.
"""

import math

import torch
from torch import nn


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def rot_yxz(a):
    """Rotation matrices for intrinsic Y-X-Z Cardan angles (..., 3) -> (..., 3, 3)."""
    y, x, z = a[..., 0], a[..., 1], a[..., 2]
    cy, sy, cx, sx, cz, sz = (torch.cos(y), torch.sin(y), torch.cos(x),
                              torch.sin(x), torch.cos(z), torch.sin(z))
    one, zero = torch.ones_like(y), torch.zeros_like(y)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(*y.shape, 3, 3)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(*y.shape, 3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(*y.shape, 3, 3)
    return ry @ rx @ rz


def token_features(a):
    """(..., 3) radians -> (..., 12): sin, cos, first two rotation columns."""
    r = rot_yxz(a)
    six = torch.cat([r[..., :, 0], r[..., :, 1]], dim=-1)
    return torch.cat([torch.sin(a), torch.cos(a), six], dim=-1)


def sinusoidal(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class GaitMAE(nn.Module):
    def __init__(self, n_joints=12, n_frames=7, d_model=288, n_heads=12,
                 enc_layers=8, dec_layers=2, dropout=0.1, feat_dim=12):
        super().__init__()
        self.J, self.T, self.d = n_joints, n_frames, d_model
        self.proj = nn.Linear(feat_dim, d_model)
        self.vel = nn.Linear(3, d_model)
        self.joint_emb = nn.Embedding(n_joints, d_model)
        self.frame_emb = nn.Embedding(n_frames, d_model)
        self.enc_mask = nn.Parameter(torch.zeros(d_model))
        self.dec_mask = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("pe", sinusoidal(n_joints * n_frames, d_model), persistent=False)
        self.register_buffer("jidx", torch.arange(n_joints).repeat_interleave(n_frames), persistent=False)
        self.register_buffer("tidx", torch.arange(n_frames).repeat(n_joints), persistent=False)

        def stack(n):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                               batch_first=True, norm_first=False)
            return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

        self.encoder = stack(enc_layers)
        self.decoder = stack(dec_layers)
        self.head = nn.Linear(d_model, feat_dim)
        nn.init.normal_(self.enc_mask, std=0.02)
        nn.init.normal_(self.dec_mask, std=0.02)

    def _context(self, vel_tok):
        return self.pe + self.joint_emb(self.jidx) + self.frame_emb(self.tidx) + self.vel(vel_tok)

    def forward(self, angles, mask):
        B, J, T, _ = angles.shape
        feats = token_features(angles).reshape(B, J * T, -1)
        # angular velocity within the window; zero at frame 0 and wherever the
        # current or previous token is hidden, so masked values do not leak
        v = torch.zeros_like(angles)
        v[:, :, 1:] = wrap(angles[:, :, 1:] - angles[:, :, :-1])
        hide = mask.clone()
        hide[:, :, 1:] |= mask[:, :, :-1]
        v = v.masked_fill(hide[..., None], 0.0).reshape(B, J * T, 3)
        m = mask.reshape(B, J * T)

        ctx = self._context(v)
        x = self.proj(feats)
        x = torch.where(m[..., None], self.enc_mask.expand_as(x), x) + ctx
        memory = self.encoder(x)
        y = torch.where(m[..., None], self.dec_mask.expand_as(memory), memory) + ctx
        out = self.head(self.decoder(y))
        return out.reshape(B, J, T, -1)


def angles_from_pred(pred):
    """Recover angles (..., 3) from the predicted sin-cos part of the features."""
    return torch.atan2(pred[..., 0:3], pred[..., 3:6])
