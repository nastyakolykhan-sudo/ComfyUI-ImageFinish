"""Brightness / colour split in CIELAB (D65), pure torch.

Images are ComfyUI IMAGE tensors (B, H, W, 3), sRGB-encoded floats in 0-1. L* carries brightness, a*/b* carry colour,
so taking L* from one image and a*/b* from another moves colour without touching brightness (or the other way).
"""

import torch

_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = torch.linalg.inv(_M_RGB2XYZ)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_EPS, _KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def _to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def _mat3(x, m):
    """(..., 3) x 3x3 as elementwise sums: full float32 even where matmuls run in reduced precision (TF32 GPUs)."""
    m = m.tolist()
    return torch.stack([x[..., 0] * r[0] + x[..., 1] * r[1] + x[..., 2] * r[2] for r in m], -1)


def rgb_to_lab(rgb):
    """(..., 3) sRGB 0-1 -> (..., 3) L* 0-100, a*, b*."""
    xyz = _mat3(_to_linear(rgb.clamp(0.0, 1.0)), _M_RGB2XYZ) / _WHITE.to(rgb)
    f = torch.where(xyz > _EPS, xyz.clamp_min(1e-12) ** (1 / 3), (_KAPPA * xyz + 16.0) / 116.0)
    return torch.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx, fz = fy + lab[..., 1] / 500.0, fy - lab[..., 2] / 200.0
    f = torch.stack([fx, fy, fz], -1)
    xyz = torch.where(f ** 3 > _EPS, f ** 3, (116.0 * f - 16.0) / _KAPPA)
    xyz = torch.stack([xyz[..., 0], torch.where(lab[..., 0] > _KAPPA * _EPS, fy ** 3, lab[..., 0] / _KAPPA), xyz[..., 2]], -1)
    return _to_srgb(_mat3(xyz * _WHITE.to(lab), _M_XYZ2RGB))


@torch.no_grad()
def combine(brightness_from, color_from, chunk=8, device=None):
    """The brightness (L*) of one image with the colour (a*, b*) of another.

    Uses: keep an image's own brightness after a colour change (brightness_from = the original), or sharpen
    brightness only (brightness_from = the sharpened copy, color_from = the original). Frame i pairs with frame i; a
    batch of one is reused for every frame. Both images must be the same size.

    Applied as a change on top of color_from (its pixels plus the difference that the new L* makes), so where the
    brightness is the same the pixels come back exactly, with no conversion round-off.
    """
    if brightness_from.shape[1:3] != color_from.shape[1:3]:
        raise ValueError("Combine Brightness and Color: both images must be the same size "
                         f"({tuple(brightness_from.shape[1:3])} vs {tuple(color_from.shape[1:3])}).")
    device = device or color_from.device
    n = max(brightness_from.shape[0], color_from.shape[0])
    outs = []
    for i in range(0, n, chunk):
        idx = range(i, min(n, i + chunk))
        b = torch.stack([brightness_from[min(j, brightness_from.shape[0] - 1)] for j in idx]).to(device, torch.float32)
        c = torch.stack([color_from[min(j, color_from.shape[0] - 1)] for j in idx]).to(device, torch.float32)
        if torch.equal(b, c):
            outs.append(c.to(color_from.device))
            continue
        lb, lc = rgb_to_lab(b), rgb_to_lab(c)
        change = lab_to_rgb(torch.cat([lb[..., :1], lc[..., 1:]], -1)) - lab_to_rgb(lc)
        outs.append((c + change).clamp(0.0, 1.0).to(color_from.device))
    return torch.cat(outs, 0)
