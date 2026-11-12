"""Smart Grain core: film-like grain whose strength follows the brightness of the image.

Pure torch, no ComfyUI imports, so it can be tested on its own. Images are ComfyUI IMAGE tensors:
(B, H, W, 3) float in 0-1, display-referred (sRGB-encoded). Grain is added in that space, which is close to
perceptually even, so the same Amount reads about the same in every tone before the tonal response is applied.
"""

import math

import torch
import torch.nn.functional as F

REF_SHORT_SIDE = 1080.0  # Size 1.0 is defined on an image with this short side; other sizes scale with it.
MAX_SIGMA = 0.12         # grain standard deviation (0-1 units) at Amount 1 and tonal weight 1
EDGE = 0.06              # channel values this close to pure black / white get proportionally less grain
COARSE_MIX = 0.3         # share of a second, coarser grain layer (gives film-like clumping)
COARSE_FACTOR = 2.2      # how much larger the coarse layer is


def _kernel1d(sigma, device, dtype):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(x, sigma):
    """Separable Gaussian blur of x (N, C, H, W). Returns (blurred, std factor for white noise)."""
    if sigma < 0.05:
        return x, 1.0
    k = _kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    mode = "reflect" if r < min(x.shape[-2:]) else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    # White noise of unit std, blurred by a separable kernel k, has std = sum(k^2) (1D sum squared, square-rooted).
    return x, float((k ** 2).sum())


def _grain_field(channels, h, w, sigma, generator, device):
    """Unit-std grain: a fine layer plus a coarser layer, both Gaussian-blurred white noise."""
    def layer(s):
        n = torch.randn((1, channels, h, w), generator=generator, device=generator.device).to(device)
        n, f = _blur(n, s)
        return n / f

    fine = layer(sigma)
    coarse = layer(sigma * COARSE_FACTOR + 0.35)
    a, b = 1.0 - COARSE_MIX, COARSE_MIX
    return (a * fine + b * coarse) / math.sqrt(a * a + b * b)


def tone_curve(y, contrast, midpoint):
    """0 in the darkest tones -> 1 in the brightest. Midpoint is where it crosses 0.5; contrast sets how steep."""
    p = min(max(float(midpoint), 0.01), 0.99)
    s = y.clamp(1e-6, 1.0) ** (math.log(0.5) / math.log(p))  # gamma that maps the midpoint to 0.5
    k = 20.0 * max(float(contrast), 0.0) ** 1.5
    if k < 1e-2:
        return s
    lo, hi = 1.0 / (1.0 + math.exp(0.5 * k)), 1.0 / (1.0 + math.exp(-0.5 * k))
    return ((torch.sigmoid(k * (s - 0.5)) - lo) / (hi - lo)).clamp(0.0, 1.0)


def tone_weight(y, shadows, highlights, contrast, midpoint):
    """Grain strength multiplier per pixel: Shadows in the dark tones, Highlights in the bright ones."""
    return highlights + (shadows - highlights) * (1.0 - tone_curve(y, contrast, midpoint))


def _generator(device, seed):
    gen_device = "cuda" if torch.device(device).type == "cuda" else "cpu"
    return torch.Generator(device=gen_device).manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


@torch.no_grad()
def apply_smart_grain(images, amount=0.3, size=1.0, color=0.15, shadows=1.0, highlights=0.3,
                      contrast=0.5, midpoint=0.5, seed=0, device=None, out_device=None):
    """Add grain to every image in the batch. Frame i uses seed + i, so a video gets moving grain.

    Returns (images, grain_map). grain_map is the tonal weight scaled to 0-1 (white = most grain).
    """
    device = torch.device(device) if device is not None else images.device
    out_device = torch.device(out_device) if out_device is not None else images.device
    b, h, w, _ = images.shape
    peak = max(shadows, highlights, 1e-6)
    if amount <= 0 or peak <= 1e-6:
        return images.to(out_device), torch.zeros((b, h, w), device=out_device)

    scale = min(h, w) / REF_SHORT_SIDE
    sigma = 0.5 * float(size) * scale
    color = min(max(float(color), 0.0), 1.0)
    mono_w, col_w = 1.0 - color, color
    norm = 1.0 / math.sqrt(mono_w ** 2 + col_w ** 2)
    luma_coef = torch.tensor([0.2126, 0.7152, 0.0722], device=device).view(1, 3, 1, 1)

    outs, maps = [], []
    for i in range(b):
        x = images[i:i + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)  # 1,3,H,W
        y = (x * luma_coef).sum(1, keepdim=True)
        y, _ = _blur(y, max(0.6, scale))  # tones follow local brightness, not single-pixel texture
        weight = tone_weight(y, shadows, highlights, contrast, midpoint)  # 1,1,H,W

        gen = _generator(device, seed + i)
        g = 0.0
        if mono_w > 0:
            g = g + mono_w * _grain_field(1, h, w, sigma, gen, device)
        if col_w > 0:
            g = g + col_w * _grain_field(3, h, w, sigma, gen, device)
        g = g * norm

        headroom = (torch.minimum(x, 1.0 - x) / EDGE).clamp(0.0, 1.0)  # per channel: protect pure black / white
        x = (x + float(amount) * MAX_SIGMA * weight * headroom * g).clamp(0.0, 1.0)

        outs.append(x.permute(0, 2, 3, 1).to(out_device))
        maps.append((weight[:, 0] / peak).clamp(0.0, 1.0).to(out_device))
    return torch.cat(outs, 0), torch.cat(maps, 0)
