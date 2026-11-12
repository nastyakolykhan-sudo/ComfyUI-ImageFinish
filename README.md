# ComfyUI-ImageFinish

Small finishing nodes for ComfyUI: grain that follows brightness, brightness/colour recombination, optional-upload checks.
batches (video frames).

## Smart Grain

Film-like grain whose strength follows the brightness of the image.

| Input | What it does |
|---|---|
| amount | Overall grain strength. 0 turns grain off. |
| size | Grain size, measured on a 1080-pixel short side and scaled with the image, so the look is the same at any resolution. |
| color | 0 = monochrome grain (brightness only), 1 = fully coloured grain. |
| shadows | Grain strength in the dark tones, as a multiple of amount. |
| highlights | Grain strength in the bright tones, as a multiple of amount. |
| contrast | How sharply strength changes from shadows to highlights: 0 = gradually across all tones, 1 = quickly around midpoint. |
| midpoint | Brightness where strength is halfway between shadows and highlights. |
| seed | Grain pattern. In a batch, frame i uses seed + i, so video gets moving grain. |
| style (optional) | A preset for shadows, highlights, contrast, midpoint and color: Fine, Classic, Heavy, Even. Custom (default) uses the sliders. Amount, size and seed always apply. |

| Style | Character |
|---|---|
| Fine | Subtle, almost monochrome, a light shadow/highlight split. |
| Classic | Film-like: clearly more grain in the shadows than in the highlights. |
| Heavy | Pushed film: gritty shadows, clean highlights, an abrupt change between them. |
| Even | The same strength in every tone. |

Outputs: `image`, and `grain_map` (MASK, white = strongest grain) for checking where the grain lands.

How it works: two layers of Gaussian-blurred noise (fine + a coarser one for clumping) are normalised to a fixed
strength, so Amount means the same thing at every Size. The strength per pixel comes from a tone curve of the local
brightness (shadows → highlights, an S-curve whose steepness is `contrast`, centred on `midpoint`). Channel values
close to pure black or white get proportionally less grain, so blacks are not lifted and whites are not dirtied.

## Combine Brightness and Color

The brightness of one image with the colour of another: CIELAB L* from `brightness_from`, a*/b* from `color_from`
(same size; a batch of one is reused for every frame). Two common uses:

- **Keep brightness after a colour change:** `brightness_from` = the original, `color_from` = the colour-matched
  copy. The colours move, the light and contrast stay.
- **Sharpen brightness only:** `brightness_from` = a sharpened copy, `color_from` = the original. Detail gets
  sharper without coloured fringes on edges.

## Is Image Provided

Returns `True` when the connected image comes from an upload that has a file (Workflow IO image input or Load
Image), or from any other node; `False` for an empty upload. The image itself is never loaded, so an empty
optional upload never runs and never fails. Feed the result to a core If/Else Switch to skip a step when an optional
image is left empty.
