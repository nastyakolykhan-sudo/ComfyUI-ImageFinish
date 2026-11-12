"""ComfyUI-ImageFinish nodes: Smart Grain and Is Image Provided."""

from .grain_core import apply_smart_grain

# Upload nodes and the widget that holds their file name. An upload with an empty file name means "not given".
_UPLOAD_NODES = {"WorkflowIOImageInput": "value", "LoadImage": "image"}


class SmartGrain:
    """Film-like grain whose strength follows the brightness of the image: set how much lands in the shadows and
    how much in the highlights, and how sharply it changes between them."""

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, step, tip: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step, "tooltip": tip})
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": f(0.3, 0.0, 1.0, 0.01, "Overall grain strength. 0 turns grain off."),
                "size": f(1.0, 0.25, 4.0, 0.05, "Grain size. Measured on a 1080-pixel short side and scaled with the "
                                                "image, so the look stays the same at any resolution."),
                "color": f(0.15, 0.0, 1.0, 0.01, "0 = monochrome grain (brightness only), 1 = fully coloured grain."),
                "shadows": f(1.0, 0.0, 2.0, 0.01, "Grain strength in the dark tones, as a multiple of Amount."),
                "highlights": f(0.3, 0.0, 2.0, 0.01, "Grain strength in the bright tones, as a multiple of Amount."),
                "contrast": f(0.5, 0.0, 1.0, 0.01, "How sharply grain strength changes from Shadows to Highlights: "
                                                   "0 = gradually across all tones, 1 = quickly around Midpoint."),
                "midpoint": f(0.5, 0.05, 0.95, 0.01, "Brightness where grain strength is halfway between Shadows and "
                                                     "Highlights. Lower it to treat only the deepest tones as shadows."),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Grain pattern. In a batch or video, frame i uses seed + i."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "grain_map")
    OUTPUT_TOOLTIPS = ("The image with grain.", "Where grain lands: white = the strongest grain.")
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"

    def run(self, image, amount, size, color, shadows, highlights, contrast, midpoint, seed):
        try:
            import comfy.model_management as mm
            device, out_device = mm.get_torch_device(), mm.intermediate_device()
        except Exception:  # outside ComfyUI
            device, out_device = image.device, image.device
        out, grain_map = apply_smart_grain(image, amount, size, color, shadows, highlights, contrast, midpoint,
                                           seed, device=device, out_device=out_device)
        return (out, grain_map)


class IsImageProvided:
    """True when the connected image comes from an upload that has a file, or from any other node. An empty upload
    (Workflow IO image input or Load Image with no file) gives False, and its node never runs, so it can't fail.
    Feed the result to an If/Else Switch to skip a step when an optional image is left empty."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"image": ("IMAGE", {"lazy": True, "tooltip": "The optional image to check. It is never loaded."})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("provided",)
    FUNCTION = "run"
    CATEGORY = "utils"

    @staticmethod
    def _given(prompt, unique_id):
        node = (prompt or {}).get(str(unique_id), {})
        link = node.get("inputs", {}).get("image")
        if not isinstance(link, (list, tuple)) or not link:
            return False
        src = (prompt or {}).get(str(link[0]), {})
        key = _UPLOAD_NODES.get(src.get("class_type"))
        if key is None:
            return True
        return bool(str(src.get("inputs", {}).get(key, "") or "").strip())

    def check_lazy_status(self, prompt=None, unique_id=None, **kw):
        return []  # never load the image: only whether it was given matters

    def run(self, image=None, prompt=None, unique_id=None):
        return (self._given(prompt, unique_id),)


NODE_CLASS_MAPPINGS = {"SmartGrain": SmartGrain, "IsImageProvided": IsImageProvided}
NODE_DISPLAY_NAME_MAPPINGS = {"SmartGrain": "Smart Grain", "IsImageProvided": "Is Image Provided"}
