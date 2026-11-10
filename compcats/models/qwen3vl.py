"""
Qwen3-VL vision encoder (the `visual` tower) as a compcats model.

Units are token x channel: every hookable layer outputs (B, N_tok, C) in RASTER
token order (row-major over the h x w patch grid); no pooling over tokens.

Hookable layer names: ``taps.block_{i}`` (output of vision block i, 0-based,
(B, (S/16)^2, hidden)) and ``taps.merger`` (patch-merger output,
(B, (S/32)^2, out_hidden), merge-group = raster order of the coarse grid).
All outputs are float32.

Usage:
    model = get_model("Qwen/Qwen3-VL-2B-Instruct", image_size=224, dtype="bfloat16")
    localizer.get_froi_mask(FLOC_ROOT, "grillspector", MODELS_DIR, model,
                            "taps.block_11", "FFA", 1.96)
"""
import torch
from torch import nn
from PIL import Image

from .basemodel import BaseModel

DEFAULT_REVISIONS = {
    "Qwen/Qwen3-VL-2B-Instruct": "89644892e4d85e24eaac8bacfd4f463576704203",
}

def get_model(model_id="Qwen/Qwen3-VL-2B-Instruct", revision=None, image_size=224,
              dtype="float32", **kwargs):
    """Get the Qwen3-VL vision encoder as a compcats model"""
    return Qwen3VLVision(model_id=model_id, revision=revision, image_size=image_size,
                         dtype=dtype, **kwargs)

def raster_permutation(h, w, merge=2):
    """
    Qwen's processor orders patches as (h/m, w/m, m, m): each m x m merge group is
    contiguous. Returns perm such that packed[perm] is row-major over the h x w grid.
    """
    idx = torch.arange(h * w).reshape(h // merge, w // merge, merge, merge)
    # idx[bh, bw, i, j] = packed position of the patch at row bh*m+i, col bw*m+j
    return idx.permute(0, 2, 1, 3).reshape(-1)

class QwenImageTransform:
    """PIL RGB -> (N_tok, patch_dim) Qwen pixel patches (square pad, bicubic resize, mean=std=0.5)"""
    def __init__(self, processor, size):
        self.processor, self.size = processor, size

    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        if w != h:  # pad to square with black, like compcats' SquarePad
            m = max(w, h)
            canvas = Image.new("RGB", (m, m), (0, 0, 0))
            canvas.paste(img, ((m - w) // 2, (m - h) // 2))
            img = canvas
        img = img.resize((self.size, self.size), Image.BICUBIC)
        out = self.processor(images=img, do_resize=False, return_tensors="pt")
        assert tuple(out["image_grid_thw"][0].tolist()) == (1, self.size // 16, self.size // 16)
        return out["pixel_values"]  # (N, 1536)

class _Tower(nn.Module):
    """Wraps `visual`: takes (B, N, patch_dim), exposes per-image raster taps"""
    def __init__(self, visual, size, dtype):
        super().__init__()
        self.visual, self.dtype = visual, dtype
        cfg = visual.config
        self.g = size // cfg.patch_size
        self.merge = cfg.spatial_merge_size
        self.register_buffer("perm", raster_permutation(self.g, self.g, self.merge), persistent=False)
        self.taps = nn.Module()
        for i in range(cfg.depth):
            self.taps.add_module(f"block_{i}", nn.Identity())
        self.taps.add_module("merger", nn.Identity())
        self._B = None
        for i, blk in enumerate(visual.blocks):
            blk.register_forward_hook(self._make_hook(f"block_{i}", True))
        visual.merger.register_forward_hook(self._make_hook("merger", False))

    def _make_hook(self, name, reorder):
        def hook(module, inp, out):
            tap = getattr(self.taps, name)
            if len(tap._forward_hooks) == 0:  # nobody listening
                return
            if isinstance(out, tuple):
                out = out[0]
            x = out.reshape(self._B, -1, out.shape[-1])
            if reorder:
                x = x[:, self.perm]
            tap(x.float())
        return hook

    def forward(self, x):
        B, N, D = x.shape
        assert N == self.g ** 2, f"expected {self.g**2} tokens per image, got {N}"
        self._B = B
        p = next(self.visual.parameters())
        x = x.reshape(B * N, D).to(device=p.device, dtype=self.dtype)
        grid = torch.tensor([[1, self.g, self.g]] * B, device=p.device)
        return self.visual(x, grid_thw=grid).pooler_output

class Qwen3VLVision(BaseModel):
    """
    Qwen3-VL vision encoder. Cache name encodes id, revision, image size, dtype.
    """
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct", revision=None, image_size=224,
                 dtype="float32", **kwargs):
        assert image_size % 32 == 0, "image_size must be a multiple of 32"
        self.model_id = model_id
        self.revision = revision or DEFAULT_REVISIONS.get(model_id)
        assert self.revision, f"no default revision for {model_id}; pass revision="
        self.image_size = image_size
        self.dtype_name = str(dtype).replace("torch.", "")
        self.dtype = getattr(torch, self.dtype_name)
        super().__init__(**kwargs)
        self.relevant_layer_names = [n for n in self.all_layer_names
                                     if n.startswith("taps.") and n != "taps"]
        self.irrelevant_layer_names = [n for n in self.all_layer_names
                                       if n not in self.relevant_layer_names]

    def _load_model(self):
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        cfg = AutoConfig.from_pretrained(self.model_id, revision=self.revision).vision_config
        cfg._attn_implementation = "sdpa"
        # Load only the vision tower weights ("model.visual.*") from the checkpoint
        # (single-file checkpoints only; sharded ones would need the index file).
        path = hf_hub_download(self.model_id, "model.safetensors", revision=self.revision)
        with safe_open(path, "pt") as f:
            sd = {k[len("model.visual."):]: f.get_tensor(k)
                  for k in f.keys() if k.startswith("model.visual.")}
        visual = Qwen3VLVisionModel(cfg)
        visual.load_state_dict(sd, strict=True)
        visual = visual.to(self.dtype).eval()
        processor = AutoImageProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.transforms = QwenImageTransform(processor, self.image_size)
        tag = self.model_id.split("/")[-1].lower().replace("-", "").replace("_", "")
        name = f"{tag}vis_rev{self.revision[:8]}_s{self.image_size}_{self.dtype_name}"
        return _Tower(visual, self.image_size, self.dtype), name

    def _hook_error_handling(self, hook_output, batch_size):
        # outputs are already (B, N, C); skip the dim-swapping heuristics
        assert hook_output.size(0) == batch_size, hook_output.shape
        return hook_output
