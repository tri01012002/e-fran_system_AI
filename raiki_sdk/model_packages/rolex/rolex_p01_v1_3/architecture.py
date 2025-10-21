import timm
import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class EfficientNetLoG(nn.Module):
    """
    Returns logits [B, num_classes].
    Expects normalized RGB input [B,3,H,W] with ImageNet stats.
    If in_chans=4, builds a 4th LoG channel from de-normalized grayscale inside forward().
    """
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b4_ns",
        checkpoint: str = None,
        in_chans: int = 4,
        num_classes: int = 3,
        log_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes

        # Backbone
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans
        )

        # Buffers for (de)normalization — keep on CPU; they'll move with .to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        # LoG kernel (CPU-safe)
        if in_chans == 4:
            log_w, pad = self.create_log_kernel_cpu(log_sigma)
            self.register_buffer("log_w", log_w, persistent=False)  # [1,1,ks,ks]
            self.pad = pad
        else:
            self.log_w = None
            self.pad = 0

        # auto-load checkpoint
        if checkpoint:
            self.load_checkpoint(
                checkpoint,
                adapt_first_conv=True,
                ignore_head_if_mismatch=True,
            )

    @staticmethod
    def create_log_kernel_cpu(sigma: float = 1.0):
        r = int(2 * math.ceil(3 * sigma)) + 1
        r //= 2
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1),
            torch.arange(-r, r + 1),
            indexing='ij'
        )
        yy = yy.float(); xx = xx.float()
        s2 = sigma * sigma
        k = -1.0/(math.pi*s2*s2) * (1 - (xx*xx + yy*yy)/(2*s2)) * torch.exp(-(xx*xx + yy*yy)/(2*s2))
        k -= k.mean()
        return k.unsqueeze(0).unsqueeze(0), r  # [1,1,K,K], pad radius

    def load_checkpoint(
        self,
        path: str,
        adapt_first_conv: bool = True,
        ignore_head_if_mismatch: bool = True,
        map_location: str = "cpu",
    ):
        """Load a checkpoint, optionally adapting first conv (3→4 ch) and ignoring head mismatch."""
        ckpt = torch.load(path, map_location=map_location)

        # unwrap common formats
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt

        # strip 'module.' if saved with DataParallel
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

        model_sd = self.model.state_dict()

        # Handle classifier head mismatch
        head_w_key = "classifier.weight"
        head_b_key = "classifier.bias"
        if ignore_head_if_mismatch:
            if head_w_key in state and state[head_w_key].shape != model_sd[head_w_key].shape:
                state.pop(head_w_key, None)
            if head_b_key in state and state[head_b_key].shape != model_sd[head_b_key].shape:
                state.pop(head_b_key, None)

        # Adapt first conv (3→4) if needed (EfficientNet stem)
        if adapt_first_conv and self.in_chans == 4:
            stem_key = "conv_stem.weight"  # timm tf_efficientnet
            if stem_key in state:
                w = state[stem_key]  # [out, C_in, k, k]
                if w.ndim == 4 and w.shape[1] == 3 and model_sd[stem_key].shape[1] == 4:
                    # expand to 4 channels by mean of RGB for 4th channel
                    extra = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                    w4 = torch.cat([w, extra], dim=1)    # [out,4,k,k]
                    state[stem_key] = w4

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[EfficientNetLoG] load_state_dict: missing={missing}, unexpected={unexpected}")

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is normalized RGB
        if self.in_chans == 4:
            x0 = x * self.std + self.mean           # back to [0,1]
            xg = x0.mean(1, keepdim=True)           # grayscale
            lg = F.conv2d(xg, self.log_w, padding=self.pad)
            # per-sample min-max normalize
            lg_min = lg.amin(dim=(2,3), keepdim=True)
            lg_max = lg.amax(dim=(2,3), keepdim=True)
            lg = (lg - lg_min) / (lg_max - lg_min + 1e-8)
            x = torch.cat([x, lg], dim=1)           # [B,4,H,W]
        return self.model(x)                         # logits