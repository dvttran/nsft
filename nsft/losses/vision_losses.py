from .data_losses import AdaLpLoss
from kornia.filters import spatial_gradient
from typing import List


def _reshape_image(image):
    # image (B, H, W, C)
    assert image.ndim == 4, f"Image must have 4 dimensions: image.shape = {image.shape}"
    image = image.contiguous()
    image = image.permute(0, 3, 1, 2)  # (B, C, H, W)
    return image


class RGBLoss:
    def __init__(self, p=1, mode="abs", **kwargs):
        self.lp_loss = AdaLpLoss(p=p, mode=mode, **kwargs)

    def __call__(self, render_rgb, target_rgb, **kwargs):
        return self.lp_loss(render_rgb, target_rgb)
        # return torch.abs(render_rgb - target_rgb).mean()


class SilLoss:
    def __init__(self, p=1, mode="abs", **kwargs):
        self.lp_loss = AdaLpLoss(p=p, mode=mode, **kwargs)

    def __call__(self, render_sil, target_sil, **kwargs):
        return self.lp_loss(render_sil, target_sil)
        # return torch.abs(render_sil - target_sil).mean()


class ImageDerivativeLoss:
    def __init__(self, p: int = 1, mode: str = "abs", kernel: str = "sobel", orders: List[int] = (1, ), **kwargs):
        self.lp_loss = AdaLpLoss(p=p, mode=mode, **kwargs)
        assert kernel == "sobel" or kernel == "diff", f"Kernel {kernel} not supported"
        for order in orders:
            self.filters = [lambda x: spatial_gradient(x, order=order, mode=kernel) for order in orders]

    def _apply_filter(self, img, **kwargs):
        # image (B, H, W, C)
        assert img.ndim == 4, f"Image must have 4 dimensions: {img.ndim}"
        # img = img.permute(0, 2, 3, 1)  # (B, C, H, W)
        img = img.contiguous()
        img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
        deriv_img = [filter_fn(img) for filter_fn in self.filters]

        return deriv_img

    def __call__(self, render_rgb, target_rgb, **kwargs):
        deriv_render_img = self._apply_filter(render_rgb)
        deriv_target_img = self._apply_filter(target_rgb)

        loss = 0.0
        for order in range(len(deriv_render_img)):
            loss += self.lp_loss(deriv_render_img[order], deriv_target_img[order])

        return loss
