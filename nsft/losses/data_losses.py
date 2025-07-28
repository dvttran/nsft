import torch


class LpLoss:
    def __init__(self, p=2, mode="rel", reduce: str="mean"):
        self.p = p
        self.mode = mode
        self.reduce = reduce if isinstance(reduce, str) else reduce
        self.values = None

    def _return(self, values):
        if self.reduce == "mean":
            return values.mean()
        elif self.reduce == "sum":
            return values.sum()
        elif self.reduce == "none" or self.reduce is None:
            return values
        else:
            raise ValueError(f"Unknown reduce mode: {self.reduce}")

    def abs(self, x, y, start_dim=1, **kwargs):
        assert x.ndim == y.ndim, f"Dimension of x and y must match: {x.ndim} != {y.ndim}"
        if y.ndim == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        diff = torch.flatten(x, start_dim=start_dim) - torch.flatten(y, start_dim=start_dim)
        n = diff.shape[-1]

        self.values = (1. / n) * torch.norm(diff, p=self.p, dim=-1)
        return self._return(self.values)

    def rel(self, x, y, start_dim=1, **kwargs):
        assert x.ndim == y.ndim, f"Dimension of x and y must match: {x.ndim} != {y.ndim}"
        if y.ndim == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        diff = torch.flatten(x, start_dim=start_dim) - torch.flatten(y, start_dim=start_dim)
        y_norm = torch.norm(torch.flatten(y, start_dim=start_dim), p=self.p, dim=-1)

        self.values = torch.norm(diff, p=self.p, dim=-1) / y_norm
        return self._return(self.values)

    def __call__(self, x_pred, x, **kwargs):
        if self.mode == "abs":
            return self.abs(x_pred, x, **kwargs)
        elif self.mode == "rel":
            return self.rel(x_pred, x, **kwargs)


class AdaLpLoss:
    def __init__(self, p=2, mode="rel", adaptive=None, alpha=1.0, sigma=1.0, tau=0.04, **kwargs):
        self.p = p
        self.mode = mode

        if adaptive == "exp":
            self.factor_fn = lambda diff: self._exp_factor(diff, alpha=alpha, sigma=sigma)
        elif adaptive == "max":
            self.factor_fn = lambda diff: self._max_factor(diff, alpha=alpha, tau=tau)
        else:
            self.factor_fn = lambda diff: 1.0

        self.values = None

    def _exp_factor(self, diff, alpha=1.0, sigma=1.0):
        return alpha * torch.exp(torch.abs(diff) / sigma)    # min = alpha * exp(0) = alpha, max = alpha / exp(sigma)

    def _max_factor(self, diff, alpha=1.0, tau=0.04):
        # tau ≈ 10 / 255. ≈ 0.04
        return alpha * torch.exp(torch.nn.ReLU()(torch.abs(diff) - tau))

    def abs(self, x, y, start_dim=1, **kwargs):
        assert x.ndim == y.ndim, f"Dimension of x and y must match: {x.ndim} != {y.ndim}"
        if y.ndim == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        diff = torch.flatten(x, start_dim=start_dim) - torch.flatten(y, start_dim=start_dim)
        n = diff.shape[-1]

        factor = self.factor_fn(diff)
        weighted_diff = factor * diff
        self.values = (1. / n) * torch.norm(weighted_diff, p=self.p, dim=-1)
        return self.values.mean()

    def rel(self, x, y, start_dim=1, **kwargs):
        assert x.ndim == y.ndim, f"Dimension of x and y must match: {x.ndim} != {y.ndim}"
        if y.ndim == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        diff = torch.flatten(x, start_dim=start_dim) - torch.flatten(y, start_dim=start_dim)
        y_norm = torch.norm(torch.flatten(y, start_dim=start_dim), p=self.p, dim=-1)

        factor = self.factor_fn(diff)
        weighted_diff = factor * diff
        self.values = torch.norm(weighted_diff, p=self.p, dim=-1) / y_norm
        return self.values.mean()

    def __call__(self, x_pred, x, **kwargs):
        if self.mode == "abs":
            return self.abs(x_pred, x, **kwargs)
        elif self.mode == "rel":
            return self.rel(x_pred, x, **kwargs)
