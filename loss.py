import torch
from torch import nn


def _sigmoid_prob(z, gamma=1.0):
    return torch.sigmoid(-gamma * z)


def constsum_stats(loss_fn, C, zmin=-10.0, zmax=10.0, step=0.01, device="cpu"):
    z = torch.arange(zmin, zmax + step, step, device=device)
    s = loss_fn(z) + loss_fn(-z) - C
    a = s.abs()
    p90 = torch.quantile(a, 0.90).item()
    p99 = torch.quantile(a, 0.99).item()
    return {
        "max_abs": float(a.max().item()),
        "mean_abs": float(a.mean().item()),
        "p90": float(p90),
        "p99": float(p99),
    }


class mpan_loss(nn.Module):
    def __init__(self, class_prior, mode, nosiy=1.0,
                 gamma=1.0,
                 const_C=1.0,
                 scale_unhinged=True
                 ):
        super().__init__()
        self.mode = mode
        self.gamma = float(gamma)
        self.const_C = float(const_C)
        self.scale_unhinged = bool(scale_unhinged)

        pi = torch.as_tensor(class_prior, dtype=torch.float32)
        K = pi.numel()
        self.num_classes = K

        if isinstance(nosiy, (int, float)):
            r = torch.ones(K, dtype=torch.float32)
            r[-1] = float(nosiy)
        else:
            r = torch.as_tensor(nosiy, dtype=torch.float32)
            assert r.numel() == K, "nosiy must be scalar or length-K vector."

        hat = (pi * r).clamp_min(1e-12)
        hat = hat / hat.sum()

        self.true_prior = pi
        self.class_prior = hat.tolist()
        self.delta_prior = (hat - pi).tolist()

        self.loss_fn = self._build_base_loss()

    def _build_base_loss(self):
        g = self.gamma
        return lambda z: _sigmoid_prob(z, gamma=g)

    def make_loss(self, y):
        return self.loss_fn(y)

    def make_hinge_loss(self, y):
        return 0.5 * torch.clamp(1.0 - y, min=0.0, max=1.0)

    def check_const_sum(self, device=None):
        dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        return constsum_stats(self.loss_fn, C=self.const_C, device=dev)

    def make_multi_loss(self, y_pred, y_true, loss_id=0):
        if loss_id == 0:
            C = y_pred.shape[1]
            device = y_pred.device
            i = int(y_true)
            idx_i = torch.full((y_pred.size(0), 1), i, dtype=torch.long, device=device)
            pos = y_pred.gather(1, idx_i)
            mask_neg = torch.ones(C, dtype=torch.bool, device=device); mask_neg[i] = False
            neg = y_pred[:, mask_neg]
            Loss = self.make_loss(pos) + (1.0 / (C - 1)) * self.make_loss(-neg)
            return Loss.mean()
        else:
            raise NotImplementedError

    def forward(self, pred, y):
        device = pred.device
        K = self.num_classes
        prior = torch.as_tensor(self.class_prior, dtype=torch.float32, device=device)

        total = torch.tensor(0.0, dtype=torch.float32, device=device)
        N = pred.size(0)
        for i in range(K):
            idx = (y == i).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0: continue
            temp_pred = pred.index_select(0, idx)
            col_i = torch.full((temp_pred.size(0), 1), i, dtype=torch.long, device=device)
            gi = self.make_loss(temp_pred.gather(1, col_i))
            gk = self.make_loss(-temp_pred[:, [-1]])

            if i != K - 1:
                total = total + (float(idx.numel()) / float(N)) * torch.mean(gi + gk)
            else:
                vals, _ = torch.max(temp_pred[:, :K - 1], dim=1)
                gi_last = self.make_loss(-vals.unsqueeze(1))
                gk_last = self.make_loss(temp_pred[:, [-1]])
                core = torch.mean(gi_last + gk_last) - (1.0 - prior[i])
                if self.mode == 'ABS':
                    total = total + (float(idx.numel()) / float(N)) * torch.abs(core)
                elif self.mode == 'NN':
                    total = total + (float(idx.numel()) / float(N)) * torch.clamp(core, min=0.0)
                elif self.mode == 'URE':
                    total = total + (float(idx.numel()) / float(N)) * core
        los = total
        return los
