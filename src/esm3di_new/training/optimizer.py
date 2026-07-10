import torch


class Lion(torch.optim.Optimizer):
    """Lion optimizer (EvoLved Sign Momentum)."""

    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0: state['exp_avg'] = torch.zeros_like(p)

                exp_avg, beta1, beta2 = state['exp_avg'], *group['betas']
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])

                update = exp_avg * beta1 + p.grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss