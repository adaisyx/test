import math
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('RAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            radam(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'])
        return loss


def radam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        # if weight_decay != 0:
        #     param.add_(param, alpha=-weight_decay*lr)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = exp_avg / bias_correction1

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

        if rho_t > 5.:
            # Compute the variance rectification term and update parameters accordingly
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq.sqrt().add_(eps)

            param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)
        else:
            param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)

