# from here https://github.com/davda54/sam
import torch


__all__ = ['SAM', 'TopKSAM']


class SAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class TopKSAM(SAM):
    
    def __init__(
        self, 
        params, 
        base_optimizer, 
        rho=0.05, 
        adaptive=False, 
        topk=0.0, 
        global_sparsity=False, 
        **kwargs
    ):
        super(TopKSAM, self).__init__(params, base_optimizer, rho, adaptive, **kwargs)
        self.global_sparsity = global_sparsity
        self.topk = topk

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        # make a step w -> w + e(w)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  
        # prune topk
        if self.global_sparsity:
            # collect global statistic
            param_list = []
            for group in self.param_groups:
                for p in group["params"]:
                    param_list.append(p.view(-1))
            param_list = torch.cat(param_list, dim=0)
            # find global threshold
            n_nonzero = int((1 - self.topk) * len(param_list))
            threshold = torch.topk(param_list.abs(), k=n_nonzero)[0][-1]
            # prune weights
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = torch.where(p.abs() > threshold, p, torch.zeros_like(p))
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    # find local threshold
                    n_nonzero = int((1 - self.topk) * p.numel())
                    threshold = torch.topk(p.view(-1).abs(), k=n_nonzero)[0][-1]
                    p.data = torch.where(p.abs() > threshold, p, torch.zeros_like(p))

        if zero_grad: 
            self.zero_grad()
