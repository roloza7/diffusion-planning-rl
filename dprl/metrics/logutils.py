import torch


def grad_norm(model : torch.nn.Module, device : str = 'cpu') -> torch.Tensor:
    total_norm = torch.zeros((1,), device=device)
    for p in model.parameters():
        param_norm  = p.grad.data.norm(2)
        total_norm += param_norm
    total_norm = total_norm.sqrt()
    return total_norm

def get_n_param(model : torch.nn.Module) -> int:
    for name, module in model.named_children():
        print(f"[{name}] -> {sum(p.numel() for p in module.parameters())} Params")