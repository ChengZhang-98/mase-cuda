import torch


def quantize1d_simulated(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert weights.ndim == 1, "Weights tensor must be 1D"
    width = 8
    assert group_size > 0, "Group size must be positive"
    numel = weights.numel()
    assert numel % group_size == 0, "Number of elements in the weights tensor must be divisible by the group size"

    num_groups = numel // group_size
    weights = weights.bfloat16().float()
    w_g = weights.flatten().reshape(num_groups, group_size)

    sign = torch.where(w_g < 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))
    w_g = w_g.abs()
    w_g = torch.where(w_g < torch.finfo(torch.bfloat16).smallest_normal, 0.0, w_g)

    is_zeros = torch.all(w_g == 0.0, dim=1, keepdim=True)

    exponent = (w_g.view(torch.int32) >> 23) & 0xFF
    group_exp = exponent.max(dim=1, keepdim=True).values
    scales = group_exp.to(torch.uint8).flatten()
    group_exp = torch.where(is_zeros, 1, group_exp)  # avoids division by zero
    group_exp = (group_exp << 23).view(torch.float32)

    w_g = w_g / group_exp
    w_g = w_g * (2 ** (width - 2))
    w_g = w_g.round().clamp(0, 2 ** (width - 1) - 1)

    mantissa = (w_g.to(dtype=torch.int8) * sign).flatten()
    return mantissa, scales
