import math
import torch

from .quantize import quantize1d_simulated
from .dequantize import dequantize1d, dequantize1d_simulated


class PackedWeight:
    def __init__(self, shape: tuple[int], group_size: int, device=None, dtype=torch.bfloat16):
        numel = math.prod(shape)
        assert numel % group_size == 0, "Number of elements in the weights tensor must be divisible by the group size"
        self.shape = shape
        self.numel = numel
        self.group_size = group_size
        self.dtype = dtype

        self.weight = torch.empty(self.numel, device=device, dtype=torch.uint8)
        self.scales = torch.empty(self.numel // group_size, device=device, dtype=torch.uint8)

    def unpack(self) -> torch.Tensor:
        try:
            return self.unpack_accelerated()
        except NotImplementedError:
            return self.unpack_simulated()

    def unpack_accelerated(self) -> torch.Tensor:
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()

        w = dequantize1d(self.weight, self.scales, self.group_size).reshape(self.shape).to(self.dtype)
        return w

    def unpack_simulated(self) -> torch.Tensor:
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()

        w = dequantize1d_simulated(self.weight, self.scales, self.group_size).reshape(self.shape).to(self.dtype)
        return w

    @classmethod
    def pack(cls, weights: torch.Tensor, group_size: int) -> "PackedWeight":
        try:
            return cls.pack_accelerated(weights, group_size)
        except NotImplementedError:
            return cls.pack_simulated(weights, group_size)

    @classmethod
    def pack_accelerated(cls, weights: torch.Tensor, group_size: int) -> "PackedWeight":
        raise NotImplementedError

    @classmethod
    def pack_simulated(cls, weights: torch.Tensor, group_size: int) -> "PackedWeight":
        weights = weights.contiguous()
        device = weights.device
        ori_shape = weights.size()
        ori_dtype = weights.dtype
        weights = weights.flatten()
        w, s = quantize1d_simulated(weights, group_size)
        packed = cls(ori_shape, group_size, device, ori_dtype)
        packed.weight = w
        packed.scales = s
        return packed

    @property
    def nbytes(self) -> float:
        return self.weight.nbytes + self.scales.nbytes


class QLinearPacked(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.bfloat16,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.packed_weight = PackedWeight((out_features, in_features), group_size, device=device, dtype=dtype)
        if bias:
            self.packed_bias = PackedWeight((out_features,), group_size, device=device, dtype=dtype)
        else:
            self.packed_bias = None

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            input, self.packed_weight.unpack(), self.packed_bias.unpack() if self.packed_bias is not None else None
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.packed_bias is not None}, group_size={self.packed_weight.group_size}, nbits={self.nbits}"

    @property
    def nbits(self) -> float:
        nbytes = self.packed_weight.nbytes
        if self.packed_bias is not None:
            nbytes += self.packed_bias.nbytes

        numels = self.in_features * self.out_features
        if self.packed_bias is not None:
            numels += self.out_features
        return nbytes * 8 / numels

    @classmethod
    def build_from_linear(cls, linear: torch.nn.Linear, group_size: int = 16) -> "QLinearPacked":
        device = linear.weight.device
        qlinear = cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None, device=device, group_size=group_size
        )
        qlinear.packed_weight = PackedWeight.pack_simulated(linear.weight, group_size)
        if linear.bias is not None:
            qlinear.packed_bias = PackedWeight.pack_simulated(linear.bias, group_size)
        return qlinear
