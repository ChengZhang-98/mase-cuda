import time
import logging
import random
import tabulate
import numpy as np
import torch

from mase_cuda.utils import seed_everything
from mase_cuda.mxint8.linear import PackedWeight, QLinearPacked

logger = logging.getLogger(__name__)
seed_everything(42)


def test_packed_weight():
    num_random_tests = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes = [(128,), (32, 32), (1024, 1024)]
    group_sizes = [4, 8, 32, 64]

    rows = []
    rows_sim = []
    for shape in shapes:
        for group_size in group_sizes:
            avg_error = 0
            avg_error_sim = 0
            for _ in range(num_random_tests):
                w = 100 * torch.rand(shape, device=device)
                packed_w = PackedWeight.pack_simulated(w, group_size)

                w_unpacked_sim = packed_w.unpack_simulated()
                error_sim = torch.abs(w - w_unpacked_sim).mean().item()
                avg_error_sim += error_sim

                w_unpacked = packed_w.unpack()
                error = torch.abs(w - w_unpacked).mean().item()
                avg_error += error
                assert torch.all(
                    w_unpacked == w_unpacked_sim
                ), f"Mismatch between simulated and cuda-acclerated unpacked weights"
                if not (error < 0.5):
                    logger.warning(
                        f"Shape: {shape}, Group Size: {group_size}, Error: {error}, Error Simulated: {error_sim}"
                    )

            avg_error /= num_random_tests
            avg_error_sim /= num_random_tests
            rows.append([shape, group_size, avg_error])
            rows_sim.append([shape, group_size, avg_error_sim])

    headers = ["Shape", "Group Size", "MeanAbsError"]
    table_sim = tabulate.tabulate(rows, headers=headers, tablefmt="pipe")
    logger.info(f"Simulated\n{table_sim}")
    table = tabulate.tabulate(rows_sim, headers=headers, tablefmt="pipe")
    logger.info(f"Simulated\n{table}")


def test_qlinear_init():
    in_features = 512
    out_features = 256
    group_size = 32
    dtypes = [torch.bfloat16, torch.float32, torch.float16]
    devices = [torch.device("cuda"), torch.device("cpu")]
    enable_bias = [True, False]
    for dtype in dtypes:
        for device in devices:
            for bias in enable_bias:
                fc = QLinearPacked(
                    in_features, out_features, bias=True, device=device, dtype=dtype, group_size=group_size
                )
                x = torch.randn(2, in_features, device=device, dtype=dtype)
                with torch.no_grad():
                    y = fc(x)
                assert y.shape == (2, out_features)


def test_qlinear_build():
    in_features = 512
    out_features = 256
    group_size = 32
    batch_size = 16
    dtypes = [torch.bfloat16, torch.float32, torch.float16]
    devices = [torch.device("cuda"), torch.device("cpu")]
    enable_bias = [True, False]
    rows = []
    for dtype in dtypes:
        for device in devices:
            for bias in enable_bias:
                fc = torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
                qfc = QLinearPacked.build_from_linear(fc, group_size=group_size)
                x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
                with torch.no_grad():
                    y = qfc(x)
                    y_ref = fc(x)
                error = torch.abs(y - y_ref).mean().item()
                rows.append([dtype, device, bias, error])

    headers = ["dtype", "device", "bias", "MeanAbsError"]
    table = tabulate.tabulate(rows, headers=headers, tablefmt="pipe")
    logger.info(f"Output error\n{table}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_packed_weight()
