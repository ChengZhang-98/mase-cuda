import time
import logging
import random
import tabulate
import numpy as np
import pytest
import ml_dtypes
import torch
from mase_cuda.constants import MASE_CUDA_ROOT_PATH

from mase_cuda.mxint8.dequantize import dequantize1d, dequantize1d_simulated
from mase_cuda.mxint8.quantize import quantize1d_simulated
from mase_cuda.utils import seed_everything

logger = logging.getLogger(__name__)

seed_everything(42)


def test_ext_dequantize1d():
    group_sizes = [8, 16, 32, 64, 128, 256, 512]
    num_elements = [1024, 2048, 4096, 8192, 16384]
    num_random_tests = 100

    for group_size in group_sizes:
        for _ in range(num_random_tests):
            for m in num_elements:
                # m = 4096
                # m = 8192
                num_groups = m // group_size
                x = torch.empty(m, dtype=torch.uint8).random_(0, 256)
                scales = torch.empty(num_groups, dtype=torch.uint8).random_(0, 255)  # max=254 to avoid NaN
                scales_dup = scales.repeat_interleave(group_size)

                # view as uint16 to avoid NaN comparison
                out_cpu = dequantize1d(x, scales, group_size)
                out_ref = dequantize1d_simulated(x, scales, group_size)

                # find mismatch idx
                if not torch.equal(out_cpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_cpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_cpu[mismatch_idx]: {out_cpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                out_gpu = dequantize1d(x.cuda(), scales.cuda(), group_size).cpu()

                if not torch.equal(out_gpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_gpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_gpu[mismatch_idx]: {out_gpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                assert torch.equal(out_cpu, out_ref)
                assert torch.equal(out_gpu, out_ref)
    logger.info("test_ext_dequantize1d: PASS")


def test_ext_dequantize1d_predication_fast():
    def get_num_elements(group_size):
        yield group_size
        yield group_size * 2
        yield group_size * 16
        yield group_size * 32
        yield group_size * random.randint(1, 1024)

    group_sizes = [8, 16, 32, 64, 128]
    num_random_tests = 10

    for group_size in group_sizes:
        for _ in range(num_random_tests):
            for m in get_num_elements(group_size):
                num_groups = m // group_size
                x = torch.empty(m, dtype=torch.uint8).random_(0, 256)
                scales = torch.empty(num_groups, dtype=torch.uint8).random_(0, 255)  # max=254 to avoid NaN
                scales_dup = scales.repeat_interleave(group_size)

                # view as uint16 to avoid NaN comparison
                out_cpu = dequantize1d(x, scales, group_size)
                out_ref = dequantize1d_simulated(x, scales, group_size)

                # find mismatch idx
                if not torch.equal(out_cpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_cpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_cpu[mismatch_idx]: {out_cpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                out_gpu = dequantize1d(x.cuda(), scales.cuda(), group_size).cpu()

                if not torch.equal(out_gpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_gpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_gpu[mismatch_idx]: {out_gpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                assert torch.equal(out_cpu, out_ref)
                assert torch.equal(out_gpu, out_ref)
    logger.info("test_ext_dequantize1d: PASS")


@pytest.mark.slow
def test_ext_dequantize1d_predication():
    def get_num_elements(group_size):
        yield group_size
        yield group_size * 2
        yield group_size * 16
        yield group_size * 32
        yield group_size * random.randint(1, 1024)
        yield group_size * random.randint(1, 1024)
        yield group_size * random.randint(1, 1024 * 1024)

    group_sizes = [8, 16, 32, 64, 128, 256, 512]
    num_random_tests = 100

    for group_size in group_sizes:
        for _ in range(num_random_tests):
            for m in get_num_elements(group_size):
                num_groups = m // group_size
                x = torch.empty(m, dtype=torch.uint8).random_(0, 256)
                scales = torch.empty(num_groups, dtype=torch.uint8).random_(0, 255)  # max=254 to avoid NaN
                scales_dup = scales.repeat_interleave(group_size)

                # view as uint16 to avoid NaN comparison
                out_cpu = dequantize1d(x, scales, group_size)
                out_ref = dequantize1d_simulated(x, scales, group_size)

                # find mismatch idx
                if not torch.equal(out_cpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_cpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_cpu[mismatch_idx]: {out_cpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                out_gpu = dequantize1d(x.cuda(), scales.cuda(), group_size).cpu()

                if not torch.equal(out_gpu, out_ref):
                    mismatch_idx = torch.where(torch.logical_not(torch.eq(out_gpu, out_ref)))
                    logger.error(f"m: {m}, group_size: {group_size}")
                    logger.error(f"mismatch_idx: {mismatch_idx}")
                    logger.error(f"x[mismatch_idx]: {x[mismatch_idx]}")
                    logger.error(f"scales[mismatch_idx]: {scales_dup[mismatch_idx]}")
                    logger.error(f"out_gpu[mismatch_idx]: {out_gpu[mismatch_idx]}")
                    logger.error(f"out_ref[mismatch_idx]: {out_ref[mismatch_idx]}")

                assert torch.equal(out_cpu, out_ref)
                assert torch.equal(out_gpu, out_ref)
    logger.info("test_ext_dequantize1d: PASS")


@pytest.mark.slow
def test_ext_dequantize1d_latency():
    # measure latecy of dequantize1d on cpu and gpu

    group_sizes = [8, 16, 32, 64, 128, 256, 512]
    num_elements = [1024, 512 * 4096, 4096 * 4096, 4096 * 14336, 8192 * 28672]
    n_repeats = 20

    results = []
    for m in num_elements:
        for group_size in group_sizes:
            num_groups = m // group_size
            x = torch.empty(m, dtype=torch.uint8).random_(0, 256)
            scales = torch.empty(num_groups, dtype=torch.uint8).random_(0, 256)

            # cpu
            start = time.time()
            for _ in range(n_repeats):
                out_cpu = dequantize1d(x, scales, group_size)
            end = time.time()
            latency_cpu = (end - start) / n_repeats  # s

            # gpu
            x = x.cuda()
            scales = scales.cuda()
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
            for i in range(n_repeats):
                start_events[i].record()
                out_gpu = dequantize1d(x, scales, group_size)
                end_events[i].record()
            torch.cuda.synchronize()
            latencies_gpu = [start_events[i].elapsed_time(end_events[i]) for i in range(n_repeats)]
            latency_gpu = sum(latencies_gpu) / n_repeats / 1000  # ms -> s
            results.append([m, group_size, latency_cpu, latency_gpu, latency_cpu / latency_gpu])

    headers = ["m", "group_size", "latency_cpu", "latency_gpu", "GPU speedup"]
    logger.info("\n{}".format(tabulate.tabulate(results, headers=headers, tablefmt="pretty", floatfmt=".3E")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ext_dequantize1d()
    test_ext_dequantize1d_latency()
