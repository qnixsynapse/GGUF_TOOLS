#!/usr/bin/env python3
"""
gguf_weight_compare_gpu.py - compare stored weights of GGUF files against a reference GGUF
using PyTorch device-accelerated ggml dequantization kernels and fused reductions.
"""
import argparse
import fnmatch
import re
import statistics as st
from collections import defaultdict

import numpy as np
import torch
from gguf import GGMLQuantizationType as T
from gguf import GGUFReader
from gguf.quants import dequantize as cpu_dequantize

# Optional MXFP4 enum lookup (depends on installed gguf-py version)
T_MXFP4 = getattr(T, "MXFP4", None)


# ---------------------------------------------------------------------------
# Backend & Device Management
# ---------------------------------------------------------------------------


def detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_backend(device_str: str, force_fp32_acc: bool = False):
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()

    # Check if backend natively supports float64 reductions
    acc_dtype = torch.float32 if force_fp32_acc else torch.float64
    if acc_dtype == torch.float64:
        try:
            _ = torch.zeros(2, dtype=torch.float64, device=device).sum()
        except (RuntimeError, TypeError):
            print(
                f"[backend] {device} does not support float64; falling back to float32 accumulators."
            )
            acc_dtype = torch.float32

    # Test if pinned memory is supported on this backend (CUDA/ROCm/XPU usually yes, MPS/CPU no)
    use_pin = False
    if device.type in ("cuda", "xpu", "hip"):
        try:
            _ = torch.empty(1, dtype=torch.uint8, pin_memory=True)
            use_pin = True
        except Exception:
            use_pin = False

    return device, acc_dtype, use_pin


def _writable(np_arr: np.ndarray) -> np.ndarray:
    """GGUFReader memory-maps files read-only; torch.from_numpy warns on (and mis-handles) such arrays."""
    return np_arr if np_arr.flags.writeable else np_arr.copy()


def _device_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class HostTransferBuffer:
    """
    Ring of reusable pinned host buffers for async H2D copies.

    A non_blocking copy from pinned memory only *enqueues* the DMA. The host buffer must not be
    overwritten until that copy has actually run, so every slot gets an event that is waited on
    before the slot is reused (or reallocated). A single shared buffer, as used previously, let the
    chunk for tensor B overwrite the chunk for tensor A before A's copy executed.
    """

    def __init__(self, device: torch.device, use_pin: bool, n_slots: int = 4):
        self.device = device
        self.use_pin = use_pin
        self.slots = [None] * n_slots
        self.events = [None] * n_slots
        self.next = 0
        self.event_cls = None
        if device.type == "cuda":
            self.event_cls = torch.cuda.Event
        elif (
            device.type == "xpu"
            and hasattr(torch, "xpu")
            and hasattr(torch.xpu, "Event")
        ):
            self.event_cls = torch.xpu.Event

    def _wait_slot(self, i: int):
        ev = self.events[i]
        if ev is not None:
            ev.synchronize()
            self.events[i] = None

    def to_device(self, np_arr: np.ndarray, view_dtype=torch.uint8) -> torch.Tensor:
        # np_arr is C-contiguous; reinterpret through uint8 so any element type (incl. uint16 for bf16) works
        if self.device.type == "cpu" or not self.use_pin:
            raw = torch.from_numpy(_writable(np_arr).view(np.uint8)).view(view_dtype)
            return (
                raw if self.device.type == "cpu" else raw.to(self.device)
            )  # blocking copy: safe

        nbytes = np_arr.nbytes
        i = self.next
        self.next = (self.next + 1) % len(self.slots)
        self._wait_slot(
            i
        )  # previous copy out of this slot must be finished before we overwrite it
        if self.slots[i] is None or self.slots[i].numel() < nbytes:
            self.slots[i] = torch.empty(
                max(nbytes, 16 * 1024 * 1024), dtype=torch.uint8, pin_memory=True
            )
        host = self.slots[i][:nbytes]
        host.numpy()[:] = np_arr.view(np.uint8).reshape(
            -1
        )  # numpy assignment: reads the read-only memmap, no torch warning
        out = (
            host.view(view_dtype)
            .reshape(np_arr.shape)
            .to(self.device, non_blocking=True)
        )
        if self.event_cls is not None:
            ev = self.event_cls()
            ev.record()
            self.events[i] = ev
        else:
            _device_sync(self.device)
        return out


# ---------------------------------------------------------------------------
# Torch-Accelerated GGML Dequantization Kernels (matching ggml-quants.c)
# ---------------------------------------------------------------------------

# Constant Lookup Table for MXFP4 (kvalues_mxfp4 in ggml-common.h: 2 * E2M1)
_MXFP4_LUT = {}


def get_mxfp4_lut(device: torch.device) -> torch.Tensor:
    if device not in _MXFP4_LUT:
        _MXFP4_LUT[device] = torch.tensor(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                8.0,
                12.0,
                0.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
                -8.0,
                -12.0,
            ],
            dtype=torch.float32,
            device=device,
        )
    return _MXFP4_LUT[device]


def _dequant_q4_0(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q4_0: fp16 d (2B), uint8 qs[16] -> 18B per 32 weights
    d = blocks[:, :, :2].contiguous().view(torch.float16).to(torch.float32)
    qs = blocks[:, :, 2:]
    lo = (qs & 0x0F).to(torch.int8) - 8
    hi = (qs >> 4).to(torch.int8) - 8
    return (d * torch.cat([lo, hi], dim=-1).to(torch.float32)).reshape(n_rows, -1)


def _dequant_q4_1(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q4_1: fp16 d (2B), fp16 m (2B), uint8 qs[16] -> 20B per 32 weights
    d = blocks[:, :, :2].contiguous().view(torch.float16).to(torch.float32)
    m = blocks[:, :, 2:4].contiguous().view(torch.float16).to(torch.float32)
    qs = blocks[:, :, 4:]
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    return (d * torch.cat([lo, hi], dim=-1) + m).reshape(n_rows, -1)


def _dequant_q5_0(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q5_0: fp16 d (2B), uint8 qh[4], uint8 qs[16] -> 22B per 32 weights
    d = blocks[:, :, :2].contiguous().view(torch.float16).to(torch.float32)
    qh = blocks[:, :, 2:6].contiguous().view(torch.int32)  # [n_rows, n_blocks, 1]
    qs = blocks[:, :, 6:]  # [n_rows, n_blocks, 16]
    shifts = torch.arange(16, device=blocks.device, dtype=torch.int32)
    xh_0 = ((qh >> shifts) << 4) & 0x10
    xh_1 = (qh >> (shifts + 12)) & 0x10
    lo = ((qs & 0x0F).to(torch.int32) | xh_0) - 16
    hi = ((qs >> 4).to(torch.int32) | xh_1) - 16
    return (d * torch.cat([lo, hi], dim=-1).to(torch.float32)).reshape(n_rows, -1)


def _dequant_q5_1(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q5_1: fp16 d (2B), fp16 m (2B), uint8 qh[4], uint8 qs[16] -> 24B per 32 weights
    d = blocks[:, :, :2].contiguous().view(torch.float16).to(torch.float32)
    m = blocks[:, :, 2:4].contiguous().view(torch.float16).to(torch.float32)
    qh = blocks[:, :, 4:8].contiguous().view(torch.int32)
    qs = blocks[:, :, 8:]
    shifts = torch.arange(16, device=blocks.device, dtype=torch.int32)
    xh_0 = ((qh >> shifts) << 4) & 0x10
    xh_1 = (qh >> (shifts + 12)) & 0x10
    lo = (qs & 0x0F).to(torch.int32) | xh_0
    hi = (qs >> 4).to(torch.int32) | xh_1
    return (d * torch.cat([lo, hi], dim=-1).to(torch.float32) + m).reshape(n_rows, -1)


def _dequant_q8_0(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q8_0: fp16 d (2B), int8 qs[32] -> 34B per 32 weights
    d = blocks[:, :, :2].contiguous().view(torch.float16).to(torch.float32)
    qs = blocks[:, :, 2:].view(torch.int8).to(torch.float32)
    return (d * qs).reshape(n_rows, -1)


def _dequant_q8_K(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q8_K: float d (4B), int8 qs[256], int16 bsums[16] (32B) -> 292B per 256 weights
    d = blocks[:, :, :4].contiguous().view(torch.float32)
    qs = blocks[:, :, 4:260].view(torch.int8).to(torch.float32)
    return (d * qs).reshape(n_rows, -1)


def _dequant_mxfp4(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_mxfp4: uint8 e (1B), uint8 qs[16] -> 17B per 32 weights
    e = blocks[:, :, :1].to(torch.int32)
    # ggml_e8m0_to_fp32_half(e): if e < 2 -> 2^(-127) * e, else 2^(e - 128)
    d = torch.where(
        e < 2,
        torch.ldexp(torch.ones_like(e, dtype=torch.float32), torch.full_like(e, -127))
        * e.to(torch.float32),
        torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 128),
    )
    qs = blocks[:, :, 1:]
    lut = get_mxfp4_lut(blocks.device)
    lo = lut[(qs & 0x0F).long()]
    hi = lut[(qs >> 4).long()]
    return (d * torch.cat([lo, hi], dim=-1)).reshape(n_rows, -1)


def _unpack_k4_scales(scales: torch.Tensor):
    """
    Vectorized equivalent of get_scale_min_k4(j, scales, &sc, &m) in ggml-quants.c.
    scales: [n_rows, n_blocks, 12] uint8 -> returns sc, m each of shape [n_rows, n_blocks, 8] float32.
    """
    sc_lo = scales[:, :, 0:4] & 63
    m_lo = scales[:, :, 4:8] & 63
    sc_hi = (scales[:, :, 8:12] & 0x0F) | ((scales[:, :, 0:4] >> 6) << 4)
    m_hi = (scales[:, :, 8:12] >> 4) | ((scales[:, :, 4:8] >> 6) << 4)
    sc = torch.cat([sc_lo, sc_hi], dim=-1).to(torch.float32)
    m = torch.cat([m_lo, m_hi], dim=-1).to(torch.float32)
    return sc, m


def _dequant_q4_K(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q4_K: fp16 d (2B), fp16 dmin (2B), uint8 scales[12], uint8 qs[128] -> 144B / 256 weights
    n_blocks = blocks.shape[1]
    d = blocks[:, :, 0:2].contiguous().view(torch.float16).to(torch.float32)
    dmin = blocks[:, :, 2:4].contiguous().view(torch.float16).to(torch.float32)
    sc, m = _unpack_k4_scales(blocks[:, :, 4:16])

    dl = (d * sc).reshape(n_rows, n_blocks, 4, 2, 1)
    ml = (dmin * m).reshape(n_rows, n_blocks, 4, 2, 1)

    qs = blocks[:, :, 16:144].reshape(n_rows, n_blocks, 4, 32)
    q = torch.stack([qs & 0x0F, qs >> 4], dim=3).to(
        torch.float32
    )  # [n_rows, n_blocks, 4, 2, 32]
    return (dl * q - ml).reshape(n_rows, -1)


def _dequant_q5_K(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q5_K: fp16 d (2B), fp16 dmin (2B), uint8 scales[12], uint8 qh[32], uint8 qs[128] -> 176B / 256
    n_blocks = blocks.shape[1]
    d = blocks[:, :, 0:2].contiguous().view(torch.float16).to(torch.float32)
    dmin = blocks[:, :, 2:4].contiguous().view(torch.float16).to(torch.float32)
    sc, m = _unpack_k4_scales(blocks[:, :, 4:16])

    dl = (d * sc).reshape(n_rows, n_blocks, 4, 2, 1)
    ml = (dmin * m).reshape(n_rows, n_blocks, 4, 2, 1)

    qh = blocks[:, :, 16:48].reshape(n_rows, n_blocks, 1, 32)
    shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8).reshape(
        1, 1, 4, 2, 1
    )
    qh_bits = ((qh.unsqueeze(3) >> shifts) & 1) << 4  # [n_rows, n_blocks, 4, 2, 32]

    qs = blocks[:, :, 48:176].reshape(n_rows, n_blocks, 4, 32)
    ql = torch.stack([qs & 0x0F, qs >> 4], dim=3)
    q = (ql | qh_bits).to(torch.float32)
    return (dl * q - ml).reshape(n_rows, -1)


def _dequant_q6_K(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q6_K: uint8 ql[128], uint8 qh[64], int8 scales[16], fp16 d (2B) -> 210B / 256 weights
    n_blocks = blocks.shape[1]
    ql = blocks[:, :, 0:128].reshape(n_rows, n_blocks, 2, 64)
    qh = blocks[:, :, 128:192].reshape(n_rows, n_blocks, 2, 32)
    sc = (
        blocks[:, :, 192:208]
        .view(torch.int8)
        .to(torch.float32)
        .reshape(n_rows, n_blocks, 2, 8)
    )
    d = (
        blocks[:, :, 208:210]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
        .unsqueeze(-1)
    )

    ql0 = ql[:, :, :, 0:32]
    ql1 = ql[:, :, :, 32:64]

    q1 = ((ql0 & 0x0F) | (((qh >> 0) & 3) << 4)).to(torch.int8) - 32
    q2 = ((ql1 & 0x0F) | (((qh >> 2) & 3) << 4)).to(torch.int8) - 32
    q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)).to(torch.int8) - 32
    q4 = ((ql1 >> 4) | (((qh >> 6) & 3) << 4)).to(torch.int8) - 32

    # [n_rows, n_blocks, 2, 4, 32] -> split 32 into (2, 16) to match 8 scales per 128-half
    q = (
        torch.stack([q1, q2, q3, q4], dim=3)
        .reshape(n_rows, n_blocks, 2, 8, 16)
        .to(torch.float32)
    )
    return (d.unsqueeze(-1) * sc.unsqueeze(-1) * q).reshape(n_rows, -1)


def _dequant_q2_K(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q2_K: uint8 scales[16], uint8 qs[64], fp16 d (2B), fp16 dmin (2B) -> 84B / 256 weights
    n_blocks = blocks.shape[1]
    scales = blocks[:, :, 0:16].reshape(n_rows, n_blocks, 2, 4, 2, 1)
    qs = blocks[:, :, 16:80].reshape(n_rows, n_blocks, 2, 1, 32)
    d = (
        blocks[:, :, 80:82]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
        .reshape(n_rows, n_blocks, 1, 1, 1, 1)
    )
    dmin = (
        blocks[:, :, 82:84]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
        .reshape(n_rows, n_blocks, 1, 1, 1, 1)
    )

    shifts = torch.tensor(
        [0, 2, 4, 6], device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 1, 4, 1)
    q = ((qs >> shifts) & 3).reshape(n_rows, n_blocks, 2, 4, 2, 16).to(torch.float32)

    dl = d * (scales & 0x0F).to(torch.float32)
    ml = dmin * (scales >> 4).to(torch.float32)
    return (dl * q - ml).reshape(n_rows, -1)


def _dequant_q3_K(blocks: torch.Tensor, n_rows: int) -> torch.Tensor:
    # struct block_q3_K: uint8 hmask[32], uint8 qs[64], uint8 scales[12], fp16 d (2B) -> 110B / 256 weights
    n_blocks = blocks.shape[1]
    hm = blocks[:, :, 0:32].reshape(n_rows, n_blocks, 1, 1, 32)
    qs = blocks[:, :, 32:96].reshape(n_rows, n_blocks, 2, 1, 32)
    sc_raw = blocks[:, :, 96:108]
    d = blocks[:, :, 108:110].contiguous().view(torch.float16).to(torch.float32)

    # Unpack 16 6-bit scales from 12 bytes (matches ggml dequantize_row_q3_K)
    a0 = sc_raw[:, :, 0:4]
    a1 = sc_raw[:, :, 4:8]
    t = sc_raw[:, :, 8:12]
    s0 = (a0 & 0x0F) | (((t >> 0) & 3) << 4)
    s1 = (a1 & 0x0F) | (((t >> 2) & 3) << 4)
    s2 = (a0 >> 4) | (((t >> 4) & 3) << 4)
    s3 = (a1 >> 4) | (((t >> 6) & 3) << 4)
    sc = (torch.cat([s0, s1, s2, s3], dim=-1).to(torch.int8) - 32).to(torch.float32)
    dl = (d * sc).reshape(n_rows, n_blocks, 2, 4, 2, 1)

    q_shifts = torch.tensor(
        [0, 2, 4, 6], device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 1, 4, 1)
    q2 = (qs >> q_shifts) & 3  # [n_rows, n_blocks, 2, 4, 32]

    h_shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8).reshape(
        1, 1, 2, 4, 1
    )
    hbit = ((hm >> h_shifts) & 1) ^ 1  # 1 if bit NOT set -> subtract 4
    q = (
        (q2.to(torch.int8) - (hbit.to(torch.int8) << 2))
        .reshape(n_rows, n_blocks, 2, 4, 2, 16)
        .to(torch.float32)
    )
    return (dl * q).reshape(n_rows, -1)


# Map GGMLQuantizationType -> (bytes_per_block, dequant_fn)
_GPU_DEQUANT_TABLE = {
    T.Q4_0: (18, _dequant_q4_0),
    T.Q4_1: (20, _dequant_q4_1),
    T.Q5_0: (22, _dequant_q5_0),
    T.Q5_1: (24, _dequant_q5_1),
    T.Q8_0: (34, _dequant_q8_0),
    T.Q2_K: (84, _dequant_q2_K),
    T.Q3_K: (110, _dequant_q3_K),
    T.Q4_K: (144, _dequant_q4_K),
    T.Q5_K: (176, _dequant_q5_K),
    T.Q6_K: (210, _dequant_q6_K),
    T.Q8_K: (292, _dequant_q8_K),
}
if T_MXFP4 is not None:
    _GPU_DEQUANT_TABLE[T_MXFP4] = (17, _dequant_mxfp4)


def dequantize_tensor_chunk(
    raw_np: np.ndarray, qtype: T, hbuf: HostTransferBuffer
) -> torch.Tensor:
    n_rows = raw_np.shape[0]
    if qtype == T.F32:
        return hbuf.to_device(raw_np.view(np.float32), torch.float32)
    if qtype == T.F16:
        return hbuf.to_device(raw_np.view(np.float16), torch.float16).to(torch.float32)
    if qtype == T.BF16:
        return hbuf.to_device(raw_np.view(np.uint16), torch.bfloat16).to(torch.float32)

    if qtype in _GPU_DEQUANT_TABLE:
        block_bytes, fn = _GPU_DEQUANT_TABLE[qtype]
        raw_gpu = hbuf.to_device(raw_np.view(np.uint8), torch.uint8)
        blocks = raw_gpu.reshape(n_rows, -1, block_bytes)
        return fn(blocks, n_rows)

    # Fallback only for non-linear LUT IQ* or TQ* formats
    deq_cpu = cpu_dequantize(raw_np, qtype).astype(np.float32, copy=False)
    return hbuf.to_device(deq_cpu, torch.float32)


def accumulate_chunk_stats(
    a: torch.Tensor,
    sr: torch.Tensor,
    b: torch.Tensor,
    acc_k: np.ndarray,
):
    """
    Compute 2D reductions on the GPU (in float32 or float64), then accumulate
    scalars in CPU float64. Uses direct residual vectors to prevent fp32
    catastrophic cancellation on XPU/MPS.
    """
    d = a - b
    er = torch.sum(d * d, dim=1)
    dr = torch.sum(a * b, dim=1)
    nr_ = torch.sum(b * b, dim=1)
    # <a - b, b> per row: used for cancellation-free fit1 residual
    dbr = torch.sum(d * b, dim=1)

    # Direct row-wise projection residual: ||a_r - (dr / nr_) * b_r||^2
    scale_r = torch.where(
        nr_ > 0, dr / torch.clamp(nr_, min=1e-30), torch.zeros_like(dr)
    )
    rres_vec = a - scale_r.unsqueeze(1) * b
    rres_r = torch.sum(rres_vec * rres_vec, dim=1)

    # Single transfer of 5 scalar chunk totals to CPU float64
    chunk_totals = (
        torch.stack(
            [
                torch.sum(sr),
                torch.sum(er),
                torch.sum(dr),
                torch.sum(nr_),
                torch.sum(rres_r),
                torch.sum(dbr),
            ]
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    acc_k += chunk_totals


def _db(sig: float, err: float) -> float:
    """10*log10(sig/err) that keeps NaN as NaN (a NaN used to print as inf) and 0-error as +inf."""
    if sig != sig or err != err:
        return float("nan")
    if err <= 0:
        return float("inf")
    if sig <= 0:
        return float("-inf")
    return float(10 * np.log10(sig / err))


# ---------------------------------------------------------------------------
# Reader & CLI Main
# ---------------------------------------------------------------------------


def load_tensors(path):
    r = GGUFReader(path, "r")
    return {t.name: t for t in r.tensors}


def rows_of(t):
    return int(t.shape[1]) if len(t.shape) >= 2 else 1


def read_rows_gpu(t, idx, is_contig: bool, hbuf: HostTransferBuffer) -> torch.Tensor:
    n_rows = rows_of(t)
    reshaped = t.data.reshape(n_rows, -1)
    if is_contig:
        sub = np.ascontiguousarray(reshaped[idx[0] : idx[-1] + 1])
    else:
        sub = np.ascontiguousarray(reshaped[idx])
    return dequantize_tensor_chunk(sub, t.tensor_type, hbuf)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ref", help="reference GGUF (signal)")
    ap.add_argument(
        "others", nargs="+", help="GGUF(s) to compare against the reference"
    )
    ap.add_argument(
        "--pattern",
        default="*.weight",
        help="fnmatch on tensor name (default *.weight)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=2048,
        help="evenly sample at most N rows per tensor (0 = all)",
    )
    ap.add_argument(
        "--chunk-rows", type=int, default=2048, help="rows per GPU chunk (default 2048)"
    )
    ap.add_argument(
        "--device",
        default=detect_default_device(),
        help="torch device (cuda, xpu, mps, cpu)",
    )
    ap.add_argument(
        "--fp32-acc",
        action="store_true",
        help="use float32 accumulators instead of float64",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="JIT-compile dequant & reduction kernels with torch.compile",
    )
    ap.add_argument(
        "--per-tensor", action="store_true", help="also print one line per tensor"
    )
    ap.add_argument(
        "--include-1d",
        action="store_true",
        help="also compare 1D tensors (norm weights, layer_output_scale); use with --pattern '*norm*'",
    )
    args = ap.parse_args()

    device, acc_dtype, use_pin = setup_backend(args.device, args.fp32_acc)
    hbuf = HostTransferBuffer(device, use_pin)

    if args.compile:
        for qtype, (bsize, fn) in list(_GPU_DEQUANT_TABLE.items()):
            _GPU_DEQUANT_TABLE[qtype] = (bsize, torch.compile(fn, dynamic=True))

    others = list(args.others)
    if (
        others
        and any(c in others[-1] for c in "*?[")
        and not others[-1].endswith(".gguf")
    ):
        args.pattern = others.pop()
    if not others:
        ap.error("need at least one file to compare against the reference")

    ref = load_tensors(args.ref)
    cmp_files = [(p, load_tensors(p)) for p in others]

    names = [
        n
        for n in ref
        if fnmatch.fnmatch(n, args.pattern)
        and (args.include_1d or len(ref[n].shape) >= 2)
        and all(n in c for _, c in cmp_files)
    ]
    names.sort(
        key=lambda n: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", n)]
    )
    print(
        f"{len(names)} tensors compared  (ref: {args.ref} | backend: {device} | acc: {acc_dtype} | pinned_dma: {use_pin})"
    )

    results = [defaultdict(list) for _ in cmp_files]
    types = [defaultdict(set) for _ in cmp_files]

    for n in names:
        tr = ref[n]
        nr = rows_of(tr)
        if args.max_rows and nr > args.max_rows:
            idx_all = np.unique(np.linspace(0, nr - 1, args.max_rows).astype(np.int64))
            is_contig = False
        else:
            idx_all = np.arange(nr, dtype=np.int64)
            is_contig = True

        cls = re.sub(r"blk\.\d+\.", "", n).replace(".weight", "")
        # Inside `for n in names:`
        # 6 float64 accumulators on CPU: [sig, err, dot, nx, rres, db_dot]
        acc = [np.zeros(6, dtype=np.float64) for _ in cmp_files]
        shapes_ok = [tuple(c[n].shape) == tuple(tr.shape) for _, c in cmp_files]

        for s in range(0, len(idx_all), args.chunk_rows):
            idx = idx_all[s : s + args.chunk_rows]
            a = read_rows_gpu(tr, idx, is_contig, hbuf).to(acc_dtype)
            sr = torch.sum(a * a, dim=1)

            for k, (_, c) in enumerate(cmp_files):
                if not shapes_ok[k]:
                    continue
                tc = c[n]
                b = read_rows_gpu(tc, idx, is_contig, hbuf).to(acc_dtype)
                accumulate_chunk_stats(a, sr, b, acc[k])

        line = [f"{n:42s}"]
        for k, (p, c) in enumerate(cmp_files):
            if not shapes_ok[k]:
                line.append("shape mismatch")
                continue
            sig, err, dot, nx, rres, db_dot = acc[k].tolist()

            sqnr = 10 * np.log10(sig / err) if err > 0 else float("inf")
            cos = dot / np.sqrt(sig * nx) if sig > 0 and nx > 0 else float("nan")
            rel = np.sqrt(err / sig) if sig > 0 else float("nan")
            gain = np.sqrt(nx / sig) if sig > 0 else float("nan")

            # Cancellation-free fit1 residual: ||a - alpha*b||^2 = ||a - b||^2 - <a - b, b>^2 / ||b||^2
            fit1_err = max(0.0, err - (db_dot**2) / nx) if nx > 0 else 0.0
            fit1 = (
                10 * np.log10(sig / fit1_err)
                if nx > 0 and fit1_err > 0
                else float("inf")
            )
            fitr = 10 * np.log10(sig / rres) if rres > 0 else float("inf")

            results[k][cls].append((sqnr, cos, rel, gain, fit1, fitr))
            types[k][cls].add(c[n].tensor_type.name)
            line.append(f"{c[n].tensor_type.name:>6s} {sqnr:6.1f}dB cos {cos:.5f}")
        if args.per_tensor:
            print("  |  ".join(line))

    for k, (p, _) in enumerate(cmp_files):
        print(f"\n=== {p}  vs reference ===")
        print(
            f"{'class':14s} {'n':>3s} {'type(s)':>10s} {'med SQNR':>9s} {'min SQNR':>9s} {'med cos':>9s} {'min cos':>9s} {'med relRMS':>10s} {'med gain':>9s} {'fit1 dB':>8s} {'fitRow dB':>9s}"
        )
        allv = []
        for cls in sorted(results[k]):
            v = results[k][cls]
            allv += v
            sq = [x[0] for x in v]
            co = [x[1] for x in v]
            re_ = [x[2] for x in v]
            print(
                f"{cls:14s} {len(v):3d} {','.join(sorted(types[k][cls])):>10s} "
                f"{st.median(sq):9.1f} {min(sq):9.1f} {st.median(co):9.5f} {min(co):9.5f} {st.median(re_):10.4f} "
                f"{st.median(x[3] for x in v):9.3f} {st.median(x[4] for x in v):8.1f} {st.median(x[5] for x in v):9.1f}"
            )
        if allv:
            print(
                f"{'ALL':14s} {len(allv):3d} {'':>10s} {st.median(x[0] for x in allv):9.1f} "
                f"{min(x[0] for x in allv):9.1f} {st.median(x[1] for x in allv):9.5f} "
                f"{min(x[1] for x in allv):9.5f} {st.median(x[2] for x in allv):10.4f} "
                f"{st.median(x[3] for x in allv):9.3f} {st.median(x[4] for x in allv):8.1f} {st.median(x[5] for x in allv):9.1f}"
            )


if __name__ == "__main__":
    main()
