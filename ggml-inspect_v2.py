#!/usr/bin/env python3
"""
ggml_quant_inspect.py - inspect any tensor of a GGUF file and measure how well
each GGML quantization format would represent it.

* Streams rows straight from disk with explicit reads (no mmap residency) on a
  background thread, so peak host RAM is ~ one chunk, independent of tensor size.
* All quantizer simulations and statistics run in PyTorch on the selected device:
  CUDA (incl. ROCm/HIP builds), Intel XPU, Apple MPS, and other backends that expose a
  torch device (npu, musa, hpu, mtia ...), or plain CPU.  --device auto picks the first
  available accelerator.  Device out-of-memory halves the chunk and retries automatically.
* Source tensor may be F32 / F16 / BF16 or any quantized type gguf-py can dequantize.
* Simulates (quantize -> dequantize), following the llama.cpp reference quantizers:
  MXFP4, NVFP4, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS,
  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K  (+ F16 / BF16 casts for reference).
  Round trips are simulated, not bit-packed, but include fp16 scale rounding and
  the 6-bit / 4-bit super-block scale quantization of the K-quants.
  Not covered: IQ1/IQ2/IQ3 (need lattice grids + importance matrix), TQ*, Q1_0.
* Without --imatrix, numbers match a plain `llama-quantize` without imatrix. With
  --imatrix FILE (llama.cpp .dat or GGUF imatrix) SQNR is weighted per input column by
  the calibration importance (the quantizers themselves are still the unweighted ones).
* --budget-gib / --budget-bpw / --budget-sweep: size-budget allocation with --recommend.
* Note: GPU results can differ from CPU in a handful of elements where a value sits
  exactly on a rounding boundary (fused multiply-add etc.); --selftest reports this.

Requires: torch>=2.0, numpy, gguf

Usage
  python ggml_quant_inspect.py model.gguf --list
  python ggml_quant_inspect.py model.gguf token_embd.weight
  python ggml_quant_inspect.py model.gguf "blk.*.ffn_down*.weight" --max-rows 2048 --device cuda
  python ggml_quant_inspect.py model.gguf token_embd.weight --max-rows 0 --formats all --device xpu
  python ggml_quant_inspect.py model.gguf --recommend --min-sqnr 22 --emit-cmd quant.sh
  python ggml_quant_inspect.py --selftest --device auto
"""
import argparse
import contextlib
import fnmatch
import gc
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    sys.exit("PyTorch is required: see https://pytorch.org/get-started/locally/")

from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import dequantize
from gguf.constants import GGML_QUANT_SIZES

# ============================================================================
# device handling
# ============================================================================
F32 = torch.float32
NPF32 = np.float32

# Probed in this order for --device auto ("cuda" also covers ROCm builds of torch).
DEVICE_ORDER = ("cuda", "xpu", "mps", "npu", "musa", "hpu", "mtia")
_PLUGINS = ("torch_npu", "torch_musa", "habana_frameworks.torch.core")


def _load_plugins():
    """Out-of-tree backends register their device type on import; ignore if absent."""
    import importlib

    for m in _PLUGINS:
        try:
            importlib.import_module(m)
        except Exception:  # noqa: BLE001
            pass


def _available(kind):
    mod = getattr(torch, kind, None)
    try:
        return bool(mod is not None and mod.is_available())
    except Exception:  # noqa: BLE001
        return False


def device_label(dev):
    mod = getattr(torch, dev.type, None)
    try:
        return f"{dev} ({mod.get_device_name(dev.index or 0)})"
    except Exception:  # noqa: BLE001
        return str(dev)


def sync(dev):
    f = getattr(getattr(torch, dev.type, None), "synchronize", None)
    if f is not None:
        try:
            f()
        except Exception:  # noqa: BLE001
            pass


def free_cache(dev):
    gc.collect()
    f = getattr(getattr(torch, dev.type, None), "empty_cache", None)
    if f is not None:
        try:
            f()
        except Exception:  # noqa: BLE001
            pass


def is_oom(ex):
    oom = getattr(torch, "OutOfMemoryError", None)
    if oom is not None and isinstance(ex, oom):
        return True
    m = str(ex).lower()
    return "memory" in m and ("out of" in m or "alloc" in m)


def resolve_device(spec):
    if spec in (None, "", "auto"):
        _load_plugins()
        for k in DEVICE_ORDER:
            if _available(k):
                return torch.device(k), True
        return torch.device("cpu"), True
    try:
        dev = torch.device(spec)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"bad --device {spec!r}: {e}")
    if dev.type not in ("cpu", "cuda", "xpu", "mps"):
        _load_plugins()
    if dev.type != "cpu" and not _available(dev.type):
        sys.exit(f"--device {spec}: backend '{dev.type}' is not available in this torch build")
    return dev, False


def probe_device(dev):
    """Run every format on a tiny tensor on `dev` and on CPU.  Raises if an op is missing;
    returns the list of formats whose output differs noticeably from the CPU result."""
    rng = np.random.default_rng(1)
    xc = torch.from_numpy((rng.standard_normal((8, 256)) * 0.02).astype(NPF32))
    xd = xc.to(dev)
    cpu = torch.device("cpu")
    bad = []
    for f in FORMATS.values():
        try:
            yc = f.fn(xc, Ctx(cpu))
            yd = f.fn(xd, Ctx(dev)).cpu()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{f.name}: {type(e).__name__}: {e}") from e
        if float((yc != yd).float().mean()) > 0.02:
            bad.append(f.name)
    return bad


def setup_device(spec):
    dev, auto = resolve_device(spec)
    if dev.type != "cpu":
        try:
            bad = probe_device(dev)
        except Exception as e:  # noqa: BLE001
            if not auto:
                sys.exit(f"device {dev} failed its self-check: {e}")
            print(f"Device {dev} failed its self-check ({e}); falling back to CPU", file=sys.stderr)
            dev = torch.device("cpu")
        else:
            if bad:
                print(
                    f"WARNING: on {dev} these formats deviate from CPU on >2% of elements: "
                    + ", ".join(bad),
                    file=sys.stderr,
                )
    print(f"Device: {device_label(dev)}   torch {torch.__version__}")
    return dev


# ============================================================================
# small numeric helpers (all operate on torch tensors)
# ============================================================================
_CONST = {}


def const(name, vals, dev):
    """Cached float32 constant table on `dev`."""
    k = (name, str(dev))
    t = _CONST.get(k)
    if t is None:
        t = torch.tensor(np.asarray(vals, dtype=NPF32), dtype=F32, device=dev)
        _CONST[k] = t
    return t


def f16r(a):
    """Round to fp16 and back (scales are stored as fp16 in most formats)."""
    return a.to(torch.float16).to(F32)


def inv(a):
    """1/a with 1/0 -> 0 (matches `id = d ? 1/d : 0` in ggml)."""
    nz = a != 0
    return torch.where(nz, 1.0 / torch.where(nz, a, 1.0), 0.0)


def signed_max(b):
    """Element with the largest |value| (keeps sign) along axis 1."""
    return b.gather(1, b.abs().argmax(1, keepdim=True))[:, 0]


def exp_bits(v):
    """floor(log2(v)) for positive normal float32 v, exact (read from the exponent bits)."""
    return (((v.contiguous().view(torch.int32) >> 23) & 0xFF) - 127).to(F32)


def pow2i(e):
    """2**e for integer-valued float e, exact, built from exponent bits."""
    e = e.clamp(-126, 127).to(torch.int32)
    return ((e + 127) << 23).view(F32)


def bincount(idx, n):
    idx = idx.reshape(-1)
    try:
        return torch.bincount(idx, minlength=n)
    except (RuntimeError, NotImplementedError):  # backend without bincount
        return torch.bincount(idx.cpu(), minlength=n).to(idx.device)


# ============================================================================
# quantizers: fn(x2d (n_rows, cols) float32 tensor, ctx) -> reconstruction, same shape
# ============================================================================
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=NPF32)
E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=NPF32)


def fp8_ue4m3(v):
    """Round non-negative v to the nearest unsigned E4M3 value (bias 7, max 448)."""
    v = v.clamp(max=448.0)
    e = exp_bits(v.clamp(min=2.0**-9)).clamp(min=-6)
    step = pow2i(e - 3)
    return (torch.round(v / step) * step).clamp(max=448.0)


# ---- trivial casts ---------------------------------------------------------
def q_f16(x, ctx):
    return x.to(torch.float16).to(F32)


def q_bf16(x, ctx):
    return x.to(torch.bfloat16).to(F32)  # round-to-nearest-even


# ---- legacy 32-element formats ---------------------------------------------
def q_q4_0(x, ctx):
    b = x.reshape(-1, 32)
    d = signed_max(b) / -8.0
    q = torch.floor(b * inv(d)[:, None] + 8.5).clamp(max=15)
    return ((q - 8) * f16r(d)[:, None]).reshape(x.shape)


def q_q5_0(x, ctx):
    b = x.reshape(-1, 32)
    d = signed_max(b) / -16.0
    q = torch.floor(b * inv(d)[:, None] + 16.5).clamp(max=31)
    return ((q - 16) * f16r(d)[:, None]).reshape(x.shape)


def _q_minmax(x, nlev):
    b = x.reshape(-1, 32)
    mn, mx = b.amin(1), b.amax(1)
    d = (mx - mn) / nlev
    q = torch.floor((b - mn[:, None]) * inv(d)[:, None] + 0.5).clamp(max=nlev)
    return (q * f16r(d)[:, None] + f16r(mn)[:, None]).reshape(x.shape)


def q_q4_1(x, ctx):
    return _q_minmax(x, 15)


def q_q5_1(x, ctx):
    return _q_minmax(x, 31)


def q_q8_0(x, ctx):
    b = x.reshape(-1, 32)
    d = b.abs().amax(1) / 127.0
    return (torch.round(b * inv(d)[:, None]) * f16r(d)[:, None]).reshape(x.shape)


# ---- FP4 microscaling formats ----------------------------------------------
def q_mxfp4(x, ctx):
    """OCP MXFP4: E2M1 elements, shared power-of-two (E8M0) scale per 32.
    ggml reference scale: 2^(floor(log2(amax)) - 2).  --opt-scale also tries +1."""
    dev = x.device
    tab, bounds = const("e2m1", E2M1, dev), const("e2m1_bounds", E2M1_BOUNDS, dev)
    b = x.reshape(-1, 32)
    a = b.abs()
    amax = a.amax(1)
    e0 = (exp_bits(torch.where(amax > 0, amax, 1.0)) - 2).clamp(-127, 127)

    def run(e):
        sc = pow2i(e)[:, None]
        idx = torch.searchsorted(bounds, (a / sc).contiguous())
        return torch.copysign(tab[idx], b) * sc

    y = run(e0)
    if ctx.opt_scale:
        y1 = run(e0 + 1)
        use1 = ((b - y1) ** 2).sum(1) < ((b - y) ** 2).sum(1)
        y = torch.where(use1[:, None], y1, y)
    y = torch.where((amax == 0)[:, None], 0.0, y)
    return y.reshape(x.shape)


def q_nvfp4(x, ctx):
    """NVFP4 as in ggml: E2M1 elements, one UE4M3 scale per 16 elements, no
    per-tensor scale. Scale = ue4m3(amax/6). (gguf-py can only dequantize this
    type, so the quantizer rule is simulated, not cross-checked.)"""
    dev = x.device
    tab, bounds = const("e2m1", E2M1, dev), const("e2m1_bounds", E2M1_BOUNDS, dev)
    b = x.reshape(-1, 16)
    a = b.abs()
    sc = fp8_ue4m3(a.amax(1) / 6.0)[:, None]
    q = tab[torch.searchsorted(bounds, (a * inv(sc)).contiguous())]
    return (torch.copysign(q, b) * sc).reshape(x.shape)


# ---- IQ4 (non-linear 4-bit codebook) ----------------------------------------
KV_IQ4 = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=NPF32,
)
KV_IQ4_MID = ((KV_IQ4[:-1] + KV_IQ4[1:]) / 2).astype(NPF32)


def _iq4_idx(al, mid):
    # ties -> upper, like best_index_int8
    return torch.searchsorted(mid, al.contiguous(), right=True)


def _iq4_search(b, ntry=7):
    """Per-32-block scale search from quantize_row_iq4_nl_impl (weights = x^2)."""
    kv, mid = const("kv_iq4", KV_IQ4, b.device), const("kv_iq4_mid", KV_IQ4_MID, b.device)
    w = b * b
    mxv = signed_max(b)
    zero = mxv.abs() < 1e-15
    mxs = torch.where(zero, 1.0, mxv)
    q = kv[_iq4_idx(b * inv(mxs / 127.0)[:, None], mid)]
    sqx, sq2 = (w * q * b).sum(1), (w * q * q).sum(1)
    d = torch.where(sq2 > 0, sqx / torch.where(sq2 > 0, sq2, 1.0), 0.0)
    best = d * sqx
    for itry in range(-ntry, ntry + 1):
        q = kv[_iq4_idx(b * ((itry - 127.0) / mxs)[:, None], mid)]
        sqx, sq2 = (w * q * b).sum(1), (w * q * q).sum(1)
        ok = (sq2 > 0) & (sqx * sqx > best * sq2)
        d = torch.where(ok, sqx / torch.where(sq2 > 0, sq2, 1.0), d)
        best = torch.where(ok, d * sqx, best)
    return torch.where(zero, 0.0, d)


def q_iq4_nl(x, ctx):
    dev = x.device
    kv, mid = const("kv_iq4", KV_IQ4, dev), const("kv_iq4_mid", KV_IQ4_MID, dev)
    b = x.reshape(-1, 32)
    d = _iq4_search(b)
    q = kv[_iq4_idx(b * inv(d)[:, None], mid)]
    return (f16r(d)[:, None] * q).reshape(x.shape)


def q_iq4_xs(x, ctx):
    dev = x.device
    kv, mid = const("kv_iq4", KV_IQ4, dev), const("kv_iq4_mid", KV_IQ4_MID, dev)
    sub = x.reshape(-1, 32)
    sc = _iq4_search(sub).reshape(-1, 8)
    msc = signed_max(sc)
    d_raw = -msc / 32.0
    l = torch.round(inv(d_raw)[:, None] * sc).clamp(-32, 31)
    dl = (f16r(d_raw)[:, None] * l).reshape(-1, 1)
    q = kv[_iq4_idx(sub * inv(dl), mid)]
    return (dl * q).reshape(x.shape)


# ---- K-quants ---------------------------------------------------------------
def _qkx2(x, w, nmax, rmin, rdelta, nstep, use_mad):
    """Vectorised make_qkx2_quants. x, w: (M, n). Returns (scale, -min) per row."""
    mn = x.amin(1).clamp(max=0)
    mx = x.amax(1)
    degenerate = mx == mn
    sum_w, sum_x = w.sum(1), (w * x).sum(1)
    rng = torch.where(degenerate, 1.0, mx - mn)
    iscale = nmax / rng
    scale = 1.0 / iscale
    l = torch.round(iscale[:, None] * (x - mn[:, None])).clamp(0, nmax)
    diff = scale[:, None] * l + mn[:, None] - x
    best = (w * (diff.abs() if use_mad else diff * diff)).sum(1)
    for s in range(nstep + 1):
        rng = torch.where(degenerate, 1.0, mx - mn)  # min is updated as better fits are found
        iscale = (rmin + rdelta * s + nmax) / rng
        l = torch.round(iscale[:, None] * (x - mn[:, None])).clamp(0, nmax)
        sum_l, sum_l2, sum_xl = (w * l).sum(1), (w * l * l).sum(1), (w * l * x).sum(1)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        Ds = torch.where(ok, D, 1.0)
        t_scale = (sum_w * sum_xl - sum_x * sum_l) / Ds
        t_min = (sum_l2 * sum_x - sum_l * sum_xl) / Ds
        pos = t_min > 0
        t_min = torch.where(pos, 0.0, t_min)
        t_scale = torch.where(pos, sum_xl / torch.where(sum_l2 > 0, sum_l2, 1.0), t_scale)
        diff = t_scale[:, None] * l + t_min[:, None] - x
        mad = (w * (diff.abs() if use_mad else diff * diff)).sum(1)
        better = ok & (mad < best) & ~degenerate
        best = torch.where(better, mad, best)
        scale = torch.where(better, t_scale, scale)
        mn = torch.where(better, t_min, mn)
    scale = torch.where(degenerate, 0.0, scale)
    return scale, -mn


def _q_kasym(x, sub, nmax, smax, rmin, rdelta, nstep, use_mad, wmode):
    """Q2_K / Q4_K / Q5_K: y = d*sc*q - dmin*m with 256-element super-blocks."""
    xb = x.reshape(-1, 256)
    N, ns = xb.shape[0], 256 // sub
    xs = xb.reshape(N * ns, sub)
    if wmode == "abs":
        w = xs.abs()
    else:
        av = torch.sqrt((xb * xb).sum(1) / 256.0)
        w = torch.repeat_interleave(av, ns)[:, None] + xs.abs()
    sc, mi = _qkx2(xs, w, nmax, rmin, rdelta, nstep, use_mad)
    sc, mi = sc.reshape(N, ns), mi.reshape(N, ns)
    max_s, max_m = sc.amax(1).clamp(min=0), mi.amax(1).clamp(min=0)
    ls = torch.round(
        torch.where(max_s > 0, smax / torch.where(max_s > 0, max_s, 1.0), 0.0)[:, None] * sc
    ).clamp(0, smax)
    lm = torch.round(
        torch.where(max_m > 0, smax / torch.where(max_m > 0, max_m, 1.0), 0.0)[:, None] * mi
    ).clamp(0, smax)
    d, dmin = f16r(max_s / smax), f16r(max_m / smax)
    dsc, dm = d[:, None] * ls, dmin[:, None] * lm
    xv = xb.reshape(N, ns, sub)
    q = torch.round((xv + dm[:, :, None]) * inv(dsc)[:, :, None]).clamp(0, nmax)
    return (dsc[:, :, None] * q - dm[:, :, None]).reshape(x.shape)


def q_q2_k(x, ctx):
    return _q_kasym(x, 16, 3, 15, -0.5, 0.1, 15, True, "abs")


def q_q4_k(x, ctx):
    return _q_kasym(x, 32, 15, 63, -1.0, 0.1, 20, False, "avx")


def q_q5_k(x, ctx):
    return _q_kasym(x, 32, 31, 63, -0.5, 0.1, 15, False, "avx")


def _make_qx(xs, nmax):
    """Vectorised make_qx_quants(rmse_type=1) for Q6_K. xs: (M, n) -> scale (M,)."""
    mxv = signed_max(xs)
    zero = mxv.abs() < 1e-15
    mxs = torch.where(zero, 1.0, mxv)
    w = xs * xs

    def trial(iscale):
        l = torch.round(iscale[:, None] * xs).clamp(-nmax, nmax - 1)
        return (w * xs * l).sum(1), (w * l * l).sum(1)

    sumlx, suml2 = trial(-nmax / mxs)
    scale = torch.where(suml2 > 0, sumlx / torch.where(suml2 > 0, suml2, 1.0), 0.0)
    best = scale * sumlx
    for i in range(-9, 10):
        if i == 0:
            continue
        sumlx, suml2 = trial(-(nmax + 0.1 * i) / mxs)
        ok = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        s2 = torch.where(suml2 > 0, suml2, 1.0)
        scale = torch.where(ok, sumlx / s2, scale)
        best = torch.where(ok, sumlx * sumlx / s2, best)
    return torch.where(zero, 0.0, scale)


def q_q6_k(x, ctx):
    xb = x.reshape(-1, 256)
    N = xb.shape[0]
    sc = _make_qx(xb.reshape(N * 16, 16), 32).reshape(N, 16)
    msc = signed_max(sc)
    iscale = torch.where(msc != 0, -128.0 / torch.where(msc != 0, msc, 1.0), 0.0)
    s8 = torch.round(iscale[:, None] * sc).clamp(-128, 127)
    d = f16r(inv(iscale))
    dsc = d[:, None] * s8
    xv = xb.reshape(N, 16, 16)
    q = torch.round(xv * inv(dsc)[:, :, None]).clamp(-32, 31)
    return (dsc[:, :, None] * q).reshape(x.shape)


def q_q3_k(x, ctx):
    """make_q3_quants(do_rmse=True): sequential refinement, vectorised across sub-blocks."""
    nmax = 4
    xb = x.reshape(-1, 256)
    N = xb.shape[0]
    xs = xb.reshape(N * 16, 16)
    mxv = signed_max(xs)
    zero = mxv.abs() < 1e-15
    mxs = torch.where(zero, 1.0, mxv)
    L = torch.round((-nmax / mxs)[:, None] * xs).clamp(-nmax, nmax - 1)
    w = xs * xs
    sumlx, suml2 = (w * xs * L).sum(1), (w * L * L).sum(1)
    for _ in range(5):
        for i in range(16):
            xi, wi, Li = xs[:, i], w[:, i], L[:, i].clone()
            slx, sl2 = sumlx - wi * xi * Li, suml2 - wi * Li * Li
            valid = slx > 0
            nl = torch.round(xi * sl2 / torch.where(valid, slx, 1.0)).clamp(-nmax, nmax - 1)
            slx2, sl22 = slx + wi * xi * nl, sl2 + wi * nl * nl
            acc = valid & (nl != Li) & (sl22 > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            L[:, i] = torch.where(acc, nl, Li)
            sumlx, suml2 = torch.where(acc, slx2, sumlx), torch.where(acc, sl22, suml2)
    sc = torch.where(suml2 > 0, sumlx / torch.where(suml2 > 0, suml2, 1.0), 0.0)
    sc = torch.where(zero, 0.0, sc).reshape(N, 16)
    msc = signed_max(sc)
    iscale = torch.where(msc != 0, -32.0 / torch.where(msc != 0, msc, 1.0), 0.0)
    s6 = torch.round(iscale[:, None] * sc).clamp(-32, 31)
    d = f16r(inv(iscale))
    dsc = d[:, None] * s6
    xv = xb.reshape(N, 16, 16)
    q = torch.round(xv * inv(dsc)[:, :, None]).clamp(-4, 3)
    return (dsc[:, :, None] * q).reshape(x.shape)


class Fmt:
    def __init__(self, name, block, bpw, fn, desc):
        self.name, self.block, self.bpw, self.fn, self.desc = name, block, bpw, fn, desc


FORMATS = {
    f.name: f
    for f in [
        Fmt("f16", 1, 16.0, q_f16, "IEEE half"),
        Fmt("bf16", 1, 16.0, q_bf16, "bfloat16 (RNE)"),
        Fmt("q8_0", 32, 8.5, q_q8_0, "8-bit, fp16 scale / 32"),
        Fmt("q6_k", 256, 6.5625, q_q6_k, "6-bit, int8 scales / 16"),
        Fmt("q5_k", 256, 5.5, q_q5_k, "5-bit, 6-bit scale+min / 32"),
        Fmt("q5_1", 32, 6.0, q_q5_1, "5-bit, fp16 scale+min / 32"),
        Fmt("q5_0", 32, 5.5, q_q5_0, "5-bit, fp16 scale / 32"),
        Fmt("q4_k", 256, 4.5, q_q4_k, "4-bit, 6-bit scale+min / 32"),
        Fmt("q4_1", 32, 5.0, q_q4_1, "4-bit, fp16 scale+min / 32"),
        Fmt("q4_0", 32, 4.5, q_q4_0, "4-bit, fp16 scale / 32"),
        Fmt("iq4_nl", 32, 4.5, q_iq4_nl, "4-bit non-linear codebook, fp16 scale / 32"),
        Fmt("nvfp4", 16, 4.5, q_nvfp4, "E2M1, UE4M3 scale / 16"),
        Fmt(
            "iq4_xs", 256, 4.25, q_iq4_xs, "4-bit non-linear codebook, 6-bit scale / 32"
        ),
        Fmt("mxfp4", 32, 4.25, q_mxfp4, "E2M1, E8M0 (2^k) scale / 32"),
        Fmt("q3_k", 256, 3.4375, q_q3_k, "3-bit, 6-bit scales / 16"),
        Fmt("q2_k", 256, 2.625, q_q2_k, "2-bit, 4-bit scale+min / 16"),
    ]
}
DEFAULT_FORMATS = [
    "mxfp4",
    "nvfp4",
    "iq4_nl",
    "iq4_xs",
    "q4_0",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
]


class Ctx:
    def __init__(self, dev, opt_scale=False):
        self.dev = dev
        self.opt_scale = opt_scale
        self.imp = None  # per-input-column importance (float32 tensor on dev, mean 1) or None
        # float64 reductions where supported cheaply; GPUs reduce in float32 (MPS/Intel
        # have no/slow fp64) and the per-row results are summed in float64 on the host.
        self.acc = torch.float64 if dev.type == "cpu" else torch.float32


# ============================================================================
# disk streaming
# ============================================================================
class TensorSource:
    """Reads row ranges with explicit file reads (no mmap pages kept resident)."""

    def __init__(self, path, t):
        self.t = t
        d = t.data  # np.memmap view: shape/dtype only, never sliced
        self.dtype = d.dtype
        self.raw_cols = int(d.shape[-1])
        self.n_rows = int(np.prod(d.shape[:-1])) if d.ndim > 1 else 1
        self.row_bytes = self.raw_cols * d.dtype.itemsize
        self.base = int(t.data_offset)  # absolute file offset in current gguf-py
        self.cols = int(t.shape[0])
        self.f = open(path, "rb", buffering=0)

    def close(self):
        self.f.close()

    def _read(self, off, nrows, out):
        mv = memoryview(out).cast("B")
        self.f.seek(off)
        got = 0
        while got < len(mv):
            n = self.f.readinto(mv[got:])
            if not n:
                raise IOError("unexpected EOF (wrong data offset?)")
            got += n

    def rows(self, ids):
        """ids: sorted int64 array of row indices -> float32 numpy (len(ids), cols)."""
        n = len(ids)
        raw = np.empty((n, self.raw_cols), dtype=self.dtype)
        if n == 1 or (ids[-1] - ids[0] == n - 1):
            self._read(self.base + int(ids[0]) * self.row_bytes, n, raw)
        else:
            for i, r in enumerate(ids):
                self._read(self.base + int(r) * self.row_bytes, 1, raw[i])
        return np.ascontiguousarray(self._to_f32(raw), dtype=NPF32)

    def _to_f32(self, raw):
        qt = self.t.tensor_type
        if qt == QT.F32:
            return raw
        if qt == QT.F16:
            return raw.astype(NPF32)
        if qt == QT.BF16:
            return (
                (raw.view(np.uint16).astype(np.uint32) << 16)
                .view(NPF32)
                .reshape(raw.shape[0], self.cols)
            )
        return (
            dequantize(raw, qt).astype(NPF32, copy=False).reshape(raw.shape[0], self.cols)
        )


# ============================================================================
# streaming statistics (device compute -> host accumulate)
# ============================================================================
class LogHist:
    """Log-spaced histogram of |x| (1/32-octave bins, ~2% accuracy) in O(1) memory."""

    LO, HI, PER = -40, 16, 32

    def __init__(self):
        self.n = (self.HI - self.LO) * self.PER
        self.counts = np.zeros(self.n, dtype=np.int64)
        self.zeros = 0

    def _bin(self, v):
        return int(np.clip((np.log2(v) - self.LO) * self.PER, 0, self.n - 1))

    def compute(self, a):
        """a: |x| tensor on device -> host int64 array [bin counts..., n_zero]."""
        idx = ((torch.log2(a.clamp(min=1e-30)) - self.LO) * self.PER).to(torch.int64)
        idx = idx.clamp(0, self.n - 1)
        cnt = bincount(idx, self.n)
        z = (a == 0).sum().reshape(1)
        return torch.cat([cnt, z]).cpu().numpy()

    def commit(self, r):
        z = int(r[-1])
        c = r[:-1].copy()
        c[0] -= z  # exact zeros were clamped into bin 0; they are counted separately
        self.zeros += z
        self.counts += c

    def total(self):
        return self.zeros + int(self.counts.sum())

    def percentile(self, p):
        target = p / 100.0 * self.total()
        if target <= self.zeros:
            return 0.0
        cum = self.zeros + np.cumsum(self.counts)
        i = min(int(np.searchsorted(cum, target)), self.n - 1)
        return float(2.0 ** (self.LO + (i + 1) / self.PER))

    def frac_above(self, v):
        return float(self.counts[self._bin(v) + 1 :].sum()) / max(self.total(), 1)


class RangeHist:
    STEP, MAXB = 0.05, 64.0

    def __init__(self):
        self.counts = np.zeros(int(self.MAXB / self.STEP) + 1, dtype=np.int64)

    def compute(self, ab):
        """ab: |x| reshaped (-1, 32) on device -> host counts of log2(max/min nonzero)."""
        L = len(self.counts)
        bmax = ab.amax(1)
        bmin = torch.where(ab > 0, ab, float("inf")).amin(1)
        ok = torch.isfinite(bmin) & (bmax > 0)
        r = torch.log2(torch.where(ok, bmax, 1.0) / torch.where(ok, bmin, 1.0))
        idx = (r / self.STEP).to(torch.int64).clamp(0, L - 1)
        idx = torch.where(ok, idx, L)  # invalid blocks go to a discarded extra bin
        return bincount(idx, L + 1).cpu().numpy()[:L]

    def commit(self, c):
        self.counts += c

    def percentile(self, p):
        cum = np.cumsum(self.counts)
        if cum[-1] == 0:
            return float("nan")
        return int(np.searchsorted(cum, p / 100.0 * cum[-1])) * self.STEP


class QuantAcc:
    def __init__(self, fmt, n_rows):
        self.fmt = fmt
        self.cos = np.empty(n_rows, dtype=NPF32)
        self.rel = np.empty(n_rows, dtype=NPF32)
        self.pos = 0
        self.sum_err = self.sum_x2 = 0.0
        self.max_abs_err = 0.0
        self.flushed = self.nonzero = self.fl_blocks = self.blocks = 0
        self.seconds = 0.0
        self.has_imp = False
        self.sum_werr = self.sum_wx2 = 0.0

    def compute(self, x, ctx):
        """Heavy part (runs on the device); mutates nothing, so a chunk can be retried."""
        dev, acc = x.device, ctx.acc
        sync(dev)
        t0 = time.time()
        y = self.fmt.fn(x, ctx)
        sync(dev)
        dt = time.time() - t0
        e = x - y
        e2 = e * e
        rows = [(x * y).sum(1, dtype=acc), (x * x).sum(1, dtype=acc),
                (y * y).sum(1, dtype=acc), e2.sum(1, dtype=acc)]
        if ctx.imp is not None:
            rows += [(e2 * ctx.imp).sum(1, dtype=acc), ((x * x) * ctx.imp).sum(1, dtype=acc)]
        R = torch.stack(rows).cpu().double().numpy()
        blk = min(max(self.fmt.block, 1), 32)
        nzm = x != 0
        fl = nzm & (y == 0)
        fb = fl.reshape(-1, blk).sum(1)
        sc = (
            torch.stack(
                [e.abs().max().to(F32), fl.sum().to(F32), nzm.sum().to(F32), (fb > blk // 2).sum().to(F32)]
            )
            .cpu().double().numpy()
        )
        return dict(R=R, sc=sc, dt=dt, nblocks=int(fb.numel()), n=int(x.shape[0]))

    def commit(self, r):
        R, sc, n = r["R"], r["sc"], r["n"]
        dot, nx, ny, err = R[0], R[1], R[2], R[3]
        sl = slice(self.pos, self.pos + n)
        self.cos[sl] = dot / np.sqrt(nx * ny + 1e-300)
        self.rel[sl] = err / (nx + 1e-300)
        self.pos += n
        self.sum_err += float(err.sum())
        self.sum_x2 += float(nx.sum())
        if R.shape[0] > 4:
            self.has_imp = True
            self.sum_werr += float(R[4].sum())
            self.sum_wx2 += float(R[5].sum())
        self.max_abs_err = max(self.max_abs_err, float(sc[0]))
        self.flushed += int(sc[1])
        self.nonzero += int(sc[2])
        self.fl_blocks += int(sc[3])
        self.blocks += r["nblocks"]
        self.seconds += r["dt"]

    def summary(self):
        c, r = self.cos[: self.pos], self.rel[: self.pos]
        plain_mse = self.sum_err / (self.sum_x2 + 1e-300)
        mse = self.sum_werr / (self.sum_wx2 + 1e-300) if self.has_imp else plain_mse
        return dict(
            weighted=self.has_imp,
            plain_sqnr_db=float(min(10 * np.log10(1.0 / (plain_mse + 1e-300)), 200.0)),
            fmt=self.fmt.name,
            bpw=self.fmt.bpw,
            rel_mse=mse,
            sqnr_db=float(
                min(10 * np.log10(1.0 / (mse + 1e-300)), 200.0)
            ),  # capped: 200 = lossless
            cos_mean=float(c.mean()),
            cos_median=float(np.median(c)),
            cos_p1=float(np.percentile(c, 1)),
            cos_min=float(c.min()),
            row_mse_mean=float(r.mean()),
            row_mse_p95=float(np.percentile(r, 95)),
            row_mse_max=float(r.max()),
            flushed_pct=100.0 * self.flushed / max(self.nonzero, 1),
            flushed_blocks_pct=100.0 * self.fl_blocks / max(self.blocks, 1),
            max_abs_err=self.max_abs_err,
            worst_rows=[(int(i), float(c[i])) for i in np.argsort(c)[:5]],
            seconds=self.seconds,
        )


# ============================================================================
# per-tensor analysis
# ============================================================================
def pick_rows(n_total, max_rows):
    if not max_rows or n_total <= max_rows:
        return np.arange(n_total, dtype=np.int64)
    return np.unique(np.linspace(0, n_total - 1, max_rows).astype(np.int64))


def inspect_tensor(path, t, fmts, args):
    dev = args.dev
    src = TensorSource(path, t)
    pool = ThreadPoolExecutor(1)  # reads the next chunk while the device works on this one
    try:
        cols, n_total = src.cols, src.n_rows
        exp_rows = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1
        if exp_rows != n_total:
            raise RuntimeError(f"row mismatch {n_total} vs {exp_rows}")
        ids = pick_rows(n_total, args.max_rows)
        n_rows = len(ids)
        rpc = max(1, args.chunk_elems // cols)

        usable = [f for f in fmts if cols % max(f.block, 1) == 0]
        skipped = [f.name for f in fmts if f not in usable]
        print(
            f"\n{'=' * 78}\n{t.name}  shape={[int(s) for s in t.shape]}  type={t.tensor_type.name}"
        )
        print(
            f"Rows analysed: {n_rows:,}/{n_total:,}   row length {cols}   "
            f"chunk {rpc} rows (~{rpc * cols * 4 / 2 ** 20:.0f} MiB per float32 temp on {dev})"
        )
        if skipped:
            print(
                f"Skipping (row length {cols} not a multiple of block): {', '.join(skipped)}"
            )

        hist, rhist = LogHist(), RangeHist()
        ctx = Ctx(dev, args.opt_scale)
        acc = ctx.acc
        imp, imp_reason = lookup_imp(args, t.name, cols)
        ctx.imp = None if imp is None else torch.from_numpy(imp).to(dev)
        col_ss = np.zeros(cols, dtype=np.float64)
        st = dict(
            s1=0.0, s2=0.0, s4=0.0, vmin=np.inf, vmax=-np.inf, count=0, nonfinite=0, done=0,
            col_amax=torch.zeros(cols, dtype=F32, device=dev),
        )
        accs = [QuantAcc(f, n_rows) for f in usable]
        if getattr(args, "imat", None) is not None:
            print(
                "SQNR metric: "
                + ("imatrix-weighted" if ctx.imp is not None else f"plain (imatrix fallback: {IMP_WHY.get(imp_reason, imp_reason)})")
            )
        do32 = cols % 32 == 0
        t_start = time.time()
        quiet = getattr(args, "quiet", False)

        def process(xn):
            """Compute everything for one chunk on the device; return a commit closure.
            Nothing global is touched until the closure runs, so an OOM retry is safe."""
            x = torch.from_numpy(xn).to(dev)
            bad = ~torch.isfinite(x)
            nbad = int(bad.sum())
            if nbad:
                x = torch.where(bad, 0.0, x)
            a = x.abs()
            x2 = x * x
            sc = torch.stack(
                [x.sum(dtype=acc), x2.sum(dtype=acc), (x2 * x2).sum(dtype=acc),
                 x.min().to(acc), x.max().to(acc)]
            ).cpu().double().numpy()
            css = x2.sum(0, dtype=acc).cpu().double().numpy()
            camax = a.amax(0)
            h = hist.compute(a)
            rh = rhist.compute(a.reshape(-1, 32)) if do32 else None
            del x2
            res = [q.compute(x, ctx) for q in accs]
            count, nrow = x.numel(), x.shape[0]

            def commit():
                st["s1"] += float(sc[0])
                st["s2"] += float(sc[1])
                st["s4"] += float(sc[2])
                st["vmin"] = min(st["vmin"], float(sc[3]))
                st["vmax"] = max(st["vmax"], float(sc[4]))
                st["count"] += count
                st["nonfinite"] += nbad
                col_ss[:] += css
                st["col_amax"] = torch.maximum(st["col_amax"], camax)
                hist.commit(h)
                if rh is not None:
                    rhist.commit(rh)
                for q, r in zip(accs, res):
                    q.commit(r)
                st["done"] += nrow

            return commit

        def run_piece(xn):
            nonlocal rpc
            try:
                commit = process(xn)
            except Exception as ex:  # noqa: BLE001
                if not is_oom(ex) or len(xn) == 1:
                    raise
                free_cache(dev)
                h = (len(xn) + 1) // 2
                rpc = max(1, min(rpc, h))
                print(f"\n  device OOM: retrying with {h}-row pieces", file=sys.stderr)
                run_piece(xn[:h])
                run_piece(xn[h:])
                return
            commit()
            if not quiet:
                print(
                    f"\r  progress {100 * st['done'] / n_rows:5.1f}%  ({time.time() - t_start:.0f}s)",
                    end="", file=sys.stderr, flush=True,
                )

        s = 0
        fut = pool.submit(src.rows, ids[0 : min(rpc, n_rows)])
        cur = min(rpc, n_rows)
        while s < n_rows:
            xn = fut.result()
            nxt_s = s + cur
            if nxt_s < n_rows:
                nxt_n = min(rpc, n_rows - nxt_s)
                fut = pool.submit(src.rows, ids[nxt_s : nxt_s + nxt_n])
            else:
                nxt_n = 0
            run_piece(xn)
            s, cur = nxt_s, nxt_n
        if not quiet:
            print("\r" + " " * 50 + "\r", end="", file=sys.stderr)

        count = st["count"]
        vmin, vmax, nonfinite = st["vmin"], st["vmax"], st["nonfinite"]
        col_amax = st["col_amax"].cpu().numpy()
        mean = st["s1"] / count
        var = max(st["s2"] / count - mean * mean, 0.0)
        std = var**0.5
        kurt = (
            (st["s4"] / count) / (var * var) - 3 if var > 0 else float("nan")
        )  # raw 4th moment (mean ~ 0)
        p99, p999 = hist.percentile(99.0), hist.percentile(99.9)
        amax = max(abs(vmin), abs(vmax))
        rms_cols = np.sqrt(col_ss / (count / cols))

        print(f"\n--- Weight Distribution ({count:,} elements) ---")
        print(f"Min / Max:        {vmin:+.6f} / {vmax:+.6f}")
        print(f"Mean / Std Dev:   {mean:+.6f} / {std:.6f}")
        print(f"99.0% Abs:        {p99:.6f}  (~{p99 / std:.1f} sigma)")
        print(f"99.9% Abs:        {p999:.6f}  (~{p999 / std:.1f} sigma)")
        print(f"Max Abs:          {amax:.6f}  (~{amax / std:.1f} sigma)")
        print(f"Outlier Ratio:    {amax / (p999 + 1e-8):.2f}x   (max / p99.9)")
        print(f"Excess kurtosis:  {kurt:.2f}   (0 = Gaussian)")
        print(
            "Elements beyond:  "
            + "  ".join(
                f"{k}σ {100 * hist.frac_above(k * std):.4f}%" for k in (4, 6, 10, 50)
            )
        )
        print(f"Exact zeros:      {100 * hist.zeros / count:.3f}%")
        if nonfinite:
            print(f"WARNING: {nonfinite} non-finite values replaced with 0")
        top = np.argsort(col_amax)[::-1][:5]
        print(
            "Top cols by max|x|: "
            + ", ".join(f"{int(i)}:{col_amax[i]:.4f}" for i in top)
            + f"   (median col max {np.median(col_amax):.4f})"
        )
        topr = np.argsort(rms_cols)[::-1][:5]
        print(
            "Top cols by RMS:    "
            + ", ".join(f"{int(i)}:{rms_cols[i]:.4f}" for i in topr)
            + f"   (median col RMS {np.median(rms_cols):.4f})"
        )

        if do32:
            print(
                "\n--- 32-element block dynamic range (log2 max|x| / min nonzero |x|) ---"
            )
            print(
                f"Median {rhist.percentile(50):.1f} bits | p95 {rhist.percentile(95):.1f} | "
                f"p99 {rhist.percentile(99):.1f}   (E2M1 spans ~3.6 bits above its smallest level)"
            )

        results = [a.summary() for a in accs]
        for r in results:
            print(
                f"\n--- {r['fmt'].upper()} round trip ({r['bpw']} bpw, {FORMATS[r['fmt']].desc}) ---"
            )
            print(
                f"Global rel. MSE {r['rel_mse']:.3e}  (SQNR {r['sqnr_db']:.1f} dB)   "
                f"max |err| {r['max_abs_err']:.5f} ({r['max_abs_err'] / std:.1f} sigma)"
            )
            print(
                f"Row cosine: mean {r['cos_mean']:.5f} | median {r['cos_median']:.5f} | "
                f"p1 {r['cos_p1']:.5f} | min {r['cos_min']:.5f}   "
                f"(mean angle {np.degrees(np.arccos(min(r['cos_mean'], 1.0))):.2f} deg)"
            )
            print(
                f"Row rel. MSE: mean {r['row_mse_mean']:.3e} | p95 {r['row_mse_p95']:.3e} | max {r['row_mse_max']:.3e}"
            )
            print(
                f"Flushed to 0: {r['flushed_pct']:.2f}% of nonzero elements | "
                f">50% flushed blocks {r['flushed_blocks_pct']:.2f}%"
            )
            print(
                "Worst rows:   "
                + ", ".join(f"{i}:{c:.4f}" for i, c in r["worst_rows"])
                + f"   [{r['seconds']:.1f}s]"
            )

        stats = dict(
            name=t.name,
            type=t.tensor_type.name,
            shape=[int(s) for s in t.shape],
            rows_analysed=n_rows,
            imp_reason=imp_reason,
            rows_total=n_total,
            elements_analysed=count,
            elements_total=int(t.n_elements),
            mean=mean,
            std=std,
            min=vmin,
            max=vmax,
            p99=p99,
            p999=p999,
            outlier_ratio=amax / (p999 + 1e-8),
            excess_kurtosis=kurt,
            top_cols_amax=[(int(i), float(col_amax[i])) for i in top],
            block32_range_bits=(
                dict(
                    median=rhist.percentile(50),
                    p95=rhist.percentile(95),
                    p99=rhist.percentile(99),
                )
                if do32
                else None
            ),
        )
        return dict(stats=stats, formats=results)
    finally:
        pool.shutdown(wait=True)
        src.close()
        free_cache(dev)


def print_comparison(res):
    """Size/quality table for one tensor; * marks the Pareto frontier (bpw vs SQNR)."""
    rows = sorted(res["formats"], key=lambda r: (r["bpw"], -r["sqnr_db"]))
    n = res["stats"]["elements_total"]
    best, front = -1e9, set()
    for r in rows:
        if r["sqnr_db"] > best + 1e-9:
            front.add(r["fmt"])
            best = r["sqnr_db"]
    print(
        f"\n--- Summary: {res['stats']['name']}  ({n:,} elements; * = Pareto-optimal) ---"
    )
    print(
        f"{'format':8s} {'bpw':>7s} {'size MiB':>9s} {'SQNR dB':>8s} {'rel MSE':>10s} "
        f"{'cos mean':>9s} {'cos p1':>8s} {'cos min':>8s} {'flush%':>7s}"
    )
    for r in rows:
        print(
            f"{r['fmt']:8s} {r['bpw']:7.3f} {n * r['bpw'] / 8 / 2**20:9.1f} {r['sqnr_db']:8.1f} "
            f"{r['rel_mse']:10.3e} {r['cos_mean']:9.5f} {r['cos_p1']:8.5f} {r['cos_min']:8.5f} "
            f"{r['flushed_pct']:7.2f} {'*' if r['fmt'] in front else ''}"
        )


def print_multi(results, fmt_names):
    print(
        f"\n{'=' * 78}\nSQNR (dB) per tensor\n{'tensor':44s} "
        + " ".join(f"{n:>7s}" for n in fmt_names)
    )
    for res in results:
        by = {r["fmt"]: r["sqnr_db"] for r in res["formats"]}
        print(
            f"{res['stats']['name'][:44]:44s} "
            + " ".join(f"{by[n]:7.1f}" if n in by else f"{'-':>7s}" for n in fmt_names)
        )


# ============================================================================
# per-tensor format recommendation
# ============================================================================
DEFAULT_CANDIDATES = "q2_k,q3_k,mxfp4,iq4_xs,q4_k,q5_k,q6_k,q8_0"
# Heuristic extra SQNR margins (dB) for tensors that tend to be more sensitive.
DEFAULT_MARGINS = [(r"token_embd", 4.0), (r"^output\.weight$", 4.0), (r"attn_v", 2.0)]
# ggml type -> llama-quantize file type (only used for the positional fallback argument)
FTYPE = {
    "q2_k": "Q2_K",
    "q3_k": "Q3_K_M",
    "q4_k": "Q4_K_M",
    "q5_k": "Q5_K_M",
    "q6_k": "Q6_K",
    "q8_0": "Q8_0",
    "q4_0": "Q4_0",
    "q4_1": "Q4_1",
    "q5_0": "Q5_0",
    "q5_1": "Q5_1",
    "iq4_nl": "IQ4_NL",
    "iq4_xs": "IQ4_XS",
    "mxfp4": "MXFP4_MOE",
}


def parse_margin(s):
    pat, _, db = s.rpartition("=")
    if not pat:
        sys.exit(f"--margin expects REGEX=DB, got {s!r}")
    return pat, float(db)


def parse_kv(s, flag):
    pat, _, v = s.rpartition("=")
    if not pat:
        sys.exit(f"{flag} expects REGEX=VALUE, got {s!r}")
    return pat, v


def source_bpw(qt):
    """Bits per weight of a tensor kept in its source type."""
    block, size = GGML_QUANT_SIZES[qt]
    return size * 8.0 / block


def bpw_of(name):
    return FORMATS[name].bpw if name in FORMATS else 99.0  # 'keep' sorts last


def choose_format(res, target_db, min_cos):
    """Smallest bpw whose SQNR >= target and worst-1%-row cosine >= min_cos (ties: higher SQNR)."""
    ok = [
        r
        for r in res["formats"]
        if r["sqnr_db"] >= target_db and r["cos_p1"] >= min_cos
    ]
    if ok:
        return min(ok, key=lambda r: (r["bpw"], -r["sqnr_db"])), True
    return max(res["formats"], key=lambda r: r["sqnr_db"]), False


def emit_command(path, gguf, rows, imatrix=None):
    sized = {}
    for r in rows:
        if r["rec"] and r["rec"] != "keep":
            sized[r["rec"]] = sized.get(r["rec"], 0) + r["elements"]
    default = max(sized, key=sized.get) if sized else "q8_0"
    out = [
        "#!/bin/sh",
        "# Generated by ggml_quant_inspect.py --recommend. Verify flags with `llama-quantize --help`.",
        "# Source GGUF should be F16/BF16/F32 (or add --allow-requantize).",
    ]
    fb = [r["name"] for r in rows if r.get("imp_reason") in IMP_WHY]
    if imatrix and fb:
        out.append(
            f"# {len(fb)} tensor(s) have no usable imatrix data and are quantized without it: "
            + ", ".join(fb[:8]) + (" ..." if len(fb) > 8 else "")
        )
    out.append("llama-quantize \\")
    for r in rows:
        if r["rec"]:
            # 'keep' = pinned to the source type: needs an explicit override, otherwise
            # llama-quantize would quantize it with the default type of the file type.
            typ = r["cur"] if r["rec"] == "keep" else r["rec"].upper()
            out.append(f"  --tensor-type '^{re.escape(r['name'])}$={typ}' \\")
    if imatrix:
        out.append(f"  --imatrix '{imatrix}' \\")
    out.append(f"  {gguf} OUT.gguf {FTYPE.get(default, default.upper())}")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def run_recommend(args, reader):
    cands = parse_formats(args.candidates)
    margins = [parse_margin(m) for m in args.margin] + (
        [] if args.no_default_margins else DEFAULT_MARGINS
    )
    pins = [(p, v.strip().lower()) for p, v in (parse_kv(x, "--pin") for x in args.pin)]
    for p, v in pins:
        if v != "keep" and v not in FORMATS:
            sys.exit(f"--pin {p}={v}: unknown format (use one of {', '.join(FORMATS)} or 'keep')")
    floors = [(p, float(v)) for p, v in (parse_kv(x, "--floor") for x in args.floor)]
    pat = args.pattern or "*"
    todo = [
        t
        for t in reader.tensors
        if fnmatch.fnmatch(t.name, pat)
        and len(t.shape) >= 2
        and t.name.endswith(".weight")
    ]
    if not todo:
        sys.exit("no 2D+ '.weight' tensors matched")
    args.quiet = True
    rows, requant, pin_missed = [], 0, []
    for i, t in enumerate(todo, 1):
        print(
            f"\r[{i}/{len(todo)}] {t.name[:60]:60s}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        margin = next((db for p, db in margins if re.search(p, t.name)), 0.0)
        pin = next((v for p, v in pins if re.search(p, t.name)), None)
        floor = next((db for p, db in floors if re.search(p, t.name)), None)
        target = args.min_sqnr + margin
        if floor is not None:
            target = max(target, floor)
        if t.tensor_type.name not in ("F32", "F16", "BF16"):
            requant += 1
        base = dict(
            name=t.name, cur=t.tensor_type.name, elements=int(t.n_elements),
            margin=margin, target=target, floor=floor, pin=None, excluded=False,
            rec=None, bpw=None, sqnr=None, cos_p1=None, met=False, imp_reason=None,
            by_fmt={}, outlier_ratio=None, kurtosis=None, weighted=False, fmts=[],
        )
        excl = any(re.search(p, t.name) for p in args.budget_exclude)

        if pin == "keep":  # no analysis needed: the tensor stays in its source type
            bpw = source_bpw(t.tensor_type)
            base.update(
                pin="keep", excluded=excl, rec="keep", bpw=bpw, sqnr=200.0, cos_p1=1.0,
                met=True, fmts=[dict(fmt="keep", bpw=bpw, rel_mse=0.0, sqnr_db=200.0, cos_p1=1.0)],
            )
            rows.append(base)
            continue

        fmts_t = list(cands)
        if pin and FORMATS[pin] not in fmts_t:
            fmts_t.append(FORMATS[pin])  # evaluate the pinned format even if not a candidate
        with contextlib.redirect_stdout(io.StringIO()):
            res = inspect_tensor(args.gguf, t, fmts_t, args)
        by = {r["fmt"]: r for r in res["formats"]}
        if pin and pin not in by:
            pin_missed.append(f"{t.name} ({pin})")
            pin = None
        base.update(
            pin=pin, excluded=excl and pin is not None,
            by_fmt={k: v["sqnr_db"] for k, v in by.items()},
            outlier_ratio=res["stats"]["outlier_ratio"],
            kurtosis=res["stats"]["excess_kurtosis"],
            weighted=bool(res["formats"] and res["formats"][0].get("weighted")),
            imp_reason=res["stats"]["imp_reason"],
            fmts=[
                dict(fmt=r["fmt"], bpw=r["bpw"], rel_mse=r["rel_mse"],
                     sqnr_db=r["sqnr_db"], cos_p1=r["cos_p1"])
                for r in res["formats"]
            ],
        )
        if res["formats"]:
            if pin:
                r, met = by[pin], True
            else:
                r, met = choose_format(res, target, args.min_cos)
            base.update(rec=r["fmt"], bpw=r["bpw"], sqnr=r["sqnr_db"], cos_p1=r["cos_p1"], met=met)
        rows.append(base)
    print("\r" + " " * 70 + "\r", end="", file=sys.stderr)

    alloc_msg = ""
    if getattr(args, "imat", None) is not None:
        n_w = sum(1 for r in rows if r["weighted"])
        fb = [r for r in rows if r.get("imp_reason") in IMP_WHY]
        print(
            f"\nSQNR metric: imatrix-weighted for {n_w}/{len(rows)} tensors, plain for the rest"
        )
        if fb:
            print(
                f"{len(fb)} tensor(s) have no usable imatrix data. llama-quantize quantizes these "
                "without importance weights, so they are scored with the plain (non-imatrix) metric:"
            )
            for r in fb[:12]:
                print(f"  {r['name']}  ({IMP_WHY[r['imp_reason']]})")
            if len(fb) > 12:
                print(f"  ... and {len(fb) - 12} more")
    if pin_missed:
        print(f"WARNING: pin ignored (row length not divisible by the format's block): {', '.join(pin_missed)}")
    sized_rows = [r for r in rows if r.get("fmts")]
    mw = not args.no_margin_weights
    if args.budget_sweep:
        gibs = [float(v) for v in args.budget_sweep.split(",") if v.strip()]
        print_sweep(rows, gibs, mw, args.min_cos)
    if args.budget_gib or args.budget_bpw:
        n_el = sum(r["elements"] for r in sized_rows if not r["excluded"])
        bits_budget = (
            args.budget_gib * 8 * 2**30 if args.budget_gib else args.budget_bpw * n_el
        )
        chosen, used = allocate_budget(rows, bits_budget, mw, args.min_cos)
        if chosen is None:
            sys.exit(
                f"budget {bits_budget / 8 / 2**30:.2f} GiB is below the smallest possible mix "
                f"({used / 8 / 2**30:.2f} GiB) for these candidates/pins/floors"
            )
        apply_allocation(rows, chosen)
        alloc_msg = (
            f"{bits_budget / 8 / 2**30:.2f} GiB budget, mix {used / 8 / 2**30:.2f} GiB, "
            + ("margins used as sensitivity weights 10^(dB/10)" if mw else "uniform sensitivity")
        )

    names = [f.name for f in cands]
    print(
        (
            f"\nRecommendation: budget allocation ({alloc_msg})"
            if alloc_msg
            else f"\nRecommendation: smallest format with SQNR >= {args.min_sqnr:g} dB + per-tensor margin"
        )
        + (f" and p1 row cosine >= {args.min_cos:g}" if args.min_cos else "")
        + f"   (candidates: {', '.join(names)}; {args.max_rows} rows/tensor)"
    )
    print(
        f"{'tensor':42s} {'cur':>5s} {'rec':>7s} {'bpw':>6s} {'SQNR':>6s} {'need':>6s}  "
        + " ".join(f"{n:>6s}" for n in names)
    )
    for r in rows:
        flag = (
            ""
            if r["met"]
            else (
                "  <- below target, best available"
                if r["rec"]
                else "  (no usable candidate)"
            )
        )
        if r["pin"]:
            flag = "  [pinned" + (", not counted in budget]" if r["excluded"] else "]")
        elif r["floor"] is not None and r["met"]:
            flag += f"  [floor {r['floor']:g} dB]"
        if r.get("imp_reason") in IMP_WHY:
            flag += "  [no imatrix data: plain SQNR]"
        print(
            f"{r['name'][:42]:42s} {r['cur']:>5s} {(r['rec'] or '-'):>7s} "
            f"{r['bpw'] if r['bpw'] else 0:6.3f} {r['sqnr'] if r['sqnr'] is not None else 0:6.1f} {r['target']:6.1f}  "
            + " ".join(
                f"{r['by_fmt'][n]:6.1f}" if n in r["by_fmt"] else f"{'-':>6s}"
                for n in names
            )
            + flag
        )

    tot = sum(r["elements"] for r in rows)
    sized = [r for r in rows if r["rec"]]
    bits = sum(r["elements"] * r["bpw"] for r in sized)
    print(f"\n{len(rows)} weight tensors, {tot / 1e9:.3f}B elements")
    print(
        f"Recommended mix: {bits / 8 / 2 ** 30:.2f} GiB  (avg {bits / max(sum(r['elements'] for r in sized), 1):.2f} bpw)"
    )
    ex_bits = sum(r["elements"] * r["bpw"] for r in sized if r["excluded"])
    if ex_bits:
        print(
            f"  budget-excluded (pinned): {ex_bits / 8 / 2 ** 30:.2f} GiB; "
            f"counted against the budget: {(bits - ex_bits) / 8 / 2 ** 30:.2f} GiB"
        )
    for n in ("q4_k", "q8_0"):
        if n in FORMATS:
            print(
                f"  for comparison, uniform {n}: {tot * FORMATS[n].bpw / 8 / 2 ** 30:.2f} GiB"
            )
    by_type = {}
    for r in sized:
        d = by_type.setdefault(r["rec"], [0, 0])
        d[0] += 1
        d[1] += r["elements"]
    print(
        "Mix: "
        + ", ".join(
            f"{k} x{v[0]} ({100 * v[1] / tot:.1f}% of weights)"
            for k, v in sorted(by_type.items(), key=lambda kv: bpw_of(kv[0]))
        )
    )
    nocand = [r["name"] for r in rows if not r["rec"]]
    if nocand:
        print(
            f"{len(nocand)} tensor(s) have a row length that no candidate block size divides "
            f"(llama-quantize picks its own fallback): "
            + ", ".join(nocand[:6])
            + (" ..." if len(nocand) > 6 else "")
        )
    unmet = [r["name"] for r in rows if r["rec"] and not r["met"]]
    if unmet:
        print(
            f"{len(unmet)} tensor(s) miss the target even at the best candidate: "
            + ", ".join(unmet[:6])
            + (" ..." if len(unmet) > 6 else "")
        )
    if requant:
        print(
            f"NOTE: {requant} tensor(s) are already quantized in this file, so the numbers measure "
            "re-quantization of dequantized values; analyse the BF16/F16 original for real answers."
        )
    if args.emit_cmd:
        emit_command(args.emit_cmd, args.gguf, rows, args.imatrix)
        print(f"Wrote llama-quantize command to {args.emit_cmd}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)
        print(f"Wrote {args.json}")


# ============================================================================
# imatrix support + size-budget allocation
# ============================================================================
def load_imatrix(path):
    """Load a llama.cpp imatrix -> {tensor name: flat float32 mean(x^2) per input column}.
    Supports the legacy .dat layout and the GGUF layout (<name>.in_sum2 / <name>.counts).
    MoE entries hold n_expert * n_cols values; they are averaged over experts at lookup."""
    import struct

    with open(path, "rb") as fh:
        head = fh.read(4)
    out = {}
    if head == b"GGUF":
        r = GGUFReader(path, "r")
        sums, counts = {}, {}
        for t in r.tensors:
            if t.name.endswith(".in_sum2"):
                sums[t.name[: -len(".in_sum2")]] = np.asarray(t.data, dtype=np.float64)
            elif t.name.endswith(".counts"):
                counts[t.name[: -len(".counts")]] = np.asarray(t.data, dtype=np.float64).reshape(-1)
        if not sums:
            raise ValueError(
                "no '*.in_sum2' tensors found; tensors present: "
                + ", ".join(t.name for t in r.tensors[:5])
            )
        for n, s in sums.items():
            s = s.reshape(-1, s.shape[-1])
            c = counts.get(n)
            if c is not None and c.size == s.shape[0]:
                s = s / np.maximum(c, 1.0)[:, None]
            out[n] = s.reshape(-1).astype(NPF32)
        return out
    with open(path, "rb") as fh:
        (n_entries,) = struct.unpack("<i", fh.read(4))
        if not 0 < n_entries < 1_000_000:
            raise ValueError("not a legacy imatrix .dat (bad entry count)")
        for _ in range(n_entries):
            (ln,) = struct.unpack("<i", fh.read(4))
            name = fh.read(ln).decode("utf-8")
            ncall, nval = struct.unpack("<ii", fh.read(8))
            v = np.frombuffer(fh.read(4 * nval), dtype="<f4")
            if v.size != nval:
                raise ValueError(f"truncated imatrix entry {name!r}")
            out[name] = (v / max(ncall, 1)).astype(NPF32)
    return out


IMP_WHY = {
    "missing": "no entry in the imatrix",
    "shape": "imatrix entry size does not match the row length",
    "empty": "imatrix entry is all zero / non-finite",
}


def lookup_imp(args, name, cols):
    """-> (per-input-column importance normalised to mean 1 or None, reason).
    reason: 'off' (no imatrix given), 'ok', or one of IMP_WHY (tensor falls back to the plain
    metric, which is also what llama-quantize does: it quantizes that tensor without weights)."""
    imat = getattr(args, "imat", None)
    if not imat:
        return None, "off"
    v = imat.get(name)
    if v is None:
        return None, "missing"
    if v.size == 0 or v.size % cols:
        return None, "shape"
    imp = v.reshape(-1, cols).mean(0).astype(np.float64)
    if not np.isfinite(imp).all() or imp.sum() <= 0:
        return None, "empty"
    return (imp / imp.mean()).astype(NPF32), "ok"


def _hull(points):
    """points: [(bpw, loss, payload)] -> lower convex hull of the Pareto points (bpw ascending)."""
    best = {}
    for b, l, p in points:
        if b not in best or l < best[b][0]:
            best[b] = (l, p)
    pts, lo = [], float("inf")
    for b in sorted(best):
        l, p = best[b]
        if l < lo:  # keep only options that beat every cheaper one
            pts.append((b, l, p))
            lo = l
    h = []
    for q in pts:
        while len(h) >= 2:
            (b0, l0, _), (b1, l1, _) = h[-2], h[-1]
            if (l1 - l0) * (q[0] - b1) > (q[1] - l1) * (b1 - b0):  # h[-1] lies above chord
                h.pop()
            else:
                break
        h.append(q)
    return h


def allocate_budget(rows, budget_bits, margin_weights=True, min_cos=0.0):
    """Equal-slope allocation: start every tensor at its cheapest candidate, then repeatedly buy
    the upgrade with the largest (weighted relative-MSE reduction) per extra bit until the budget
    is spent. Loss per tensor = 10^(margin/10) * relMSE, so a +4 dB margin counts 2.5x.
    Per-tensor constraints: formats whose worst-1%-row cosine is below min_cos are excluded
    (unless none qualifies); a --floor removes formats below that SQNR (best available if none
    reaches it); a --pin fixes the format outright; pinned tensors marked `excluded` cost 0.
    Returns ({tensor name: format dict}, used_bits), or (None, min_bits) if infeasible."""
    items = [r for r in rows if r.get("fmts")]
    hulls, n_el = [], []
    for r in items:
        w = 10 ** (r["margin"] / 10.0) if margin_weights else 1.0
        ok = [f for f in r["fmts"] if f["cos_p1"] >= min_cos] or r["fmts"]
        fl = r.get("floor")
        if fl is not None:
            ok = [f for f in ok if f["sqnr_db"] >= fl] or [max(ok, key=lambda f: f["sqnr_db"])]
        if r.get("pin"):
            ok = [f for f in r["fmts"] if f["fmt"] == r["pin"]] or ok
        hulls.append(_hull([(f["bpw"], w * f["rel_mse"], f) for f in ok]))
        n_el.append(0 if r.get("excluded") else r["elements"])
    used = sum(n * h[0][0] for n, h in zip(n_el, hulls))
    if used > budget_bits + 1e-6:
        return None, used
    level = [0] * len(items)
    steps = []
    for i, h in enumerate(hulls):
        for k in range(len(h) - 1):
            cost = n_el[i] * (h[k + 1][0] - h[k][0])
            steps.append(((h[k][1] - h[k + 1][1]) / max(cost, 1e-9), i, k, cost))
    steps.sort(key=lambda s: -s[0])  # stable: hull slopes already decrease within a tensor
    for ratio, i, k, cost in steps:
        if level[i] == k and used + cost <= budget_bits + 1e-6:
            level[i] = k + 1
            used += cost
    return {r["name"]: h[l][2] for r, h, l in zip(items, hulls, level)}, used


def apply_allocation(rows, chosen):
    for r in rows:
        f = chosen.get(r["name"])
        if f:
            r.update(
                rec=f["fmt"],
                bpw=f["bpw"],
                sqnr=f["sqnr_db"],
                cos_p1=f["cos_p1"],
                met=bool(r.get("pin")) or f["sqnr_db"] >= r["target"],
            )


def print_sweep(rows, budgets, margin_weights, min_cos=0.0):
    sized = [r for r in rows if r.get("fmts")]
    tot = sum(r["elements"] for r in sized if not r.get("excluded"))
    print(f"\nBudget sweep ({len(sized)} tensors, {tot / 1e9:.3f}B counted elements)")
    print(f"{'budget':>8s} {'used':>8s} {'avg bpw':>8s} {'min SQNR':>9s} {'< need':>7s}  mix")
    for g in budgets:
        chosen, used = allocate_budget(rows, g * 8 * 2**30, margin_weights, min_cos)
        if chosen is None:
            print(f"{g:7.2f}G  below the smallest possible mix ({used / 8 / 2**30:.2f} GiB)")
            continue
        by, below, mn = {}, 0, 1e9
        for r in sized:
            f = chosen[r["name"]]
            by[f["fmt"]] = by.get(f["fmt"], 0) + 1
            mn = min(mn, f["sqnr_db"])
            below += (f["sqnr_db"] < r["target"]) and not r.get("pin")
        mix = " ".join(f"{k}x{v}" for k, v in sorted(by.items(), key=lambda kv: bpw_of(kv[0])))
        print(
            f"{g:7.2f}G {used / 8 / 2**30:7.2f}G {used / tot:8.2f} {mn:9.1f} {below:7d}  {mix}"
        )


# ============================================================================
# self test
# ============================================================================
def selftest(args):
    from gguf.quants import quantize

    dev = args.dev
    cpu = torch.device("cpu")
    rng = np.random.default_rng(0)
    xn = (rng.standard_normal((512, 1024)) * 0.0166).astype(NPF32)
    xn[:, 7] *= 40
    x = torch.from_numpy(xn).to(dev)
    print(f"1) cross-check against gguf-py reference quantizers (legacy formats) on {dev}")
    for name, qt in [
        ("q4_0", QT.Q4_0),
        ("q4_1", QT.Q4_1),
        ("q5_0", QT.Q5_0),
        ("q5_1", QT.Q5_1),
        ("q8_0", QT.Q8_0),
    ]:
        try:
            ref = dequantize(quantize(xn, qt), qt)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:6s} gguf-py quantize unavailable ({type(e).__name__})")
            continue
        mine = FORMATS[name].fn(x, Ctx(dev)).cpu().numpy()
        frac = float((np.abs(mine - ref) > 1e-6 * np.abs(xn).max()).mean())
        print(
            f"  {name:6s} max|diff| {np.abs(mine - ref).max():.2e}   elements differing {100 * frac:.4f}%"
        )
    print(f"\n2) reference SQNR on iid Gaussian data (std 0.0166, no outliers) on {dev}")
    g = torch.from_numpy((rng.standard_normal((256, 2048)) * 0.0166).astype(NPF32)).to(dev)
    gc_ = g.cpu()
    nx = float((gc_.double() ** 2).sum())
    for f in sorted(FORMATS.values(), key=lambda f: f.bpw):
        sync(dev)
        t0 = time.time()
        y = f.fn(g, Ctx(dev))
        sync(dev)
        dt = time.time() - t0
        e = float(((gc_ - y.cpu()).double() ** 2).sum())
        line = (
            f"  {f.name:7s} {f.bpw:7.3f} bpw  SQNR {10 * np.log10(nx / max(e, 1e-300)):6.1f} dB   ({dt:.2f}s)"
        )
        if dev.type != "cpu":
            yc = f.fn(gc_, Ctx(cpu))
            line += f"   differs from CPU in {100 * float((yc != y.cpu()).float().mean()):.4f}% of elements"
        print(line)


# ============================================================================
def parse_formats(s):
    if s == "default":
        return [FORMATS[n] for n in DEFAULT_FORMATS]
    if s == "all":
        return list(FORMATS.values())
    out = []
    for n in s.split(","):
        n = n.strip().lower()
        if n not in FORMATS:
            sys.exit(f"unknown format {n!r}; known: {', '.join(FORMATS)}")
        out.append(FORMATS[n])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("gguf", nargs="?")
    ap.add_argument("pattern", nargs="?", help="tensor name or fnmatch pattern")
    ap.add_argument("--list", action="store_true", help="list tensors and exit")
    ap.add_argument("--list-formats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--device",
        default="auto",
        help="auto (default: first of cuda, xpu, mps, npu, musa, hpu, mtia, else cpu), cpu, "
        "cuda, cuda:1, xpu, mps, ... (any torch device string; ROCm uses 'cuda')",
    )
    ap.add_argument(
        "--formats",
        default="default",
        help="comma list, 'default' or 'all' (see --list-formats)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="evenly sample at most N rows per tensor (0 = all rows; "
        "default 8192, or 1024 with --recommend)",
    )
    ap.add_argument(
        "--chunk-elems",
        type=int,
        default=None,
        help="elements per chunk; memory ~ 40-60 bytes x this on the device "
        "(default 2M on CPU, 8M on accelerators; halved automatically on device OOM)",
    )
    ap.add_argument(
        "--opt-scale",
        action="store_true",
        help="MXFP4: also try scale exponent+1 per block",
    )
    ap.add_argument("--json", help="write full results to this JSON file")
    g = ap.add_argument_group("recommendation mode")
    g.add_argument(
        "--recommend",
        action="store_true",
        help="scan all 2D+ .weight tensors (or those matching PATTERN) and pick the smallest "
        "format that meets the quality target",
    )
    g.add_argument(
        "--candidates", default=DEFAULT_CANDIDATES, help="formats to choose from"
    )
    g.add_argument(
        "--min-sqnr", type=float, default=22.0, help="required SQNR in dB (default 22)"
    )
    g.add_argument(
        "--min-cos",
        type=float,
        default=0.0,
        help="also require this cosine similarity on the worst 1%% of rows (e.g. 0.995)",
    )
    g.add_argument(
        "--margin",
        action="append",
        default=[],
        metavar="REGEX=DB",
        help="extra dB required for tensors matching REGEX (repeatable; checked before defaults)",
    )
    g.add_argument(
        "--no-default-margins",
        action="store_true",
        help="drop built-in margins (token_embd +4, output +4, attn_v +2 dB)",
    )
    g.add_argument(
        "--imatrix",
        metavar="FILE",
        help="llama.cpp imatrix (.dat or GGUF): weight each input column's error by calibration "
        "importance; tensors without an entry (e.g. token_embd) fall back to plain SQNR",
    )
    g.add_argument(
        "--budget-gib",
        type=float,
        help="choose formats to fit this many GiB of weights (replaces the --min-sqnr threshold; "
        "--min-cos still applies as a per-tensor floor)",
    )
    g.add_argument(
        "--budget-bpw",
        type=float,
        help="same as --budget-gib but as an average bits-per-weight over the analysed tensors",
    )
    g.add_argument(
        "--budget-sweep",
        metavar="G1,G2,..",
        help="print the size/quality curve for these budgets in GiB (analysis only)",
    )
    g.add_argument(
        "--no-margin-weights",
        action="store_true",
        help="budget mode: treat all tensors as equally sensitive instead of weighting by margin",
    )
    g.add_argument(
        "--emit-cmd",
        metavar="FILE",
        help="write a llama-quantize command with per-tensor types",
    )
    g.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="REGEX=FORMAT",
        help="force a format for matching tensors (any name from --list-formats, or 'keep' to leave "
        "the source type untouched, e.g. '^token_embd=keep'); repeatable, first match wins",
    )
    g.add_argument(
        "--floor",
        action="append",
        default=[],
        metavar="REGEX=DB",
        help="hard minimum SQNR for matching tensors: removes formats below it in budget mode "
        "(unlike --margin, which is only a weight there) and raises the target in threshold mode",
    )
    g.add_argument(
        "--budget-exclude",
        action="append",
        default=[],
        metavar="REGEX",
        help="pinned tensors matching REGEX do not count against --budget-gib/--budget-bpw "
        "(e.g. a CPU-resident token_embd)",
    )
    args = ap.parse_args()

    if args.list_formats:
        for f in FORMATS.values():
            print(f"{f.name:8s} {f.bpw:7.4f} bpw  block {f.block:<4d} {f.desc}")
        return
    if args.selftest:
        args.dev = setup_device(args.device)
        return selftest(args)
    if not args.gguf:
        ap.error("gguf path required")

    args.imat = None
    if args.imatrix:
        try:
            args.imat = load_imatrix(args.imatrix)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"cannot read imatrix {args.imatrix}: {e}")
        print(f"Loaded imatrix {args.imatrix}: {len(args.imat)} tensors")
    if args.max_rows is None:
        args.max_rows = 1024 if args.recommend else 8192
    print(f"Reading header from {args.gguf}...")
    reader = GGUFReader(args.gguf, "r")
    if not args.recommend and (args.list or not args.pattern):
        for t in reader.tensors:
            print(f"{t.name:50s} {t.tensor_type.name:8s} {[int(s) for s in t.shape]}")
        return

    args.dev = setup_device(args.device)
    if args.chunk_elems is None:
        args.chunk_elems = 2_000_000 if args.dev.type == "cpu" else 8_000_000
    args.chunk_elems = max(1, min(args.chunk_elems, 1 << 24))  # keep float32 counters exact

    if args.recommend:
        run_recommend(args, reader)
        return
    matches = [t for t in reader.tensors if fnmatch.fnmatch(t.name, args.pattern)]
    if not matches:
        sys.exit(f"No tensor matches {args.pattern!r} (use --list)")
    fmts = parse_formats(args.formats)

    results = []
    for t in matches:
        res = inspect_tensor(args.gguf, t, fmts, args)
        if res["formats"]:
            print_comparison(res)
        results.append(res)
    if len(results) > 1:
        print_multi(results, [f.name for f in fmts])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"\nWrote {args.json}")
    try:
        import resource

        print(
            f"\nPeak host RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB"
        )
    except ImportError:
        pass
    try:
        if args.dev.type != "cpu":
            peak = getattr(torch, args.dev.type).max_memory_allocated(args.dev)
            print(f"Peak device memory: {peak / 2**20:.0f} MiB")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
