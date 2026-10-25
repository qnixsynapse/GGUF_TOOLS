#!/usr/bin/env python3
"""
gguf_weight_compare.py - compare stored weights of GGUF files against a reference GGUF.

Every tensor is dequantized to float32 (any type gguf-py can dequantize: F32/F16/BF16/Q4_0/...)
and compared against the same tensor in the reference file, row by row, streamed from disk.

Per tensor: SQNR (dB, signal = reference), cosine similarity, and relative RMS error, plus
  gain   = |other| / |ref| (overall scale ratio)
  fit1   = SQNR after the best single scalar multiplier on the whole tensor
  fitRow = SQNR after the best scalar multiplier per row (per output channel)
A large gap between SQNR and fit1/fitRow means the files differ mainly by scale, not direction.
Per tensor class (ffn_down, attn_q, ...): median / min SQNR across layers.

Examples
  # drift of QAT fine-tune + Q4_0 error, both measured against the baseline
  python gguf_weight_compare.py baseline-bf16.gguf qat-bf16.gguf qat-Q4_0.gguf

  # pure quantization error of the QAT Q4_0 vs its own BF16
  python gguf_weight_compare.py qat-bf16.gguf qat-Q4_0.gguf

  # only some tensors, all rows, per-tensor table
  python gguf_weight_compare.py base.gguf qat-Q4_0.gguf "blk.*.ffn_down.weight" --max-rows 0 --per-tensor
"""
import argparse
import fnmatch
import re
import statistics as st
from collections import defaultdict

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf import GGMLQuantizationType as T


def load_tensors(path):
    r = GGUFReader(path, "r")
    return {t.name: t for t in r.tensors}


def rows_of(t):
    # gguf-py stores shape reversed (ne order): shape[0] = row length, shape[1] = number of rows
    return int(t.shape[1]) if len(t.shape) >= 2 else 1


def read_rows(t, idx):
    """Dequantize the selected rows of tensor t -> float32 [len(idx), row_len]."""
    data = t.data
    n_rows = rows_of(t)
    data = data.reshape(n_rows, -1)[idx]
    if t.tensor_type in (T.F32, T.F16):
        return data.astype(np.float32)
    return dequantize(np.ascontiguousarray(data), t.tensor_type).astype(np.float32)


def stats(ref, x):
    ref = ref.astype(np.float64)
    x = x.astype(np.float64)
    sig = float(np.sum(ref * ref))
    err = float(np.sum((ref - x) ** 2))
    dot = float(np.sum(ref * x))
    nx = float(np.sum(x * x))
    sqnr = 10 * np.log10(sig / err) if err > 0 else float("inf")
    cos = dot / np.sqrt(sig * nx) if sig > 0 and nx > 0 else float("nan")
    rel = np.sqrt(err / sig) if sig > 0 else float("nan")
    return sqnr, cos, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="reference GGUF (signal)")
    ap.add_argument("others", nargs="+", help="GGUF(s) to compare against the reference")
    ap.add_argument("--pattern", default="*.weight", help="fnmatch on tensor name (default *.weight)")
    ap.add_argument("--max-rows", type=int, default=2048, help="evenly sample at most N rows per tensor (0 = all)")
    ap.add_argument("--chunk-rows", type=int, default=256, help="rows per read chunk")
    ap.add_argument("--per-tensor", action="store_true", help="also print one line per tensor")
    args = ap.parse_args()

    # positional trick: a trailing arg that looks like a glob is the pattern
    others = list(args.others)
    if others and any(c in others[-1] for c in "*?[") and not others[-1].endswith(".gguf"):
        args.pattern = others.pop()
    if not others:
        ap.error("need at least one file to compare against the reference")

    ref = load_tensors(args.ref)
    cmp_files = [(p, load_tensors(p)) for p in others]

    names = [n for n in ref
             if fnmatch.fnmatch(n, args.pattern) and len(ref[n].shape) >= 2
             and all(n in c for _, c in cmp_files)]
    names.sort(key=lambda n: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", n)])
    print(f"{len(names)} tensors compared  (ref: {args.ref})")

    # results[file_index][class] -> list of (sqnr, cos, rel)
    results = [defaultdict(list) for _ in cmp_files]
    types = [defaultdict(set) for _ in cmp_files]

    for n in names:
        tr = ref[n]
        nr = rows_of(tr)
        if args.max_rows and nr > args.max_rows:
            idx_all = np.unique(np.linspace(0, nr - 1, args.max_rows).astype(np.int64))
        else:
            idx_all = np.arange(nr)
        cls = re.sub(r"blk\.\d+\.", "", n).replace(".weight", "")

        acc = [dict(sig=0.0, err=0.0, dot=0.0, nx=0.0, rres=0.0) for _ in cmp_files]
        shapes_ok = [True] * len(cmp_files)
        for s in range(0, len(idx_all), args.chunk_rows):
            idx = idx_all[s:s + args.chunk_rows]
            a = read_rows(tr, idx).astype(np.float64)
            for k, (_, c) in enumerate(cmp_files):
                tc = c[n]
                if tuple(tc.shape) != tuple(tr.shape):
                    shapes_ok[k] = False
                    continue
                b = read_rows(tc, idx).astype(np.float64)
                d = a - b
                acc[k]["sig"] += float(np.sum(a * a))
                acc[k]["err"] += float(np.sum(d * d))
                acc[k]["dot"] += float(np.sum(a * b))
                acc[k]["nx"] += float(np.sum(b * b))
                # residual after the best scalar fit per row: sum_r (sig_r - dot_r^2 / nx_r)
                sr = np.sum(a * a, axis=1)
                nr_ = np.sum(b * b, axis=1)
                dr = np.sum(a * b, axis=1)
                acc[k]["rres"] += float(np.sum(sr - np.where(nr_ > 0, dr * dr / np.maximum(nr_, 1e-300), 0.0)))

        line = [f"{n:42s}"]
        for k, (p, c) in enumerate(cmp_files):
            if not shapes_ok[k]:
                line.append("shape mismatch")
                continue
            m = acc[k]
            sqnr = 10 * np.log10(m["sig"] / m["err"]) if m["err"] > 0 else float("inf")
            cos = m["dot"] / np.sqrt(m["sig"] * m["nx"]) if m["sig"] > 0 and m["nx"] > 0 else float("nan")
            rel = np.sqrt(m["err"] / m["sig"]) if m["sig"] > 0 else float("nan")
            gain = np.sqrt(m["nx"] / m["sig"]) if m["sig"] > 0 else float("nan")
            fit1 = 10 * np.log10(m["sig"] / (m["sig"] - m["dot"] ** 2 / m["nx"])) if m["nx"] > 0 and m["sig"] > m["dot"] ** 2 / m["nx"] else float("inf")
            fitr = 10 * np.log10(m["sig"] / m["rres"]) if m["rres"] > 0 else float("inf")
            results[k][cls].append((sqnr, cos, rel, gain, fit1, fitr))
            types[k][cls].add(c[n].tensor_type.name)
            line.append(f"{c[n].tensor_type.name:>6s} {sqnr:6.1f}dB cos {cos:.5f}")
        if args.per_tensor:
            print("  |  ".join(line))

    for k, (p, _) in enumerate(cmp_files):
        print(f"\n=== {p}  vs reference ===")
        print(f"{'class':14s} {'n':>3s} {'type(s)':>10s} {'med SQNR':>9s} {'min SQNR':>9s} {'med cos':>9s} {'min cos':>9s} {'med relRMS':>10s} {'med gain':>9s} {'fit1 dB':>8s} {'fitRow dB':>9s}")
        allv = []
        for cls in sorted(results[k]):
            v = results[k][cls]
            allv += v
            sq = [x[0] for x in v]
            co = [x[1] for x in v]
            re_ = [x[2] for x in v]
            print(f"{cls:14s} {len(v):3d} {','.join(sorted(types[k][cls])):>10s} "
                  f"{st.median(sq):9.1f} {min(sq):9.1f} {st.median(co):9.5f} {min(co):9.5f} {st.median(re_):10.4f} "
                  f"{st.median(x[3] for x in v):9.3f} {st.median(x[4] for x in v):8.1f} {st.median(x[5] for x in v):9.1f}")
        if allv:
            print(f"{'ALL':14s} {len(allv):3d} {'':>10s} {st.median(x[0] for x in allv):9.1f} "
                  f"{min(x[0] for x in allv):9.1f} {st.median(x[1] for x in allv):9.5f} "
                  f"{min(x[1] for x in allv):9.5f} {st.median(x[2] for x in allv):10.4f} "
                  f"{st.median(x[3] for x in allv):9.3f} {st.median(x[4] for x in allv):8.1f} {st.median(x[5] for x in allv):9.1f}")


if __name__ == "__main__":
    main()
