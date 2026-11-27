#!/usr/bin/env python3
"""Apply migtissera/Tess-4.5-26B-A3B task vectors (diff vs thinkingmachines/Inkling-2-Mini)
onto MLX loopling (TML-d24-varloop-research-MLX / loopling-p3).

Tess-4.5-26B-A3B fine-tuned 145 BF16 tensors (all 24 layers' 6 attention projections
plus model.llm.unembed.weight), leaving all 1,099 other tensors (NVFP4/MXFP8 experts,
routers, loop controllers, embeddings, norms, sconv, attn_res, vision, mtp.safetensors)
byte-identical to Inkling-2-Mini.

Avoids Q8 double-quantization error by sourcing unquantized BF16 weights from
thinkingmachines/TML-d24-varloop-research (exact upstream source of loopling-p3),
applying the FP32 task-vector delta, and quantizing once with loopling-p3's exact
affine 8-bit group_size=64 recipe (stream=mx.cpu).

NEVER run MLX/Metal on the local Mac — run on 512S2 via:
    LANE=tess RHOST=512S2 scripts/r [--gpu] "HF_HUB_OFFLINE=0 python3 scripts/apply_tess_vectors.py ..."
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# Pinned HF revisions for reproducibility
DEFAULT_TESS_REPO = "migtissera/Tess-4.5-26B-A3B"
DEFAULT_TESS_REV = "aa4682763e42c4efd0bf8209d8aa5986f731c691"

DEFAULT_MINI_REPO = "thinkingmachines/Inkling-2-Mini"
DEFAULT_MINI_REV = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"

DEFAULT_VARLOOP_REPO = "thinkingmachines/TML-d24-varloop-research"
DEFAULT_VARLOOP_REV = "657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2"

SAFETENSORS_FILE = "model-00001-of-00001.safetensors"

TARGET_ATTN_SUFFIXES = (
    ".attn.wq_du.weight",
    ".attn.wk_dv.weight",
    ".attn.wv_dv.weight",
    ".attn.wr_du.weight",
    ".attn.wo_ud.weight",
    ".attn.gate_attn.weight",
)

# 16 representative untouched expert & vision tensors across L0..23 for full bit-for-bit spot-check
SPOT_CHECK_UNTOUCHED_KEYS = [
    "model.llm.layers.0.mlp.experts.w13_weight_scale",
    "model.llm.layers.0.mlp.experts.w2_weight_amax",
    "model.llm.layers.6.mlp.experts.w13_weight_scale",
    "model.llm.layers.6.mlp.experts.w2_weight_scale",
    "model.llm.layers.12.mlp.experts.w2_weight_scale",
    "model.llm.layers.17.mlp.experts.w13_weight_scale",
    "model.llm.layers.20.mlp.experts.w2_weight_scale",
    "model.llm.layers.23.mlp.experts.w13_weight_scale",
    "model.visual.backbone.embeddings.patch_embedding.weight",
    "model.visual.backbone.embeddings.position_embedding",
    "model.visual.backbone.layers.0.self_attn.q_proj.weight",
    "model.visual.backbone.layers.6.mlp.c_fc.weight",
    "model.visual.backbone.layers.12.self_attn.out_proj.weight",
    "model.visual.backbone.layers.17.mlp.c_proj.weight",
    "model.visual.backbone.layers.22.self_attn.k_proj.weight",
    "model.visual.proj_out.weight",
]

# Huge untouched tensors audited via 5x 1MB stratified byte-slice SHA-256 checks (0%, 25%, 50%, 75%, end)
SLICE_AUDIT_LARGE_KEYS = [
    "model.llm.embed.weight",
    "model.llm.layers.0.mlp.experts.w13_weight",
    "model.llm.layers.6.mlp.experts.w2_weight",
    "model.llm.layers.23.mlp.experts.w2_weight",
]


def _hf_resolve_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def _fetch_bytes_range(url: str, start: int, end_inclusive: int, retries: int = 8) -> bytes:
    """Fetch exact closed byte range [start, end_inclusive] with retries."""
    expected = end_inclusive - start + 1
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end_inclusive}",
                    "User-Agent": "loopling-tess-merge/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP {resp.status} for {url}")
                data = resp.read()
                if resp.status == 200 and start == 0:
                    data = data[:expected]
                if len(data) != expected:
                    raise RuntimeError(
                        f"Short read: got {len(data)} bytes, expected {expected} ({start}-{end_inclusive})"
                    )
                return data
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0 * (attempt + 1), 15.0))
    raise RuntimeError(f"Failed fetching bytes={start}-{end_inclusive} from {url}: {last_err}")


def read_remote_safetensors_header(repo: str, rev: str, filename: str = SAFETENSORS_FILE) -> tuple[dict, int]:
    """Return (header_dict_without_metadata, data_base_offset)."""
    url = _hf_resolve_url(repo, rev, filename)
    len_bytes = _fetch_bytes_range(url, 0, 7)
    hlen = struct.unpack("<Q", len_bytes)[0]
    hdr_bytes = _fetch_bytes_range(url, 8, 8 + hlen - 1)
    header = json.loads(hdr_bytes.decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def build_coalesced_ranges(
    header: dict,
    keys: list[str],
    data_base: int,
    max_gap: int = 256 * 1024,
    max_chunk: int = 32 * 1024 * 1024,
) -> list[tuple[int, int, list[tuple[str, int, int]]]]:
    """Coalesce sorted tensor byte ranges into fewer HTTP Range requests."""
    items = []
    for k in keys:
        off0, off1 = header[k]["data_offsets"]
        items.append((data_base + off0, data_base + off1, k))
    items.sort(key=lambda x: x[0])

    chunks: list[tuple[int, int, list[tuple[str, int, int]]]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_members: list[tuple[str, int, int]] = []

    for s, e, k in items:
        length = e - s
        if length > max_chunk:
            if cur_start is not None:
                chunks.append((cur_start, cur_end - 1, cur_members))
                cur_start = cur_end = None
                cur_members = []
            chunks.append((s, e - 1, [(k, 0, length)]))
            continue

        if cur_start is None:
            cur_start, cur_end = s, e
            cur_members = [(k, 0, length)]
        elif (s - cur_end) <= max_gap and (e - cur_start) <= max_chunk:
            cur_members.append((k, s - cur_start, length))
            cur_end = max(cur_end, e)
        else:
            chunks.append((cur_start, cur_end - 1, cur_members))
            cur_start, cur_end = s, e
            cur_members = [(k, 0, length)]

    if cur_start is not None:
        chunks.append((cur_start, cur_end - 1, cur_members))
    return chunks


def fetch_subset_safetensors(
    repo: str,
    rev: str,
    header: dict,
    data_base: int,
    keys: list[str],
    out_path: Path,
    workers: int = 8,
    max_sub_range: int = 32 * 1024 * 1024,
) -> None:
    """Range-fetch `keys` from remote safetensors and write a valid local `.safetensors`."""
    keys = sorted(set(keys))
    if out_path.exists():
        with open(out_path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            local_hdr = json.loads(f.read(hlen).decode("utf-8"))
        local_hdr.pop("__metadata__", None)
        if all(
            k in local_hdr
            and local_hdr[k]["dtype"] == header[k]["dtype"]
            and local_hdr[k]["shape"] == header[k]["shape"]
            for k in keys
        ):
            print(f"[fetch] Cached subset valid ({len(keys)} keys): {out_path}")
            return
        print(f"[fetch] Existing cache incomplete, re-fetching: {out_path}")

    url = _hf_resolve_url(repo, rev, SAFETENSORS_FILE)
    chunks = build_coalesced_ranges(header, keys, data_base)
    total_bytes = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in keys)
    fetch_bytes = sum(end - start + 1 for start, end, _ in chunks)
    print(
        f"[fetch] {repo}@{rev[:8]}: {len(keys)} tensors "
        f"({total_bytes / 1e6:.1f} MB payload, {fetch_bytes / 1e6:.1f} MB over {len(chunks)} HTTP range chunks)..."
    )

    out_header: dict[str, dict] = {}
    out_offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for k in keys:
        sz = header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
        out_header[k] = {
            "dtype": header[k]["dtype"],
            "shape": header[k]["shape"],
            "data_offsets": [cursor, cursor + sz],
        }
        out_offsets[k] = (cursor, sz)
        cursor += sz
    out_header["__metadata__"] = {"source_repo": repo, "source_rev": rev}
    hdr_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    out_data_base = 8 + len(hdr_bytes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".safetensors.tmp")
    with open(tmp_path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        if cursor > 0:
            f.seek(out_data_base + cursor - 1)
            f.write(b"\x00")

    t0 = time.time()
    done_bytes = 0

    def _download_chunk(start: int, end_inc: int, members: list[tuple[str, int, int]]) -> int:
        if len(members) == 1 and (end_inc - start + 1) > max_sub_range:
            k, _, length = members[0]
            out_off, _ = out_offsets[k]
            pos = 0
            while pos < length:
                step = min(max_sub_range, length - pos)
                buf = _fetch_bytes_range(url, start + pos, start + pos + step - 1)
                with open(tmp_path, "r+b") as f:
                    f.seek(out_data_base + out_off + pos)
                    f.write(buf)
                pos += step
            return length
        buf = _fetch_bytes_range(url, start, end_inc)
        with open(tmp_path, "r+b") as f:
            for k, rel, length in members:
                out_off, exp_len = out_offsets[k]
                assert length == exp_len
                f.seek(out_data_base + out_off)
                f.write(buf[rel : rel + length])
        return end_inc - start + 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_chunk, s, e, m) for s, e, m in chunks]
        completed = 0
        for fut in concurrent.futures.as_completed(futs):
            done_bytes += fut.result()
            completed += 1
            if completed % 10 == 0 or completed == len(chunks):
                dt = max(time.time() - t0, 0.01)
                print(
                    f"[fetch]   {out_path.name}: {completed}/{len(chunks)} chunks, "
                    f"{done_bytes / 1e6:.1f}/{fetch_bytes / 1e6:.1f} MB ({done_bytes / 1e6 / dt:.1f} MB/s)"
                )

    elapsed = time.time() - t0
    os.replace(tmp_path, out_path)
    print(
        f"[fetch] Wrote {out_path} ({total_bytes / 1e6:.1f} MB in {elapsed:.1f}s, "
        f"{done_bytes / 1e6 / max(elapsed, 0.01):.1f} MB/s)"
    )


def _sha256_file_slice(path: Path, offset: int, length: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        rem = length
        while rem > 0:
            chunk = f.read(min(rem, 8 * 1024 * 1024))
            if not chunk:
                raise EOFError(f"Unexpected EOF in {path}")
            h.update(chunk)
            rem -= len(chunk)
    return h.hexdigest()


def read_local_st_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(hlen).decode("utf-8"))
    h.pop("__metadata__", None)
    return h, 8 + hlen


def _select_tess_edited_keys(header: dict) -> list[str]:
    edited = sorted(
        k
        for k in header
        if k == "model.llm.unembed.weight" or k.endswith(TARGET_ATTN_SUFFIXES)
    )
    assert len(edited) == 145, f"Expected 145 Tess edited keys (144 attn + unembed), got {len(edited)}"
    return edited


def run_fetch_and_audit(args: argparse.Namespace) -> dict:
    staging = Path(args.staging).expanduser()
    staging.mkdir(parents=True, exist_ok=True)

    print("[stage] Reading remote safetensors headers (Tess, Inkling-2-Mini, Varloop)...")
    h_tess, base_tess = read_remote_safetensors_header(args.tess_repo, args.tess_rev)
    h_mini, base_mini = read_remote_safetensors_header(args.mini_repo, args.mini_rev)
    h_varloop, base_varloop = read_remote_safetensors_header(args.varloop_repo, args.varloop_rev)

    assert set(h_tess) == set(h_mini) == set(h_varloop), "Key set mismatch across upstream repos!"
    for k in h_tess:
        assert (
            h_tess[k]["shape"] == h_mini[k]["shape"] == h_varloop[k]["shape"]
            and h_tess[k]["dtype"] == h_mini[k]["dtype"] == h_varloop[k]["dtype"]
        ), f"Shape/dtype mismatch on {k}"
    print(f"[stage] Verified {len(h_tess)} identical keys/shapes/dtypes across Tess, Mini, and Varloop.")

    edited_keys = _select_tess_edited_keys(h_tess)
    for k in edited_keys:
        assert h_tess[k]["dtype"] == "BF16", f"Expected BF16 on edited key {k}, got {h_tess[k]['dtype']}"

    all_text_small = sorted(
        k
        for k in h_tess
        if not k.startswith(("model.visual.", "model.llm.embed.", "model.llm.unembed."))
        and ".mlp.experts." not in k
    )
    assert len(all_text_small) == 644, f"Expected 644 non-expert/non-vision/non-embed keys, got {len(all_text_small)}"
    for k in SPOT_CHECK_UNTOUCHED_KEYS + SLICE_AUDIT_LARGE_KEYS:
        assert k in h_tess, f"Spot-check key missing: {k}"

    audit_keys = sorted(set(all_text_small) | {"model.llm.unembed.weight"} | set(SPOT_CHECK_UNTOUCHED_KEYS))

    tess_st = staging / "tess_subset.safetensors"
    mini_st = staging / "mini_subset.safetensors"
    varloop_st = staging / "varloop_subset.safetensors"

    if args.fetch:
        fetch_subset_safetensors(args.tess_repo, args.tess_rev, h_tess, base_tess, audit_keys, tess_st, workers=args.workers)
        fetch_subset_safetensors(args.mini_repo, args.mini_rev, h_mini, base_mini, audit_keys, mini_st, workers=args.workers)
        fetch_subset_safetensors(args.varloop_repo, args.varloop_rev, h_varloop, base_varloop, edited_keys, varloop_st, workers=args.workers)
        (staging / "tess_edited_keys.json").write_text(json.dumps(edited_keys, indent=2))

    summary = {
        "tess_repo": args.tess_repo,
        "tess_rev": args.tess_rev,
        "edited_keys_count": len(edited_keys),
        "audited_full_keys": len(audit_keys),
        "audited_untouched_text": len(all_text_small) - 144,
        "audited_spot_check_expert_vision_full": len(SPOT_CHECK_UNTOUCHED_KEYS),
        "audited_large_tensors_stratified_slices": len(SLICE_AUDIT_LARGE_KEYS),
    }

    if args.audit:
        print(f"[audit] Comparing {len(audit_keys)} full tensors bit-for-bit between Tess and Inkling-2-Mini...")
        ht, bt = read_local_st_header(tess_st)
        hm, bm = read_local_st_header(mini_st)

        edited_set = set(edited_keys)
        spot_set = set(SPOT_CHECK_UNTOUCHED_KEYS)
        changed_in_edited = []
        unchanged_in_edited = []
        hidden_edits = []
        spot_check_diffs = []

        for k in audit_keys:
            t0, t1 = ht[k]["data_offsets"]
            m0, m1 = hm[k]["data_offsets"]
            assert (t1 - t0) == (m1 - m0)
            sha_t = _sha256_file_slice(tess_st, bt + t0, t1 - t0)
            sha_m = _sha256_file_slice(mini_st, bm + m0, m1 - m0)
            differs = sha_t != sha_m
            if k in edited_set:
                if differs:
                    changed_in_edited.append(k)
                else:
                    unchanged_in_edited.append(k)
            elif k in spot_set:
                if differs:
                    spot_check_diffs.append(k)
            else:
                if differs:
                    hidden_edits.append(k)

        print(f"[audit] Stratified 5x 1MB slice audit on {len(SLICE_AUDIT_LARGE_KEYS)} huge untouched tensors...")
        url_t = _hf_resolve_url(args.tess_repo, args.tess_rev, SAFETENSORS_FILE)
        url_m = _hf_resolve_url(args.mini_repo, args.mini_rev, SAFETENSORS_FILE)
        slice_diffs = []
        slice_sz = 1024 * 1024
        for k in SLICE_AUDIT_LARGE_KEYS:
            t0, t1 = h_tess[k]["data_offsets"]
            m0, m1 = h_mini[k]["data_offsets"]
            length = t1 - t0
            assert length == (m1 - m0)
            for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
                rel = min(int(frac * max(length - slice_sz, 0)), max(length - slice_sz, 0))
                n = min(slice_sz, length - rel)
                bt_bytes = _fetch_bytes_range(url_t, base_tess + t0 + rel, base_tess + t0 + rel + n - 1)
                bm_bytes = _fetch_bytes_range(url_m, base_mini + m0 + rel, base_mini + m0 + rel + n - 1)
                if hashlib.sha256(bt_bytes).digest() != hashlib.sha256(bm_bytes).digest():
                    slice_diffs.append((k, frac))

        summary.update(
            {
                "edited_actually_changed": len(changed_in_edited),
                "edited_byte_identical": len(unchanged_in_edited),
                "hidden_unlisted_text_edits": hidden_edits,
                "spot_check_expert_vision_diffs": spot_check_diffs,
                "large_tensor_slice_diffs": slice_diffs,
            }
        )
        untouched_text_cnt = len(all_text_small) - 144
        print(
            f"[audit] Result: edited_changed={len(changed_in_edited)}/145, "
            f"hidden_text_edits={len(hidden_edits)}/{untouched_text_cnt}, "
            f"spot_check_expert_vision_diffs={len(spot_check_diffs)}/{len(SPOT_CHECK_UNTOUCHED_KEYS)}, "
            f"large_slice_diffs={len(slice_diffs)}/{len(SLICE_AUDIT_LARGE_KEYS) * 5}"
        )
        assert len(changed_in_edited) == 145, f"Expected all 145 edited tensors changed, got {len(changed_in_edited)} (unchanged: {unchanged_in_edited[:5]})"
        assert not hidden_edits, f"Unexpected hidden text edits: {hidden_edits}"
        assert not spot_check_diffs, f"Unexpected expert/vision edits: {spot_check_diffs}"
        assert not slice_diffs, f"Unexpected diffs in large untouched tensors: {slice_diffs}"
        (staging / "audit_report.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[audit] PASS: 145/145 target tensors changed, 0 hidden edits across {untouched_text_cnt} untouched text tensors, "
            f"{len(SPOT_CHECK_UNTOUCHED_KEYS)} full expert/vision tensors, and {len(SLICE_AUDIT_LARGE_KEYS) * 5} stratified 1MB slices!"
        )

    return summary


def _load_mlx_modules():
    """Import MLX lazily only when running on remote Apple Silicon host."""
    import mlx.core as mx
    from mlx_lm.models.inkling2.checkpoint import _rename

    return mx, _rename


def _group_name(hf_key: str) -> str:
    if hf_key == "model.llm.unembed.weight":
        return "lm_head_q8"
    if ".attn." in hf_key:
        return "attn_q8_" + hf_key.split(".")[-2]
    return "other"


def compute_merged_updates(
    staging: Path,
    base_mlx_dir: Path,
    alpha: float,
) -> tuple[dict, dict]:
    """Compute all MLX-layout updated tensors for the 145 edited keys at scale `alpha`."""
    mx, _rename = _load_mlx_modules()

    edited_keys = json.loads((staging / "tess_edited_keys.json").read_text())
    assert len(edited_keys) == 145, f"Expected 145 edited keys, got {len(edited_keys)}"

    cfg = json.loads((base_mlx_dir / "config.json").read_text())
    qcfg = cfg["quantization"]

    print(f"[merge] Loading subset tensors from {staging} (alpha={alpha})...")
    w_tess = mx.load(str(staging / "tess_subset.safetensors"))
    w_mini = mx.load(str(staging / "mini_subset.safetensors"))
    w_varloop = mx.load(str(staging / "varloop_subset.safetensors"))

    updates: dict[str, object] = {}
    group_stats: dict[str, list[dict]] = defaultdict(list)
    q8_retention_rows: list[dict] = []

    for hf_k in edited_keys:
        vt = w_tess[hf_k].astype(mx.float32)
        vm = w_mini[hf_k].astype(mx.float32)
        vv = w_varloop[hf_k].astype(mx.float32)

        delta = vt - vm
        scaled_delta = delta * alpha
        merged_f32 = vv + scaled_delta

        if bool(mx.any(mx.isnan(merged_f32)).item()) or bool(mx.any(mx.isinf(merged_f32)).item()):
            raise ValueError(f"NaN/Inf detected in merged tensor for {hf_k}")

        v_norm = float(mx.sqrt(mx.sum(vv * vv)).item())
        d_norm = float(mx.sqrt(mx.sum(scaled_delta * scaled_delta)).item())
        max_abs_d = float(mx.max(mx.abs(scaled_delta)).item())
        rel_change = d_norm / max(v_norm, 1e-12)
        grp = _group_name(hf_k)
        group_stats[grp].append(
            {
                "key": hf_k,
                "v_norm": v_norm,
                "delta_norm": d_norm,
                "rel_change": rel_change,
                "max_abs_delta": max_abs_d,
            }
        )

        mlx_k = _rename(hf_k)
        assert mlx_k.endswith(".weight"), f"Expected .weight key, got {mlx_k}"
        mod_path = mlx_k[:-7]

        assert mod_path in qcfg and isinstance(qcfg[mod_path], dict), f"Missing quant config for {mod_path}"
        qc = qcfg[mod_path]
        gs, bits, mode = qc["group_size"], qc["bits"], qc["mode"]
        assert (gs, bits, mode) == (64, 8, "affine"), f"Unexpected quant config on {mod_path}: {qc}"

        w_new_bf16 = merged_f32.astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(w_new_bf16, group_size=gs, bits=bits, mode=mode, stream=mx.cpu)
        mx.eval(wq, scales, biases)
        updates[f"{mod_path}.weight"] = wq
        updates[f"{mod_path}.scales"] = scales
        updates[f"{mod_path}.biases"] = biases

        if alpha != 0.0:
            wq0, sc0, bi0 = mx.quantize(vv.astype(mx.bfloat16), group_size=gs, bits=bits, mode=mode, stream=mx.cpu)
            deq0 = mx.dequantize(wq0, sc0, bi0, group_size=gs, bits=bits, mode=mode).astype(mx.float32)
            deq1 = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits, mode=mode).astype(mx.float32)
            eff_d = deq1 - deq0
            dot = float(mx.sum(eff_d * scaled_delta).item())
            eff_norm = float(mx.sqrt(mx.sum(eff_d * eff_d)).item())
            cos = dot / max(eff_norm * d_norm, 1e-12)
            l2_ratio = eff_norm / max(d_norm, 1e-12)
            q8_retention_rows.append(
                {
                    "key": mlx_k,
                    "rel_change": rel_change,
                    "cos_eff_vs_true": cos,
                    "l2_ratio_eff_vs_true": l2_ratio,
                }
            )

    agg_groups = {}
    for grp, rows in sorted(group_stats.items()):
        rels = [r["rel_change"] for r in rows]
        dnorms = [r["delta_norm"] for r in rows]
        maxds = [r["max_abs_delta"] for r in rows]
        agg_groups[grp] = {
            "count": len(rows),
            "rel_change_min": min(rels),
            "rel_change_mean": sum(rels) / len(rels),
            "rel_change_max": max(rels),
            "delta_l2_mean": sum(dnorms) / len(dnorms),
            "max_abs_delta": max(maxds),
        }

    q8_summary = {}
    if q8_retention_rows:
        coss = [r["cos_eff_vs_true"] for r in q8_retention_rows]
        rats = [r["l2_ratio_eff_vs_true"] for r in q8_retention_rows]
        q8_summary = {
            "count": len(q8_retention_rows),
            "cos_min": min(coss),
            "cos_mean": sum(coss) / len(coss),
            "cos_max": max(coss),
            "l2_ratio_min": min(rats),
            "l2_ratio_mean": sum(rats) / len(rats),
            "l2_ratio_max": max(rats),
        }

    stats = {
        "alpha": alpha,
        "num_hf_edited_keys": len(edited_keys),
        "num_mlx_updated_tensors": len(updates),  # 145 * 3 = 435
        "groups": agg_groups,
        "q8_retention": q8_summary,
    }
    return updates, stats


def verify_alpha0(staging: Path, base_mlx_dir: Path) -> dict:
    """Prove alpha=0 through our exact pipeline matches base_mlx_dir 100% bit-for-bit."""
    mx, _ = _load_mlx_modules()
    updates, _ = compute_merged_updates(staging, base_mlx_dir, alpha=0.0)
    assert len(updates) == 435, f"Expected 435 MLX tensors (145*3), got {len(updates)}"

    idx = json.loads((base_mlx_dir / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in updates:
        assert k in wmap, f"Missing MLX key in index: {k}"
        by_shard[wmap[k]].append(k)

    mismatches = []
    checked = 0
    for shard_name, keys in sorted(by_shard.items()):
        shard_w = mx.load(str(base_mlx_dir / shard_name))
        for k in keys:
            a = updates[k]
            b = shard_w[k]
            if a.dtype != b.dtype or a.shape != b.shape:
                mismatches.append((k, f"meta mismatch: {a.dtype} {a.shape} vs {b.dtype} {b.shape}"))
                continue
            eq = bool(mx.all(a.view(mx.uint8) == b.view(mx.uint8)).item())
            if not eq:
                max_diff = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
                mismatches.append((k, f"value mismatch max_diff={max_diff}"))
            checked += 1

    res = {
        "checked_mlx_tensors": checked,
        "bit_exact_matches": checked - len(mismatches),
        "mismatches": mismatches,
    }
    print(f"[alpha=0] Checked {checked} MLX tensors against {base_mlx_dir}: {checked - len(mismatches)}/{checked} BIT-EXACT!")
    if mismatches:
        for m in mismatches[:10]:
            print("  MISMATCH:", m)
        raise AssertionError(f"alpha=0 verification FAILED with {len(mismatches)} mismatches!")
    return res


def build_output_checkpoint(
    staging: Path,
    base_mlx_dir: Path,
    output_dir: Path,
    alpha: float,
    overwrite: bool = False,
) -> dict:
    mx, _ = _load_mlx_modules()

    base_resolved = base_mlx_dir.resolve()
    out_resolved = output_dir.resolve()
    if out_resolved == base_resolved:
        raise ValueError(f"Refusing to overwrite base checkpoint in place: {out_resolved}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists (pass --overwrite to replace): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updates, stats = compute_merged_updates(staging, base_mlx_dir, alpha=alpha)
    assert len(updates) == 435, f"Expected 435 MLX updated tensors (145*3), got {len(updates)}"

    idx = json.loads((base_mlx_dir / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shards = sorted(set(wmap.values()))

    for item in sorted(base_mlx_dir.iterdir()):
        if item.name.endswith(".safetensors") or item.name.startswith(".") and item.name != ".gitattributes":
            continue
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)

    total_written = 0
    total_replaced = 0
    for shard_name in shards:
        src_shard = base_mlx_dir / shard_name
        with open(src_shard, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            orig_hdr = json.loads(f.read(hlen).decode("utf-8"))
        orig_meta = orig_hdr.get("__metadata__") or {}

        shard_w = mx.load(str(src_shard))
        replaced_here = 0
        for k in list(shard_w.keys()):
            if k in updates:
                new_arr = updates[k]
                old_arr = shard_w[k]
                assert new_arr.shape == old_arr.shape, f"Shape mismatch on {k}: {new_arr.shape} vs {old_arr.shape}"
                assert new_arr.dtype == old_arr.dtype, f"Dtype mismatch on {k}: {new_arr.dtype} vs {old_arr.dtype}"
                if new_arr.dtype in (mx.float32, mx.bfloat16, mx.float16):
                    assert not bool(mx.any(mx.isnan(new_arr)).item()), f"NaN in {k}"
                    assert not bool(mx.any(mx.isinf(new_arr)).item()), f"Inf in {k}"
                shard_w[k] = new_arr
                replaced_here += 1

        out_shard = output_dir / shard_name
        mx.save_safetensors(str(out_shard), shard_w, metadata=orig_meta)
        total_written += len(shard_w)
        total_replaced += replaced_here
        print(f"[build] Wrote {out_shard.name}: {len(shard_w)} tensors ({replaced_here} updated)")

    assert total_replaced == len(updates), f"Replaced {total_replaced} != {len(updates)}"
    stats["base_mlx_dir"] = str(base_mlx_dir)
    stats["output_dir"] = str(output_dir)
    stats["total_tensors_written"] = total_written
    stats["total_tensors_replaced"] = total_replaced
    (output_dir / "tess_merge_report.json").write_text(json.dumps(stats, indent=2))
    print(f"[build] Complete: {output_dir} ({total_replaced}/{total_written} MLX tensors updated, 0 NaN/Inf)")
    return stats


def clone_pass_variant(src_dir: Path, dst_dir: Path, num_passes: int, overwrite: bool = False) -> None:
    if dst_dir.resolve() == src_dir.resolve():
        raise ValueError("Clone destination equals source")
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Clone dir already exists (pass --overwrite): {dst_dir}")
        shutil.rmtree(dst_dir)
    subprocess.run(["cp", "-cR", str(src_dir), str(dst_dir)], check=True)
    cfg_path = dst_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["text_config"]["loop_config"]["num_passes"] = int(num_passes)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[clone] Created APFS clone {dst_dir} with num_passes={num_passes}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply migtissera/Tess-4.5-26B-A3B task vectors to MLX loopling.")
    ap.add_argument("--tess-repo", default=DEFAULT_TESS_REPO)
    ap.add_argument("--tess-rev", default=DEFAULT_TESS_REV)
    ap.add_argument("--mini-repo", default=DEFAULT_MINI_REPO)
    ap.add_argument("--mini-rev", default=DEFAULT_MINI_REV)
    ap.add_argument("--varloop-repo", default=DEFAULT_VARLOOP_REPO)
    ap.add_argument("--varloop-rev", default=DEFAULT_VARLOOP_REV)
    ap.add_argument("--base-mlx", default=os.path.expanduser("~/loopling/models/loopling-p3"))
    ap.add_argument("--staging", default=os.path.expanduser("~/loopling/staging/tess_merge"))
    ap.add_argument("--output", default=os.path.expanduser("~/loopling/models/loopling-p3-tess"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fetch", action="store_true", help="Range-fetch required tensors into --staging")
    ap.add_argument("--audit", action="store_true", help="Audit 145 edited and 0 hidden edits between Tess and Mini")
    ap.add_argument("--verify-alpha0", action="store_true", help="Verify alpha=0 matches --base-mlx bit-for-bit")
    ap.add_argument("--build", action="store_true", help="Build --output checkpoint with --alpha")
    ap.add_argument("--clone-p4", default=None, help="Optional APFS clone path with num_passes=4")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing --output / --clone-p4")
    args = ap.parse_args()

    if not any([args.fetch, args.audit, args.verify_alpha0, args.build, args.clone_p4]):
        ap.error("Specify at least one action: --fetch, --audit, --verify-alpha0, --build, or --clone-p4")

    if args.fetch or args.audit:
        run_fetch_and_audit(args)

    staging = Path(args.staging).expanduser()
    base_mlx = Path(args.base_mlx).expanduser()
    output = Path(args.output).expanduser()

    if args.verify_alpha0:
        verify_alpha0(staging, base_mlx)

    if args.build:
        stats = build_output_checkpoint(staging, base_mlx, output, alpha=args.alpha, overwrite=args.overwrite)
        print("\n=== Merge Summary ===")
        print(json.dumps(stats, indent=2))

    if args.clone_p4:
        clone_pass_variant(output, Path(args.clone_p4).expanduser(), num_passes=4, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
