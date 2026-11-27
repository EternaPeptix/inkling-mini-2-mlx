"""Runtime integration patches hooking vllm-mlx onto Inkling 2 (`inkling2_loop`)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = False

_INKLING2_SPECIAL_RE = (
    r"<\|message_model\|>|<\|message_user\|>|<\|message_system\|>|<\|message_tool\|>|"
    r"<\|content_text\|>|<\|content_thinking\|>|<\|end_message\|>|"
    r"<\|content_model_end_sampling\|>|<\|content_image\|>|<\|content_xml\|>|"
    r"<\|content_tool_error\|>|<\|begin_of_text\|>"
)


def install() -> None:
    """Apply idempotent patches for Inkling 2 model routing, template prep, SSD cache, and perf."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _patch_mllm_detection_and_tokens()
    _patch_chat_template_safety()
    _patch_ssd_cache_list()
    _patch_scheduler_insert_need_logprobs()
    logger.info("[vllm_mlx_inkling2] Inkling 2 patches installed")


def _patch_mllm_detection_and_tokens() -> None:
    from vllm_mlx.api import utils as api_utils

    orig_config_vlm = api_utils._config_indicates_vlm

    def _patched_config_indicates_vlm(config: dict) -> bool:
        if config.get("model_type") == "inkling2_loop":
            return False
        text_cfg = config.get("text_config")
        if isinstance(text_cfg, dict) and text_cfg.get("model_type") == "inkling2_loop":
            return False
        return orig_config_vlm(config)

    api_utils._config_indicates_vlm = _patched_config_indicates_vlm

    base_pattern = re.compile(api_utils.SPECIAL_TOKENS_PATTERN.pattern + r"|<\|content_model_end_sampling\|>|<\|begin_of_text\|>")
    api_utils.SPECIAL_TOKENS_PATTERN = base_pattern

    # Only strip message/content/end_message in server post-parser cleanup (NOT in batched.clean_output_text,
    # which runs BEFORE _extract_reasoning_and_tool_calls!).
    post_parse_pattern = re.compile(base_pattern.pattern + "|" + _INKLING2_SPECIAL_RE)

    if hasattr(api_utils, "_TOOL_CALL_TAGS"):
        for pair in (
            ("<|content_invoke_tool_json|>", "<|end_message|>"),
            ("<|content_invoke_tool_text|>", "<|end_message|>"),
        ):
            if pair not in api_utils._TOOL_CALL_TAGS:
                api_utils._TOOL_CALL_TAGS.append(pair)

    import vllm_mlx.server as vserver

    orig_clean = vserver.clean_output_text

    def _post_clean_output_text(text: str) -> str:
        cleaned = orig_clean(text)
        return post_parse_pattern.sub("", cleaned).strip() if cleaned else cleaned

    vserver.clean_output_text = _post_clean_output_text
    vserver.SPECIAL_TOKENS_PATTERN = post_parse_pattern
    if hasattr(vserver, "_TOOL_MARKUP_PATTERN"):
        vserver._TOOL_MARKUP_PATTERN = re.compile(
            vserver._TOOL_MARKUP_PATTERN.pattern
            + r"|<\|content_invoke_tool_json\|>|<\|content_invoke_tool_text\|>|<\|message_model\|>|<\|content_text\|>|<\|end_message\|>|<\|content_model_end_sampling\|>"
        )
    if hasattr(vserver, "_STREAMING_TOOL_MARKERS"):
        markers = tuple(vserver._STREAMING_TOOL_MARKERS)
        for m in ("<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>"):
            if m not in markers:
                markers = markers + (m,)
        vserver._STREAMING_TOOL_MARKERS = markers


def _patch_chat_template_safety() -> None:
    from vllm_mlx.engine import batched as engine_batched
    from vllm_mlx.engine import chat_template_safety as cts

    orig_norm = cts.normalize_messages_for_chat_template

    def _patched_normalize(messages: list[Any]) -> list[dict]:
        normalized = orig_norm(messages)
        for msg in normalized:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if not msg.get("reasoning_content") and msg.get("reasoning"):
                msg["reasoning_content"] = msg["reasoning"]
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        args = fn.get("arguments")
                        if args is None or args == "" or args == {"value": ""}:
                            fn["arguments"] = {}
                        elif isinstance(args, str):
                            s = args.strip()
                            if not s:
                                fn["arguments"] = {}
                            else:
                                parsed = json.loads(s)
                                fn["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
        return normalized

    cts.normalize_messages_for_chat_template = _patched_normalize
    engine_batched.normalize_messages_for_chat_template = _patched_normalize


def _patch_ssd_cache_list() -> None:
    """Allow CacheList(KVCache/RotatingKVCache, ArraysCache) layers to spill/reload on SSD."""
    from safetensors.numpy import load_file, save_file
    import mlx.core as mx
    import numpy as np

    from vllm_mlx import scheduler as vsched
    from vllm_mlx import ssd_cache as vssd

    kv_ser = vssd.KVCacheSerializer()
    arr_ser = vssd.ArraysCacheSerializer()

    def _snapshot_arrays_sub(sub: Any) -> dict[str, Any]:
        raw_list = sub.cache if hasattr(sub, "cache") else sub.state
        state_np: list[np.ndarray] = []
        original_dtypes: list[str | None] = []
        for arr in raw_list:
            np_arr, orig = vssd._mx_to_numpy_safe(mx.contiguous(arr))
            state_np.append(np.ascontiguousarray(np_arr))
            original_dtypes.append(orig)
        snap: dict[str, Any] = {"state_np": state_np}
        if any(d is not None for d in original_dtypes):
            snap["state_original_dtypes"] = original_dtypes
        return snap

    def _snapshot_layer(self, layer: Any) -> dict[str, Any]:
        sub_snapshots = []
        for sub in layer.caches:
            if hasattr(sub, "keys") and hasattr(sub, "values") and hasattr(sub, "offset"):
                k_np, k_orig = vssd._mx_to_numpy_safe(mx.contiguous(sub.keys))
                v_np, v_orig = vssd._mx_to_numpy_safe(mx.contiguous(sub.values))
                snap: dict[str, Any] = {
                    "keys_np": np.ascontiguousarray(k_np),
                    "values_np": np.ascontiguousarray(v_np),
                    "offset": sub.offset,
                }
                k_orig = getattr(sub, "_ssd_keys_original_dtype", None) or k_orig
                v_orig = getattr(sub, "_ssd_values_original_dtype", None) or v_orig
                if k_orig is not None:
                    snap["keys_original_dtype"] = k_orig
                if v_orig is not None:
                    snap["values_original_dtype"] = v_orig
                for attr in kv_ser._ROTATING_ATTRS:
                    if hasattr(sub, attr):
                        snap[attr] = getattr(sub, attr)
                sub_snapshots.append(("kv", snap))
            elif hasattr(sub, "cache") or hasattr(sub, "state"):
                sub_snapshots.append(("arrays", _snapshot_arrays_sub(sub)))
            else:
                raise ValueError(f"Unsupported CacheList sub-cache {type(sub).__name__} for SSD spill")
        return {"sub_snapshots": sub_snapshots}

    def _serialize_layer(self, snapshot: dict[str, Any], layer_idx: int, file_path: str) -> dict[str, Any]:
        tensors: dict[str, np.ndarray] = {}
        sub_meta: list[dict[str, Any]] = []
        for j, (kind, sub) in enumerate(snapshot["sub_snapshots"]):
            if kind == "kv":
                keys_np = sub["keys_np"]
                values_np = sub["values_np"]
                m: dict[str, Any] = {"sub_kind": "kv", "offset": sub["offset"]}
                if getattr(keys_np, "size", 1) == 0:
                    m["keys_empty"] = [list(keys_np.shape), str(keys_np.dtype)]
                else:
                    tensors[f"sub_{j}_keys"] = keys_np
                if getattr(values_np, "size", 1) == 0:
                    m["values_empty"] = [list(values_np.shape), str(values_np.dtype)]
                else:
                    tensors[f"sub_{j}_values"] = values_np
                for k in ("keys_original_dtype", "values_original_dtype"):
                    if k in sub:
                        m[k] = sub[k]
                for attr in kv_ser._ROTATING_ATTRS:
                    if attr in sub:
                        m[attr] = sub[attr]
                sub_meta.append(m)
            else:
                state_np = sub["state_np"]
                m = {"sub_kind": "arrays", "num_arrays": len(state_np)}
                for k, arr in enumerate(state_np):
                    if getattr(arr, "size", 1) == 0:
                        m.setdefault("empty_arrays", {})[str(k)] = [list(arr.shape), str(arr.dtype)]
                    else:
                        tensors[f"sub_{j}_state_{k}"] = arr
                if "state_original_dtypes" in sub:
                    m["state_original_dtypes"] = sub["state_original_dtypes"]
                sub_meta.append(m)

        if not tensors:
            tensors["__placeholder__"] = np.zeros((1,), dtype=np.uint8)
        save_file(tensors, file_path)
        return {
            "layer_type": "CacheList",
            "layer_idx": layer_idx,
            "sub_count": len(snapshot["sub_snapshots"]),
            "sub_meta": sub_meta,
        }

    def _deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
        sub_count = metadata["sub_count"]
        sub_meta = metadata["sub_meta"]
        tensors = load_file(file_path)
        subs = []
        for j in range(sub_count):
            m = sub_meta[j]
            kind = m.get("sub_kind", "kv")
            if kind == "arrays":
                num_arrays = m["num_arrays"]
                empty_map = m.get("empty_arrays", {})
                state = []
                for k in range(num_arrays):
                    skey = f"sub_{j}_state_{k}"
                    if skey in tensors:
                        state.append(tensors[skey])
                    else:
                        shape, dt = empty_map[str(k)]
                        state.append(np.zeros(shape, dtype=np.dtype(dt)))
                sd: dict[str, Any] = {"sub_kind": "arrays", "state": state}
                if "state_original_dtypes" in m:
                    sd["state_original_dtypes"] = m["state_original_dtypes"]
                subs.append(sd)
            else:
                kkey, vkey = f"sub_{j}_keys", f"sub_{j}_values"
                if kkey in tensors:
                    keys = tensors[kkey]
                else:
                    shape, dt = m["keys_empty"]
                    keys = np.zeros(shape, dtype=np.dtype(dt))
                if vkey in tensors:
                    values = tensors[vkey]
                else:
                    shape, dt = m["values_empty"]
                    values = np.zeros(shape, dtype=np.dtype(dt))
                sd = {"sub_kind": "kv", "keys": keys, "values": values, "offset": m["offset"]}
                for k in ("keys_original_dtype", "values_original_dtype", *kv_ser._ROTATING_ATTRS):
                    if k in m:
                        sd[k] = m[k]
                subs.append(sd)
        return {"cachelist_subs": subs}

    vssd.CacheListSerializer.snapshot_layer = _snapshot_layer
    vssd.CacheListSerializer.serialize_layer = _serialize_layer
    vssd.CacheListSerializer.deserialize_layer = _deserialize_layer

    orig_reconstruct = vsched.Scheduler._reconstruct_ssd_layers

    def _patched_reconstruct(self, layer_dicts: list[dict]) -> list | None:
        has_arrays_sub = any(
            isinstance(ld, dict)
            and "cachelist_subs" in ld
            and any(s.get("sub_kind") == "arrays" for s in ld["cachelist_subs"])
            for ld in layer_dicts
        )
        if not has_arrays_sub:
            return orig_reconstruct(self, layer_dicts)
        try:
            from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

            def _dt(name: str | None):
                return getattr(mx, name, None) if name else None

            def _mk_sub(d: dict):
                if d.get("sub_kind") == "arrays":
                    arrs = []
                    odts = d.get("state_original_dtypes")
                    for idx, a in enumerate(d["state"]):
                        t = mx.array(a)
                        if odts and idx < len(odts):
                            dt = _dt(odts[idx])
                            if dt is not None:
                                t = t.astype(dt)
                        arrs.append(t)
                    ac = ArraysCache(len(arrs))
                    ac.cache = arrs
                    return ac
                is_rot = "max_size" in d and d["max_size"] is not None
                kv = RotatingKVCache(max_size=d["max_size"], keep=d.get("keep", 0)) if is_rot else KVCache()
                kv.keys = mx.array(d["keys"])
                kv.values = mx.array(d["values"])
                for field, orig_key in (("keys", "keys_original_dtype"), ("values", "values_original_dtype")):
                    dt = _dt(d.get(orig_key))
                    if dt is not None:
                        setattr(kv, field, getattr(kv, field).astype(dt))
                kv.offset = d["offset"]
                for attr in ("step", "_idx"):
                    if attr in d and d[attr] is not None:
                        setattr(kv, attr, d[attr])
                return kv

            out = []
            for ld in layer_dicts:
                if "cachelist_subs" in ld:
                    out.append(CacheList(*[_mk_sub(s) for s in ld["cachelist_subs"]]))
                else:
                    sub = orig_reconstruct(self, [ld])
                    if not sub:
                        return None
                    out.extend(sub)
            return out
        except Exception as exc:
            logger.warning("[ssd_cache] CacheList reconstruction failed: %s", exc)
            return None

    vsched.Scheduler._reconstruct_ssd_layers = _patched_reconstruct


def _patch_scheduler_insert_need_logprobs() -> None:
    """Default `need_logprobs=[False]` in BatchGenerator.insert so BatchDecoder skips full-vocab logprobs."""
    import inspect
    from vllm_mlx import scheduler as vsched

    orig_create = vsched.Scheduler._create_batch_generator

    def _patched_create_bg(self, sampling_params):
        bg = orig_create(self, sampling_params)
        if "need_logprobs" in inspect.signature(bg.insert).parameters:
            orig_insert = bg.insert

            def _insert_with_need_lp(prompts, *args, **kwargs):
                if "need_logprobs" not in kwargs or kwargs["need_logprobs"] is None:
                    kwargs["need_logprobs"] = [False] * len(prompts)
                return orig_insert(prompts, *args, **kwargs)

            bg.insert = _insert_with_need_lp
        return bg

    vsched.Scheduler._create_batch_generator = _patched_create_bg
