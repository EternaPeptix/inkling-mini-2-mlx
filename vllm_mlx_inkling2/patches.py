"""Runtime integration patches hooking vllm-mlx onto Inkling 2 (`inkling2_loop`)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
import inspect
import json
import logging
import os
import re
import time
from typing import Any
import weakref

logger = logging.getLogger(__name__)

_INSTALLED = False
_ACTIVE_SCHEDULERS: weakref.WeakSet[Any] = weakref.WeakSet()

_INKLING2_SPECIAL_RE = (
    r"<\|message_model\|>|<\|message_user\|>|<\|message_system\|>|<\|message_tool\|>|"
    r"<\|content_text\|>|<\|content_thinking\|>|<\|end_message\|>|"
    r"<\|content_model_end_sampling\|>|<\|content_image\|>|<\|content_xml\|>|"
    r"<\|content_tool_error\|>|<\|begin_of_text\|>"
)


def install() -> None:
    """Apply idempotent patches for Inkling 2 model routing, template prep, cache, SSD, and warmup."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _patch_mllm_detection_and_tokens()
    _patch_chat_template_safety()
    _patch_ssd_cache_list()
    _patch_cache_list_trim_and_config()
    _patch_prompt_cache_and_batch_hooks()
    _patch_cascade_prefix_pinning()
    _patch_startup_warmup()
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

    base_pattern = re.compile(
        api_utils.SPECIAL_TOKENS_PATTERN.pattern + r"|<\|content_model_end_sampling\|>|<\|begin_of_text\|>"
    )
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
    """Bit-exact zero-upcast (`bf16->uint16`) + batched-eval + parallel `mx.load()` SSD spill/reload for CacheList."""
    from safetensors.numpy import load_file, save_file
    import mlx.core as mx
    import numpy as np

    from vllm_mlx import scheduler as vsched
    from vllm_mlx import ssd_cache as vssd

    kv_ser = vssd.KVCacheSerializer()

    def _prep_mx_array(arr: Any, eval_targets: list[Any]) -> tuple[Any, str | None, str | None]:
        """Prepare an mx.array for zero-upcast numpy copy; returns (arr_or_np, view_dtype, orig_dtype)."""
        if isinstance(arr, mx.array):
            if arr.dtype == mx.bfloat16:
                a = mx.contiguous(arr).view(mx.uint16)
                eval_targets.append(a)
                return a, "bfloat16", None
            a = mx.contiguous(arr)
            eval_targets.append(a)
            return a, None, None
        return np.ascontiguousarray(arr), None, None

    def _to_host_np(arr: Any) -> np.ndarray:
        if isinstance(arr, mx.array):
            # np.array(arr) copies into a standalone C-contiguous host buffer so MLX
            # releases the unified-memory allocation immediately on RAM eviction.
            out = np.array(arr)
            return out if out.flags.c_contiguous else np.ascontiguousarray(out)
        return arr if (isinstance(arr, np.ndarray) and arr.flags.c_contiguous) else np.ascontiguousarray(arr)

    def _collect_cachelist_plans(layers: list[Any]) -> tuple[list[tuple[Any, list[Any]]], list[Any]]:
        eval_targets: list[Any] = []
        plans: list[tuple[Any, list[Any]]] = []
        for layer in layers:
            sub_plans = []
            for sub in layer.caches:
                if hasattr(sub, "keys") and hasattr(sub, "values") and hasattr(sub, "offset"):
                    if getattr(sub, "keys", None) is None or getattr(sub, "values", None) is None:
                        raise ValueError(
                            f"{type(sub).__name__} keys/values are not materialized (None); cannot spill to SSD"
                        )
                    k_arr, k_view, k_orig = _prep_mx_array(sub.keys, eval_targets)
                    v_arr, v_view, v_orig = _prep_mx_array(sub.values, eval_targets)
                    k_orig = getattr(sub, "_ssd_keys_original_dtype", None) or k_orig
                    v_orig = getattr(sub, "_ssd_values_original_dtype", None) or v_orig
                    rot_attrs = {
                        attr: getattr(sub, attr) for attr in kv_ser._ROTATING_ATTRS if hasattr(sub, attr)
                    }
                    sub_plans.append(("kv", k_arr, v_arr, k_view, v_view, k_orig, v_orig, sub.offset, rot_attrs))
                elif hasattr(sub, "cache") or hasattr(sub, "state"):
                    raw_list = sub.cache if hasattr(sub, "cache") else sub.state
                    arr_items = []
                    for a in raw_list:
                        if a is None:
                            arr_items.append((np.zeros((0,), dtype=np.float32), None, None))
                        else:
                            arr_items.append(_prep_mx_array(a, eval_targets))
                    sub_plans.append(("arrays", arr_items))
                else:
                    raise ValueError(f"Unsupported CacheList sub-cache {type(sub).__name__} for SSD spill")
            plans.append((vssd.CacheListSerializer(), sub_plans))
        return plans, eval_targets

    def _materialize_cachelist_plans(plans: list[tuple[Any, list[Any]]]) -> list[tuple[Any, dict[str, Any]]]:
        layer_snapshots: list[tuple[Any, dict[str, Any]]] = []
        for serializer, sub_plans in plans:
            sub_snapshots = []
            for item in sub_plans:
                if item[0] == "kv":
                    _, k_arr, v_arr, k_view, v_view, k_orig, v_orig, offset, rot_attrs = item
                    snap: dict[str, Any] = {
                        "keys_np": _to_host_np(k_arr),
                        "values_np": _to_host_np(v_arr),
                        "offset": offset,
                    }
                    if k_view is not None:
                        snap["keys_view_dtype"] = k_view
                    if v_view is not None:
                        snap["values_view_dtype"] = v_view
                    if k_orig is not None:
                        snap["keys_original_dtype"] = k_orig
                    if v_orig is not None:
                        snap["values_original_dtype"] = v_orig
                    snap.update(rot_attrs)
                    sub_snapshots.append(("kv", snap))
                else:
                    _, arr_items = item
                    state_np: list[np.ndarray] = []
                    view_dtypes: list[str | None] = []
                    orig_dtypes: list[str | None] = []
                    for a, vd, od in arr_items:
                        state_np.append(_to_host_np(a))
                        view_dtypes.append(vd)
                        orig_dtypes.append(od)
                    snap = {"state_np": state_np}
                    if any(d is not None for d in view_dtypes):
                        snap["state_view_dtypes"] = view_dtypes
                    if any(d is not None for d in orig_dtypes):
                        snap["state_original_dtypes"] = orig_dtypes
                    sub_snapshots.append(("arrays", snap))
            layer_snapshots.append((serializer, {"sub_snapshots": sub_snapshots}))
        return layer_snapshots

    def _snapshot_layer(self, layer: Any) -> dict[str, Any]:
        plans, eval_targets = _collect_cachelist_plans([layer])
        if eval_targets:
            mx.eval(*eval_targets)
        return _materialize_cachelist_plans(plans)[0][1]

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
                for k in (
                    "keys_view_dtype",
                    "values_view_dtype",
                    "keys_original_dtype",
                    "values_original_dtype",
                ):
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
                for k in ("state_view_dtypes", "state_original_dtypes"):
                    if k in sub:
                        m[k] = sub[k]
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
        use_mx_load = os.environ.get("LOOPLING_SSD_MX_LOAD", "1") not in ("0", "", "false", "False")
        tensors: dict[str, Any]
        if use_mx_load:
            try:
                tensors = mx.load(file_path)
                if tensors:
                    # Materialize on the loader thread so the returned buffers have
                    # zero lazy-graph / cross-thread stream affinity AND I/O runs
                    # genuinely in parallel across ThreadPoolExecutor workers.
                    mx.eval(*tensors.values())
            except Exception:
                tensors = load_file(file_path)
        else:
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
                for k in ("state_view_dtypes", "state_original_dtypes"):
                    if k in m:
                        sd[k] = m[k]
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
                for k in (
                    "keys_view_dtype",
                    "values_view_dtype",
                    "keys_original_dtype",
                    "values_original_dtype",
                    *kv_ser._ROTATING_ATTRS,
                ):
                    if k in m:
                        sd[k] = m[k]
                subs.append(sd)
        return {"cachelist_subs": subs}

    vssd.CacheListSerializer.snapshot_layer = _snapshot_layer
    vssd.CacheListSerializer.serialize_layer = _serialize_layer
    vssd.CacheListSerializer.deserialize_layer = _deserialize_layer

    orig_enqueue_spill = vssd.SSDCacheTier.enqueue_spill

    def _patched_enqueue_spill(
        self,
        tokens: tuple[int, ...],
        cache: list[Any],
        memory_bytes: int,
    ) -> bool:
        """Fast path for all-CacheList stacks: single batched mx.eval across all layers + uint16 bf16 view."""
        if not cache or not all(isinstance(getattr(layer, "caches", None), (list, tuple)) for layer in cache):
            return orig_enqueue_spill(self, tokens, cache, memory_bytes)
        with self._lifecycle_lock:
            if not self._accepting_spills:
                return False
        try:
            plans, eval_targets = _collect_cachelist_plans(cache)
            if eval_targets:
                mx.eval(*eval_targets)
            layer_snapshots = _materialize_cachelist_plans(plans)
        except Exception:
            logger.exception(
                "[ssd_cache] failed to snapshot CacheList layers for spill (%d tokens) — entry dropped",
                len(tokens),
            )
            return False

        import queue

        with self._lifecycle_lock:
            if not self._accepting_spills:
                return False
            try:
                self._spill_queue.put_nowait((tokens, layer_snapshots, memory_bytes))
                return True
            except queue.Full:
                logger.warning(
                    "[ssd_cache] spill queue full, dropping entry (%d tokens, %d bytes)",
                    len(tokens),
                    memory_bytes,
                )
                return False

    vssd.SSDCacheTier.enqueue_spill = _patched_enqueue_spill

    orig_read_entry = vssd.SSDCacheTier._read_entry

    def _patched_read_entry(self, tokens: tuple[int, ...], relative_path: str) -> list | None:
        entry_dir = os.path.join(self._data_dir, relative_path)
        manifest_path = os.path.join(entry_dir, "manifest.json")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[ssd_cache] corrupt manifest for %s: %s", relative_path, e)
            self._quarantine_entry(tokens, relative_path)
            return None

        layers_meta = manifest.get("layers", [])
        if len(layers_meta) <= 1 or not all(lm.get("layer_type") == "CacheList" for lm in layers_meta):
            return orig_read_entry(self, tokens, relative_path)

        ser = vssd.CacheListSerializer()

        def _load_one(lm: dict[str, Any]) -> dict[str, Any]:
            lpath = os.path.join(entry_dir, f"layer_{lm['layer_idx']}.safetensors")
            return ser.deserialize_layer(lpath, lm)

        workers = min(int(os.environ.get("LOOPLING_SSD_READ_WORKERS", "8")), len(layers_meta))
        try:
            if workers <= 1:
                return [_load_one(lm) for lm in layers_meta]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_load_one, layers_meta))
        except Exception as e:
            logger.warning("[ssd_cache] parallel read failed in %s: %s", relative_path, e)
            self._quarantine_entry(tokens, relative_path)
            return None

    vssd.SSDCacheTier._read_entry = _patched_read_entry

    orig_reconstruct = vsched.Scheduler._reconstruct_ssd_layers

    def _patched_reconstruct(self, layer_dicts: list[dict]) -> list | None:
        has_cachelist = any(isinstance(ld, dict) and "cachelist_subs" in ld for ld in layer_dicts)
        if not has_cachelist:
            return orig_reconstruct(self, layer_dicts)
        try:
            from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

            def _dt(name: str | None):
                return getattr(mx, name, None) if name else None

            def _restore_arr(a: Any, view_name: str | None, orig_name: str | None) -> mx.array:
                t = a if isinstance(a, mx.array) else mx.array(a)
                if view_name is not None:
                    vdt = _dt(view_name)
                    if vdt is not None and t.dtype == mx.uint16:
                        return t.view(vdt)
                if orig_name is not None:
                    odt = _dt(orig_name)
                    if odt is not None and t.dtype != odt:
                        return t.astype(odt)
                return t

            def _mk_sub(d: dict):
                if d.get("sub_kind") == "arrays":
                    arrs = []
                    vdts = d.get("state_view_dtypes")
                    odts = d.get("state_original_dtypes")
                    for idx, a in enumerate(d["state"]):
                        vd = vdts[idx] if (vdts and idx < len(vdts)) else None
                        od = odts[idx] if (odts and idx < len(odts)) else None
                        arrs.append(_restore_arr(a, vd, od))
                    ac = ArraysCache(len(arrs))
                    ac.cache = arrs
                    return ac
                is_rot = "max_size" in d and d["max_size"] is not None
                kv = RotatingKVCache(max_size=d["max_size"], keep=d.get("keep", 0)) if is_rot else KVCache()
                kv.keys = _restore_arr(d["keys"], d.get("keys_view_dtype"), d.get("keys_original_dtype"))
                kv.values = _restore_arr(d["values"], d.get("values_view_dtype"), d.get("values_original_dtype"))
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


def _patch_cache_list_trim_and_config() -> None:
    """Recurse `_trim_to_offset` into `CacheList` children and lower default `min_prefix_tokens`."""
    from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache
    from vllm_mlx import memory_cache as vmc

    orig_needs_trim = vmc._needs_kv_trim
    orig_trim = vmc._trim_to_offset

    def _sub_needs_trim(layer: Any) -> bool:
        if isinstance(layer, CacheList) or (
            hasattr(layer, "caches") and isinstance(getattr(layer, "caches", None), (list, tuple))
        ):
            return any(_sub_needs_trim(sub) for sub in layer.caches)
        if isinstance(layer, RotatingKVCache):
            keys = getattr(layer, "keys", None)
            if keys is None or not hasattr(keys, "shape") or len(keys.shape) < 3:
                return False
            off = layer.offset
            # Never trim a wrapped rotating window; only strip unused pre-allocated tail slots.
            return 0 < off < layer.max_size and off < keys.shape[2] and getattr(layer, "_idx", off) == off
        return orig_needs_trim(layer)

    def _trim_single(layer: Any) -> Any:
        if isinstance(layer, CacheList) or (
            hasattr(layer, "caches") and isinstance(getattr(layer, "caches", None), (list, tuple))
        ):
            if not any(_sub_needs_trim(sub) for sub in layer.caches):
                return layer
            trimmed_subs = [_trim_single(sub) for sub in layer.caches]
            out = copy.copy(layer)
            out.caches = tuple(trimmed_subs) if isinstance(layer.caches, tuple) else trimmed_subs
            return out
        if isinstance(layer, RotatingKVCache) and layer.keys is not None:
            off = layer.offset
            if 0 < off < layer.max_size and off < layer.keys.shape[2] and getattr(layer, "_idx", off) == off:
                tc = RotatingKVCache(max_size=layer.max_size, keep=layer.keep)
                tc.keys = layer.keys[:, :, :off, :]
                tc.values = layer.values[:, :, :off, :]
                tc.offset = off
                tc._idx = off
                tc.step = layer.step
                return tc
            return layer
        if isinstance(layer, KVCache) and layer.keys is not None:
            off = layer.offset
            if 0 < off < layer.keys.shape[2]:
                tc = KVCache()
                tc.keys = layer.keys[:, :, :off, :]
                tc.values = layer.values[:, :, :off, :]
                tc.offset = off
                return tc
            return layer
        return layer

    def _patched_trim_to_offset(cache: list[Any]) -> list[Any]:
        if not any(_sub_needs_trim(layer) for layer in cache):
            return cache
        if any(isinstance(getattr(layer, "caches", None), (list, tuple)) for layer in cache):
            return [_trim_single(layer) for layer in cache]
        return orig_trim(cache)

    vmc._needs_kv_trim = _sub_needs_trim
    vmc._trim_to_offset = _patched_trim_to_offset

    # Lower default min_prefix_tokens from 128 -> 32 (configurable via env) so short prompts cache too.
    default_min_tok = int(os.environ.get("LOOPLING_MIN_PREFIX_TOKENS", "32"))
    orig_mc_init = vmc.MemoryAwarePrefixCache.__init__

    def _patched_mc_init(self, model: Any, config: Any = None, *args, **kwargs):
        if config is None:
            config = vmc.MemoryCacheConfig(min_prefix_tokens=default_min_tok)
        elif isinstance(config, vmc.MemoryCacheConfig) and config.min_prefix_tokens == 128:
            config = dataclasses.replace(config, min_prefix_tokens=default_min_tok)
        return orig_mc_init(self, model, config, *args, **kwargs)

    vmc.MemoryAwarePrefixCache.__init__ = _patched_mc_init


def _clone_cache_wrappers(cache: list[Any]) -> list[Any]:
    """Zero-copy shallow clone of cache wrappers (and `ArraysCache.cache` lists).

    MLX arrays are immutable (writes rebind attributes/list entries), so cloning
    the Python wrapper objects guarantees BatchDecoder's 1-token direct-admit
    path (`pack_b1`/`update_packed_b1`) can never mutate a stored prefix-cache
    entry in place, at <0.05ms and 0 GPU bytes copied.
    """

    def _clone_layer(layer: Any) -> Any:
        children = getattr(layer, "caches", None)
        if isinstance(children, (list, tuple)):
            c = copy.copy(layer)
            cloned_kids = [_clone_layer(k) for k in children]
            c.caches = tuple(cloned_kids) if isinstance(children, tuple) else cloned_kids
            return c
        c = copy.copy(layer)
        if hasattr(c, "cache") and isinstance(c.cache, list):
            c.cache = list(c.cache)
        return c

    return [_clone_layer(layer) for layer in cache]


class _SuppressPromptCacheSaveWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Prompt cache save disabled: prompt-cache-save is unavailable" not in record.getMessage()


def _patch_prompt_cache_and_batch_hooks() -> None:
    """Install bit-exact N-1 prompt-cache capture, 1-tok wrapper clone, and need_logprobs=[False]."""
    from mlx_lm.generate import PromptProcessingBatch
    from vllm_mlx import scheduler as vsched

    # 1. Hook PromptProcessingBatch.generate to snapshot exact N-1 (prompt[:-1]) caches right before
    #    GenerationBatch feeds prompt[-1] and allocates +256 step padding.
    orig_ppb_generate = PromptProcessingBatch.generate

    def _patched_ppb_generate(self, tokens: list[list[int]]):
        if any(len(t) > 1 for t in tokens):
            self.prompt([t[:-1] for t in tokens])
            tokens = [[t[-1]] for t in tokens]

        if self.uids and self.prompt_cache:
            owner = None
            for sched in list(_ACTIVE_SCHEDULERS):
                bg = getattr(sched, "batch_generator", None)
                if (
                    bg is not None
                    and bg.model is self.model
                    and sched.memory_aware_cache is not None
                    and any(
                        (rid := sched.uid_to_request_id.get(uid)) is not None
                        and getattr(sched.running.get(rid), "batch_uid", None) == uid
                        for uid in self.uids
                    )
                ):
                    owner = sched
                    break

            if owner is not None:
                mc = owner.memory_aware_cache
                min_toks = mc._config.min_prefix_tokens
                env_refresh = os.environ.get("LOOPLING_SNAPSHOT_REFRESH_TOKENS", "")
                for i, uid in enumerate(self.uids):
                    try:
                        rid = owner.uid_to_request_id.get(uid)
                        if not rid:
                            continue
                        req = owner.running.get(rid)
                        if req is None or getattr(req, "batch_uid", None) != uid:
                            continue
                        prompt_ids = req.prompt_token_ids
                        if not prompt_ids or len(prompt_ids) < 2:
                            continue
                        key = list(self.tokens[i])
                        if len(key) != len(prompt_ids) - 1 or key != list(prompt_ids[:-1]):
                            continue
                        if len(key) < min_toks:
                            req._prompt_cache_saved = True
                            continue
                        covered = getattr(req, "cached_tokens", 0) or 0
                        if env_refresh:
                            refresh_thresh = int(env_refresh)
                        elif len(key) < 4096:
                            refresh_thresh = 64
                        elif len(key) < 32768:
                            refresh_thresh = 512
                        else:
                            refresh_thresh = 4096
                        if covered > 0 and (len(key) - covered) <= refresh_thresh:
                            req._prompt_cache_saved = True
                            continue
                        if tuple(key) in mc._entries:
                            req._prompt_cache_saved = True
                            continue
                        extracted = self.extract_cache(i)
                        if not owner._prompt_output_entry_is_useless(extracted):
                            continue
                        cov = owner._cache_coverage(extracted)
                        if cov != len(key):
                            logger.debug(
                                "[cache_store_prompt_n1] coverage %s != key %d; skipping",
                                cov,
                                len(key),
                            )
                            continue
                        stored = mc.store(key, extracted, evict_prefixes=True)
                        req._prompt_cache_saved = True
                        logger.info(
                            "[cache_store_prompt_n1] request=%s key_tokens=%d prompt_tokens=%d stored=%s entries=%d",
                            rid[:12],
                            len(key),
                            len(prompt_ids),
                            stored,
                            len(mc._entries),
                        )
                    except Exception as exc:
                        logger.warning("[cache_store_prompt_n1] uid=%s snapshot failed: %s", uid, exc)

        return orig_ppb_generate(self, tokens)

    PromptProcessingBatch.generate = _patched_ppb_generate

    # 2. Skip redundant token-0 _store_prompt_only_cache (and its synchronous _flush_bd() + unpack_b1() stall)
    #    when N-1 capture already handled it or prompt is below min_prefix_tokens.
    orig_store_prompt_only = vsched.Scheduler._store_prompt_only_cache

    def _patched_store_prompt_only(self, request: Any, response: Any) -> None:
        if getattr(request, "_prompt_cache_saved", False):
            return
        mc = self.memory_aware_cache
        if mc is None:
            return
        prompt_ids = getattr(request, "prompt_token_ids", None)
        if not prompt_ids or len(prompt_ids) < mc._config.min_prefix_tokens:
            return
        return orig_store_prompt_only(self, request, response)

    vsched.Scheduler._store_prompt_only_cache = _patched_store_prompt_only

    # 3. Wrap _create_batch_generator to:
    #    - register active Scheduler for N-1 capture
    #    - silence legacy `_process_prompts` warning (replaced by our N-1 hook)
    #    - default `need_logprobs=[False]`
    #    - shallow-clone prefix-hit cache wrappers passed into `bg.insert`
    orig_create = vsched.Scheduler._create_batch_generator
    sched_logger = logging.getLogger("vllm_mlx.scheduler")
    warn_filter = _SuppressPromptCacheSaveWarning()

    def _patched_create_bg(self, sampling_params):
        _ACTIVE_SCHEDULERS.add(self)
        sched_logger.addFilter(warn_filter)
        try:
            bg = orig_create(self, sampling_params)
        finally:
            sched_logger.removeFilter(warn_filter)

        sig_params = inspect.signature(bg.insert).parameters
        has_need_lp = "need_logprobs" in sig_params
        orig_insert = bg.insert

        def _wrapped_insert(prompts, *args, **kwargs):
            if has_need_lp and ("need_logprobs" not in kwargs or kwargs["need_logprobs"] is None):
                kwargs["need_logprobs"] = [False] * len(prompts)
            caches_kw = kwargs.get("caches")
            if caches_kw is not None:
                kwargs["caches"] = [
                    _clone_cache_wrappers(c) if c is not None else None for c in caches_kw
                ]
            elif len(args) >= 2 and args[1] is not None:
                args_list = list(args)
                args_list[1] = [
                    _clone_cache_wrappers(c) if c is not None else None for c in args_list[1]
                ]
                args = tuple(args_list)
            return orig_insert(prompts, *args, **kwargs)

        bg.insert = _wrapped_insert
        return bg

    vsched.Scheduler._create_batch_generator = _patched_create_bg


def _patch_cascade_prefix_pinning() -> None:
    """Zero-copy Shared-Prefix Cascade pinning & `kv._cascade_ref` propagation (Phase 3b).

    For prefix-cache HITs with `cached_tokens >= LOOPLING_CASCADE_MIN_TOKENS` (default 32768):
    - Refcount-pins the matched `MemoryAwarePrefixCache` entry while requests reference it
      (skipping pinned entries in LRU eviction and prefix eviction; unpinned on finish,
      abort, error recovery, discard, and reset).
    - Tags every full-attention `KVCache` (`type(sub) is KVCache`) with:
      `sub._cascade_ref = (prefix_id, L_shared_blk, k_shared, v_shared)`
      where `L_shared_blk = (cached_tokens // 256) * 256` and `(k_shared, v_shared)` are
      the immutable read-only stored prefix buffers (zero 32 GiB copy).
    - Propagates `_cascade_ref` across `BatchKVCache.merge/extend/filter/extract` so
      concurrent requests with distinct private suffixes (`len(remaining) > 1`) retain
      `_cascade_ref` when entering `BatchDecoder`.
    """
    import itertools
    import threading

    from mlx_lm.models.cache import BatchKVCache, KVCache
    from vllm_mlx import memory_cache as vmc
    from vllm_mlx import scheduler as vsched

    _prefix_id_counter = itertools.count(1)
    _prefix_id_lock = threading.Lock()

    def _get_prefix_id(entry: Any) -> int:
        pid = getattr(entry, "_cascade_prefix_id", None)
        if pid is None:
            with _prefix_id_lock:
                pid = getattr(entry, "_cascade_prefix_id", None)
                if pid is None:
                    pid = next(_prefix_id_counter)
                    entry._cascade_prefix_id = pid
        return pid

    def _cascade_min_tokens() -> int:
        return int(os.environ.get("LOOPLING_CASCADE_MIN_TOKENS", "32768"))

    # 1. Propagate per-row `_cascade_ref` through BatchKVCache lifecycle.
    orig_bkv_merge = BatchKVCache.merge.__func__
    orig_bkv_extend = BatchKVCache.extend
    orig_bkv_filter = BatchKVCache.filter
    orig_bkv_extract = BatchKVCache.extract

    @classmethod
    def _patched_bkv_merge(cls, caches):
        bkv = orig_bkv_merge(cls, caches)
        refs = [getattr(c, "_cascade_ref", None) for c in caches]
        if any(r is not None for r in refs):
            bkv._cascade_refs = refs
        return bkv

    def _patched_bkv_extend(self, other):
        n_self = int(self.left_padding.shape[0]) if self.left_padding is not None else 0
        n_other = int(other.left_padding.shape[0]) if other.left_padding is not None else 0
        r_self = getattr(self, "_cascade_refs", None)
        r_other = getattr(other, "_cascade_refs", None)
        orig_bkv_extend(self, other)
        if r_self is not None or r_other is not None:
            a = list(r_self) if r_self is not None else [None] * n_self
            b = list(r_other) if r_other is not None else [None] * n_other
            self._cascade_refs = a + b

    def _patched_bkv_filter(self, batch_indices):
        r_self = getattr(self, "_cascade_refs", None)
        orig_bkv_filter(self, batch_indices)
        if r_self is not None:
            self._cascade_refs = [r_self[i] for i in batch_indices]

    def _patched_bkv_extract(self, idx):
        kv = orig_bkv_extract(self, idx)
        refs = getattr(self, "_cascade_refs", None)
        if refs is not None and 0 <= idx < len(refs):
            ref = refs[idx]
            if ref is not None and kv.offset >= ref[1]:
                kv._cascade_ref = ref
        return kv

    BatchKVCache.merge = _patched_bkv_merge
    BatchKVCache.extend = _patched_bkv_extend
    BatchKVCache.filter = _patched_bkv_filter
    BatchKVCache.extract = _patched_bkv_extract

    # 2. Protect pinned MemoryAwarePrefixCache entries from LRU eviction & prefix eviction.
    class _AllEntriesPinned(Exception):
        pass

    orig_evict_lru = vmc.MemoryAwarePrefixCache._evict_lru

    def _patched_evict_lru(self) -> None:
        evicted_key = None
        evicted_entry = None
        with self._memory_lock:
            if not self._entries:
                return
            for k, e in self._entries.items():
                if getattr(e, "_pin_count", 0) <= 0:
                    evicted_key = k
                    evicted_entry = e
                    break
            if evicted_key is None:
                raise _AllEntriesPinned()
            self._entries.pop(evicted_key)
            self._current_memory -= evicted_entry.memory_bytes
            self._remove_from_sorted(evicted_key)
            self._stats.evictions += 1
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        if self._ssd_tier is not None and evicted_entry is not None:
            self._ssd_tier.enqueue_spill(evicted_key, evicted_entry.cache, evicted_entry.memory_bytes)

    vmc.MemoryAwarePrefixCache._evict_lru = _patched_evict_lru

    orig_commit_prepared = vmc.MemoryAwarePrefixCache.commit_prepared

    def _patched_commit_prepared(self, entry, evict_prefixes: bool = True, **kwargs) -> bool:
        import bisect
        from contextlib import nullcontext

        commit_lock = kwargs.get("commit_lock", None)
        commit_guard = kwargs.get("commit_guard", None)

        # Fast-path when no entries in cache are pinned: delegate untouched to upstream.
        with self._memory_lock:
            any_pinned = any(getattr(e, "_pin_count", 0) > 0 for e in self._entries.values())
        if not any_pinned:
            return orig_commit_prepared(self, entry, evict_prefixes=evict_prefixes, **kwargs)

        tokens_key = entry.tokens
        commit_context = commit_lock if commit_lock is not None else nullcontext()
        try:
            with self._memory_lock, commit_context:
                if commit_guard is not None and not commit_guard():
                    return False
                if entry.memory_bytes > self._max_memory:
                    self._stats.store_rejections += 1
                    return False
                if tokens_key in self._entries:
                    self._entries.move_to_end(tokens_key)
                    return True

                if evict_prefixes and self._sorted_keys:
                    to_remove = []
                    idx = bisect.bisect_left(self._sorted_keys, tokens_key)
                    for i in range(idx - 1, -1, -1):
                        key = self._sorted_keys[i]
                        klen = len(key)
                        if klen >= len(tokens_key):
                            continue
                        if tokens_key[:klen] == key:
                            old_e = self._entries.get(key)
                            if old_e is not None and getattr(old_e, "_pin_count", 0) <= 0:
                                to_remove.append(key)
                        elif key[0] != tokens_key[0]:
                            break
                    for key in to_remove:
                        old = self._entries.pop(key)
                        self._current_memory -= old.memory_bytes
                        self._stats.evictions += 1
                        self._remove_from_sorted(key)

                while (
                    self._current_memory + entry.memory_bytes > self._max_memory
                    or len(self._entries) >= self._config.max_entries
                ) and self._entries:
                    self._evict_lru()

                self._entries[tokens_key] = entry
                self._current_memory += entry.memory_bytes
                bisect.insort(self._sorted_keys, tokens_key)
                self._stats.entry_count = len(self._entries)
                self._stats.current_memory_bytes = self._current_memory
            return True
        except _AllEntriesPinned:
            self._stats.store_rejections += 1
            logger.debug(
                "[cascade_pin] skipped store (%d tokens): all cached entries are pinned",
                len(tokens_key),
            )
            return False

    vmc.MemoryAwarePrefixCache.commit_prepared = _patched_commit_prepared

    # 3. Pin + tag full-attention KVCache layers on prefix HIT; unpin on all exit paths.
    def _tag_full_kv_layers(cloned_cache: list[Any], entry_cache: list[Any], prefix_id: int, L_shared_blk: int) -> int:
        tagged = 0
        for dst_layer, src_layer in zip(cloned_cache, entry_cache):
            dst_subs = getattr(dst_layer, "caches", None)
            src_subs = getattr(src_layer, "caches", None)
            if isinstance(dst_subs, (list, tuple)) and isinstance(src_subs, (list, tuple)):
                for dsub, ssub in zip(dst_subs, src_subs):
                    if type(dsub) is KVCache and type(ssub) is KVCache:
                        sk, sv = ssub.keys, ssub.values
                        if (
                            sk is not None
                            and sv is not None
                            and sk.ndim >= 3
                            and sk.shape[-2] >= L_shared_blk
                            and dsub.offset >= L_shared_blk
                        ):
                            dsub._cascade_ref = (prefix_id, L_shared_blk, sk, sv)
                            tagged += 1
            elif type(dst_layer) is KVCache and type(src_layer) is KVCache:
                sk, sv = src_layer.keys, src_layer.values
                if (
                    sk is not None
                    and sv is not None
                    and sk.ndim >= 3
                    and sk.shape[-2] >= L_shared_blk
                    and dst_layer.offset >= L_shared_blk
                ):
                    dst_layer._cascade_ref = (prefix_id, L_shared_blk, sk, sv)
                    tagged += 1
        return tagged

    def _maybe_pin_and_tag(sched: Any, request: Any) -> None:
        if getattr(request, "_cascade_pin", None) is not None:
            return
        mc = getattr(sched, "memory_aware_cache", None)
        if mc is None:
            return
        pc = getattr(request, "prompt_cache", None)
        if not pc:
            return
        cached_toks = getattr(request, "cached_tokens", 0) or 0
        min_toks = _cascade_min_tokens()
        if cached_toks < min_toks:
            return
        L_shared_blk = (cached_toks // 256) * 256
        if L_shared_blk < 256:
            return
        key_list = getattr(request, "prompt_cache_key", None)
        if not key_list:
            return
        tokens_key = tuple(key_list)
        with mc._memory_lock:
            entry = mc._entries.get(tokens_key)
            if entry is None:
                return
            entry._pin_count = getattr(entry, "_pin_count", 0) + 1
            prefix_id = _get_prefix_id(entry)
            entry_cache = entry.cache

        cloned = _clone_cache_wrappers(pc)
        tagged = _tag_full_kv_layers(cloned, entry_cache, prefix_id, L_shared_blk)
        if tagged <= 0:
            with mc._memory_lock:
                entry._pin_count = max(0, getattr(entry, "_pin_count", 0) - 1)
            return
        request.prompt_cache = cloned
        request._cascade_pin = (mc, entry)
        logger.debug(
            "[cascade_pin] request=%s pinned prefix_id=%d L_shared_blk=%d tagged_layers=%d pin_count=%d",
            request.request_id[:12],
            prefix_id,
            L_shared_blk,
            tagged,
            entry._pin_count,
        )

    def _unpin_request(request: Any) -> None:
        if request is None:
            return
        pin = getattr(request, "_cascade_pin", None)
        if pin is None:
            return
        request._cascade_pin = None
        mc, entry = pin
        with mc._memory_lock:
            entry._pin_count = max(0, getattr(entry, "_pin_count", 0) - 1)

    orig_add_request = vsched.Scheduler.add_request

    def _patched_add_request(self, request: Any) -> None:
        orig_add_request(self, request)
        _maybe_pin_and_tag(self, request)

    vsched.Scheduler.add_request = _patched_add_request

    if hasattr(vsched.Scheduler, "_try_promote_ssd_pending"):
        orig_promote_ssd = vsched.Scheduler._try_promote_ssd_pending

        def _patched_promote_ssd(self) -> None:
            orig_promote_ssd(self)
            for req in self.waiting:
                if getattr(req, "cache_hit_type", None) == "ssd_hit":
                    _maybe_pin_and_tag(self, req)

        vsched.Scheduler._try_promote_ssd_pending = _patched_promote_ssd

    orig_schedule_waiting = vsched.Scheduler._schedule_waiting

    def _patched_schedule_waiting(self):
        scheduled = orig_schedule_waiting(self)
        for req in scheduled:
            if (getattr(req, "cached_tokens", 0) or 0) <= 0:
                _unpin_request(req)
        return scheduled

    vsched.Scheduler._schedule_waiting = _patched_schedule_waiting

    orig_cleanup_finished = vsched.Scheduler._cleanup_finished

    def _patched_cleanup_finished(self, finished_ids):
        reqs = [self.running.get(rid) or self.requests.get(rid) for rid in finished_ids]
        try:
            return orig_cleanup_finished(self, finished_ids)
        finally:
            for req in reqs:
                _unpin_request(req)

    vsched.Scheduler._cleanup_finished = _patched_cleanup_finished

    orig_do_abort = vsched.Scheduler._do_abort_request

    def _patched_do_abort(self, request_id: str) -> bool:
        req = self.requests.get(request_id) or self.running.get(request_id)
        try:
            return orig_do_abort(self, request_id)
        finally:
            _unpin_request(req)

    vsched.Scheduler._do_abort_request = _patched_do_abort

    for method_name in ("_recover_from_generation_error", "_recover_from_cache_error", "reset", "deep_reset"):
        orig_fn = getattr(vsched.Scheduler, method_name, None)
        if orig_fn is None:
            continue

        def _make_unpin_wrapper(fn):
            def _wrapped(self, *args, **kwargs):
                all_reqs = list(self.requests.values()) + list(self.running.values()) + list(self.waiting)
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    for req in all_reqs:
                        _unpin_request(req)
            return _wrapped

        setattr(vsched.Scheduler, method_name, _make_unpin_wrapper(orig_fn))


def _run_inkling2_warmup(model: Any) -> None:
    """Pre-compile Inkling 2 prefill Metal PSOs (cold + prefix-hit regimes) and BatchDecoder C=1."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.inkling2.batchdecode import BatchDecoder

    t0 = time.perf_counter()
    # 1. Short prefill cold (HAS_CS=0) + prefix-hit (HAS_CS=1) + BatchDecoder C=1 (2 steps)
    c16 = make_prompt_cache(model)
    tok16 = mx.ones((1, 16), dtype=mx.uint32)
    y = model(tok16, cache=c16)
    mx.eval(y, [c.state for c in c16])
    y = model(tok16, cache=c16)
    mx.eval(y, [c.state for c in c16])
    bd = BatchDecoder(model, [c16])
    t_in = mx.array([1], dtype=mx.uint32)
    for _ in range(2):
        t_in = bd.step(t_in)
        mx.eval(t_in)
    bd.flush()
    bd.release_stage()
    del bd, c16, y, t_in

    # 2. L<16 short_conv/dyn_short_conv (L=8 cold+cont), aff8_wk (L=40), GMM bm=8 (L=65 cold+cont), fg_mix+bm=16 (L=256)
    for seq_len, do_cont in ((8, True), (40, False), (65, True), (256, False)):
        cache = make_prompt_cache(model)
        toks = mx.ones((1, seq_len), dtype=mx.uint32)
        y = model(toks, cache=cache)
        mx.eval(y, [c.state for c in cache])
        if do_cont:
            y = model(toks, cache=cache)
            mx.eval(y, [c.state for c in cache])
        del cache, y, toks

    mx.clear_cache()
    logger.info("[vllm_mlx_inkling2] Startup warmup complete in %.2fs", time.perf_counter() - t0)


def _patch_startup_warmup() -> None:
    """Warm Inkling 2 Metal pipelines on the engine's worker thread at boot (before serving traffic)."""
    from vllm_mlx.engine import batched as engine_batched

    orig_prepare = engine_batched.BatchedEngine.prepare_for_start

    def _patched_prepare_for_start(self) -> None:
        orig_prepare(self)
        if os.environ.get("LOOPLING_STARTUP_WARMUP", "1") in ("0", "", "false", "False"):
            return
        model = getattr(self, "_model", None)
        if model is None or getattr(model, "model_type", None) != "inkling2_loop":
            return
        try:
            _run_inkling2_warmup(model)
        except Exception as exc:
            logger.warning("[vllm_mlx_inkling2] Startup warmup skipped: %s", exc)

    engine_batched.BatchedEngine.prepare_for_start = _patched_prepare_for_start
