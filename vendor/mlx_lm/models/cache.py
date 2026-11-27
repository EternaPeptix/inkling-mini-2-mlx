# Copyright © 2023 Apple Inc.

import copy
import hashlib
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from queue import Empty as QueueEmpty, Full as QueueFull, Queue
from threading import Thread
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

from mlx_lm.layers.attention import create_causal_mask


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def save_prompt_cache(
    file_name: str, cache: List[Any], metadata: Optional[Dict[str, str]] = None
):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    scalars = [""]  # must be non-empty otherwise tree_flatten drops it

    # Convert scalars to mx.array so they can be serialized
    def map_data(item):
        i, data = item
        if isinstance(data, mx.array):
            return item
        elif isinstance(data, (int, float, bool)):
            scalars.append((i, "scalar"))
            return (i, mx.array(data))
        elif isinstance(data, str):
            scalars.append((i, "string"))
            return (i, mx.array([ord(char) for char in data]))
        elif data is None:
            scalars.append((i, "none"))
            return (i, mx.array([]))
        else:
            raise ValueError(f"Model state is not serializable: {data}")

    cache_data = tree_flatten([c.state for c in cache])
    cache_data = dict(map(map_data, cache_data))
    cache_classes = [type(c).__name__ for c in cache]
    cache_metadata = [metadata or {}, cache_classes, scalars]
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    metadata, classes, scalars = cache_metadata
    for pos, type in scalars[1:]:
        array = arrays[pos]
        if type == "scalar":
            arrays[pos] = array.item()
        elif type == "string":
            arrays[pos] = "".join(chr(i) for i in array.tolist())
        elif type == "none":
            arrays[pos] = None
        else:
            raise ValueError(f"[load_prompt_cache] Unknown type {type}")
    arrays = tree_unflatten(list(arrays.items()))
    cache = [globals()[c].from_state(state) for c, state in zip(classes, arrays)]
    if return_metadata:
        return cache, metadata
    return cache


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: List[Any], num_tokens: int) -> List[Any]:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    def is_trimmable(self):
        return False

    def size(self):
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self):
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state):
        # Create an instance of cls without calling __init__
        obj = cls.__new__(cls)
        obj.state = state
        return obj


class ConcatenateKVCache(_BaseCache):
    """ConcatenateKVCache the simplest KV cache implementation.

    Can be used as a mock KV cache or when large blocks are being processed at
    a time in which case KVCache isn't necessarily faster. Consider using the
    KVCache with a larger step size before using this cache.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]
        return self.keys_and_values()

    def keys_and_values(self):
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[-2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.group_size, self.bits

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.group_size, self.bits = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class KVCache(_BaseCache):
    step = 256
    _keys = None
    _values = None
    _packed = None

    def __init__(self):
        self._keys = None
        self._values = None
        self._packed = None
        self.offset = 0

    @property
    def keys(self):
        return self._packed[0][None] if self._packed is not None else self._keys

    @keys.setter
    def keys(self, v):
        if self._packed is not None:
            self._values = mx.contiguous(self._packed[1][None])
            self._packed = None
        self._keys = v

    @property
    def values(self):
        return self._packed[1][None] if self._packed is not None else self._values

    @values.setter
    def values(self, v):
        if self._packed is not None:
            self._keys = mx.contiguous(self._packed[0][None])
            self._packed = None
        self._values = v

    def pack_b1(self):
        if self._packed is not None or self._keys is None:
            return
        self._packed = mx.contiguous(mx.stack([self._keys[0], self._values[0]], axis=0))
        self._keys = self._values = None

    def unpack_b1(self):
        if self._packed is not None:
            self._keys = mx.contiguous(self._packed[0][None])
            self._values = mx.contiguous(self._packed[1][None])
            self._packed = None

    def update_packed_b1(self, kv_new, wpos_fn):
        prev = self.offset
        buf = self._packed
        if buf is None:
            self.pack_b1()
            buf = self._packed
        self._packed = None
        cap = buf.shape[2]
        if prev + 1 > cap:
            if prev % self.step != 0:
                buf = buf[:, :, :prev, :]
            ext = mx.zeros((2, buf.shape[1], self.step, buf.shape[3]), buf.dtype)
            buf = mx.concatenate([buf, ext], axis=2)
            cap = buf.shape[2]
        self.offset = prev + 1
        self._packed = mx.slice_update(buf, kv_new, wpos_fn(prev), axes=(2,))
        return self._packed, self.offset, self.offset, 0, 0, cap

    def commit_packed_w(self, kv_new, keep, wpos_fn):
        """Commit first `keep` slots from kv_new (2, Hkv, W, 128) into packed KV buffer."""
        src = kv_new[:, :, :keep, :]
        prev = self.offset
        buf = self._packed
        self._packed = None
        cap = buf.shape[2]
        if prev + keep > cap:
            if prev % self.step != 0:
                buf = buf[:, :, :prev, :]
            need = prev + keep - buf.shape[2]
            n_steps = (need + self.step - 1) // self.step
            ext = mx.zeros((2, buf.shape[1], n_steps * self.step, buf.shape[3]), buf.dtype)
            buf = mx.concatenate([buf, ext], axis=2)
        self.offset = prev + keep
        self._packed = mx.slice_update(buf, src, wpos_fn(prev), axes=(2,))
        return self._packed

    def commit_unpacked_w(self, k_chunk, v_chunk):
        """Commit `(B, Hkv, keep, D)` keys/values into unpacked `_keys`/`_values`, advancing `offset`."""
        if self._packed is not None:
            self.unpack_b1()
        keep = int(k_chunk.shape[2])
        prev = self.offset
        if self._keys is None or (prev + keep) > self._keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = k_chunk.shape
            v_head_dim = v_chunk.shape[3]
            if self._keys is None:
                n_steps = (self.step + keep - 1) // self.step
                self._keys = mx.zeros((B, n_kv_heads, n_steps * self.step, k_head_dim), k_chunk.dtype)
                self._values = mx.zeros((B, n_kv_heads, n_steps * self.step, v_head_dim), v_chunk.dtype)
            else:
                cap = self._keys.shape[2]
                if prev % self.step != 0 and cap % self.step != 0:
                    self._keys = self._keys[..., :prev, :]
                    self._values = self._values[..., :prev, :]
                    cap = prev
                target_cap = ((prev + keep + self.step - 1) // self.step) * self.step
                add = target_cap - cap
                ext_k = mx.zeros((B, n_kv_heads, add, k_head_dim), k_chunk.dtype)
                ext_v = mx.zeros((B, n_kv_heads, add, v_head_dim), v_chunk.dtype)
                self._keys = mx.concatenate([self._keys, ext_k], axis=2)
                self._values = mx.concatenate([self._values, ext_v], axis=2)
        self.offset = prev + keep
        self._keys[..., prev : self.offset, :] = k_chunk
        self._values[..., prev : self.offset, :] = v_chunk

    def update_and_fetch(self, keys, values):
        if self._packed is not None:
            self.unpack_b1()
        prev = self.offset
        if self._keys is None or (prev + keys.shape[2]) > self._keys.shape[2]:
            B, n_kv_heads, S, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            if self._keys is not None:
                if prev % self.step != 0:
                    self._keys = self._keys[..., :prev, :]
                    self._values = self._values[..., :prev, :]
                cur = self._keys.shape[2]
                need = prev + S
                if cur >= 4096:
                    target = max(
                        ((need + self.step - 1) // self.step) * self.step,
                        ((cur * 2 + self.step - 1) // self.step) * self.step,
                    )
                    if need <= 1048576 < target:
                        target = 1048576
                    add = target - cur
                else:
                    add = ((self.step + S - 1) // self.step) * self.step
                new_k = mx.zeros((B, n_kv_heads, add, k_head_dim), keys.dtype)
                new_v = mx.zeros((B, n_kv_heads, add, v_head_dim), values.dtype)
                self._keys = mx.concatenate([self._keys, new_k], axis=2)
                self._values = mx.concatenate([self._values, new_v], axis=2)
            else:
                n_steps = (self.step + S - 1) // self.step
                k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
                v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
                self._keys = mx.zeros(k_shape, keys.dtype)
                self._values = mx.zeros(v_shape, values.dtype)

        self.offset += keys.shape[2]
        self._keys[..., prev : self.offset, :] = keys
        self._values[..., prev : self.offset, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        k, v = self.keys, self.values
        if self.offset < k.shape[2]:
            return k[..., : self.offset, :], v[..., : self.offset, :]
        else:
            return k, v

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, v):
        self._packed = None
        self._keys, self._values, self.offset = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        if self._packed is not None:
            self.unpack_b1()
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self._keys is not None:
            quant_cache.keys = mx.quantize(self._keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self._values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchKVCache.merge(caches)

    def empty(self):
        return self._packed is None and self._keys is None

    @property
    def nbytes(self):
        if self._packed is not None:
            return self._packed.nbytes
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes


class RotatingKVCache(_BaseCache):
    step = 256
    _keys = None
    _values = None
    _packed = None

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self._keys = None
        self._values = None
        self._packed = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    @property
    def keys(self):
        return self._packed[0][None] if self._packed is not None else self._keys

    @keys.setter
    def keys(self, v):
        if self._packed is not None:
            self._values = mx.contiguous(self._packed[1][None])
            self._packed = None
        self._keys = v

    @property
    def values(self):
        return self._packed[1][None] if self._packed is not None else self._values

    @values.setter
    def values(self, v):
        if self._packed is not None:
            self._keys = mx.contiguous(self._packed[0][None])
            self._packed = None
        self._values = v

    def pack_b1(self):
        if self._packed is not None or self._keys is None:
            return
        k, v = self._keys, self._values
        trim_size = k.shape[2] - self.max_size
        if trim_size > 0:
            k = self._trim(trim_size, k)
            v = self._trim(trim_size, v)
            self._idx = self.max_size
        self._packed = mx.contiguous(mx.stack([k[0], v[0]], axis=0))
        self._keys = self._values = None

    def unpack_b1(self):
        if self._packed is not None:
            self._keys = mx.contiguous(self._packed[0][None])
            self._values = mx.contiguous(self._packed[1][None])
            self._packed = None

    def update_packed_b1(self, kv_new, wpos_fn):
        prev = self.offset
        buf = self._packed
        if buf is None:
            self.pack_b1()
            buf = self._packed
        self._packed = None
        cap = buf.shape[2]
        if prev >= cap and cap < self.max_size:
            new_size = min(self.step, self.max_size - prev)
            ext = mx.zeros((2, buf.shape[1], new_size, buf.shape[3]), buf.dtype)
            buf = mx.concatenate([buf, ext], axis=2)
            cap = buf.shape[2]
            self._idx = prev
        if self._idx == self.max_size:
            self._idx = self.keep
        pos = self._idx
        self.offset = prev + 1
        self._idx = pos + 1
        self._packed = mx.slice_update(buf, kv_new, wpos_fn(pos), axes=(2,))
        S = min(self.offset, self.max_size)
        return self._packed, S, self.offset, self._idx, 1, cap

    def commit_packed_w(self, kv_new, keep, wpos_fn):
        """Commit first `keep` slots from kv_new (2, Hkv, W, 128) into rotating packed KV buffer."""
        assert self.keep == 0
        src = kv_new[:, :, :keep, :]
        prev = self.offset
        buf = self._packed
        self._packed = None
        cap = buf.shape[2]
        if cap < self.max_size and prev + keep > cap:
            target_cap = min(self.max_size, ((prev + keep + self.step - 1) // self.step) * self.step)
            ext = mx.zeros((2, buf.shape[1], target_cap - cap, buf.shape[3]), buf.dtype)
            buf = mx.concatenate([buf, ext], axis=2)
            self._idx = prev
        slot0 = 0 if self._idx == self.max_size else self._idx
        if slot0 + keep <= self.max_size:
            buf = mx.slice_update(buf, src, wpos_fn(slot0), axes=(2,))
        else:
            n1 = self.max_size - slot0
            buf = mx.slice_update(buf, src[:, :, :n1, :], wpos_fn(slot0), axes=(2,))
            buf = mx.slice_update(buf, src[:, :, n1:, :], wpos_fn(0), axes=(2,))
        self.offset = prev + keep
        self._idx = ((slot0 + keep - 1) % self.max_size) + 1
        self._packed = buf
        return self._packed

    def commit_unpacked_w(self, k_chunk, v_chunk):
        """Commit `(B, Hkv, keep, D)` keys/values into unpacked rotating buffers, preserving ring slots."""
        assert self.keep == 0
        if self._packed is not None:
            self.unpack_b1()
        B, n_kv_heads, keep, k_head_dim = k_chunk.shape
        v_head_dim = v_chunk.shape[3]
        prev = self.offset
        if self._keys is None:
            cap = min(self.max_size, ((keep + self.step - 1) // self.step) * self.step)
            self._keys = mx.zeros((B, n_kv_heads, cap, k_head_dim), k_chunk.dtype)
            self._values = mx.zeros((B, n_kv_heads, cap, v_head_dim), v_chunk.dtype)
            self._idx = 0
        else:
            trim_size = self._keys.shape[2] - self.max_size
            if trim_size > 0:
                self._keys = self._trim(trim_size, self._keys)
                self._values = self._trim(trim_size, self._values)
                self._idx = self.max_size
            cap = self._keys.shape[2]
            if cap < self.max_size and prev + keep > cap:
                add = min(self.max_size - cap, ((prev + keep - cap + self.step - 1) // self.step) * self.step)
                if self._idx >= cap:
                    self._idx = prev
                ext_k = mx.zeros((B, n_kv_heads, add, k_head_dim), k_chunk.dtype)
                ext_v = mx.zeros((B, n_kv_heads, add, v_head_dim), v_chunk.dtype)
                self._keys = mx.concatenate([self._keys, ext_k], axis=2)
                self._values = mx.concatenate([self._values, ext_v], axis=2)
        slot0 = 0 if self._idx == self.max_size else self._idx
        if slot0 + keep <= self.max_size:
            self._keys[..., slot0 : slot0 + keep, :] = k_chunk
            self._values[..., slot0 : slot0 + keep, :] = v_chunk
        else:
            n1 = self.max_size - slot0
            self._keys[..., slot0 : self.max_size, :] = k_chunk[..., :n1, :]
            self._keys[..., 0 : keep - n1, :] = k_chunk[..., n1:, :]
            self._values[..., slot0 : self.max_size, :] = v_chunk[..., :n1, :]
            self._values[..., 0 : keep - n1, :] = v_chunk[..., n1:, :]
        self.offset = prev + keep
        self._idx = ((slot0 + keep - 1) % self.max_size) + 1

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self._packed is not None:
            self.unpack_b1()
        if self._keys is None:
            self._keys = keys
            self._values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._keys = self._temporal_order(self._keys)
            self._values = self._temporal_order(self._values)
            self._idx = self._keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self._keys = self._trim(trim_size, self._keys, keys)
            self._values = self._trim(trim_size, self._values, values)
        self.offset += keys.shape[2]
        self._idx = self._keys.shape[2]
        return self._keys, self._values

    def _update_in_place(self, keys, values):
        if self._packed is not None:
            self.unpack_b1()
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self._keys is None or (
            prev >= self._keys.shape[2] and self._keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self._keys is not None:
                self._keys = mx.concatenate([self._keys, new_k], axis=2)
                self._values = mx.concatenate([self._values, new_v], axis=2)
            else:
                self._keys, self._values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self._keys.shape[2] - self.max_size
        if trim_size > 0:
            self._keys = self._trim(trim_size, self._keys)
            self._values = self._trim(trim_size, self._values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self._keys[..., self._idx : self._idx + S, :] = keys
        self._values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        return self.keys_and_values()

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def keys_and_values(self):
        k, v = self.keys, self.values
        if self.offset < k.shape[2]:
            return k[..., : self.offset, :], v[..., : self.offset, :]
        else:
            return k, v

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.keep, self.max_size, self._idx

    @state.setter
    def state(self, v):
        self._packed = None
        self._keys, self._values, self.offset, self.keep, self.max_size, self._idx = v

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("RotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingKVCache.merge(caches)

    def empty(self):
        return self._packed is None and self._keys is None

    @property
    def nbytes(self):
        if self._packed is not None:
            return self._packed.nbytes
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes


class ArraysCache(_BaseCache):
    _zero_pad = False

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        self.left_padding = mx.array(left_padding) if left_padding else None
        self.lengths = None

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self.left_padding is not None:
            return self.left_padding.size
        elif self.lengths is not None:
            return self.lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache, self.left_padding, self.lengths

    @state.setter
    def state(self, v):
        self.cache, self.left_padding, self.lengths = v

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a, b):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [mx.contiguous(c[idx : idx + 1]) if c is not None else None for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        pos = mx.arange(N)
        mask = None
        if self.left_padding is not None:
            mask = pos >= self.left_padding[:, None]
        if self.lengths is not None:
            in_length = pos < self.lengths[:, None]
            mask = in_length if mask is None else (mask & in_length)
        return mask

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        # All caches are empty so return early
        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            cache._zero_pad = B == 1  # fastdecode: the conv mask of this cache is all True
            return cache

        for e in range(n_state):
            c_init = next(iter(c[e] for c in caches if c[e] is not None))
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                cache[e][i : i + 1] = caches[i][e]
        return cache

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


class ChunkedKVCache(_BaseCache):
    step = 256

    def __init__(self, chunk_size):
        self.keys = None
        self.values = None
        self.offset = 0
        self.chunk_size = chunk_size
        self.start_position = 0

    def maybe_trim_front(self):
        # Maintain the cache below the chunk size.
        if self.keys is None:
            return
        valid = self.offset - self.start_position
        if valid > self.chunk_size:
            trim = valid - self.chunk_size
            self.start_position += trim
            self.keys = self.keys[..., trim:valid, :]
            self.values = self.values[..., trim:valid, :]

    def update_and_fetch(self, keys, values):
        prev = self.offset - self.start_position
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        end = self.offset - self.start_position
        self.keys[..., prev:end, :] = keys
        self.values[..., prev:end, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        end = self.offset - self.start_position
        return self.keys[..., :end, :], self.values[..., :end, :]

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.chunk_size, self.start_position

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.chunk_size, self.start_position = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset - self.start_position, n)
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches

    def __getitem__(self, idx):
        return self.caches[idx]

    def is_trimmable(self):
        return all(c.is_trimmable() for c in self.caches)

    def trim(self, n):
        for c in self.caches:
            m = c.trim(n)
        return m

    @property
    def state(self):
        return [(c.state, type(c).__name__) for c in self.caches]

    @state.setter
    def state(self, v):
        self.caches = [globals()[c].from_state(s) for s, c in v]

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        for c in self.caches:
            c.filter(batch_indices)

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        for c, o in zip(self.caches, other.caches):
            c.extend(o)

    @classmethod
    def merge(cls, caches):
        cache = cls()
        cache.caches = tuple(
            caches[0].caches[i].merge([c.caches[i] for c in caches])
            for i in range(len(caches[0].caches))
        )
        return cache

    def extract(self, idx):
        return CacheList(*(c.extract(idx) for c in self.caches))

    def prepare(self, **kwargs):
        for c in self.caches:
            c.prepare(**kwargs)

    def finalize(self):
        for c in self.caches:
            c.finalize()

    def size(self):
        return max(c.size() for c in self.caches)

    def empty(self):
        return self.caches[0].empty()

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.caches)


def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        # fastdecode: a single unpadded sequence behaves exactly like a plain KVCache whose host offset is _idx + 0
        self._fd_delta = 0 if (len(left_padding) == 1 and left_padding[0] == 0) else None

        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, S, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                cur = self.keys.shape[2]
                need = prev + S
                if cur >= 4096:
                    target = max(
                        ((need + self.step - 1) // self.step) * self.step,
                        ((cur * 2 + self.step - 1) // self.step) * self.step,
                    )
                    if need <= 1048576 < target:
                        target = 1048576
                    add = target - cur
                else:
                    add = ((self.step + S - 1) // self.step) * self.step
                new_k = mx.zeros((B, n_kv_heads, add, k_head_dim), keys.dtype)
                new_v = mx.zeros((B, n_kv_heads, add, v_head_dim), values.dtype)
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                n_steps = (self.step + S - 1) // self.step
                k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
                v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
                self.keys = mx.zeros(k_shape, keys.dtype)
                self.values = mx.zeros(v_shape, values.dtype)

        self.offset = self.offset + keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset = self.offset - padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.left_padding, self._idx

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding, self._idx = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = min(self.left_padding.tolist())
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)
        dtype = (self.keys if self.keys is not None else other.keys).dtype

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.zeros((Bc, H, 0, D), dtype=dtype)
                v = mx.zeros((Bc, H, 0, M), dtype=dtype)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        mx.eval(self.left_padding)
        padding = self.left_padding.tolist()[idx]
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return BatchKVCache([0] * len(caches))

        if len(caches) == 1:
            c = caches[0]
            cache = cls([0])
            cache.keys = c.keys[..., : c.offset, :]
            cache.values = c.values[..., : c.offset, :]
            cache.offset += cache.keys.shape[2]
            cache._idx = cache.keys.shape[2]
            return cache

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache._idx = keys.shape[2]

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class BatchRotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])

        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False
        # fastdecode: host offset of a single unpadded sequence is _offset + _fd_delta (None = not eligible)
        self._fd_delta = 0 if (len(left_padding) == 1 and left_padding[0] == 0) else None

        # Lengths for right_padded inputs to make sure that padding tokens do
        # not evict valid tokens.
        self._lengths = None

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return mx.contiguous(v) if trim_size > 0 else v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = mx.roll(self.keys, -self._idx, axis=2)
            self.values = mx.roll(self.values, -self._idx, axis=2)
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()

            # Slice off the end if needed
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            # Roll right sequences that are padded to make sure that we don't
            # trim valid cache entries
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
                self.values = dynamic_roll(self.values, roll[:, None], axis=2)
                self.left_padding += roll
                self.offset = self.offset - roll

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset = self.offset + keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        return self.keys, self.values

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before deocoding with BatchRotatingKVCache"
            )

        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset = self.offset + S
        self._idx += S

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        return self.keys_and_values()

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def keys_and_values(self):
        if self._offset < self.keys.shape[2]:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        else:
            return self.keys, self.values

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
            self.values = dynamic_roll(self.values, roll[:, None], axis=2)
            self.left_padding += roll
            self.offset = self.offset - roll
            self._lengths = None

    @property
    def state(self):
        return (
            self.keys,
            self.values,
            self.offset,
            self.left_padding,
            self.max_size,
            self._offset,
            self._idx,
            self.rotated,
        )

    @state.setter
    def state(self, v):
        (
            self.keys,
            self.values,
            self.offset,
            self.left_padding,
            self.max_size,
            self._offset,
            self._idx,
            self.rotated,
        ) = v

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset = self.offset - n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("BatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        if (self.rotated != other.rotated) or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)
        dtype = (self.keys if self.keys is not None else other.keys).dtype

        def pad(c):
            left = max_idx - c._idx
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.zeros((Bc, H, 0, D), dtype=dtype)
                v = mx.zeros((Bc, H, 0, M), dtype=dtype)
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        cache = RotatingKVCache(self.max_size)
        padding = max(0, self.left_padding.tolist()[idx])
        offset = self.offset.tolist()[idx]
        cache.keys = self.keys[idx : idx + 1]
        cache.values = self.values[idx : idx + 1]
        cache._idx = self._idx
        if self.rotated:
            cache.keys = mx.roll(cache.keys, -self._idx, axis=2)
            cache.values = mx.roll(cache.values, -self._idx, axis=2)
            cache._idx = self.max_size
        cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
        cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
        if cache.keys.shape[2] > cache.max_size:
            cache.keys = mx.contiguous(cache.keys[:, :, -cache.max_size :])
            cache.values = mx.contiguous(cache.values[:, :, -cache.max_size :])
        cache.offset = offset
        cache._idx = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all(c.max_size == caches[0].max_size for c in caches):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )

        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))

        if len(caches) == 1:
            c = caches[0]
            l = lengths[0]
            keys = c._temporal_order(c.keys)[..., -l:, :]
            values = c._temporal_order(c.values)[..., -l:, :]
            cache = cls(caches[0].max_size, [0])
            cache.keys = keys
            cache.values = values
            cache.offset = mx.array(offsets)
            cache._idx = keys.shape[2]
            cache._offset = keys.shape[2]
            if cache._fd_delta is not None:
                cache._fd_delta = int(offsets[0]) - keys.shape[2]
            return cache

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + l] = c._temporal_order(c.keys)[..., -l:, :]
            values[i : i + 1, :, p : p + l] = c._temporal_order(c.values)[..., -l:, :]

        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = keys.shape[2]
        cache._offset = keys.shape[2]
        if cache._fd_delta is not None:
            cache._fd_delta = int(offsets[0]) - keys.shape[2]

        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


def _drop_fd_flag(cls, attr, value, names):
    """fastdecode eligibility flags describe a freshly merged single unpadded sequence; anything that can add padding
    or change the batch (prepare/extend/filter/state assignment) drops them, which sends decode to the reference path."""

    def wrap(f):
        def w(self, *a, **k):
            # filter([0]) of a single sequence keeps the batch as is (the server's prompt -> generation split)
            if not (f.__name__ == "filter" and a and list(a[0]) == [0]):
                setattr(self, attr, value)
            return f(self, *a, **k)

        w.__name__, w.__doc__ = f.__name__, f.__doc__
        return w

    for n in names:
        m = cls.__dict__[n]
        if isinstance(m, property):
            setattr(cls, n, m.setter(wrap(m.fset)))
        else:
            setattr(cls, n, wrap(m))


def _quantize_kv_int8(x: mx.array):
    amax = mx.max(mx.abs(x.astype(mx.float32)), axis=-1)
    scale = mx.maximum(amax / 127.0, 1e-12)
    inv = 1.0 / scale[..., None]
    q = mx.round(x.astype(mx.float32) * inv).astype(mx.int8)
    return q, scale


def _dequantize_kv_int8(q: mx.array, scale: mx.array, dtype=mx.bfloat16) -> mx.array:
    return (q.astype(mx.float32) * scale[..., None]).astype(dtype)


class Int8KVCache(_BaseCache):
    """Per-token, per-head symmetric INT8 KV cache for FullDecoderLayer (opt-in via LOOPLING_KV_INT8=1)."""

    step = 256

    def __init__(self):
        self.k_q = None
        self.v_q = None
        self.k_s = None
        self.v_s = None
        self.offset = 0
        self._dtype = mx.bfloat16

    def _ensure_cap(self, B, Hkv, D, need):
        if self.k_q is None:
            n_steps = (self.step + need - 1) // self.step
            cap = n_steps * self.step
            self.k_q = mx.zeros((B, Hkv, cap, D), mx.int8)
            self.v_q = mx.zeros((B, Hkv, cap, D), mx.int8)
            self.k_s = mx.zeros((B, Hkv, cap), mx.float32)
            self.v_s = mx.zeros((B, Hkv, cap), mx.float32)
        elif need > self.k_q.shape[2]:
            prev = self.offset
            add_need = need - prev
            n_steps = (self.step + add_need - 1) // self.step
            ext = n_steps * self.step
            if prev % self.step != 0:
                self.k_q = self.k_q[..., :prev, :]
                self.v_q = self.v_q[..., :prev, :]
                self.k_s = self.k_s[..., :prev]
                self.v_s = self.v_s[..., :prev]
            self.k_q = mx.concatenate([self.k_q, mx.zeros((B, Hkv, ext, D), mx.int8)], axis=2)
            self.v_q = mx.concatenate([self.v_q, mx.zeros((B, Hkv, ext, D), mx.int8)], axis=2)
            self.k_s = mx.concatenate([self.k_s, mx.zeros((B, Hkv, ext), mx.float32)], axis=2)
            self.v_s = mx.concatenate([self.v_s, mx.zeros((B, Hkv, ext), mx.float32)], axis=2)

    @property
    def keys(self):
        return None if self.k_q is None else _dequantize_kv_int8(self.k_q, self.k_s, self._dtype)

    @property
    def values(self):
        return None if self.v_q is None else _dequantize_kv_int8(self.v_q, self.v_s, self._dtype)

    def update_b1_int8(self, kq_new, vq_new, ks_new, vs_new, wpos_fn):
        prev = self.offset
        B, Hkv, _, D = kq_new.shape
        self._ensure_cap(B, Hkv, D, prev + 1)
        wp = wpos_fn(prev)
        self.offset = prev + 1
        self.k_q = mx.slice_update(self.k_q, kq_new, wp, axes=(2,))
        self.v_q = mx.slice_update(self.v_q, vq_new, wp, axes=(2,))
        self.k_s = mx.slice_update(self.k_s, ks_new, wp, axes=(2,))
        self.v_s = mx.slice_update(self.v_s, vs_new, wp, axes=(2,))
        return self.k_q, self.v_q, self.k_s, self.v_s, self.offset, self.k_q.shape[2]

    def update_and_fetch(self, keys, values):
        prev = self.offset
        B, Hkv, S, D = keys.shape
        self._dtype = keys.dtype
        self._ensure_cap(B, Hkv, D, prev + S)
        kq, ks = _quantize_kv_int8(keys)
        vq, vs = _quantize_kv_int8(values)
        self.offset = prev + S
        self.k_q[..., prev : self.offset, :] = kq
        self.v_q[..., prev : self.offset, :] = vq
        self.k_s[..., prev : self.offset] = ks
        self.v_s[..., prev : self.offset] = vs
        if prev == 0:
            return keys, values
        return self.keys_and_values()

    def keys_and_values(self):
        if self.k_q is None:
            return None, None
        return (
            _dequantize_kv_int8(self.k_q[..., : self.offset, :], self.k_s[..., : self.offset], self._dtype),
            _dequantize_kv_int8(self.v_q[..., : self.offset, :], self.v_s[..., : self.offset], self._dtype),
        )

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.k_q, self.v_q, self.k_s, self.v_s, self.offset

    @state.setter
    def state(self, v):
        self.k_q, self.v_q, self.k_s, self.v_s, self.offset = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchInt8KVCache.merge(caches)

    def empty(self):
        return self.k_q is None

    @property
    def nbytes(self):
        if self.k_q is None:
            return 0
        return self.k_q.nbytes + self.v_q.nbytes + self.k_s.nbytes + self.v_s.nbytes


class BatchInt8KVCache(_BaseCache):
    """Batched INT8 KV cache mirroring BatchKVCache semantics with B=1 fastdecode support."""

    step = 256

    def __init__(self, left_padding: List[int]):
        self.k_q = None
        self.v_q = None
        self.k_s = None
        self.v_s = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._dtype = mx.bfloat16
        self._fd_delta = 0 if (len(left_padding) == 1 and left_padding[0] == 0) else None
        self._right_padding = None

    @property
    def keys(self):
        return None if self.k_q is None else _dequantize_kv_int8(self.k_q, self.k_s, self._dtype)

    @property
    def values(self):
        return None if self.v_q is None else _dequantize_kv_int8(self.v_q, self.v_s, self._dtype)

    def _ensure_cap(self, B, Hkv, D, need):
        if self.k_q is None:
            n_steps = (self.step + need - 1) // self.step
            cap = n_steps * self.step
            self.k_q = mx.zeros((B, Hkv, cap, D), mx.int8)
            self.v_q = mx.zeros((B, Hkv, cap, D), mx.int8)
            self.k_s = mx.zeros((B, Hkv, cap), mx.float32)
            self.v_s = mx.zeros((B, Hkv, cap), mx.float32)
        elif need > self.k_q.shape[2]:
            prev = self._idx
            n_steps = (self.step + need - prev - 1) // self.step
            ext = n_steps * self.step
            if prev % self.step != 0:
                self.k_q = self.k_q[..., :prev, :]
                self.v_q = self.v_q[..., :prev, :]
                self.k_s = self.k_s[..., :prev]
                self.v_s = self.v_s[..., :prev]
            self.k_q = mx.concatenate([self.k_q, mx.zeros((B, Hkv, ext, D), mx.int8)], axis=2)
            self.v_q = mx.concatenate([self.v_q, mx.zeros((B, Hkv, ext, D), mx.int8)], axis=2)
            self.k_s = mx.concatenate([self.k_s, mx.zeros((B, Hkv, ext), mx.float32)], axis=2)
            self.v_s = mx.concatenate([self.v_s, mx.zeros((B, Hkv, ext), mx.float32)], axis=2)

    def update_b1_int8(self, kq_new, vq_new, ks_new, vs_new, wpos_fn):
        prev = self._idx
        B, Hkv, _, D = kq_new.shape
        self._ensure_cap(B, Hkv, D, prev + 1)
        wp = wpos_fn(prev)
        self._idx = prev + 1
        self.offset = self.offset + 1
        self.k_q = mx.slice_update(self.k_q, kq_new, wp, axes=(2,))
        self.v_q = mx.slice_update(self.v_q, vq_new, wp, axes=(2,))
        self.k_s = mx.slice_update(self.k_s, ks_new, wp, axes=(2,))
        self.v_s = mx.slice_update(self.v_s, vs_new, wp, axes=(2,))
        return self.k_q, self.v_q, self.k_s, self.v_s, self._idx, self.k_q.shape[2]

    def update_and_fetch(self, keys, values):
        prev = self._idx
        B, Hkv, S, D = keys.shape
        self._dtype = keys.dtype
        self._ensure_cap(B, Hkv, D, prev + S)
        kq, ks = _quantize_kv_int8(keys)
        vq, vs = _quantize_kv_int8(values)
        self.offset = self.offset + S
        self._idx = prev + S
        self.k_q[..., prev : self._idx, :] = kq
        self.v_q[..., prev : self._idx, :] = vq
        self.k_s[..., prev : self._idx] = ks
        self.v_s[..., prev : self._idx] = vs
        if prev == 0:
            return keys, values
        return self.keys_and_values()

    def keys_and_values(self):
        return (
            _dequantize_kv_int8(self.k_q[..., : self._idx, :], self.k_s[..., : self._idx], self._dtype),
            _dequantize_kv_int8(self.v_q[..., : self._idx, :], self.v_s[..., : self._idx], self._dtype),
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.k_q is not None:
                raise ValueError("Left padding can only be added to an empty BatchInt8KVCache")
            lp = mx.array(left_padding)
            self.left_padding += lp
            self.offset = self.offset - lp
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            p = self._right_padding
            self.k_q = dynamic_roll(self.k_q, p[:, None], axis=2)
            self.v_q = dynamic_roll(self.v_q, p[:, None], axis=2)
            self.k_s = dynamic_roll(self.k_s, p[:, None], axis=2)
            self.v_s = dynamic_roll(self.v_s, p[:, None], axis=2)
            self.offset = self.offset - p
            self.left_padding += p
            self._right_padding = None

    @property
    def state(self):
        return self.k_q, self.v_q, self.k_s, self.v_s, self.offset, self.left_padding, self._idx

    @state.setter
    def state(self, v):
        self.k_q, self.v_q, self.k_s, self.v_s, self.offset, self.left_padding, self._idx = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(N, offset=self._idx, left_padding=self.left_padding, **kwargs)

    def filter(self, batch_indices):
        if self.k_q is not None:
            self.k_q = self.k_q[batch_indices]
            self.v_q = self.v_q[batch_indices]
            self.k_s = self.k_s[batch_indices]
            self.v_s = self.v_s[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_lp = min(self.left_padding.tolist())
        if min_lp > 0:
            if self.k_q is not None:
                self.k_q = self.k_q[..., min_lp:, :]
                self.v_q = self.v_q[..., min_lp:, :]
                self.k_s = self.k_s[..., min_lp:]
                self.v_s = self.v_s[..., min_lp:]
            self._idx -= min_lp
            self.left_padding -= min_lp

    def extend(self, other):
        if self.k_q is None and other.k_q is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        max_idx = max(self._idx, other._idx)
        L1 = self.k_q.shape[2] if self.k_q is not None else 0
        L2 = other.k_q.shape[2] if other.k_q is not None else 0
        H = self.k_q.shape[1] if self.k_q is not None else other.k_q.shape[1]
        D = self.k_q.shape[3] if self.k_q is not None else other.k_q.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            kq, vq, ks, vs = c.k_q, c.v_q, c.k_s, c.v_s
            if kq is None:
                Bc = c.offset.shape[0]
                kq = mx.zeros((Bc, H, 0, D), mx.int8)
                vq = mx.zeros((Bc, H, 0, D), mx.int8)
                ks = mx.zeros((Bc, H, 0), mx.float32)
                vs = mx.zeros((Bc, H, 0), mx.float32)
            left = max_idx - c._idx
            right = max_size - kq.shape[2] - left
            if right < 0:
                kq, vq, ks, vs = kq[..., :right, :], vq[..., :right, :], ks[..., :right], vs[..., :right]
                right = 0
            if left != 0 or right != 0:
                p4 = [(0, 0), (0, 0), (left, right), (0, 0)]
                p3 = [(0, 0), (0, 0), (left, right)]
                kq, vq = mx.pad(kq, p4), mx.pad(vq, p4)
                ks, vs = mx.pad(ks, p3), mx.pad(vs, p3)
            return kq, vq, ks, vs, c.offset, c.left_padding + left

        self.k_q, self.v_q, self.k_s, self.v_s, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = Int8KVCache()
        mx.eval(self.left_padding)
        padding = self.left_padding.tolist()[idx]
        cache.k_q = mx.contiguous(self.k_q[idx : idx + 1, :, padding : self._idx])
        cache.v_q = mx.contiguous(self.v_q[idx : idx + 1, :, padding : self._idx])
        cache.k_s = mx.contiguous(self.k_s[idx : idx + 1, :, padding : self._idx])
        cache.v_s = mx.contiguous(self.v_s[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.k_q.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls([0] * len(caches))
        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.k_q.shape[1] for c in caches if c.k_q is not None)
        D = max(c.k_q.shape[3] for c in caches if c.k_q is not None)
        k_q = mx.zeros((B, H, max_length, D), mx.int8)
        v_q = mx.zeros((B, H, max_length, D), mx.int8)
        k_s = mx.zeros((B, H, max_length), mx.float32)
        v_s = mx.zeros((B, H, max_length), mx.float32)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.k_q is None:
                continue
            k_q[i : i + 1, :, p : p + c.offset] = c.k_q[..., : c.offset, :]
            v_q[i : i + 1, :, p : p + c.offset] = c.v_q[..., : c.offset, :]
            k_s[i : i + 1, :, p : p + c.offset] = c.k_s[..., : c.offset]
            v_s[i : i + 1, :, p : p + c.offset] = c.v_s[..., : c.offset]
        cache = cls(padding)
        cache.k_q, cache.v_q, cache.k_s, cache.v_s = k_q, v_q, k_s, v_s
        cache.offset += max_length
        cache._idx = max_length
        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.k_q is None

    @property
    def nbytes(self):
        if self.k_q is None:
            return 0
        return self.k_q.nbytes + self.v_q.nbytes + self.k_s.nbytes + self.v_s.nbytes


for _c in (BatchKVCache, BatchRotatingKVCache, BatchInt8KVCache):
    _drop_fd_flag(_c, "_fd_delta", None, ("prepare", "filter", "extend", "state"))
_drop_fd_flag(ArraysCache, "_zero_pad", False, ("prepare", "filter", "extend", "state"))


class TokenBuffer:
    """A simple token buffer that can be efficiently appended to in a similar
    fashion to the KVCache.

    Perhaps these could share some logic in the future.
    """

    step = 256

    def __init__(self, tokens=None):
        tokens = [] if tokens is None else tokens
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, tokens):
        start = self._size
        end = start + len(tokens)

        new_size = ((end + self.step - 1) // self.step) * self.step
        if new_size > self._buffer.size:
            self._buffer = mx.concatenate(
                [self._buffer, mx.zeros(new_size - self._buffer.size, dtype=mx.int32)]
            )
        self._buffer[start:end] = tokens
        self._size = end

        return self._buffer[:end]

    @property
    def state(self):
        return self._buffer

    @property
    def tokens(self):
        return self._buffer[: self._size]


@dataclass
class PromptTrieResult:
    model: Any
    exact: Optional[List[int]]  # Exact match found
    shorter: Optional[List[int]]  # Longest prefix with a value
    longer: Optional[List[int]]  # Shortest value that extends beyond tokens
    common_prefix: int  # Length of common prefix with any path


class PromptTrie:
    def __init__(self):
        self._trie = {}

    def add(self, model: Any, tokens: List[int], value: Any):
        if model not in self._trie:
            self._trie[model] = {}

        current = self._trie[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, model: Any, tokens: List[int]):
        current = self._trie[model]
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, model: Any, tokens: List[int]):
        path = [self._trie[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, model: Any, tokens: List[int]):
        values = []
        current = self._trie[model]
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, model: Any, tokens: List[int], find_longer: bool = True) -> PromptTrieResult:
        if model not in self._trie:
            return PromptTrieResult(model, None, None, None, 0)

        current = self._trie[model]

        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer via O(V) BFS with parent backpointers
        longer = None
        common_prefix = index
        if index > 0 and find_longer:
            nodes = [(current, -1, None)]
            q = deque([0])
            found_idx = -1
            while q:
                ni = q.popleft()
                node, _, _ = nodes[ni]
                if "__value__" in node:
                    found_idx = ni
                    break
                for tok, child in node.items():
                    nodes.append((child, ni, tok))
                    q.append(len(nodes) - 1)
            if found_idx >= 0:
                extra = []
                cur_idx = found_idx
                while cur_idx > 0:
                    _, parent_idx, tok = nodes[cur_idx]
                    extra.append(tok)
                    cur_idx = parent_idx
                extra.reverse()
                longer = tokens[:index] + extra
        return PromptTrieResult(model, None, shorter, longer, common_prefix)


class LRUPromptCache:
    _DISK_FORMAT = 1

    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        nbytes: int
        cache_type: str

    @dataclass
    class _DiskCacheEntry:
        path: str
        nbytes: int
        cache_type: str
        model_key: Any
        tokens: List[int]

    class CacheOrder:
        def __init__(self, ordering: Optional[List[str]] = None):
            self._ordering = (
                ["checkpoint", "assistant", "user", "system"]
                if ordering is None
                else ordering
            )
            self._lrus = {k: deque() for k in self._ordering}

        def __len__(self):
            return sum(len(lru) for lru in self._lrus.values())

        def count_type(self, cache_type: str) -> int:
            return len(self._lrus.get(cache_type, ()))

        def count_non_checkpoint(self) -> int:
            return sum(len(q) for k, q in self._lrus.items() if k != "checkpoint")

        def push(self, model: Any, tokens: List[Any], cache_type: str = "assistant"):
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model: Any, tokens: List[Any]):
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove((model, tokens))
                    break
                except ValueError:
                    pass

        def pop(self, *, skip_checkpoint: bool = False):
            # Evict periodic checkpoints before conversation entries.
            if not skip_checkpoint and self._lrus.get("checkpoint"):
                return self._lrus["checkpoint"].popleft()
            i = 1 if self._ordering and self._ordering[0] == "checkpoint" else 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            return self._lrus[self._ordering[-1]].popleft()

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        max_checkpoints: int = 32,
        disk_dir: Optional[str] = None,
        disk_max_bytes: int = 8 << 30,
    ):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.max_checkpoints = max_checkpoints
        self._trie = PromptTrie()
        self._lru = LRUPromptCache.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = {k: 0 for k in self._lru._ordering}
        self._has_trimmable = False
        self.disk_dir = os.path.expanduser(disk_dir) if disk_dir else None
        self.disk_max_bytes = int(disk_max_bytes)
        self._disk_queue: Optional[Queue] = None
        self._disk_thread: Optional[Thread] = None
        if self.disk_dir:
            self._init_disk_cache()

    @staticmethod
    def _disk_key_hash(model: Any, tokens: List[int]) -> str:
        h = hashlib.sha256()
        mk = json.dumps(list(model) if isinstance(model, tuple) else model, ensure_ascii=True)
        h.update(mk.encode("utf-8"))
        h.update(b":")
        h.update(len(tokens).to_bytes(8, "little"))
        for t in tokens:
            h.update(int(t).to_bytes(4, "little", signed=True))
        return h.hexdigest()[:32]

    def _init_disk_cache(self):
        os.makedirs(self.disk_dir, exist_ok=True)
        loaded = []
        for fn in os.listdir(self.disk_dir):
            if not fn.endswith(".json"):
                continue
            jp = os.path.join(self.disk_dir, fn)
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("format") != self._DISK_FORMAT:
                    continue
                sf = os.path.join(self.disk_dir, meta["file"])
                if not os.path.isfile(sf):
                    continue
                mk = tuple(meta["model_key"]) if meta.get("model_is_tuple") else meta["model_key"]
                toks = [int(t) for t in meta["tokens"]]
                ctype = meta.get("cache_type", "assistant")
                nbytes = int(meta.get("nbytes", os.path.getsize(sf)))
                loaded.append((os.path.getmtime(sf), mk, toks, sf, nbytes, ctype))
            except Exception:
                continue
        for _, mk, toks, sf, nbytes, ctype in sorted(loaded, key=lambda x: x[0]):
            self._trie.add(mk, toks, LRUPromptCache._DiskCacheEntry(sf, nbytes, ctype, mk, toks))
        self._disk_queue = Queue(maxsize=32)
        self._disk_thread = Thread(target=self._disk_worker, daemon=True)
        self._disk_thread.start()

    def _prune_disk(self):
        if not self.disk_dir or self.disk_max_bytes <= 0:
            return
        files = []
        total = 0
        for fn in os.listdir(self.disk_dir):
            if fn.endswith(".safetensors"):
                p = os.path.join(self.disk_dir, fn)
                try:
                    st = os.stat(p)
                    files.append((st.st_mtime, st.st_size, p))
                    total += st.st_size
                except OSError:
                    pass
        if total <= self.disk_max_bytes:
            return
        files.sort()
        for _, sz, sf_path in files:
            if total <= self.disk_max_bytes:
                break
            try:
                os.remove(sf_path)
            except OSError:
                pass
            jp = sf_path[: -len(".safetensors")] + ".json"
            try:
                if os.path.exists(jp):
                    os.remove(jp)
            except OSError:
                pass
            total -= sz

    def _disk_worker(self):
        while True:
            item = self._disk_queue.get()
            try:
                if item is None:
                    return
                model, tokens, cache_snap, cache_type = item
                h = self._disk_key_hash(model, tokens)
                sf_name = f"{h}.safetensors"
                sf_path = os.path.join(self.disk_dir, sf_name)
                jp_path = os.path.join(self.disk_dir, f"{h}.json")
                if os.path.exists(sf_path) and os.path.exists(jp_path):
                    continue
                tmp_sf = os.path.join(self.disk_dir, f".{h}.{os.getpid()}.tmp.safetensors")
                tmp_jp = os.path.join(self.disk_dir, f".{h}.{os.getpid()}.tmp.json")
                save_prompt_cache(tmp_sf, cache_snap, {"cache_type": cache_type})
                os.replace(tmp_sf, sf_path)
                meta = {
                    "format": self._DISK_FORMAT,
                    "model_key": list(model) if isinstance(model, tuple) else model,
                    "model_is_tuple": isinstance(model, tuple),
                    "tokens": [int(t) for t in tokens],
                    "cache_type": cache_type,
                    "nbytes": sum(c.nbytes for c in cache_snap),
                    "file": sf_name,
                }
                with open(tmp_jp, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                os.replace(tmp_jp, jp_path)
                self._prune_disk()
            except Exception as e:
                logging.warning(f"Persistent prompt cache write failed: {e}")
            finally:
                self._disk_queue.task_done()

    def _enqueue_disk(self, model: Any, tokens: List[int], prompt_cache: List[Any], cache_type: str):
        if self._disk_queue is None or cache_type not in ("system", "user", "assistant") or len(tokens) < 512:
            return
        try:
            states = [c.state for c in prompt_cache]
            mx.eval(states)
            snap = [type(c).from_state(s) for c, s in zip(prompt_cache, states)]
            self._disk_queue.put_nowait((model, tokens[:], snap, cache_type))
        except QueueFull:
            pass
        except Exception as e:
            logging.warning(f"Persistent prompt cache enqueue failed: {e}")

    def flush_disk(self):
        if self._disk_queue is not None:
            self._disk_queue.join()

    def _materialize_entry(self, model: Any, tokens: List[int], entry: Any):
        if not isinstance(entry, LRUPromptCache._DiskCacheEntry):
            return entry
        try:
            cache = load_prompt_cache(entry.path)
        except Exception as e:
            logging.warning(f"Persistent prompt cache load failed for {entry.path}: {e}")
            try:
                self._trie.pop(model, tokens)
            except KeyError:
                pass
            return None
        ram_entry = LRUPromptCache.CacheEntry(cache, sum(c.nbytes for c in cache), entry.cache_type)
        if ram_entry.cache_type not in self._n_bytes_by_type:
            ram_entry.cache_type = "assistant"
        self._trie.add(model, tokens, ram_entry)
        self._n_bytes += ram_entry.nbytes
        self._n_bytes_by_type[ram_entry.cache_type] += ram_entry.nbytes
        self._lru.push(model, tokens, ram_entry.cache_type)
        if can_trim_prompt_cache(cache):
            self._has_trimmable = True
        self._enforce_ram_limits()
        return ram_entry

    def __len__(self):
        return len(self._lru)

    @property
    def nbytes(self):
        return self._n_bytes

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens, find_longer=self._has_trimmable)
        if result.exact is not None:
            cache_entry = self._materialize_entry(result.model, result.exact, self._trie.get(result.model, result.exact))
            if cache_entry is not None:
                return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._materialize_entry(result.model, result.longer, self._trie.get(result.model, result.longer))
            if cache_entry is not None and can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

        if short_length > 0:
            cache_entry = self._materialize_entry(result.model, result.shorter, self._trie.get(result.model, result.shorter))
            if cache_entry is not None:
                return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]

        return None, tokens

    def _evict_entry(self, *, skip_checkpoint: bool = False):
        model, tokens = self._lru.pop(skip_checkpoint=skip_checkpoint)
        entry = self._trie.pop(model, tokens)
        if isinstance(entry, LRUPromptCache.CacheEntry):
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def _enforce_ram_limits(self):
        while self._lru.count_type("checkpoint") > self.max_checkpoints:
            self._evict_entry(skip_checkpoint=False)
        while self._lru.count_non_checkpoint() > self.max_size:
            self._evict_entry(skip_checkpoint=True)
        while self._n_bytes > self.max_bytes and len(self._lru) > 0:
            self._evict_entry(skip_checkpoint=False)

    def promote_cache_type(self, model: Any, tokens: List[int], cache_type: str = "system") -> bool:
        if cache_type not in self._n_bytes_by_type:
            return False
        try:
            entry = self._trie.get(model, tokens)
        except KeyError:
            return False
        if isinstance(entry, LRUPromptCache._DiskCacheEntry):
            entry.cache_type = cache_type
            return True
        if entry.cache_type == cache_type:
            return True
        self._lru.remove(model, tokens)
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        entry.cache_type = cache_type
        self._n_bytes_by_type[cache_type] += entry.nbytes
        self._lru.push(model, tokens, cache_type)
        self._enqueue_disk(model, tokens, entry.prompt_cache, cache_type)
        return True

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        if cache_type not in self._n_bytes_by_type:
            cache_type = "assistant"
        # Make the cache entry
        entry = LRUPromptCache.CacheEntry(
            prompt_cache, sum(c.nbytes for c in prompt_cache), cache_type
        )

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(model, tokens, entry)
        if isinstance(prev, LRUPromptCache.CacheEntry):
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(model, tokens)
        self._lru.push(model, tokens, cache_type)

        # If it is a trimmable cache remove all prefixes cause they just take
        # space
        if can_trim_prompt_cache(prompt_cache):
            self._has_trimmable = True
            for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
                if isinstance(entry, LRUPromptCache.CacheEntry):
                    self._n_bytes -= entry.nbytes
                    self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                    self._lru.remove(model, tokens[:prefix_len])

        self._enforce_ram_limits()
        self._enqueue_disk(model, tokens, prompt_cache, cache_type)

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > n_sequences:
            self._evict_entry(skip_checkpoint=False)
        while self._n_bytes > n_bytes and len(self._lru) > 0:
            self._evict_entry(skip_checkpoint=False)

    def stats_by_type(self):
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result
