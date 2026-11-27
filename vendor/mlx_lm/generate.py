# Copyright © 2023 Apple Inc.

import argparse
import contextlib
import copy
import functools
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from .generate_utils import BatchCounters, BatchCountersSnapshot, BatchStats
from .models.cache import (
    QuantizedKVCache,
    TokenBuffer,
    can_trim_prompt_cache,
    load_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from .processors import load_image, load_processor
from .sample_utils import LogitsProcessor, Sampler, greedy_sampler, make_sampler
from .tokenizer_utils import TokenizerWrapper
from .utils import (
    does_model_support_input_embeddings,
    load,
    maybe_set_recommended_wired_limit,
)

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.1
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = None
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000
_SERVER_BATCHDECODE = os.environ.get("LOOPLING_SERVER_BATCHDECODE", "1") not in (
    "0",
    "",
    "false",
    "False",
)
_PREFILL_CLEAR_INTERVAL = max(1, int(os.environ.get("LOOPLING_PREFILL_CLEAR_INTERVAL", "8")))
_DECODE_BURST_STEPS = max(1, int(os.environ.get("LOOPLING_DECODE_BURST_STEPS", "16")))
_DECODE_BURST_S = max(0.0, float(os.environ.get("LOOPLING_DECODE_BURST_MS", "250")) / 1000.0)
_SHORT_PREFILL_THRESH = max(1, int(os.environ.get("LOOPLING_SHORT_PREFILL_THRESH", "2048")))


def _is_inkling2_model(model: nn.Module) -> bool:
    inner = getattr(model, "model", None)
    lm = getattr(inner, "language_model", None) if inner is not None else None
    return lm is not None and hasattr(lm, "layer_visits") and hasattr(model, "args")


def _inkling2_bd_caches_ok(model: nn.Module, caches: List[List[Any]]) -> bool:
    if not _SERVER_BATCHDECODE or not (1 <= len(caches) <= 32) or not _is_inkling2_model(model):
        return False
    from .models.inkling2 import layers as _L
    from .models.inkling2.batchdecode import BatchDecoder, _batch_cache_ok, _b1_cache_ok

    nv = len(model.model.language_model.layer_visits)
    if not caches[0] or len(caches[0]) != nv:
        return False
    if len(caches) == 1:
        return _L._FAST_ON and _L._use_kernels() and all(_b1_cache_ok(caches[0][v]) for v in range(nv))
    return _L._FAST_ON and _batch_cache_ok(caches, nv)


def _df2_batch_enabled() -> bool:
    return (
        os.environ.get("LOOPLING_MTP") == "dflash2"
        and os.environ.get("LOOPLING_DFLASH2_BATCH", "0") not in ("0", "", "false", "False")
    )


def _df2_c1_enabled() -> bool:
    return (
        os.environ.get("LOOPLING_MTP") == "dflash2"
        and os.environ.get("LOOPLING_DFLASH2_C1", "1") not in ("0", "", "false", "False")
    )


def _df2_c1_max_ctx() -> int:
    return int(os.environ.get("LOOPLING_DFLASH2_MAX_CTX", "1048576"))


def _df2_batch_auto_enabled() -> bool:
    return (
        os.environ.get("LOOPLING_MTP") == "dflash2"
        and os.environ.get("LOOPLING_DFLASH2_BATCH", "1") not in ("0", "", "false", "False")
    )


def _df2_batch_min_ctx() -> int:
    return int(os.environ.get("LOOPLING_DFLASH2_BATCH_MIN_CTX", "16384"))


def _get_df2_batch_ctx(model: nn.Module):
    from .models.inkling2 import dflash2 as DF2
    from .models.inkling2.mtp import default_taps, parse_taps

    if not os.path.isdir(DF2.DFLASH2_DIR):
        return None
    tm = model.model.language_model
    passes = tm.args.loop_config["num_passes"]
    cfg = getattr(tm, "loopcd", None)
    if cfg is not None and cfg.enabled_for(passes) and cfg.mode != "hidden":
        return None
    dr = DF2._get_drafter(model, DF2.DFLASH2_DIR, reset=False)
    taps = parse_taps(os.environ.get("LOOPLING_MTP_TAPS", default_taps(passes)), passes)
    tap_keys = [t if t in ("F", "f", "o") else int(t[1:]) for t in taps]
    tap_set = {k for k in tap_keys if isinstance(k, int)}
    max_c = int(os.environ.get("LOOPLING_DFLASH2_BATCH_MAX_C", str(DF2._DFLASH2_BATCH_MAX_C)))
    return dr, tm, tap_keys, tap_set, max(1, min(8, max_c))
DEFAULT_PREFILL_STEP_SIZE = 2048


def str2bool(string):
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--extra-eos-token",
        type=str,
        default=(),
        nargs="+",
        help="Add tokens in the list of eos tokens that stop generation.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=(),
        nargs="+",
        help="Image files for the image placeholders of the prompt, in order.",
    )
    parser.add_argument(
        "--prefill-response",
        default=None,
        help="Prefill response to be used for the chat template",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=DEFAULT_XTC_THRESHOLD,
        help="Threshold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=DEFAULT_MIN_TOKENS_TO_KEEP,
        help="Minimum tokens to keep for min-p sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        help="Additional config for `apply_chat_template`. Should be a dictionary of"
        " string keys to values represented as a JSON decodable string.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of prompt tokens to process at a time. Smaller values "
        f"lower peak memory during prefill (default: {DEFAULT_PREFILL_STEP_SIZE})",
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    return parser


# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    old_limit = maybe_set_recommended_wired_limit()
    if old_limit is None:
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    """
    The output of :func:`stream_generate`.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        token (int): The next token.
        from_draft (bool): Whether the token was generated by the draft model.
        logprobs (mx.array): A vector of log probabilities.
        prompt_tokens (int): The number of tokens in the prompt.
        prompt_tps (float): The prompt processing tokens-per-second.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        peak_memory (float): The peak memory used so far in GB.
        finish_reason (str): The reason the response is being sent: "length", "stop" or `None`
    """

    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    *,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    logits_processors: Optional[List[LogitsProcessor]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 4096,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    thinking_tracker: Optional[Any] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Sampler, optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[LogitsProcessor], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``5000``.
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = None
    _cd_cfg = getattr(model, "loopcd", None)
    _track_thinking = thinking_tracker is not None and bool(getattr(_cd_cfg, "exclude_thinking", False))
    _prompt_ids = (
        [int(t) for t in prompt.tolist()]
        if (_track_thinking and input_embeddings is None and len(prompt) > 0)
        else []
    )

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or greedy_sampler

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(stream):
            if (
                not logits_processors
                and input_tokens.size == 1
                and input_embeddings is None
                and hasattr(model, "decode_step_logprobs")
            ):
                res = model.decode_step_logprobs(
                    input_tokens[None],
                    cache=prompt_cache,
                    greedy=(sampler is greedy_sampler),
                )
                if res is not None:
                    sampled, logprobs = res
                    quantize_cache_fn(prompt_cache)
                    if sampled is None:
                        sampled = sampler(logprobs)
                    return sampled, logprobs.squeeze(0)

            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            processed = prompt[:n_to_process]
            _model_call(
                input_tokens=processed[None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            if logits_processors and len(processed) > 0:
                tokens = (
                    mx.concat([tokens, processed]) if tokens is not None else processed
                )
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()

        if _track_thinking:
            from .models.inkling2 import loopcd as _loopcd

            thinking_tracker.reset(_prompt_ids)
            _loopcd.set_cache_skip(prompt_cache, thinking_tracker.in_thinking)

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
        if _track_thinking:
            n = 0
            while True:
                mx.eval(y, logprobs)
                if n == 0:
                    prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                if n == max_tokens:
                    break
                tok_id = y.item()
                thinking_tracker.update(tok_id)
                _loopcd.set_cache_skip(prompt_cache, thinking_tracker.in_thinking)
                yield tok_id, logprobs
                if n % 256 == 0:
                    mx.clear_cache()
                if n + 1 == max_tokens:
                    break
                y, logprobs = _step(y)
                n += 1
        else:
            mx.async_eval(y, logprobs)

            n = 0
            while True:
                if n != max_tokens:
                    next_y, next_logprobs = _step(y)
                    mx.async_eval(next_y, next_logprobs)
                if n == 0:
                    mx.eval(y)
                    prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                if n == max_tokens:
                    break
                yield y.item(), logprobs
                if n % 256 == 0:
                    mx.clear_cache()
                y, logprobs = next_y, next_logprobs
                n += 1


def speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    logits_processors: Optional[List[LogitsProcessor]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        draft_model (nn.Module): The draft model for speculative decoding.
        num_draft_tokens (int, optional): The number of draft tokens for
          speculative decoding. Default: ``2``.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Sampler, optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[LogitsProcessor], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place. The cache must be trimmable.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``5000``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, a vector of log probabilities,
          and a bool indicating if the token was generated by the draft model
    """

    y = prompt.astype(mx.uint32)
    prev_tokens = y[:-1] if logits_processors else None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = make_prompt_cache(model)
        draft_cache = make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    if not can_trim_prompt_cache(model_cache):
        types = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(
            f"Speculative decoding requires a trimmable prompt cache " f"(got {types})."
        )

    sampler = sampler or greedy_sampler

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(model, cache, y, n_predict=1):
        logits = model(y[None], cache=cache)
        logits = logits[:, -n_predict:, :]

        quantize_cache_fn(cache)
        if logits_processors:
            nonlocal prev_tokens
            out_y, out_logprobs = [], []
            if n_predict > 1:
                y = y[: -(n_predict - 1)]
            for i in range(n_predict):
                prev_tokens = (
                    mx.concatenate([prev_tokens, y]) if prev_tokens is not None else y
                )
                y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                out_y.append(y)
                out_logprobs.append(logprobs)
            return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0)
        else:
            return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        while y.size > 1:
            n_to_process = min(prefill_step_size, y.size - 1)
            model(y[:n_to_process][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[n_to_process:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        trim_prompt_cache(model_cache, num_draft - num_accept)
        trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y, _ = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    with mx.stream(stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

        ntoks = 0
        # Set these so the finally block doesn't raise
        num_draft = 0
        n = 0
        try:
            while True:
                num_draft = min(max_tokens - ntoks, num_draft_tokens)
                draft_tokens = _draft_generate(draft_y, num_draft)
                if prev_tokens is not None:
                    prev_tokens = prev_tokens[
                        : prev_tokens.size - y.size - num_draft + 1
                    ]
                y = mx.concatenate([y, draft_tokens])
                tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
                mx.eval(tokens, draft_tokens)
                draft_tokens = draft_tokens.tolist()
                tokens = tokens.tolist()
                n = 0
                while n < num_draft:
                    tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                    if tn != dtn:
                        break
                    n += 1
                    ntoks += 1
                    yield tn, lpn, True
                    if ntoks == max_tokens:
                        break
                if ntoks < max_tokens:
                    ntoks += 1
                    yield tokens[n], logprobs[n], False

                if ntoks == max_tokens:
                    break

                y = mx.array([tokens[n]], mx.uint32)
                draft_y = y

                # If we accepted all the draft tokens, include the last
                # draft token in the next draft step since it hasn't been
                # processed yet by the draft model
                if n == num_draft:
                    draft_y = mx.concatenate(
                        [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                    )

                if prev_tokens is not None and num_draft > 0:
                    prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
                _rewind_cache(num_draft, n)
        finally:
            _rewind_cache(num_draft, n)


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt string or
          integer tokens.
        max_tokens (int): The maximum number of tokens to generate.
          Default: ``256``.
        draft_model (Optional[nn.Module]): An optional draft model. If provided
          then speculative decoding is used. The draft model must use the same
          tokenizer as the main model. Default: ``None``.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
    if max_tokens == 0:
        raise ValueError(
            "Maximum number of tokens must be non-zero (use -1 for no limit)."
        )

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens

    _ss = None
    if draft_model is None:
        from mlx_lm import selfspec

        _ss = selfspec.parse_env()
        if _ss is not None and not selfspec.applicable(model, *_ss):
            _ss = None
    if draft_model is None and os.environ.get("LOOPLING_MTP") in ("1", "dflash2") and not kwargs.get("logits_processors"):
        # official MTP or Splash DFlash2 drafter; default OFF
        if _ss is not None:
            raise ValueError("LOOPLING_MTP and LOOPLING_SELFSPEC cannot be combined")
        from mlx_lm.models.inkling2.mtp import mtp_generate_step

        kwargs.pop("num_draft_tokens", None)
        kwargs.setdefault("eos_token_ids", tokenizer.eos_token_ids)
        token_generator = mtp_generate_step(prompt, model, **kwargs)
        token_generator = (
            (token, logprobs, fd) for token, logprobs, fd in token_generator
        )
    elif _ss is not None:
        # LOOPLING_SELFSPEC=<draft_passes>:<k>: greedy self-speculative decoding
        for _k in ("max_kv_size", "prompt_progress_callback", "full_prompt", "checkpoint_callback", "segment_ends", "cache_state"):
            kwargs.pop(_k, None)
        token_generator = selfspec.selfspec_generate_step(
            prompt, model, _ss[0], _ss[1], stream, **kwargs
        )
    elif draft_model is None:
        for _k in ("num_draft_tokens", "full_prompt", "checkpoint_callback", "segment_ends", "cache_state"):
            kwargs.pop(_k, None)
        if kwargs.get("thinking_tracker") is None and getattr(getattr(model, "loopcd", None), "exclude_thinking", False):
            from mlx_lm.models.inkling2.loopcd import ThinkingTracker

            _tt = ThinkingTracker.from_tokenizer(tokenizer)
            if _tt is not None:
                kwargs["thinking_tracker"] = _tt
        token_generator = generate_step(prompt, model, stream, **kwargs)
        # from_draft always false for non-speculative generation
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    else:
        for _k in ("max_kv_size", "prompt_progress_callback", "full_prompt", "checkpoint_callback", "segment_ends", "cache_state"):
            kwargs.pop(_k, None)
        if kwargs.pop("input_embeddings", None) is not None:
            raise ValueError("Speculative decoding does not support input embeddings.")
        token_generator = speculative_generate_step(
            prompt, model, draft_model, stream, **kwargs
        )
    with wired_limit(model, [stream]):
        try:
            tic = time.perf_counter()
            for n, (token, logprobs, from_draft) in enumerate(token_generator):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = prompt.size / prompt_time
                    tic = time.perf_counter()
                if token in tokenizer.eos_token_ids:
                    break

                detokenizer.add_token(token)
                if (n + 1) == max_tokens:
                    break

                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    from_draft=from_draft,
                    prompt_tokens=prompt.size,
                    prompt_tps=prompt_tps,
                    generation_tokens=n + 1,
                    generation_tps=(n + 1) / (time.perf_counter() - tic),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )

            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
            )
        finally:
            if hasattr(token_generator, "close"):
                token_generator.close()


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (Union[str, List[int]]): The input prompt string or integer tokens.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       kwargs: The remaining options get passed to :func:`stream_generate`.
          See :func:`stream_generate` for more details.
    """
    if verbose:
        print("=" * 10)

    text = ""
    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text

    if verbose:
        print()
        print("=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return
        print(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {response.peak_memory:.3f} GB")
    return text


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


def _merge_caches(caches):
    batch_cache = []

    if not caches:
        return batch_cache

    for i in range(len(caches[0])):
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
        else:
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _normalize_samplers(samplers, *, n, fallback):
    """One concrete sampler per sequence."""
    if not samplers:
        return [fallback] * n
    return [fallback if s is None else s for s in samplers]


def _normalize_logits_processors(logits_processors, *, n):
    """One concrete tuple of processors per sequence."""
    if not logits_processors:
        return [() for _ in range(n)]
    return [tuple(lp) if lp else () for lp in logits_processors]


def _build_trie(sequences: Sequence[Any]) -> dict:
    """Build an Aho-Corasick trie from the provided sequences

    See https://en.wikipedia.org/wiki/Aho–Corasick_algorithm . Nodes keep their
    depth in ``__depth__``.
    """
    trie = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for depth, tok in enumerate(seq, 1):
                node = node.setdefault(tok, {"__depth__": depth})
            if node is trie:
                # An empty pattern would make the root match every token.
                continue
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {"__depth__": 1})
            node["__match__"] = ((seq,), idx)

    # BFS to set failure links and propagate matches.
    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__", "__depth__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]
    return trie


class StopSequences:
    """An immutable Aho-Corasick automaton over stop sequences.

    Any matched sequence signals stop. A :class:`StopSequences.Matcher` holds
    one stream's position in it.
    """

    class Matcher:
        """A position in a :class:`StopSequences` automaton."""

        def __init__(self, root):
            self._root = root
            self._node = root

        def advance(self, token: int) -> bool:
            """Consume one token. Returns whether a stop sequence completed."""
            node = self._node
            # Fail back until the token continues a match, or we hit the root.
            while token not in node and node is not self._root:
                node = node["__fail__"]
            if token in node:
                node = node[token]
            self._node = node
            return node.get("__match__") is not None

    def __init__(self, stop_sequences: Optional[Sequence[Sequence[int]]] = None):
        self._root = _build_trie(stop_sequences) if stop_sequences else {}

    def matcher(self) -> "StopSequences.Matcher":
        """A fresh matcher at the start of this automaton."""
        return self.Matcher(self._root)


TextState = Literal["normal", "reasoning", "tool", "stop", "header"]


def _check_state(state: str) -> None:
    if state not in get_args(TextState):
        raise ValueError(f"Unknown state {state!r}.")


class TextStateMachine:
    """Split streamed text into content, reasoning and tool calls.

    ``transitions`` maps a state to its ``(marker, next_state)`` pairs, where a
    ``next_state`` of ``None`` returns to the previous state. Markers match on
    text, so a marker split over tokens still matches.

    ``normal`` text is the content, ``reasoning`` text the reasoning and each
    visit to ``tool`` one tool call. ``stop`` ends the text: the reader is
    then ``stopped``. The text of ``header``, for the headers of a response
    format, is dropped.

    Example:

        sm = TextStateMachine(
            {
                "normal": [("<think>", "reasoning"), ("<tool_call>", "tool")],
                "reasoning": [("</think>", "normal")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        reader = sm.reader(state="normal")
        reader.feed("<think>Hmm.</think>Hi<tool_call>{}</tool_call>")
        # Delta(content="Hi", reasoning="Hmm.", tool_calls=("{}",))
    """

    @dataclass(frozen=True)
    class Delta:
        """New response text, as in a chat completion delta."""

        content: str = ""
        reasoning: str = ""
        tool_calls: tuple[str, ...] = ()

        def __add__(self, other: "TextStateMachine.Delta") -> "TextStateMachine.Delta":
            return TextStateMachine.Delta(
                content=self.content + other.content,
                reasoning=self.reasoning + other.reasoning,
                tool_calls=self.tool_calls + other.tool_calls,
            )

    class Reader:
        """One stream's position in a :class:`TextStateMachine`."""

        def __init__(self, machine: "TextStateMachine", *, state: TextState):
            _check_state(state)
            self._states = machine._states
            self._held = ""
            self._tool = ""
            self._previous = state
            self._enter(state)

        @property
        def stopped(self) -> bool:
            return self._state == "stop"

        def _enter(self, state: TextState) -> None:
            self._state = state
            self._root, self._next = self._states.get(state, ({}, []))
            self._node = self._root

        def _add(
            self, delta: "TextStateMachine.Delta", text: str
        ) -> "TextStateMachine.Delta":
            if self._state == "normal":
                return delta + TextStateMachine.Delta(content=text)
            if self._state == "reasoning":
                return delta + TextStateMachine.Delta(reasoning=text)
            if self._state == "tool":
                self._tool += text
            # Text after a stop is dropped.
            return delta

        def _move(
            self, delta: "TextStateMachine.Delta", state: TextState | None
        ) -> "TextStateMachine.Delta":
            """``None`` is the previous state. Leaving ``tool`` closes the call."""
            if state is None:
                state = self._previous
            if state != self._state:
                if self._state == "tool":
                    delta = self._close_tool_call(delta)
                self._previous = self._state
            self._enter(state)
            return delta

        def _close_tool_call(
            self, delta: "TextStateMachine.Delta"
        ) -> "TextStateMachine.Delta":
            tool_call = self._tool.strip()
            self._tool = ""
            if tool_call:
                delta += TextStateMachine.Delta(tool_calls=(tool_call,))
            return delta

        def feed(self, text: str) -> "TextStateMachine.Delta":
            """Read a chunk. Text that can still start a marker is held back."""
            delta = TextStateMachine.Delta()
            start = len(self._held)
            text = self._held + text
            begin = 0
            for i in range(start, len(text)):
                ch = text[i]
                node = self._node
                while ch not in node and node is not self._root:
                    node = node["__fail__"]
                self._node = node.get(ch, self._root)
                match = self._node.get("__match__")
                if match is not None:
                    before = text[begin : i + 1 - len(match[0])]
                    delta = self._add(delta, before)
                    delta = self._move(delta, self._next[match[1]])
                    begin = i + 1
            end = len(text) - self._node.get("__depth__", 0)
            delta = self._add(delta, text[begin:end])
            self._held = text[end:]
            return delta

        def finish(self, text: str = "") -> "TextStateMachine.Delta":
            """Read the last chunk, release the held text and close a tool call."""
            delta = self.feed(text)
            delta = self._add(delta, self._held)
            self._held = ""
            self._node = self._root
            return self._close_tool_call(delta)

    def __init__(
        self, transitions: dict[TextState, Sequence[tuple[str, TextState | None]]]
    ):
        # Per state: its marker trie and next states.
        self._states = {}
        for state, edges in transitions.items():
            _check_state(state)
            for _, next_state in edges:
                if next_state is not None:
                    _check_state(next_state)
            markers = [marker for marker, _ in edges]
            self._states[state] = (_build_trie(markers), [dst for _, dst in edges])

    def reader(self, *, state: TextState = "normal") -> "TextStateMachine.Reader":
        """A fresh reader that starts in ``state``."""
        return self.Reader(self, state=state)

    @classmethod
    def from_stop_words(cls, stop_words: Sequence[str]) -> "TextStateMachine":
        """A TextStateMachine whose stop words end the text."""
        return cls({"normal": [(w, "stop") for w in stop_words]})

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerWrapper) -> "TextStateMachine":
        """A TextStateMachine with the markers of a tokenizer."""
        transitions = {}

        if tokenizer.has_thinking:
            transitions.setdefault("normal", []).append(
                (tokenizer.think_start, "reasoning")
            )
            transitions["reasoning"] = [(tokenizer.think_end, "normal")]

        if tokenizer.has_tool_calling:
            # A response format that goes to "tool" decides where calls start.
            targets = set()
            for edges in tokenizer.response_transitions.values():
                targets.update(dst for _, dst in edges)
            if "tool" not in targets:
                transitions.setdefault("normal", []).append(
                    (tokenizer.tool_call_start, "tool")
                )
            if tokenizer.has_thinking:
                transitions["reasoning"].append((tokenizer.tool_call_start, "tool"))
            # Back to the state the tool call started in.
            transitions["tool"] = (
                [(tokenizer.tool_call_end, None)] if tokenizer.tool_call_end else []
            )

        if tokenizer.structural_markers:
            for w in tokenizer.structural_markers:
                transitions.setdefault("normal", []).append((w, "normal"))

        for state, edges in tokenizer.response_transitions.items():
            transitions.setdefault(state, []).extend(edges)

        return cls(transitions)


class PromptProcessingBatch:
    """
    A batch processor for prompt tokens with support for incremental processing.

    This class handles batched prompt processing, managing KV caches and preparing
    tokens for generation. It supports extending, filtering, and splitting batches.
    """

    @dataclass
    class Response:
        uid: int
        progress: tuple
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        caches: List[List[Any]],
        tokens: Optional[List[List[int]]] = None,
        prefill_step_size: int = 4096,
        samplers: Optional[List[Sampler]] = None,
        fallback_sampler: Optional[Sampler] = None,
        logits_processors: Optional[List[List[LogitsProcessor]]] = None,
        stop_sequences: Optional[List[StopSequences]] = None,
        max_tokens: Optional[List[int]] = None,
        need_logprobs: Optional[List[bool]] = None,
        df2_states: Optional[List[Any]] = None,
        need_final_cache: Optional[List[bool]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)
        self.tokens = tokens if tokens is not None else [[] for _ in uids]

        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or greedy_sampler
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.stop_sequences = (
            stop_sequences
            if stop_sequences is not None
            else [StopSequences()] * len(uids)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else [DEFAULT_MAX_TOKENS] * len(self.uids)
        )
        self.need_logprobs = (
            list(need_logprobs) if need_logprobs is not None else [True] * len(self.uids)
        )
        self._df2_states = (
            list(df2_states) if df2_states is not None else [None] * len(self.uids)
        )
        self.need_final_cache = (
            list(need_final_cache)
            if need_final_cache is not None
            else [True] * len(self.uids)
        )

    def __len__(self):
        return len(self.uids)

    def extract_cache(self, idx: int) -> List[Any]:
        lps = [c.left_padding for c in self.prompt_cache if getattr(c, "left_padding", None) is not None]
        if lps:
            mx.eval(lps)
        return [c.extract(idx) for c in self.prompt_cache]

    def extend(self, batch):
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_sequences.extend(batch.stop_sequences)
        self.need_logprobs.extend(batch.need_logprobs)
        self._df2_states.extend(batch._df2_states)
        self.need_final_cache.extend(batch.need_final_cache)

    def _copy(self):
        new_batch = self.__class__.__new__(self.__class__)
        new_batch.model = self.model
        new_batch.uids = list(self.uids)
        new_batch.prompt_cache = copy.deepcopy(self.prompt_cache)
        new_batch.tokens = list(self.tokens)
        new_batch.prefill_step_size = self.prefill_step_size
        new_batch.samplers = list(self.samplers)
        new_batch.fallback_sampler = self.fallback_sampler
        new_batch.logits_processors = list(self.logits_processors)
        new_batch.stop_sequences = list(self.stop_sequences)
        new_batch.max_tokens = list(self.max_tokens)
        new_batch.need_logprobs = list(self.need_logprobs)
        new_batch._df2_states = [None if s is None else s.clone() for s in self._df2_states]
        new_batch.need_final_cache = list(self.need_final_cache)
        return new_batch

    def split(self, indices: List[int]):
        indices = sorted(indices)
        indices_left = sorted(set(range(len(self.uids))) - set(indices))
        if not indices_left:
            new_batch = self.__class__.__new__(self.__class__)
            new_batch.model = self.model
            new_batch.uids = list(self.uids)
            new_batch.prompt_cache = self.prompt_cache
            self.prompt_cache = []
            new_batch.tokens = list(self.tokens)
            new_batch.prefill_step_size = self.prefill_step_size
            new_batch.samplers = list(self.samplers)
            new_batch.fallback_sampler = self.fallback_sampler
            new_batch.logits_processors = list(self.logits_processors)
            new_batch.stop_sequences = list(self.stop_sequences)
            new_batch.max_tokens = list(self.max_tokens)
            new_batch.need_logprobs = list(self.need_logprobs)
            new_batch._df2_states = list(self._df2_states)
            new_batch.need_final_cache = list(self.need_final_cache)
            self.filter([])
            if indices != list(range(len(new_batch.uids))):
                new_batch.filter(indices)
            return new_batch
        new_batch = self._copy()
        self.filter(indices_left)
        new_batch.filter(indices)

        return new_batch

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        self.samplers = [self.samplers[idx] for idx in keep]
        self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_sequences = [self.stop_sequences[idx] for idx in keep]
        self.need_logprobs = [self.need_logprobs[idx] for idx in keep]
        self._df2_states = [self._df2_states[idx] for idx in keep]
        self.need_final_cache = [self.need_final_cache[idx] for idx in keep]

    def prompt(self, tokens: List[List[int]]):
        """
        Process prompt tokens through the model.

        Args:
            tokens: List of token sequences to process.
        """
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")

        if not tokens:
            return

        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        df2_ctx = None
        df2_tail_only = False
        if _is_inkling2_model(self.model) and (
            _df2_batch_enabled()
            or (
                _df2_c1_enabled()
                and len(self.uids) == 1
                and (len(self.tokens[0]) + lengths[0]) <= _df2_c1_max_ctx()
                and not self.need_logprobs[0]
                and not self.logits_processors[0]
            )
            or (
                _df2_batch_auto_enabled()
                and 2 <= len(self.uids) <= 4
                and min(len(self.tokens[i]) + lengths[i] for i in range(len(self.uids))) >= _df2_batch_min_ctx()
                and not any(self.need_logprobs)
                and not any(self.logits_processors)
            )
        ):
            df2_ctx = _get_df2_batch_ctx(self.model)
            if df2_ctx is not None:
                df2_tail_only = not _df2_batch_enabled()
                dr = df2_ctx[0]
                for i, ti in enumerate(tokens):
                    st = self._df2_states[i]
                    if st is None:
                        st = self._df2_states[i] = dr.new_stream_state(allocate=True)
                        st.lookup.append(self.tokens[i])
                    st.lookup.append(ti)

        # Add the tokens to the self.tokens so they represent the tokens
        # contained in the KV Cache.
        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        # Calculate if we need to pad
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)

        # Prepare the caches and inputs. Right pad if needed otherwise just
        # cast to array.
        if max_padding > 0:
            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        # Actual prompt processing loop
        chunk_off = 0
        clear_counter = 0
        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            if df2_ctx is not None:
                dr, tm, tap_keys, tap_set, _ = df2_ctx
                # Only the final `dr.window` (1024) tokens enter DFlash2's sliding window.
                need_taps_chunk = (not df2_tail_only) or any(
                    chunk_off + n_to_process > max(0, l_i - dr.window) for l_i in lengths
                )
                if need_taps_chunk:
                    tm.tap_visits = tap_set
                    tm.tap_buf = {}
                try:
                    self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
                finally:
                    if need_taps_chunk:
                        tm.tap_visits = None
                        tb = tm.tap_buf
                        tm.tap_buf = {}
                    else:
                        tb = None
                v_states, v_feats = [], []
                if tb is not None:
                    for i, l_i in enumerate(lengths):
                        valid_i = max(0, min(n_to_process, l_i - chunk_off))
                        if valid_i > 0:
                            start_j = max(0, l_i - dr.window - chunk_off) if df2_tail_only else 0
                            if start_j < valid_i:
                                n_take = valid_i - start_j
                                v_states.append(self._df2_states[i])
                                v_feats.append(
                                    mx.concatenate(
                                        [tb[k][i, start_j:valid_i].reshape(n_take, 1, dr.H) for k in tap_keys],
                                        axis=1,
                                    )[None]
                                )
                if v_states:
                    dr.append_ctx_batch(v_states, v_feats)
                    mx.eval(
                        [c.state for c in self.prompt_cache],
                        *[x for st in v_states for x in st.ck + st.cv],
                    )
                else:
                    mx.eval([c.state for c in self.prompt_cache])
            else:
                self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
                mx.eval([c.state for c in self.prompt_cache])
            clear_counter += 1
            chunk_off += n_to_process
            tokens = tokens[:, n_to_process:]
            if tokens.shape[1] == 0 or clear_counter % _PREFILL_CLEAR_INTERVAL == 0:
                mx.clear_cache()

        # Finalize the cache if there was any padding
        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

    def generate(self, tokens: List[List[int]]):
        """
        Transition from prompt processing to generation.

        Args:
            tokens: Final tokens for each sequence to start generation.

        Returns:
            A GenerationBatch ready for token generation.
        """
        if any(len(t) > 1 for t in tokens):
            self.prompt([t[:-1] for t in tokens])
        last_token = mx.array([t[-1] for t in tokens])

        generation = GenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.stop_sequences,
            self.max_tokens,
            self.need_logprobs,
            df2_states=self._df2_states,
            need_final_cache=self.need_final_cache,
        )

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []
        self.need_logprobs = []
        self._df2_states = []
        self.need_final_cache = []

        return generation

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Sampler,
        prefill_step_size: int = 4096,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_sequences=[],
            need_logprobs=[],
            need_final_cache=[],
        )


class GenerationBatch:
    """
    A batched token generator that manages multiple sequences in parallel.

    This class handles the generation phase after prompt processing, managing
    KV caches, sampling, and stop sequence detection for multiple sequences.
    """

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        tokens: List[List[int]],
        samplers: Optional[List[Sampler]],
        fallback_sampler: Sampler,
        logits_processors: Optional[List[List[LogitsProcessor]]],
        stop_sequences: List[StopSequences],
        max_tokens: List[int],
        need_logprobs: Optional[List[bool]] = None,
        df2_states: Optional[List[Any]] = None,
        per_caches: Optional[List[List[Any]]] = None,
        need_final_cache: Optional[List[bool]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens

        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.stop_sequences = stop_sequences
        self.max_tokens = max_tokens
        self.need_logprobs = (
            list(need_logprobs) if need_logprobs is not None else [True] * len(self.uids)
        )
        self._df2_states = (
            list(df2_states) if df2_states is not None else [None] * len(self.uids)
        )
        self.need_final_cache = (
            list(need_final_cache)
            if need_final_cache is not None
            else [True] * len(self.uids)
        )
        self._df2_ctx = None
        self._df2_c1_runner = None

        if self.samplers and len(self.samplers) != len(self.uids):
            raise ValueError("Insufficient number of samplers provided")
        if self.logits_processors and len(self.logits_processors) != len(self.uids):
            raise ValueError("Insufficient number of logits_processors provided")

        self._current_tokens = None
        self._current_logprobs = []
        self._next_tokens = inputs
        self._next_logprobs = []
        self._token_context = [TokenBuffer(t) for t in tokens]
        self._num_tokens = [0] * len(self.uids)
        self._matchers = [ss.matcher() for ss in stop_sequences]
        self._bd = None

        if (
            _SERVER_BATCHDECODE
            and 1 <= len(self.uids) <= 32
            and _is_inkling2_model(self.model)
            and (per_caches is not None or (self.prompt_cache and not any(c.empty() for c in self.prompt_cache)))
        ):
            if per_caches is None:
                lps = [c.left_padding for c in self.prompt_cache if getattr(c, "left_padding", None) is not None]
                if lps:
                    mx.eval(lps)
                per_caches = [
                    [c.extract(i) for c in self.prompt_cache] for i in range(len(self.uids))
                ]
            if _inkling2_bd_caches_ok(self.model, per_caches):
                self._activate_bd(per_caches)
            elif not self.prompt_cache and per_caches:
                self.prompt_cache = _merge_caches(per_caches)
        elif not self.prompt_cache and per_caches:
            self.prompt_cache = _merge_caches(per_caches)

        if self.uids:
            df2_c1_ctx = self._df2_c1_eligible()
            if df2_c1_ctx is not None:
                self._step_init_df2_c1(df2_c1_ctx)
            else:
                self._step()

    def _step_init_df2_c1(self, df2_c1_ctx) -> None:
        """Initial prompt[-1] step for C=1 DFlash 2: captures last-prompt tap into drafter state."""
        from .models.inkling2.mtp import _feats_from_dict

        dr, _tm, tap_keys, tap_set, _ = df2_c1_ctx
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens
        base_sampler = (
            self.fallback_sampler
            if (not self.samplers or self.samplers[0] is None)
            else self.samplers[0]
        )
        y, tap0 = self._bd.step_with_taps(inputs, tap_visits=tap_set, sampler=base_sampler)
        st = self._df2_states[0]
        if st is None:
            st = self._df2_states[0] = dr.new_stream_state(allocate=True)
            st.lookup.append(self.tokens[0])
        f0 = _feats_from_dict(tap0, tap_keys)
        dr.append_ctx_batch([st], [f0])
        self._next_tokens = y.reshape(1).astype(mx.uint32)
        self._next_logprobs = [None]
        mx.async_eval(self._next_tokens, *st.ck, *st.cv)
        mx.eval(inputs)
        in_list = inputs.tolist()
        for sti, ti in zip(self.tokens, in_list):
            sti.append(ti)
        st.lookup.append_one(int(in_list[0]))
        st.lookup.set_prompt_end(len(st.lookup.tokens))

    def _activate_bd(self, per_caches: List[List[Any]]) -> None:
        from .models.inkling2.batchdecode import BatchDecoder

        self._bd = BatchDecoder(self.model, per_caches)
        self.prompt_cache = []

    def _flush_bd(self, *, keep_cascade_split: bool = False) -> None:
        if self._bd is None:
            return
        if self._next_tokens is not None:
            nlp = [lp for lp in self._next_logprobs if lp is not None]
            mx.eval(self._next_tokens, nlp) if nlp else mx.eval(self._next_tokens)
        self._bd.flush(keep_cascade_split=keep_cascade_split)

    def _extract_all_caches(self, *, keep_cascade_split: bool = False) -> List[List[Any]]:
        if not self.uids:
            return []
        if self._bd is not None:
            self._flush_bd(keep_cascade_split=keep_cascade_split)
            return list(self._bd.caches)
        return [self.extract_cache(i) for i in range(len(self.uids))]

    @property
    def cache_nbytes(self) -> int:
        if self._bd is not None:
            return sum(c.nbytes for seq in self._bd.caches for c in seq)
        return sum(c.nbytes for c in self.prompt_cache)

    def __len__(self):
        return len(self.uids)

    def extend(self, batch):
        """Extend this batch with another generation batch."""
        if self._df2_c1_runner is not None:
            nxt = self._df2_c1_runner.handoff_to_ar(need_next_token=True)
            self._df2_c1_runner = None
            self._next_tokens = nxt
            self._next_logprobs = [None]
        if getattr(batch, "_df2_c1_runner", None) is not None:
            nxt_b = batch._df2_c1_runner.handoff_to_ar(need_next_token=True)
            batch._df2_c1_runner = None
            batch._next_tokens = nxt_b
            batch._next_logprobs = [None]
        was_empty = len(self.uids) == 0
        if was_empty:
            self.uids.extend(batch.uids)
            if batch._bd is not None:
                self._bd = batch._bd
                batch._bd = None
                self.prompt_cache = []
            else:
                self.prompt_cache = batch.prompt_cache
                batch.prompt_cache = []
        elif (
            self._bd is not None
            or batch._bd is not None
            or (
                _SERVER_BATCHDECODE
                and 1 <= len(self.uids) + len(batch.uids) <= 32
                and _is_inkling2_model(self.model)
            )
        ):
            if self._bd is not None and batch._bd is not None:
                pend = []
                for gb in (self, batch):
                    if gb._next_tokens is not None:
                        pend.append(gb._next_tokens)
                        pend.extend(lp for lp in gb._next_logprobs if lp is not None)
                if pend:
                    mx.eval(*pend)
            all_caches = self._extract_all_caches(keep_cascade_split=True) + batch._extract_all_caches(keep_cascade_split=True)
            self.uids.extend(batch.uids)
            if self._bd is not None:
                self._bd.release_stage()
            if batch._bd is not None:
                batch._bd.release_stage()
            self._bd = None
            batch._bd = None
            if _inkling2_bd_caches_ok(self.model, all_caches):
                self._activate_bd(all_caches)
            elif all_caches:
                self.prompt_cache = _merge_caches(all_caches)
            else:
                self.prompt_cache = []
        else:
            self.uids.extend(batch.uids)
            self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_sequences.extend(batch.stop_sequences)
        self.need_logprobs.extend(batch.need_logprobs)
        self._df2_states.extend(batch._df2_states)
        self.need_final_cache.extend(batch.need_final_cache)
        if self._current_tokens is None:
            self._current_tokens = batch._current_tokens
            self._current_logprobs = batch._current_logprobs
        elif batch._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, batch._current_tokens]
            )
            self._current_logprobs.extend(batch._current_logprobs)
        if self._next_tokens is None:
            self._next_tokens = batch._next_tokens
            self._next_logprobs = batch._next_logprobs
        elif batch._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
            self._next_logprobs.extend(batch._next_logprobs)
        self._token_context.extend(batch._token_context)
        self._num_tokens.extend(batch._num_tokens)
        self._matchers.extend(batch._matchers)

    def _step(self) -> Tuple[List[int], List[mx.array]]:
        """
        Perform a single generation step.

        Returns:
            Tuple of token list and logprobs list.
        """
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        # D3: When all active sequences hit max_tokens on the token we are about to emit
        # and no caller needs the post-completion cache, skip the wasted max_tokens+1 forward pass.
        if (
            self.uids
            and not any(self.need_final_cache)
            and all(nt + 1 >= mt for nt, mt in zip(self._num_tokens, self.max_tokens))
        ):
            self._next_tokens = None
            self._next_logprobs = [None] * len(self.uids)
            cur_lp = [lp for lp in self._current_logprobs if lp is not None]
            mx.eval(inputs, cur_lp) if cur_lp else mx.eval(inputs)
            inputs = inputs.tolist()
            for sti, ti in zip(self.tokens, inputs):
                sti.append(ti)
            return inputs, self._current_logprobs

        samplers = [
            self.fallback_sampler if s is None else s
            for s in (self.samplers or [None] * len(self.uids))
        ]
        k0 = getattr(samplers[0], "_sampler_key", None) if samplers else None
        same_sampler = (not samplers) or all(
            s is samplers[0] or (k0 is not None and getattr(s, "_sampler_key", None) == k0)
            for s in samplers
        )
        base_sampler = samplers[0] if samplers else self.fallback_sampler
        need_lp = any(self.need_logprobs)

        if (
            self._bd is not None
            and not need_lp
            and not any(self.logits_processors)
            and same_sampler
        ):
            if base_sampler is greedy_sampler:
                sampled_fast = self._bd.step(inputs)
            elif self._bd.can_fast_sample(base_sampler):
                sampled_fast = self._bd.sample_step(inputs, base_sampler)
            else:
                sampled_fast = None
            if sampled_fast is not None:
                self._next_tokens = sampled_fast
                self._next_logprobs = [None] * len(self.uids)
                mx.async_eval(self._next_tokens)
                cur_lp = [lp for lp in self._current_logprobs if lp is not None]
                mx.eval(inputs, cur_lp) if cur_lp else mx.eval(inputs)
                inputs = inputs.tolist()
                for sti, ti in zip(self.tokens, inputs):
                    sti.append(ti)
                return inputs, self._current_logprobs

        # Forward pass + logprobs
        has_lp_proc = any(self.logits_processors)
        if self._bd is not None and not has_lp_proc:
            if same_sampler and base_sampler is greedy_sampler:
                sampled, logprobs = self._bd.logprobs(self._bd._hidden_pair(inputs), argmax=True)
            else:
                logprobs = self._bd.logprobs(self._bd._hidden_pair(inputs))
                if same_sampler:
                    sampled = base_sampler(logprobs)
                else:
                    sampled = mx.concatenate(
                        [s(logprobs[e : e + 1]) for e, s in enumerate(samplers)], axis=0
                    )
            token_context = []
        else:
            if self._bd is not None:
                logits = self._bd.logits(self._bd._hidden_pair(inputs))
            else:
                logits = self.model(inputs[:, None], cache=self.prompt_cache)
                logits = logits[:, -1, :]

            # Logits processors
            token_context = []
            if has_lp_proc:
                processed_logits = []
                for e in range(len(self.uids)):
                    sample_logits = logits[e : e + 1]
                    # Only sequences with processors need a context.
                    processors = self.logits_processors[e] or ()
                    if processors:
                        context = self._token_context[e].update_and_fetch(inputs[e : e + 1])
                        token_context.append(context)
                        for processor in processors:
                            sample_logits = processor(context, sample_logits)
                    processed_logits.append(sample_logits)
                logits = mx.concatenate(processed_logits, axis=0)

            # Normalize the logits
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            # One batched call when the batch shares a sampler.
            if same_sampler:
                sampled = base_sampler(logprobs)
            else:
                sampled = mx.concatenate(
                    [s(logprobs[e : e + 1]) for e, s in enumerate(samplers)], axis=0
                )

        # Assign the next step to member variables and start computing it
        # asynchronously
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs) if need_lp else [None] * len(self.uids)
        nlp = [lp for lp in self._next_logprobs if lp is not None]
        if nlp:
            mx.async_eval(self._next_tokens, nlp, token_context)
        else:
            mx.async_eval(self._next_tokens, token_context)

        # Eval the current tokens and current logprobs. After that also add
        # them to self.tokens so that it always represents the tokens contained
        # in the KV Cache.
        cur_lp = [lp for lp in self._current_logprobs if lp is not None]
        mx.eval(inputs, cur_lp) if cur_lp else mx.eval(inputs)
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs):
            sti.append(ti)
        return inputs, self._current_logprobs

    def extract_cache(self, idx: int) -> List[Any]:
        if self._df2_c1_runner is not None:
            self._df2_c1_runner.rollback_unemitted()
        if self._bd is not None:
            return self._bd.snapshot_cache(idx)
        return [c.extract(idx) for c in self.prompt_cache]

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        if self._df2_c1_runner is not None:
            if not keep:
                self._df2_c1_runner.discard()
                self._df2_c1_runner = None
            else:
                nxt = self._df2_c1_runner.handoff_to_ar(need_next_token=True)
                self._df2_c1_runner = None
                self._next_tokens = nxt
                self._next_logprobs = [None]
        self.uids = [self.uids[idx] for idx in keep]
        if self._bd is not None:
            if keep:
                self._flush_bd(keep_cascade_split=True)
                kept = [self._bd.caches[idx] for idx in keep]
                self._bd.release_stage()
                self._bd = None
                if not _inkling2_bd_caches_ok(self.model, kept):
                    self.prompt_cache = _merge_caches(kept)
                else:
                    self._activate_bd(kept)
            else:
                if any(self.need_final_cache):
                    self._flush_bd()
                    self._bd.release_stage()
                elif self._next_tokens is None:
                    self._bd.release_stage()
                else:
                    # Drop stage without returning live in-flight buffers to the reuse pool.
                    self._bd._stage_main = None
                    self._bd._stage_ref = None
                self._bd = None
                self.prompt_cache = []
        elif not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        self.samplers = [self.samplers[idx] for idx in keep]
        self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_sequences = [self.stop_sequences[idx] for idx in keep]
        self.need_logprobs = [self.need_logprobs[idx] for idx in keep]
        self.need_final_cache = [self.need_final_cache[idx] for idx in keep]

        self._next_tokens = (
            self._next_tokens[keep] if (keep and self._next_tokens is not None) else None
        )
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matchers = [self._matchers[idx] for idx in keep]
        self._df2_states = [self._df2_states[idx] for idx in keep]

    def _df2_c1_eligible(self) -> Optional[Tuple[Any, ...]]:
        if self._bd is None or not self._bd._b1_fast or len(self.uids) != 1:
            return None
        if not _df2_c1_enabled() or not _is_inkling2_model(self.model):
            return None
        if self.need_logprobs[0] or self.logits_processors[0]:
            return None
        if int(self._bd.caches[0][0][0].offset) > _df2_c1_max_ctx():
            return None
        s = self.fallback_sampler if (not self.samplers or self.samplers[0] is None) else self.samplers[0]
        is_greedy = s is greedy_sampler or getattr(s, "_sampler_key", None) == ("greedy",)
        if not is_greedy:
            from .models.inkling2.batchdecode import _fast_sample_enabled, _std_sampler_params

            if (
                not _fast_sample_enabled()
                or _std_sampler_params(s) is None
                or os.environ.get("LOOPLING_DFLASH2_COUPLED_SAMPLE", "1") in ("0", "", "false", "False")
            ):
                return None
        if self._df2_ctx is None:
            self._df2_ctx = _get_df2_batch_ctx(self.model)
            if self._df2_ctx is None:
                return None
        return self._df2_ctx

    def _next_df2_c1(self, df2_ctx) -> List["GenerationBatch.Response"]:
        from .models.inkling2 import dflash2 as DF2

        dr, _tm, tap_keys, tap_set, _ = df2_ctx
        uid = self.uids[0]
        base_sampler = (
            self.fallback_sampler
            if (not self.samplers or self.samplers[0] is None)
            else self.samplers[0]
        )

        # First token (or first token right after C>=2 -> C=1 shrink): emit already-computed `_next_tokens`
        # without launching a redundant AR step, then arm `_DFlash2C1Runner` if >=2 tokens remain.
        if self._df2_c1_runner is None:
            tok_arr = self._next_tokens
            mx.eval(tok_arr)
            tok0 = int(tok_arr.tolist()[0])
            self._num_tokens[0] += 1
            self.tokens[0].append(tok0)
            finish_reason = None
            if self._num_tokens[0] >= self.max_tokens[0]:
                finish_reason = "length"
            if self._matchers[0].advance(tok0):
                finish_reason = "stop"
            if finish_reason is not None:
                pc = None
                if self.need_final_cache[0]:
                    # Commit tok0 into KV so returned prompt_cache covers all_tokens exactly.
                    _ = self._bd.step(tok_arr.reshape(1).astype(mx.uint32))
                    mx.eval(_)
                    self._bd.flush()
                    pc = self._bd.caches[0]
                resp = self.Response(
                    uid=uid,
                    token=tok0,
                    logprobs=None,
                    finish_reason=finish_reason,
                    prompt_cache=pc,
                    all_tokens=self.tokens[0],
                )
                self._next_tokens = None
                self.filter([])
                return [resp]

            rem = self.max_tokens[0] - self._num_tokens[0]
            if rem >= 2:
                st = self._df2_states[0]
                if st is None:
                    st = self._df2_states[0] = dr.new_stream_state(allocate=True)
                self._df2_c1_runner = DF2._DFlash2C1Runner(
                    self._bd,
                    dr,
                    st,
                    tok_arr,
                    tok0,
                    all_tokens=self.tokens[0],
                    tap_keys=tap_keys,
                    tap_set=tap_set,
                    sampler=base_sampler,
                    max_tokens=self.max_tokens[0],
                    num_tokens=self._num_tokens[0],
                    track_cache_rb=bool(self.need_final_cache[0]),
                )
                self._df2_c1_runner.prefetch()
                self._next_tokens = None
                self._next_logprobs = [None]
            else:
                # Only 1 token left: run 1 AR step on tok0 and let next() emit & finish.
                anc = tok_arr.reshape(1).astype(mx.uint32)
                if base_sampler is greedy_sampler:
                    nxt = self._bd.step(anc)
                elif self._bd.can_fast_sample(base_sampler):
                    nxt = self._bd.sample_step(anc, base_sampler)
                else:
                    lp = self._bd.logprobs(self._bd._hidden_pair(anc))
                    nxt = base_sampler(lp).astype(mx.uint32)
                self._next_tokens = nxt
                self._next_logprobs = [None]
                mx.async_eval(self._next_tokens)
            return [
                self.Response(
                    uid=uid,
                    token=tok0,
                    logprobs=None,
                    finish_reason=None,
                    prompt_cache=None,
                    all_tokens=None,
                )
            ]

        # Steady-state DFlash 2 C=1 cycle.
        em = self._df2_c1_runner.step()
        responses = []
        for j, tok in enumerate(em):
            tok_i = int(tok)
            self._num_tokens[0] += 1
            self.tokens[0].append(tok_i)
            finish_reason = None
            if self._num_tokens[0] >= self.max_tokens[0]:
                finish_reason = "length"
            if self._matchers[0].advance(tok_i):
                finish_reason = "stop"
            if finish_reason is not None:
                pc = None
                if self.need_final_cache[0]:
                    self._df2_c1_runner.finalize_exact_cache(
                        emitted_kept=j + 1, total_in_burst=len(em), em_burst=em
                    )
                    pc = self._bd.caches[0]
                else:
                    self._df2_c1_runner.discard()
                self._df2_c1_runner = None
                responses.append(
                    self.Response(
                        uid=uid,
                        token=tok_i,
                        logprobs=None,
                        finish_reason=finish_reason,
                        prompt_cache=pc,
                        all_tokens=self.tokens[0],
                    )
                )
                self._next_tokens = None
                self.filter([])
                return responses
            responses.append(
                self.Response(
                    uid=uid,
                    token=tok_i,
                    logprobs=None,
                    finish_reason=None,
                    prompt_cache=None,
                    all_tokens=None,
                )
            )
        return responses

    def _df2_eligible(self) -> Optional[Tuple[Any, ...]]:
        if self._bd is None or not self._bd._batch_fast or len(self.uids) < 2:
            return None
        if not _is_inkling2_model(self.model):
            return None
        explicit_batch = _df2_batch_enabled()
        auto_batch = _df2_batch_auto_enabled()
        if not explicit_batch and not auto_batch:
            return None
        if any(self.need_logprobs) or any(self.logits_processors):
            return None
        C = len(self.uids)
        if not explicit_batch:
            if C > 4:
                return None
            # Gate automatic C=2..4 DFlash2 on min_ctx >= LOOPLING_DFLASH2_BATCH_MIN_CTX (default 16384).
            st = self._bd._stage_main
            if st is None or not st._full_visits:
                return None
            fv0 = st._full_visits[0]
            max_ctx = max(int(self._bd.caches[c][fv0][0].offset) + int(st.st_lens[c]) for c in range(C))
            if max_ctx < _df2_batch_min_ctx():
                return None
        samplers = [
            self.fallback_sampler if s is None else s
            for s in (self.samplers or [None] * C)
        ]
        all_greedy = all(
            s is greedy_sampler or getattr(s, "_sampler_key", None) == ("greedy",)
            for s in samplers
        )
        if not all_greedy:
            from .models.inkling2.batchdecode import _fast_sample_enabled, _std_sampler_params

            if (
                not _fast_sample_enabled()
                or os.environ.get("LOOPLING_DFLASH2_COUPLED_SAMPLE", "1") in ("0", "", "false", "False")
            ):
                return None
            p0 = _std_sampler_params(samplers[0])
            if p0 is None or not all(_std_sampler_params(s) == p0 for s in samplers[1:]):
                return None
        if self._df2_ctx is None:
            self._df2_ctx = _get_df2_batch_ctx(self.model)
            if self._df2_ctx is None:
                return None
        max_c = self._df2_ctx[4] if explicit_batch else min(4, self._df2_ctx[4])
        if C > max_c:
            return None
        return self._df2_ctx

    def _next_df2_batch(self, df2_ctx) -> List["GenerationBatch.Response"]:
        from .models.inkling2 import dflash2 as DF2
        from .models.inkling2.batchdecode import _std_sampler_params

        dr, _tm, tap_keys, tap_set, _ = df2_ctx
        C = len(self.uids)
        w_env = os.environ.get("LOOPLING_DFLASH2_BATCH_W")
        if w_env:
            W = max(2, min(dr.block, int(w_env), 16 // C))
        elif _df2_batch_enabled():
            W = min(dr.block, DF2._default_batch_w(C))
        else:
            # Phase 3b default for C=2..4 long context: W=4 (C*W <= 16, QPS=4 scalar ALU sweet spot).
            W = min(dr.block, 4, 16 // C)

        base_sampler = (
            self.fallback_sampler
            if (not self.samplers or self.samplers[0] is None)
            else self.samplers[0]
        )
        is_greedy = base_sampler is greedy_sampler or getattr(base_sampler, "_sampler_key", None) == ("greedy",)

        anc_arr = self._next_tokens
        mx.eval(anc_arr)
        anc_list = [int(a) for a in anc_arr.tolist()]

        for c in range(C):
            st = self._df2_states[c]
            if st is None:
                st = self._df2_states[c] = dr.new_stream_state(allocate=True)
            n_toks = len(self.tokens[c])
            n_lp = len(st.lookup.tokens)
            if n_lp < n_toks:
                st.lookup.append(self.tokens[c][n_lp:])
            if len(st.lookup.tokens) == n_toks:
                st.lookup.append_one(anc_list[c])

        if is_greedy:
            blk = dr.draft_batch_chains(self._df2_states, anc_list, W)
            t_cw, kv_news, conv_ws, tap_buf = self._bd.verify_batch_w(blk, W, tap_visits=tap_set)
            mx.eval(t_cw)
            tl = t_cw.tolist()
            bl = blk.tolist()
            keeps_raw = []
            bonus_next = []
            for c in range(C):
                row = bl[c]
                t_row = tl[c]
                k_raw = 1
                while k_raw < W and t_row[k_raw - 1] == row[k_raw]:
                    k_raw += 1
                keeps_raw.append(k_raw)
                bonus_next.append(int(t_row[k_raw - 1]))
        else:
            std_p = _std_sampler_params(base_sampler)
            dt_env = os.environ.get("LOOPLING_DFLASH2_DRAFT_TEMP", "")
            draft_temp = float(dt_env) if dt_env else min(std_p[0], 0.25)
            blk, q_ids_c, q_probs_c = dr.draft_batch_chains_coupled(
                self._df2_states, anc_list, W, temp=draft_temp
            )
            h_out_cw, kv_news, conv_ws, tap_buf = self._bd.verify_batch_w(
                blk, W, tap_visits=tap_set, return_h_out=True
            )
            keeps_raw, bonus_next = DF2._coupled_verify_chains(
                self.model, h_out_cw, blk, q_ids_c, q_probs_c, base_sampler
            )
            bl = blk.tolist()

        keeps_eff = [1] * C
        next_anc = [0] * C
        emitted: List[List[Tuple[int, Optional[str]]]] = [[] for _ in range(C)]
        keep: List[int] = []
        any_finished = False
        need_any_fin_cache = False

        for c in range(C):
            row = bl[c]
            k_raw = keeps_raw[c]
            finished_c = False
            for j in range(k_raw):
                tok = int(row[j])
                self._num_tokens[c] += 1
                finish_reason = None
                if self._num_tokens[c] >= self.max_tokens[c]:
                    finish_reason = "length"
                if self._matchers[c].advance(tok):
                    finish_reason = "stop"
                self.tokens[c].append(tok)
                emitted[c].append((tok, finish_reason))
                if j > 0:
                    self._df2_states[c].lookup.append_one(tok)
                if finish_reason is not None:
                    keeps_eff[c] = j + 1
                    finished_c = True
                    any_finished = True
                    if self.need_final_cache[c]:
                        need_any_fin_cache = True
                    break
            if not finished_c:
                keeps_eff[c] = k_raw
                next_anc[c] = int(bonus_next[c])
                keep.append(c)

        if keep or need_any_fin_cache:
            self._bd.commit_batch_w(kv_news, conv_ws, keeps_eff, W)
        if any_finished and (keep or need_any_fin_cache):
            self._bd.flush(keep_cascade_split=not need_any_fin_cache)

        responses = []
        for c in range(C):
            for tok, finish_reason in emitted[c]:
                if finish_reason is not None:
                    pc = self._bd.caches[c] if self.need_final_cache[c] else None
                    responses.append(
                        self.Response(
                            uid=self.uids[c],
                            token=tok,
                            logprobs=None,
                            finish_reason=finish_reason,
                            prompt_cache=pc,
                            all_tokens=self.tokens[c],
                        )
                    )
                else:
                    responses.append(
                        self.Response(
                            uid=self.uids[c],
                            token=tok,
                            logprobs=None,
                            finish_reason=None,
                            prompt_cache=None,
                            all_tokens=None,
                        )
                    )

        if keep:
            fw = mx.concatenate([tap_buf[k].reshape(C * W, 1, dr.H) for k in tap_keys], axis=1)
            k_states = [self._df2_states[c] for c in keep]
            k_feats = [fw[c * W : c * W + keeps_eff[c]][None] for c in keep]
            dr.append_ctx_batch(k_states, k_feats)
            mx.async_eval(*[x for st in k_states for x in st.ck + st.cv])

        self._next_tokens = mx.array(next_anc, dtype=mx.uint32)
        self._next_logprobs = [None] * C

        if len(keep) < C:
            if not keep and self._bd is not None:
                self._bd.release_stage()
                self._bd = None
                self.prompt_cache = []
            self.filter(keep)

        return responses

    def next(self) -> List[Response]:
        """
        Generate the next batch of tokens.

        Returns:
            List of Response objects for each sequence in the batch.
        """
        if not self.uids:
            return []

        df2_c1_ctx = self._df2_c1_eligible()
        if df2_c1_ctx is not None:
            return self._next_df2_c1(df2_c1_ctx)
        if self._df2_c1_runner is not None:
            # E.g. context crossed `_df2_c1_max_ctx()` mid-generation: seamless handoff to AR.
            nxt = self._df2_c1_runner.handoff_to_ar(need_next_token=True)
            self._df2_c1_runner = None
            self._next_tokens = nxt
            self._next_logprobs = [None]

        df2_ctx = self._df2_eligible()
        if df2_ctx is not None:
            return self._next_df2_batch(df2_ctx)

        tokens, logprobs = self._step()

        finish_reasons = [None] * len(self.uids)
        keep = []
        any_finished = False
        for i in range(len(self.uids)):
            finish_reason = None

            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if self._matchers[i].advance(tokens[i]):
                finish_reason = "stop"

            finish_reasons[i] = finish_reason
            if finish_reason is not None:
                any_finished = True
            else:
                keep.append(i)

        need_any_fin_cache = any(
            self.need_final_cache[i]
            for i, fr in enumerate(finish_reasons)
            if fr is not None
        )
        if any_finished and self._bd is not None and (keep or need_any_fin_cache):
            self._flush_bd()

        responses = []
        for i, finish_reason in enumerate(finish_reasons):
            if finish_reason is not None:
                if self.need_final_cache[i]:
                    pc = self._bd.caches[i] if self._bd is not None else self.extract_cache(i)
                else:
                    pc = None
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=finish_reason,
                        prompt_cache=pc,
                        all_tokens=self.tokens[i],
                    )
                )
            else:
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=None,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Sampler,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_sequences=[],
            need_logprobs=[],
            need_final_cache=[],
        )


class BatchGenerator:
    """
    A batch generator implements continuous batching.

    This class provides automatic management of prompt processing and generation
    batches, handling the transition between the two.

    It also allows for segmented prompt processing which guarantees that the
    generator will stop at these boundaries when processing an input.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Sampler] = None,
        logits_processors: Optional[List[LogitsProcessor]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 4096,
        max_kv_size: Optional[int] = None,
        stream=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.sampler = sampler or greedy_sampler
        self.logits_processors = logits_processors or []
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.max_kv_size = max_kv_size

        self._stream = stream or generation_stream

        self._default_stop_sequences = StopSequences(
            stop_tokens if stop_tokens else None,
        )
        self._uid_count = 0
        self._prompt_batch = PromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=prefill_step_size,
        )
        self._generation_batch = GenerationBatch.empty(self.model, self.sampler)
        self._unprocessed_sequences = deque()
        self._currently_processing = []
        self._prefill_turn = False
        self._decode_burst_steps = 0
        self._decode_burst_start = 0.0

        self._counters = BatchCounters()

        self._old_wired_limit = maybe_set_recommended_wired_limit()

    @property
    def stream(self):
        return self._stream

    def close(self):
        inner = getattr(self.model, "model", None)
        tm = getattr(inner, "language_model", None) if inner is not None else None
        if tm is not None:
            pool = getattr(tm, "_bd_stage_pool", None)
            if pool:
                pool.clear()
        if self._old_wired_limit is not None:
            mx.synchronize(self._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def __del__(self):
        self.close()

    @contextlib.contextmanager
    def stats(self):
        """The work this generator does while the block runs.

        Readings are taken on entry and exit and subtracted, so nested windows
        each report their own interval.
        """
        stats = BatchStats()
        start = BatchCountersSnapshot(**asdict(self._counters))
        try:
            yield stats
        finally:
            end = BatchCountersSnapshot(**asdict(self._counters))
            stats += BatchCountersSnapshot.between(start, end)

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Sampler]] = None,
        logits_processors: Optional[List[List[LogitsProcessor]]] = None,
        stop_sequences: Optional[List[StopSequences]] = None,
        need_logprobs: Optional[List[bool]] = None,
        need_final_cache: Optional[List[bool]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_sequences,
            need_logprobs=need_logprobs,
            need_final_cache=need_final_cache,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Sampler]] = None,
        logits_processors: Optional[List[List[LogitsProcessor]]] = None,
        stop_sequences: Optional[List[StopSequences]] = None,
        need_logprobs: Optional[List[bool]] = None,
        need_final_cache: Optional[List[bool]] = None,
    ):
        num_segments = len(segments)

        max_tokens = max_tokens or [self.max_tokens] * num_segments
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = _normalize_samplers(samplers, n=num_segments, fallback=self.sampler)
        logits_processors = _normalize_logits_processors(
            logits_processors or [self.logits_processors] * num_segments,
            n=num_segments,
        )
        stop_sequences = stop_sequences or (
            [self._default_stop_sequences] * num_segments
        )
        need_logprobs = (
            list(need_logprobs) if need_logprobs is not None else [True] * num_segments
        )
        need_final_cache = (
            list(need_final_cache)
            if need_final_cache is not None
            else [True] * num_segments
        )

        caches = caches or [None] * num_segments

        # Validate before touching any state: ``zip`` pairs up to the shortest.
        opts = {
            "max_tokens": max_tokens,
            "caches": caches,
            "all_tokens": all_tokens,
            "samplers": samplers,
            "logits_processors": logits_processors,
            "stop_sequences": stop_sequences,
            "need_logprobs": need_logprobs,
            "need_final_cache": need_final_cache,
        }
        for k, v in opts.items():
            if len(v) != num_segments:
                raise ValueError(
                    f"Option {k} must have one entry per segment: expected "
                    f"{num_segments}, got {len(v)}."
                )

        uids = list(range(self._uid_count, self._uid_count + num_segments))
        pending = []
        for uid, seq, m, c, at, s, lp, sm, nl, nfc in zip(
            uids,
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_sequences,
            need_logprobs,
            need_final_cache,
        ):
            seq = [segment for segment in seq if segment]
            if not seq:
                raise ValueError(f"Sequence {uid} has an empty prompt.")
            if m <= 0:
                raise ValueError(f"Sequence {uid}'s max_tokens must be > 0.")
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            if c is None:
                c = make_prompt_cache(self.model, self.max_kv_size)
            pending.append((uid, seq, m, c, at, s, lp, sm, nl, nfc))

        self._unprocessed_sequences.extend(pending)
        self._uid_count += num_segments

        return uids

    def _find_uids(self, uids):
        uids = set(uids)
        results = {}
        for i, uid_i in enumerate(self._generation_batch.uids):
            if uid_i in uids:
                results[uid_i] = (2, i)
        for i, uid_i in enumerate(self._prompt_batch.uids):
            if uid_i in uids:
                results[uid_i] = (1, i)
        for i, seq in enumerate(self._unprocessed_sequences):
            if seq[0] in uids:
                results[seq[0]] = (0, i)
        return results

    def extract_cache(self, uids):
        results = {}
        for uid, (stage, idx) in self._find_uids(uids).items():
            if stage == 0:
                results[uid] = self._unprocessed_sequences[idx][3:5]
            elif stage == 1:
                results[uid] = (
                    self._prompt_batch.extract_cache(idx),
                    self._prompt_batch.tokens[idx],
                )
            else:
                results[uid] = (
                    self._generation_batch.extract_cache(idx),
                    self._generation_batch.tokens[idx],
                )
        return results

    def remove(self, uids, return_prompt_caches=False):
        caches = {}
        if return_prompt_caches:
            caches = self.extract_cache(uids)

        keep = (
            set(range(len(self._unprocessed_sequences))),
            set(range(len(self._prompt_batch))),
            set(range(len(self._generation_batch))),
        )
        for stage, idx in self._find_uids(uids).values():
            keep[stage].remove(idx)

        if len(keep[0]) < len(self._unprocessed_sequences):
            self._unprocessed_sequences = deque(
                x for i, x in enumerate(self._unprocessed_sequences) if i in keep[0]
            )
        if len(keep[1]) < len(self._prompt_batch):
            self._prompt_batch.filter(sorted(keep[1]))
            self._currently_processing = [
                x for i, x in enumerate(self._currently_processing) if i in keep[1]
            ]
        if len(keep[2]) < len(self._generation_batch):
            self._generation_batch.filter(sorted(keep[2]))

        return caches

    @property
    def prompt_cache_nbytes(self):
        total = sum(c.nbytes for p in self._unprocessed_sequences for c in p[3])
        total += sum(c.nbytes for c in self._prompt_batch.prompt_cache)
        total += self._generation_batch.cache_nbytes
        return total

    def _make_batch_from(self, sequences):
        uids = []
        caches = []
        tokens = []
        samplers = []
        logits_processors = []
        max_tokens = []
        stop_sequences = []
        need_logprobs = []
        need_final_cache = []
        for sequence in sequences:
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            stop_sequences.append(sequence[7])
            need_logprobs.append(sequence[8])
            need_final_cache.append(sequence[9])
            self._currently_processing.append(
                [sequence[1], 0, sum(len(s) for s in sequence[1])]
            )

        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=logits_processors,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            need_logprobs=need_logprobs,
            need_final_cache=need_final_cache,
        )

    def _make_batch(self, n: int):
        sequences = [self._unprocessed_sequences.popleft() for _ in range(n)]
        return self._make_batch_from(sequences)

    @staticmethod
    def _seq_rem_tokens(sequence) -> int:
        return sum(len(s) for s in sequence[1])

    def _prereserve_solo_kv(self, total_new_tokens: int) -> None:
        """Pre-extend BatchKVCache once for solo long prefills to avoid 2x doubling concats."""
        if total_new_tokens <= 4096 or len(self._prompt_batch) != 1:
            return
        for layer in self._prompt_batch.prompt_cache:
            subs = getattr(layer, "caches", None)
            targets = subs if isinstance(subs, (list, tuple)) else (layer,)
            for sub in targets:
                if (
                    hasattr(sub, "keys")
                    and hasattr(sub, "values")
                    and hasattr(sub, "_idx")
                    and hasattr(sub, "step")
                    and not hasattr(sub, "max_size")
                    and sub.keys is not None
                ):
                    prev = int(sub._idx)
                    need = prev + total_new_tokens
                    step = int(sub.step)
                    target = ((need + step - 1) // step) * step
                    cur = int(sub.keys.shape[2])
                    if target > cur:
                        if prev % step != 0:
                            sub.keys = sub.keys[..., :prev, :]
                            sub.values = sub.values[..., :prev, :]
                            cur = prev
                        add = target - cur
                        if add > 0:
                            sub.keys = mx.concatenate(
                                [sub.keys, mx.zeros((1, sub.keys.shape[1], add, sub.keys.shape[3]), sub.keys.dtype)],
                                axis=2,
                            )
                            sub.values = mx.concatenate(
                                [sub.values, mx.zeros((1, sub.values.shape[1], add, sub.values.shape[3]), sub.values.dtype)],
                                axis=2,
                            )

    def _next(self):
        generation_responses = []
        prompt_responses = []

        # Generate tokens unless this tick is explicitly flushing a deferred prefill chunk
        prefill_turn = self._prefill_turn
        self._prefill_turn = False
        if len(self._generation_batch) > 0 and not prefill_turn:
            if self._decode_burst_steps == 0:
                self._decode_burst_start = time.perf_counter()
            tic = time.perf_counter()
            generation_responses = self._generation_batch.next()
            self._counters.decode_time += time.perf_counter() - tic
            self._counters.generation_tokens += len(generation_responses)
            self._counters.generation_steps += 1
            self._decode_burst_steps += 1
            if self._counters.generation_steps % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) == 0:
            self._decode_burst_steps = 0
            self._decode_burst_start = 0.0

        # Exit early because we already have our hands full with decoding
        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        # Split finished prompt sequences to generation first so prompt slots open immediately
        def _split_ready():
            keep = []
            split = []
            for i, seq in enumerate(self._currently_processing):
                segments = seq[0]
                if len(segments) == 1 and len(segments[0]) == 1:
                    split.append(i)
                else:
                    keep.append(i)
            if not split:
                return
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            gen_batch = self._prompt_batch.split(split).generate(last_inputs)
            for i, p in enumerate(progress):
                prompt_responses.append(
                    PromptProcessingBatch.Response(
                        gen_batch.uids[i],
                        p,
                        True,
                        True,
                    )
                )
            self._generation_batch.extend(gen_batch)

        _split_ready()

        # Direct-admit already-prefilled 1-token sequences (e.g. prefix-cache hits) anywhere
        # in the queue, even while a long prefill is active in _prompt_batch.
        def _direct_admit_ready():
            if not self._unprocessed_sequences or len(self._generation_batch) >= self.completion_batch_size:
                return
            cap_ready = self.completion_batch_size - len(self._generation_batch)
            ready_seqs = []
            kept_unproc = deque()
            while self._unprocessed_sequences:
                seq = self._unprocessed_sequences.popleft()
                if (
                    len(ready_seqs) < cap_ready
                    and len(seq[1]) == 1
                    and len(seq[1][0]) == 1
                ):
                    ready_seqs.append(seq)
                else:
                    kept_unproc.append(seq)
            self._unprocessed_sequences = kept_unproc
            if ready_seqs:
                r_uids = [s[0] for s in ready_seqs]
                r_last = mx.array([s[1][0][0] for s in ready_seqs])
                r_caches = [s[3] for s in ready_seqs]
                r_toks = [s[4] for s in ready_seqs]
                r_samps = [s[5] for s in ready_seqs]
                r_lps = [s[6] for s in ready_seqs]
                r_max = [s[2] for s in ready_seqs]
                r_stops = [s[7] for s in ready_seqs]
                r_nlp = [s[8] for s in ready_seqs]
                r_nfc = [s[9] for s in ready_seqs]
                gen_batch = GenerationBatch(
                    self.model,
                    r_uids,
                    r_last,
                    [],
                    r_toks,
                    r_samps,
                    self.sampler,
                    r_lps,
                    r_stops,
                    r_max,
                    r_nlp,
                    per_caches=r_caches,
                    need_final_cache=r_nfc,
                )
                for uid in r_uids:
                    prompt_responses.append(
                        PromptProcessingBatch.Response(uid, (1, 1), True, True)
                    )
                self._generation_batch.extend(gen_batch)

        _direct_admit_ready()

        # If we are in a decode burst and still have burst budget left, keep streaming decode
        # tokens smoothly before taking a prefill turn (unless _generation_batch just drained).
        if generation_responses and (self._currently_processing or self._unprocessed_sequences):
            burst_elapsed = time.perf_counter() - self._decode_burst_start
            burst_done = (
                self._decode_burst_steps >= _DECODE_BURST_STEPS
                or burst_elapsed >= _DECODE_BURST_S
            )
            if not burst_done:
                return prompt_responses, generation_responses
            self._prefill_turn = True
            self._decode_burst_steps = 0
            self._decode_burst_start = 0.0
            return prompt_responses, generation_responses

        # Fast-lane short uncached prompts (<= _SHORT_PREFILL_THRESH) when a long prompt is
        # already mid-prefill in _prompt_batch, avoiding left/right padding contamination and HoL blocking.
        if (
            len(self._prompt_batch) > 0
            and self._unprocessed_sequences
            and len(self._generation_batch) < self.completion_batch_size
        ):
            pb_max_rem = max(s[2] - s[1] for s in self._currently_processing)
            pb_max_cached = max(len(t) for t in self._prompt_batch.tokens)
            has_long_pb = pb_max_rem > _SHORT_PREFILL_THRESH or pb_max_cached >= 1024
            if has_long_pb and any(
                self._seq_rem_tokens(s) <= _SHORT_PREFILL_THRESH for s in self._unprocessed_sequences
            ):
                cap_short = min(
                    self.prefill_batch_size,
                    self.completion_batch_size - len(self._generation_batch),
                )
                short_seqs = []
                rem_unproc = deque()
                ref_rem = None
                ref_cached = None
                while self._unprocessed_sequences:
                    s = self._unprocessed_sequences.popleft()
                    s_rem = self._seq_rem_tokens(s)
                    s_cached = len(s[4])
                    if (
                        len(short_seqs) < cap_short
                        and s_rem <= _SHORT_PREFILL_THRESH
                        and (
                            ref_rem is None
                            or (abs(s_rem - ref_rem) <= 256 and abs(s_cached - ref_cached) <= 256)
                        )
                    ):
                        if ref_rem is None:
                            ref_rem = s_rem
                            ref_cached = s_cached
                        short_seqs.append(s)
                    else:
                        rem_unproc.append(s)
                self._unprocessed_sequences = rem_unproc
                if short_seqs:
                    saved_pb = self._prompt_batch
                    saved_cp = self._currently_processing
                    self._currently_processing = []
                    try:
                        self._prompt_batch = self._make_batch_from(short_seqs)
                        while True:
                            _split_ready()
                            if not self._currently_processing:
                                break
                            s_prompts = []
                            for i, seq in enumerate(self._currently_processing):
                                resp = PromptProcessingBatch.Response(
                                    self._prompt_batch.uids[i], 0, False, False
                                )
                                segments = seq[0]
                                take = len(segments[0]) if len(segments) > 1 else max(1, len(segments[0]) - 1)
                                take = min(take, self.prefill_step_size)
                                s_prompts.append(segments[0][:take])
                                segments[0] = segments[0][take:]
                                if len(segments[0]) == 0:
                                    segments.pop(0)
                                    resp.end_of_segment = True
                                seq[1] += len(s_prompts[-1])
                                resp.progress = (seq[1], seq[2])
                                prompt_responses.append(resp)
                            if s_prompts:
                                self._counters.prompt_tokens += sum(len(p) for p in s_prompts)
                                tic = time.perf_counter()
                                self._prompt_batch.prompt(s_prompts)
                                self._counters.prompt_time += time.perf_counter() - tic
                    finally:
                        self._prompt_batch = saved_pb
                        self._currently_processing = saved_cp
                    return prompt_responses, generation_responses

        # Admit pending sequences into newly freed prompt slots (without mixing incompatible offsets/lengths).
        cap_add = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        if cap_add > 0:
            admitted = []
            if len(self._prompt_batch) == 0:
                ref_seq = self._unprocessed_sequences[0]
                ref_rem = self._seq_rem_tokens(ref_seq)
                ref_cached = len(ref_seq[4])
                if ref_rem > _SHORT_PREFILL_THRESH:
                    admitted.append(self._unprocessed_sequences.popleft())
                else:
                    kept = deque()
                    while self._unprocessed_sequences:
                        s = self._unprocessed_sequences.popleft()
                        s_rem = self._seq_rem_tokens(s)
                        s_cached = len(s[4])
                        if (
                            len(admitted) < cap_add
                            and s_rem <= _SHORT_PREFILL_THRESH
                            and abs(s_rem - ref_rem) <= 256
                            and abs(s_cached - ref_cached) <= 256
                        ):
                            admitted.append(s)
                        else:
                            kept.append(s)
                    self._unprocessed_sequences = kept
            else:
                pb_rem0 = self._currently_processing[0][2] - self._currently_processing[0][1]
                pb_cached0 = len(self._prompt_batch.tokens[0])
                kept = deque()
                while self._unprocessed_sequences:
                    s = self._unprocessed_sequences.popleft()
                    s_rem = self._seq_rem_tokens(s)
                    s_cached = len(s[4])
                    if (
                        len(admitted) < cap_add
                        and abs(s_rem - pb_rem0) <= 2048
                        and abs(s_cached - pb_cached0) < 1024
                    ):
                        admitted.append(s)
                    else:
                        kept.append(s)
                self._unprocessed_sequences = kept
            if admitted:
                self._prompt_batch.extend(self._make_batch_from(admitted))
                _split_ready()

        # Pre-reserve BatchKVCache once for solo long prefills to skip 2x doubling copies.
        if len(self._prompt_batch) == 1 and self._currently_processing:
            total_rem = max(0, self._currently_processing[0][2] - self._currently_processing[0][1] - 1)
            self._prereserve_solo_kv(total_rem)

        # Extract the next prompts input (preserve fixed self.prefill_step_size for BF16 bit-exactness)
        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            n = min(len(segments[0]), self.prefill_step_size)
            prompts.append(segments[0][:n])
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)

        # Process the prompts
        if prompts:
            self._counters.prompt_tokens += sum(len(p) for p in prompts)
            tic = time.perf_counter()
            self._prompt_batch.prompt(prompts)
            self._counters.prompt_time += time.perf_counter() - tic

        return prompt_responses, generation_responses

    def next(self):
        """
        Get the next batch of responses.

        Returns:
            Tuple of prompt processing responses and generation responses.
        """
        with mx.stream(self._stream):
            return self._next()

    def next_generated(self):
        """
        Return only generated tokens ignoring batch generation responses.

        Returns:
            List of GenerationBatch.Response objects
        """
        with mx.stream(self._stream):
            while True:
                prompt_responses, generation_responses = self._next()
                if not generation_responses and prompt_responses:
                    continue
                return generation_responses


@dataclass
class BatchResponse:
    """
    A data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        caches: Optional prompt caches for each sequence.
        token_ids (Optional[List[List[int]]]): The generated token IDs for each
            prompt. Only present when ``return_token_ids=True``.
        logprobs (Optional[List[List[float]]]): The per-token log-probabilities
            of the sampled tokens for each prompt. Only present when
            ``return_logprobs=True``.
    """

    texts: List[str]
    stats: BatchStats
    caches: Optional[List[List[Any]]]
    token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[List[float]]] = None


def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    return_token_ids: bool = False,
    return_logprobs: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompts (List[List[int]]): The input prompts.
       prompt_caches (List[List[Any]], optional): Pre-computed prompt-caches
          for each input prompt. Note, unlike ``generate_step``, the caches
          won't be updated in-place.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       return_prompt_caches (bool): Return the prompt caches in the batch
          responses. Default: ``False``.
       return_token_ids (bool): Return the generated token IDs in the batch
          responses. Default: ``False``.
       return_logprobs (bool): Return the per-token log-probability of the
          sampled token for each generated token. Useful for reinforcement
          learning (e.g. RLOO, PPO) where behavior log-probabilities are needed
          for importance weighting. Default: ``False``.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(
        prompts,
        max_tokens,
        caches=prompt_caches,
        need_logprobs=[return_logprobs] * len(prompts),
        need_final_cache=[return_prompt_caches] * len(prompts),
    )
    results = {uid: [] for uid in uids}
    logprob_results = {uid: [] for uid in uids} if return_logprobs else None
    prompt_caches = {}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    if return_prompt_caches:
                        prompt_caches[r.uid] = r.prompt_cache
                    if verbose:
                        fin += 1
                        print(
                            f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                            end="\r",
                        )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
                    if return_logprobs:
                        logprob_results[r.uid].append(r.logprobs[r.token].item())
    gen.close()
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    caches = [prompt_caches[uid] for uid in uids] if return_prompt_caches else None
    token_ids = [results[uid] for uid in uids] if return_token_ids else None
    logprobs = [logprob_results[uid] for uid in uids] if return_logprobs else None
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats, caches, token_ids, logprobs)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = args.trust_remote_code

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer, config = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
        model_config={"quantize_activations": args.quantize_activations},
        return_config=True,
        trust_remote_code=args.trust_remote_code,
    )
    for eos_token in args.extra_eos_token:
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    use_chat_template = not args.ignore_chat_template and tokenizer.has_chat_template
    if use_chat_template:
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        has_prefill = args.prefill_response is not None
        if has_prefill:
            messages.append({"role": "assistant", "content": args.prefill_response})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **template_kwargs,
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=has_prefill,
                add_generation_prompt=not has_prefill,
            )
            prompt = prompt[test_prompt.index("<query>") :]

    if args.image:
        processor = load_processor(config)
        if processor is None:
            raise ValueError(f"The model {model_path} does not take images.")
        images = [load_image(processor, f) for f in args.image]
        prompt, media_inputs = processor(prompt, images=images)
    else:
        media_inputs = {}
    prompt = tokenizer.encode(prompt, add_special_tokens=not use_chat_template)

    if media_inputs:
        input_embeddings = model.embed_inputs(mx.array(prompt), **media_inputs)
        # The prefill runs on another stream: compute the media rows first.
        mx.eval(input_embeddings)
    else:
        input_embeddings = None

    if args.draft_model is not None:
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    else:
        draft_model = None
    sampler = make_sampler(
        args.temp,
        args.top_p,
        args.min_p,
        args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prefill_step_size=args.prefill_step_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        draft_model=draft_model,
        num_draft_tokens=args.num_draft_tokens,
        input_embeddings=input_embeddings,
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.generate...` directly is deprecated."
        " Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead."
    )
    main()
