# Copyright © 2026 Apple Inc.

from typing import Any, Callable
import os

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import shard_linear

from mlx_lm.layers.common import splice
from mlx_lm.layers.distributed import PipelineMixin, gather_last_axis
from mlx_lm.models.cache import ArraysCache, CacheList, Int8KVCache, KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import checkpoint, fastdecode, kernels, loopcd
from mlx_lm.models.inkling2.config import ModelArgs, TextArgs
from mlx_lm.models.inkling2.layers import (
    BlockAttentionResidual,
    DecoderLayer,
    FullDecoderLayer,
    MessageInput,
    ShortConvolution,
    SlidingDecoderLayer,
    _FAST_ON,
    _use_kernels,
    unweighted_rms_norm,
)
from mlx_lm.models.inkling2.rows import head as RH, qmv as RQ, rowlocal as RL
from mlx_lm.models.inkling2.vision import VisionModel
from mlx_lm.models.muse_spark.model import shard_experts_inplace

__all__ = [
    "Inkling2Model",
    "Model",
    "TextModel",
]


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # Not in embed_tokens as in transformers, so that it can be quantized.
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            (
                SlidingDecoderLayer(args, i)
                if i in args.local_layer_ids
                else FullDecoderLayer(args, i)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.final_input = BlockAttentionResidual(args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.tap_visits = None  # set of visit indices to capture (LOOPLING_MTP), else off
        self.tap_buf = {}
        self.loopcd = loopcd.from_env()
        if args.loop_config is not None:
            self.message_input = MessageInput(args.hidden_size, len(args.loop_core))

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))

    @property
    def layer_visits(self) -> list[DecoderLayer]:
        if self.args.loop_config is None:
            return self.pipeline_layers
        core = self.args.loop_core
        passes = self.args.loop_config["num_passes"]
        layers = self.layers
        return (
            layers[: core.start]
            + layers[core.start : core.stop] * passes
            + layers[core.stop :]
        )

    def pipeline(self, group, split=None) -> None:
        if self.args.loop_config is not None:
            raise ValueError("Pipeline parallel does not support looped models.")
        super().pipeline(group, split)

    def _run_loop(self, hidden_states, cache, conv_mask):
        core = self.args.loop_core
        caches = iter(cache)

        tap = getattr(self, "tap_visits", None)  # LOOPLING_MTP: visit-input capture
        self._visit = 0

        fast_b1 = (
            _FAST_ON
            and tap is None
            and self.pipeline_rank == 0
            and self.pipeline_size == 1
            and hidden_states.dtype == mx.bfloat16
            and hidden_states.shape == (1, 1, 2304)
            and _use_kernels()
            and cache[0] is not None
            and fastdecode.eligible(hidden_states, cache[0], conv_mask)
            and fastdecode._fused_b1_ok(self.layers[0], hidden_states, cache[0][1])
        )
        use_batch_msg = fast_b1 and getattr(self, "message_input", None) is not None

        def run(
            layers,
            hidden_states,
            block_states,
            bank=None,
            cache_iter=None,
            record_tap=True,
        ):
            it = caches if cache_iter is None else cache_iter
            msgs = None
            if bank is not None and use_batch_msg:
                msgs = kernels.depth_attention_batch(
                    bank[0], bank[1], self.message_input.query, scale=bank[1].shape[-1] ** -0.5
                )
            for i, layer in enumerate(layers):
                if tap is not None and record_tap:
                    if self._visit in tap:
                        self.tap_buf[self._visit] = hidden_states
                    self._visit += 1
                messages = None if bank is None else (msgs[i] if msgs is not None else self.message_input(bank, i))
                hidden_states, block_states = layer(
                    hidden_states,
                    block_states,
                    cache=next(it),
                    conv_mask=conv_mask,
                    messages=messages,
                )
            return hidden_states, block_states

        num_passes = self.args.loop_config["num_passes"]
        cfg = self.loopcd
        use_cd = cfg.enabled_for(num_passes)
        ref_pass = cfg.resolved_ref_pass(num_passes) if use_cd else -1
        ref_state = None

        prelude = run(self.layers[: core.start], hidden_states, [] if fast_b1 else None)
        hidden_states, block_states = prelude
        pre_n = len(prelude[1]) if isinstance(prelude[1], list) else prelude[1].shape[0]
        for loop_pass in range(num_passes):
            bank = None
            if loop_pass > 0:
                if isinstance(block_states, list):
                    bank = (
                        mx.stack(block_states[pre_n + 1 :], axis=0),
                        unweighted_rms_norm(hidden_states.astype(mx.float32)).reshape(1, 2304),
                    )
                else:
                    bank = (
                        block_states[pre_n + 1 :],
                        unweighted_rms_norm(hidden_states.astype(mx.float32)),
                    )
                hidden_states, block_states = prelude
            hidden_states, block_states = run(
                self.layers[core.start : core.stop], hidden_states, block_states, bank
            )
            if loop_pass == ref_pass:
                keep_bs = cfg.blend_bs or cfg.mode == "logits"
                ref_state = (
                    hidden_states,
                    (list(block_states) if isinstance(block_states, list) else block_states) if keep_bs else None,
                )

        if use_cd and cfg.mode == "hidden":
            hidden_states, block_states = loopcd.apply_hidden(
                hidden_states, block_states, ref_state[0], ref_state[1], pre_n, cfg
            )

        coda_layers = self.layers[core.stop :]
        ref_coda = (
            loopcd.get_or_init_ref_coda(cache, coda_layers, self.args.sliding_window_size)
            if (use_cd and cfg.mode == "logits")
            else None
        )

        out = run(coda_layers, hidden_states, block_states)
        if tap is not None:
            self.tap_buf["o"] = out[0]  # output of the last visit (pre final_input)

        ref_out = None
        if ref_coda is not None:
            ref_out = run(
                coda_layers,
                ref_state[0],
                ref_state[1],
                cache_iter=iter(ref_coda),
                record_tap=False,
            )
            if not fast_b1:
                loopcd.bind_ref_coda_deps(cache, ref_coda)

        return out[0], out[1], ref_out

    def _finalize(
        self,
        hidden_states: mx.array,
        block_states: Any,
        dtype: Any,
        *,
        record_tap: bool = True,
    ) -> mx.array:
        tap_on = record_tap and getattr(self, "tap_visits", None) is not None
        if (
            isinstance(block_states, list)
            and block_states
            and getattr(self, "tap_visits", None) is None
            and hidden_states.dtype == mx.bfloat16
            and hidden_states.shape == (1, 1, 2304)
            and self.norm.weight.dtype == mx.bfloat16
            and self.final_input.query.dtype == mx.float32
        ):
            return RL.k1_pre(
                hidden_states.reshape(1, 2304),
                block_states,
                None,
                self.final_input.query,
                [self.norm.weight],
                sliding=True,
                eps_ln=self.norm.eps,
            )["q_in"].reshape(1, 1, 2304)
        if isinstance(block_states, list):
            block_states = None if not block_states else mx.stack(block_states, axis=0)[:, :, None, :]
        normed_states = unweighted_rms_norm(hidden_states.astype(mx.float32))
        hidden_states = self.final_input(block_states, normed_states)[..., 0, :]
        if tap_on:
            self.tap_buf["f"] = hidden_states.astype(dtype)
        hidden_states = self.norm(hidden_states).astype(dtype)
        if tap_on:
            self.tap_buf["F"] = hidden_states
        return hidden_states

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        *,
        return_ref: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array | None]:
        if input_embeddings is None:
            input_embeddings = self.embed(inputs)
        hidden_states = input_embeddings
        dtype = hidden_states.dtype

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(self.layer_visits)
        conv_mask = None
        if cache[0] is not None:
            conv_mask = cache[0][1].make_mask(hidden_states.shape[1])

        rank, size = self.pipeline_rank, self.pipeline_size
        block_states = None
        if rank < size - 1:
            num_blocks = -(-self.start_idx // self.args.n_layers_per_attn_res_block)
            states = mx.zeros((num_blocks + 1, *hidden_states.shape), mx.float32)
            states = mx.distributed.recv_like(states, rank + 1)
            hidden_states, block_states = states[0].astype(dtype), states[1:]

        ref_out = None
        if self.args.loop_config is None:
            for layer, c in zip(layers, cache):
                hidden_states, block_states = layer(
                    hidden_states, block_states, cache=c, conv_mask=conv_mask
                )
        else:
            hidden_states, block_states, ref_out = self._run_loop(
                hidden_states, cache, conv_mask
            )

        ref_hidden = None
        if rank != 0:
            states = mx.concatenate(
                [hidden_states.astype(mx.float32)[None], block_states]
            )
            states = mx.distributed.send(states, (rank - 1) % size)
            if cache[-1] is not None:
                kv_cache = cache[-1][0]
                kv_cache.keys = mx.depends(kv_cache.keys, states)
            # Only the output of rank 0 is used.
            hidden_states = states[0].astype(dtype)
        else:
            hidden_states = self._finalize(hidden_states, block_states, dtype, record_tap=True)
            if ref_out is not None:
                ref_hidden = self._finalize(ref_out[0], ref_out[1], dtype, record_tap=False)

        if size > 1:
            batch = hidden_states.shape[0]
            hidden_states = mx.distributed.all_gather(hidden_states)[:batch]
        if return_ref:
            return hidden_states, ref_hidden
        return hidden_states


class Inkling2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = TextModel(args.text)
        if args.vision is not None:
            self.vision_tower = VisionModel(args.vision, args.text.hidden_size)

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self) -> list[DecoderLayer | None]:
        return self.language_model.layers

    @property
    def start_idx(self) -> int:
        return self.language_model.start_idx

    @property
    def end_idx(self) -> int | None:
        return self.language_model.end_idx

    def pipeline(self, group, split=None) -> None:
        self.language_model.pipeline(group, split)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.sharding_group = None

    def encode_image(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        """pixel_values: one image (num_channels, H, W) or a list of images of
        any size. Returns the rows of the image tokens of all images."""
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        assert len(pixel_values), "No images."
        return mx.concatenate([self.model.vision_tower(p) for p in pixel_values])

    def embed_inputs(
        self,
        inputs: mx.array,
        *,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        inputs_embeds = self.model.language_model.embed(inputs)
        if pixel_values is not None:
            image_features = self.encode_image(pixel_values)
            inputs_embeds = splice(
                inputs_embeds,
                inputs,
                rows=image_features,
                token_id=self.args.image_token_id,
            )
        return inputs_embeds

    @property
    def loopcd(self) -> loopcd.LoopCDConfig:
        return self.model.language_model.loopcd

    def set_loopcd(
        self,
        mode: str = "off",
        *,
        omega: float = 0.0,
        adaptive: bool = False,
        omega_max: float = 0.5,
        ref_pass: int = 0,
        blend_bs: bool = False,
    ) -> loopcd.LoopCDConfig:
        """Configure LoopCD (mode='off'|'hidden'|'logits') on this model."""
        cfg = loopcd.LoopCDConfig(
            mode=mode,
            omega=omega,
            adaptive=adaptive,
            omega_max=omega_max,
            ref_pass=ref_pass,
            blend_bs=blend_bs,
        )
        self.model.language_model.loopcd = cfg
        return cfg

    def _project_softcap(self, hidden_states: mx.array) -> mx.array:
        if (
            _FAST_ON
            and self.sharding_group is None
            and hidden_states.dtype == mx.bfloat16
            and hidden_states.shape == (1, 1, 2304)
            and _use_kernels()
            and fastdecode._is_aff8(self.lm_head)
        ):
            lm = self.lm_head
            logits = RQ.qmv_rows(
                hidden_states.reshape(1, 2304), lm["weight"], lm["scales"], lm["biases"]
            ).reshape(1, 1, -1)
        else:
            logits = self.lm_head(hidden_states)
        if self.sharding_group is not None:
            logits = gather_last_axis(logits, self.sharding_group)
        text = self.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size]

    def _logits_pair_softcap(self, h_R: mx.array, h_ref: mx.array) -> tuple[mx.array, mx.array]:
        lm = self.lm_head
        if (
            _FAST_ON
            and self.sharding_group is None
            and h_R.dtype == mx.bfloat16
            and h_R.ndim == 3
            and h_R.shape[1] == 1
            and 2 * h_R.shape[0] <= RQ.MAX_B
            and h_R.shape[-1] % 256 == 0
            and h_R.shape[-1] not in (64, 128)
            and _use_kernels()
            and fastdecode._is_aff8(lm)
        ):
            B, _, K = h_R.shape
            h_both = mx.concatenate([h_R.reshape(B, K), h_ref.reshape(B, K)], axis=0)
            raw = RQ.qmv_rows(h_both, lm["weight"], lm["scales"], lm["biases"]).astype(mx.float32)
            text = self.args.text
            softcap = text.final_logit_softcapping
            raw = mx.tanh(raw / softcap) * softcap
            raw = raw[:, None, : text.unpadded_vocab_size]
            return raw[:B], raw[B:]
        return self._project_softcap(h_R), self._project_softcap(h_ref)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        if pixel_values is not None:
            assert input_embeddings is None, "Pass pixel_values or input_embeddings."
            input_embeddings = self.embed_inputs(inputs, pixel_values=pixel_values)
        cfg = self.model.language_model.loopcd
        lc = self.args.text.loop_config
        if lc is not None and cfg.mode == "logits" and cfg.enabled_for(lc["num_passes"]):
            h_R, h_ref = self.model.language_model(
                inputs, cache=cache, input_embeddings=input_embeddings, return_ref=True
            )
            z_R, z_ref = self._logits_pair_softcap(h_R, h_ref)
            return loopcd.combine_logits(z_R, z_ref, cfg)
        hidden_states = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        return self._project_softcap(hidden_states)

    def decode_step_logprobs(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        *,
        greedy: bool = False,
        input_embeddings: mx.array | None = None,
    ):
        if (
            not _FAST_ON
            or self.sharding_group is not None
            or inputs.shape != (1, 1)
            or input_embeddings is not None
            or not _use_kernels()
        ):
            return None
        cfg = self.model.language_model.loopcd
        lc = self.args.text.loop_config
        if lc is not None and cfg.mode == "logits" and cfg.enabled_for(lc["num_passes"]):
            h_R, h_ref = self.model.language_model(inputs, cache=cache, return_ref=True)
            z_R, z_ref = self._logits_pair_softcap(h_R, h_ref)
            logits = loopcd.combine_logits(z_R.reshape(1, -1), z_ref.reshape(1, -1), cfg)
            lp = logits - mx.logsumexp(logits, keepdims=True)
            return (mx.argmax(lp, axis=-1) if greedy else None), lp
        hidden_states = self.model.language_model(inputs, cache=cache)
        lm = self.lm_head
        if (
            hidden_states.dtype == mx.bfloat16
            and hidden_states.shape == (1, 1, 2304)
            and fastdecode._is_aff8(lm)
        ):
            logits = RQ.qmv_rows(hidden_states.reshape(1, 2304), lm["weight"], lm["scales"], lm["biases"])
        else:
            logits = lm(hidden_states).reshape(1, -1)
        text = self.args.text
        softcap = text.final_logit_softcapping
        if logits.dtype == mx.bfloat16:
            if greedy:
                return RH.softcap_logprobs_rows(logits, softcap, text.unpadded_vocab_size, argmax=True)
            return None, RH.softcap_logprobs_rows(logits, softcap, text.unpadded_vocab_size, argmax=False)
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        logits = logits[..., : text.unpadded_vocab_size]
        lp = logits - mx.logsumexp(logits, keepdims=True)
        return (mx.argmax(lp, axis=-1) if greedy else None), lp

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    def shard(self, group: mx.distributed.Group | None = None) -> None:
        group = group or mx.distributed.init()
        N, r = group.size(), group.rank()
        text = self.args.text
        for n_kv_heads in (text.num_key_value_heads, text.swa_num_key_value_heads):
            assert (
                n_kv_heads % N == 0
            ), f"group size {N} must divide num_key_value_heads {n_kv_heads}"
        assert (
            text.vocab_size % N == 0
        ), f"group size {N} must divide vocab_size {text.vocab_size}"

        def part(w: mx.array, axis: int) -> mx.array:
            return mx.contiguous(mx.split(w, N, axis=axis)[r])

        self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
        self.sharding_group = group
        for layer in self.layers:
            attn = layer.self_attn
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "r_proj",
                "gate_proj",
                "sink_proj",
            ):
                proj = shard_linear(getattr(attn, name), "all-to-sharded", group=group)
                setattr(attn, name, proj)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.q_l2_clip.scale = part(attn.q_l2_clip.scale, 0)
            attn.rel_l2_clip.scale = part(attn.rel_l2_clip.scale, 0)
            for conv in (attn.k_sconv, attn.v_sconv):
                if isinstance(conv, ShortConvolution):
                    conv.conv1d.weight = part(conv.conv1d.weight, 0)
                    conv.conv1d.groups //= N
                else:
                    conv.static_weight = part(conv.static_weight, 1)
                    conv.weight_proj = shard_linear(
                        conv.weight_proj, "all-to-sharded", group=group
                    )
                    conv.num_heads //= N
            attn.num_heads //= N
            attn.num_key_value_heads //= N

            layer.mlp.sharding_group = group
            experts = layer.mlp.experts
            shard_experts_inplace(experts.gate_up_proj, "all-to-sharded", group)
            shard_experts_inplace(experts.down_proj, "sharded-to-all", group)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:

        def predicate(path: str, module: nn.Module) -> bool:
            return not path.startswith("model.vision_tower.") and (
                "_message_gate." not in path
            )

        return predicate

    @property
    def cast_predicate(self) -> Callable[[str], bool]:

        def predicate(k: str) -> bool:
            return not k.endswith(
                (
                    ".query",
                    ".mlp.gate.weight",
                    ".mlp.gate.e_score_correction_bias",
                    ".forget_gate_proj.weight",
                    ".message_gate_proj.weight",
                )
            )

        return predicate

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.language_model.pipeline_layers

    def make_cache(self) -> list[CacheList]:
        window = self.args.text.sliding_window_size
        kv_int8 = os.environ.get("LOOPLING_KV_INT8", "0") not in ("0", "", "false", "False")
        full_cls = Int8KVCache if kv_int8 else KVCache
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=window)
                    if isinstance(layer, SlidingDecoderLayer)
                    else full_cls()
                ),
                ArraysCache(4),
            )
            for layer in self.model.language_model.layer_visits
        ]
