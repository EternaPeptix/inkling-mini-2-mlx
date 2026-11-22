# MTP: the official Loopling drafter ("DSpark" / "dflare") in MLX

Flag: `LOOPLING_MTP=1` (default OFF). Code: `vendor/mlx_lm/models/inkling2/mtp.py` (drafter, verify loop, cache rollback),
hooks in `vendor/mlx_lm/generate.py::stream_generate`, tap capture in `inkling2/model.py::TextModel._run_loop`,
conv-window recording in `inkling2/layers.py::DecoderLayer.__call__`. Other env: `LOOPLING_MTP_W` (`auto` default, or a fixed
verify width 2..8), `LOOPLING_MTP_TAPS` (tap list, default `i5,i23,i<6+12p-1>,F`), `LOOPLING_MTP_DIR`.

Reference (read only): `/Users/jeweled/.gemini/antigravity/brain/78201cfd-43f0-4f1b-ae82-c573f3cd7b11/scratch/patches-s6/`
`dspark_model.py` (M), `dspark_ops.py` (O), `speculator.py` (S), plus `attention_ops.py` (A) and
`../dec160/tree_spec/staged_s8v2/model_l2pf.py` (T, the target's tap convention). That reference is for Inkling-2-Small; the
Loopling checkpoint has the same drafter family with different sizes (hidden 2304, 3 layers, 24/8 heads, head_dim 128, MLP 9216,
block 8, rank 256, `target_layer_ids [5,23,41,48]`).

## What the drafter is

A 3-layer transformer that proposes a whole block of 7 tokens in ONE forward (not autoregressively), conditioned on the target's
own hidden states, followed by a small recurrent "Markov" head that makes the 7 greedy token choices sequentially-dependent.
The target then verifies `[anchor, d1..d7]` in one 8-token forward and keeps the longest matching prefix plus one bonus token.

## The math (our port; line numbers are in the reference files)

Notation: `P` = position of the anchor token (the last sampled token, not yet run through the target); `H=2304`, `T=4` taps.

1. **Target taps** (T:800-805, T:1040-1043; M:1001-1004 `fuse_weight` shape `(num_layers+1, len(target_layer_ids))`).
   The target hands the drafter 4 feature vectors per committed position: the hidden state *entering* visits 5, 23, 41 (the vLLM
   loop appends `partial` before running layer `i` when `i in aux_hidden_state_layers`) and, for id 48 (= number of layers, never
   hit by the loop), the final normed hidden `out` that feeds the LM head (T:1040-1043 returns `[*aux, out]`).
2. **Fuse + per-layer context** (M:1033-1035 `mix`, O:22-53 `_mix_rmsnorm_kernel`, M:1277-1279 softmax of `fuse_weight`):
   `ctx_l = RMSNorm_{dflare_norm_l}( sum_t softmax(fuse_weight[l])_t * feat_t )`, fp32 math, bf16 out. Row `l` for draft layer `l`,
   the extra last row for the Markov head. Learned fuse weights (softmax) e.g. layer 0 puts 0.68 on tap 3 (visit 23)...
3. **Context K/V** (M:599-643 `store_context_kv`): `k = k_proj_ctx(ctx_l)`, `v = v_proj_ctx(ctx_l)`; `k` gets per-head
   `RMSNorm(k_norm_ctx)` then neox RoPE at the committed position; `v` is raw. One K/V entry per committed target position.
4. **Block input** (M:1037-1043 `embed_block`): slot 0 = `embed_norm(embed(anchor))` (the *target's* normalized embedding),
   slots 1..7 = learned `mask_embed`. Block positions `P..P+7` (S:3507-3509: query pos = last_valid_pos+1+off).
5. **Layers** (M:716-762, M:498-596): standard pre-norm decoder, residual stream bf16:
   `h=RMSNorm(x)`; `q,k,v=q_proj,k_proj,v_proj(h)`; `q=RMSNorm(q_norm)`, `k=RMSNorm(k_norm)` per head; neox RoPE (theta 1e6, full
   head_dim) at block positions; attention of the 8 block queries over `[ctx K/V , block K/V]`, **bidirectional inside the block**,
   no relative bias / sink / gate (A:1315-1339 `block_decode_attention`, scale head_dim^-0.5); GQA 24/8; layers 0 and 2 are local
   (`local_layer_ids`): only the last 1024 context positions + the block (A: `window = sliding_window + block`), layer 1 global.
   `o_proj`; residual add; `RMSNorm(post_attention_layernorm)`; SwiGLU `down(silu(gate)*up)` (M:662-685); residual add.
   Final `RMSNorm(norm)`; slots 1..7 are the 7 hidden outputs (S:2944 `head_hidden[1:1+k]`; slot 0 is unused).
6. **Base logits** (M:1579): `base_i = target.lm_head(hid_i)` (shared LM head, raw, pre-softcap).
7. **Markov head** (M:765-880 `Inkling2MarkovHead`, O:595-610 `_markov_gate_kernel`, S:2881-2921 + S:2951-3000):
   * `f_rnn = RMSNorm_{markov.dflare_norm}(sum_t softmax(fuse_weight[-1])_t * feat_t(P-1))`: features of the last committed
     position (S:2905-2921).
   * `ctx_i = rnn_proj.W[:, 2r:] @ [hid_i ; f_rnn] + rnn_proj.b` (fp32), `r=256` (M:824-870 `context`).
   * Sequential over i=0..6 with `state_0 = 0`, `prev_0 = anchor`: `raw = ctx_i + rnn_proj.W[:, :2r] @ [state ; markov_w1[prev]]`
     (M:884-896 `step`); `gate, cand, out = raw[:r], raw[r:2r], raw[2r:]`;
     `state = sigmoid(gate)*state + (1-sigmoid(gate))*tanh(cand)`; `latent = tanh(out)` (O:600-610).
   * `bias_logits = lm_head(hidden_out(latent))`, folded once into a `(V, r)` matrix (M:798-815 `fold`). We fold with the quantized
     MLX lm_head applied to the `hidden_out` columns, bf16 (reference folds in fp64).
   * `draft_i = argmax( 30*tanh((base_i + bias_i)/30) )` over the unpadded vocab (O:573-582 `add_softcap`, M:1679-1690); `prev = draft_i`.
8. **Verify** (S:3507-3509, the bonus token logic): target forward on `[anchor, d1..d7]`; accept `d_{j+1}` while it equals the
   target's greedy token after position `j`; the target's token after the last accepted position is the bonus and becomes the next
   anchor. Context K/V and `f_rnn` for the accepted positions come from the same verify forward (taps are returned per position).

### Our deviations / uncertainty

* `layers.0.dflare_norm` is **absent from the checkpoint** (only layers 1, 2 and markov_head have one). The reference's `_RMSNorm`
  defaults to ones (M:297); we use ones. Ablation: using layer 1's norm instead drops teacher-forced accept length 1.69 -> 0.88 (ones is right or close).
* The reference deployment caps the global layer's window at 1024 (`INKLING_DRAFT_GLOBAL_WINDOW`, M:145-153); we use the true
  global attention (identical for contexts < 1024).
* The reference's bf16 rounding emulation inside the Markov gate (O:586-610) is only approximated (bf16 raw, bf16 state).
* The mapping of the reference's 48 target layers to Loopling's visits (below) is by analogy plus an acceptance test, not documented.

## Mapping `target_layer_ids [5,23,41,48]` onto Loopling's visits

Loopling at 3 passes has 6 prelude + 12x3 loop + 6 tail = **48 visits**, exactly the 48 layers the ids index (the official
checkpoint is a 3-pass model). The taps are the *inputs* of visit 5 (= end of the prelude), 23 (pass 1, core layer 11),
41 (pass 2, last core layer) and the final normed hidden. Evidence (teacher-forced drafter acceptance, 3 bench prompts, ground truth
= baseline greedy tokens, `bench/mtp_probe.py`; mean accepted drafts / block, 7 max):

| taps | code | prose | math | mean |
|---|---|---|---|---|
| **i5,i23,i41,F** (reference convention) | 1.11 | 0.40 | 3.55 | **1.69** |
| i5,i23,i41,o (raw output of last visit) | 0.89 | 0.36 | 3.31 | 1.52 |
| i5,i23,i41,f (final_input pre-norm) | 0.63 | 0.39 | 3.45 | 1.49 |
| i6,i24,i42,F (outputs instead of inputs) | 0.89 | 0.35 | 2.99 | 1.41 |
| i4,i22,i40,F | 0.84 | 0.43 | 3.08 | 1.45 |
| i5,i23,i41,i47 | 0.68 | 0.39 | 3.43 | 1.50 |
| no Markov bias (base logits only) | 0.58 | 0.23 | 1.16 | 0.65 |

The reference convention wins on every prompt where it matters, and removing the Markov head costs 60%, so both are in use.

## Spec loop (what runs with `LOOPLING_MTP=1`)

Greedy only. Prefill (chunks of 2048, last prompt token alone as in `generate_step`) records the 4 taps for every prompt position and
builds the drafter's context K/V. Each cycle: draft 7 tokens (one lazy MLX graph, ~4.8 ms incl. the 7 Markov steps), verify
`[anchor, d1..d_{W-1}]` with ONE target forward (W<=8), accept the matching prefix, roll the caches back, append the accepted
positions' taps to the drafter context. One host sync per cycle.

* **Cache rollback** (no replay): `KVCache` -> `offset -= n`; `RotatingKVCache` after the multi-token concat update is in temporal
  order, so slice the rejected tail and set `_idx`; the 4 conv windows per visit (k/v sconv, post-attn, post-mlp) are recorded in
  `DecoderLayer.__call__` (`conv.record`, `conv.windows`, 2-line patch) and the state after `keep` tokens is `window[:, keep:keep+K-1]`.
* **Adaptive verify width** (`LOOPLING_MTP_W=auto`): the verify forward costs ~28 ms for 1 token and ~47 ms for 8 (measured:
  1/2/4/6/8 tokens = 27.9/33.4/38.3/43.2/47.1 ms, drafter 4.8 ms), so with low acceptance a full block loses to plain decode.
  An online per-position acceptance estimate picks W in 1..8 to maximize expected tokens/ms; W=1 runs a pipelined plain-decode
  burst of 8 tokens; every 64 tokens a full-width cycle re-measures acceptance. `LOOPLING_MTP_W=8` forces fixed width.

## Results (512S1, p3 = native 3 passes, greedy, 256 tokens, median of 3 reps, same session as the baseline)

| prompt | baseline tok/s | MTP tok/s | speedup | mean tokens / cycle (accept length) |
|---|---|---|---|---|
| code | 38.69 | 39.68 | +2.6% | 1.45 (stats from the last rep; mostly W<8 or plain bursts) |
| prose | 38.19 | 36.66 | -4.0% | ~1.0 (acceptance too low; exploration cycles cost ~4%) |
| math | 38.11 | 72.55 | +90.4% | 3.7 |
| geomean | 38.33 | 47.26 | +23.3% | |

Fixed full block (`LOOPLING_MTP_W=8`, first experiment, 1 rep): code 35.3 (accept length 1.9), prose 28.5 (1.5), math 73.4 (3.9):
drafter acceptance is the lever, a block cycle costs 53 ms vs 26 ms per plain step, break-even accept length 2.05.
Teacher-forced acceptance with the final taps (`bench/mtp_probe.py`, 3 passes): code 2.11, prose 1.40, math 4.55 tokens/cycle
(accepted drafts + bonus), clearly above 1, so the port is correct (the bonus-only floor is 1.0).

5 passes (taps stay the literal `5,23,41,F`; mapping variants at p5 are within 0.05 of each other, see `mtp_probe.py`):
baseline 24.54 / 24.28 / 24.23 -> MTP 27.16 / 23.96 / 51.74 tok/s (+11% / -1% / +114%); teacher-forced accept length 3-pass 2.69 vs 5-pass 2.59 mean.
The drafter was trained for 3 passes but still accepts at 5.

## Exact gate: FAIL by near-ties (inherent to any batched verify here)

`bench/exact.py` vs `runs/baseline.json`: every prompt diverges (p3: token 11 / 52 / 34; p5: 60 / 78 / 109). Evidence that this is
numerical, not a port or rollback bug (`bench/mtp_diag.py`, `bench/mtp_consistency.py`):

* The verify forward (L>1: relative-attention kernel, sorted MoE gathers) differs from the 1-token decode path by mean |logit| diff
  0.09 - 0.21 and max 1.5 - 5.5 over the first 32 steps, for ANY block size 2..32 (B=2,4,8 are bit-identical to each other).
  Changing only the *prefill* attention path (`LOOPLING_REF_ATTN=1`, multi-token reference attention instead of the kernel) moves
  the subsequent decode-path logits by the same amount (mean 0.11 - 0.19), so the model is this sensitive to benign kernel changes;
  a batched verify cannot be bit-exact against the L=1 path.
* Teacher-forcing every MTP token through the 1-token decode path: positions where its argmax differs from the MTP token:
  code 1 of 111 (gap 0.070), prose 11 of 256 (gaps 0.00 - 0.59, median 0.10), math 3 of 256 (0.12, 0.16, 0.64).
  First flips: code tok 11 baseline 150131 vs MTP 150493, decode-path top-2 gap 0.0697 (logits 27.866 / 27.796);
  prose tok 58 gap 0.0706; math tok 34 gap 0.160. Two flips (prose 62: 0.59 lead, math 92: 0.64) are larger than a rounding tie but
  are within the measured max verify-vs-decode logit diffs (up to 5.5).
* So the MTP stream is a valid greedy path up to numerical near-ties, not token-identical. It is therefore opt-in (lossy-class)
  and stays default OFF; `LOOPLING_MTP=1` needs the user's OK per OBJECTIVES.md.

## Open questions

* `layers.0.dflare_norm` missing from the checkpoint (ones assumed; ablation says ones is better than layer 1's norm).
* prose acceptance (1.40) is low even teacher-forced; code 2.11. Whether the reference-trained drafter is better on chat/prose
  with the true bf16 rounding of its Markov gate, or whether Loopling's post-training will fix it, is unknown. A retrained
  drafter for the post-trained checkpoint is the real lever; this port works unchanged once the checkpoint's `mtp.safetensors` is swapped.
* Verify cost: 8 tokens cost 1.7x one token (MoE expert reads); lane D's expert matmul work compounds with this.
* Tree/beam verification (reference `INKLING_TREE_SPEC`) and confidence-based width (reference `INKLING_VERIFY_W`) are not ported; the
  adaptive width here is a simple online estimator.
* The exact gate could only be met by verifying with the L=1 numerics (8 sequential decode steps), which removes the gain.
