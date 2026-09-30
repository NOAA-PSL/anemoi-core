# Plan: Latent-Space Autoregressive Rollout for Anemoi

**Status:** Draft / Proposal
**Scope:** `anemoi-models` and `anemoi-training` (this repo); companion plan in `NOAA-PSL/anemoi-inference`
**Motivation:** WindBorne's WeatherMesh-style architecture — encode the initial state once, roll out autoregressively in the *latent* (hidden-mesh) space, and decode only when grid-space output is needed. This contrasts with the current Anemoi pipeline, which runs the full encoder→processor→decoder at every rollout step and re-injects decoded prognostic variables back into a grid-space input window.

---

## 1. Background: how the current pipeline works

### Model side (`anemoi-models`)

- `AnemoiModelEncProcDec.forward()` (`models/src/anemoi/models/models/encoder_processor_decoder.py`) is monolithic: it assembles grid-space inputs, runs the graph **encoder** (data mesh → hidden mesh), the **processor** (hidden mesh → hidden mesh), and the **decoder** (hidden mesh → data mesh) in a single call.
- Skip connections tie decoder output to encoder-time state: `x_skip` (grid-space residual on prognostic variables) and `latent_skip` (`x_latent_proc = x_latent_proc + x_latent`).
- Time-dependent forcings (solar insolation, time-of-day/year embeddings) enter as *grid-space input variables* each step.

### Training side (`anemoi-training`)

- The autoregressive loop lives in the training task (`training/src/anemoi/training/tasks/forecaster.py`). `Forecaster._step` calls the full model each rollout step, then `advance_input()` / `_advance_dataset_input()` rolls the grid-space window `x (B, T, E, grid, vars)` with `x.roll(-keep_steps, dims=1)`, writes decoded prognostic variables into the newest slots, and refreshes **forcings from the batch**.
- `OffsetForecaster` generalizes this with a precomputed `_advance_map` (ADR-003) — still entirely grid-space.
- Rollout scheduling (`rollout.start` / `epoch_increment` / `maximum`) is handled by `RolloutConfig` and is architecture-agnostic; it can be reused.

### Why change

1. **Information bottleneck:** every step squeezes the state through latent→grid→latent, losing information the decoder discards. Latent rollout keeps the full hidden-mesh state between steps.
2. **Compute/memory at inference:** decode only at requested lead times; enables cheap high-frequency latent stepping (e.g., 1 h latent stepper decoded 6-hourly).
3. **Cleaner separation:** the processor becomes a true latent transition operator `z_{t+Δt} = P(z_t, f_t)`.

---

## 2. Target architecture

```
grid x(t0-history..t0) ──encode()──► z(t0)
                                       │
                          ┌────────────┴────────────┐
                          │  latent rollout loop     │
                          │  z ← process(z, f_t)     │   f_t = per-step forcings
                          └────────────┬────────────┘   embedded on hidden mesh
                                       │
                 decode(z_t) on demand ─► grid y(t)
```

Key design decisions (to be finalized during Phase 1):

- **D1 — Latent state contents.** Option A: single latent step `z_t` (simplest). Option B: latent history window (WeatherMesh keeps multiple latent timesteps). Start with A; keep the API shaped so B is a non-breaking extension (latent state as a dict/tuple).
- **D2 — Forcing conditioning.** Time-varying forcings (cos/sin zenith angle, julian day, local time) must be recomputed per latent step and injected on the *hidden mesh*. Approach: compute forcings analytically on hidden-mesh node coordinates (lat/lon are available via `node_attributes`), embed with a small MLP, and add/concat to `z` (or FiLM-condition the processor layers). Static hidden-node attributes are already supported.
- **D3 — Skip connections.** `x_skip` (grid-space residual) is unavailable after step 0. Options: (a) drop it and predict full states; (b) predict *tendencies in latent space* with a latent residual `z' = z + P(z, f)` (recommended default — the latent analogue of the current design); decoder learns full-state reconstruction.
- **D4 — Decoder input.** Decoder currently consumes `x_target_latent` assembled partly from encoder-time data latents. In latent rollout, the decoder must work from hidden latent + static target-node embeddings only.
- **D5 — Δt conditioning.** Fixed Δt per checkpoint initially. Leave a conditioning hook (embed Δt alongside forcings) so a variable-Δt stepper is a follow-on, not a rewrite.

---

## 3. Work plan

### Phase 1 — Model API split (`anemoi-models`) ~2–3 weeks

1. New model class `AnemoiModelLatentRollout(BaseGraphModel)` in `models/src/anemoi/models/models/latent_rollout_encoder_processor_decoder.py` exposing:
   - `encode(x, model_comm_group, grid_shard_sizes) -> LatentState`
   - `process(z: LatentState, step_forcings, model_comm_group) -> LatentState` (one Δt)
   - `decode(z: LatentState, model_comm_group, grid_shard_sizes) -> y_pred`
   - `forward(x, n_steps, decode_steps, **kw)` — convenience wrapper doing encode → N× process → decode at requested steps (keeps the existing training-module calling convention workable).
2. `LatentState` dataclass: hidden-mesh tensor(s) + shard metadata (`shard_sizes_hidden`, `model_comm_group` context) so it can live across steps under model sharding.
3. Hidden-mesh forcing embedder: module that maps `(datetime, hidden lat/lon)` → per-node forcing features → MLP → conditioning tensor. Reuse the analytic forcing computations from `anemoi-transform`/training data pipeline where possible.
4. Remove/restructure `x_skip`; implement latent residual stepping (D3).
5. Config schema: new entry under `config/model/` (e.g. `latent_rollout_gnn.yaml` / `latent_rollout_transformer.yaml`) with `model._target_` pointing at the new class; processor/encoder/decoder sub-configs unchanged.
6. Unit tests: shape/equivariance tests for `encode/process/decode`; one-step equivalence sanity test (latent model with 1 step ≈ existing pipeline minus skip connections, not bitwise).

**Out of scope for Phase 1:** transport/diffusion variants, ensemble `ens_encoder_processor_decoder`, multi-dataset/multi-encoder routing (keep single dataset first).

### Phase 2 — Training task (`anemoi-training`) ~2 weeks

1. New task `LatentForecaster` (subclass of the rollout-scheduling base) in `training/src/anemoi/training/tasks/latent_forecaster.py`:
   - `_step`: pre-process batch → `encode` once → loop `z = process(z, forcings_t)` → `decode` each step → loss vs. targets. **No `advance_input` on grid tensors.**
   - Per-step forcings sourced from batch dates (dataloader already provides date metadata) or computed analytically.
   - Reuse `RolloutConfig` (start/epoch_increment/maximum) unchanged.
2. Gradient checkpointing per latent step (wrap each `process` call; `BaseProcessor.run_layers` chunk checkpointing is reused inside).
3. Loss: decode at every rollout step for training loss (gradient path stays latent-to-latent between steps). Optional: loss weighting per lead time (existing config).
4. Curriculum: standard two-stage training still applies (rollout=1 pretrain, then increase). Document that latent models benefit from earlier multi-step training.
5. Validation/plot callbacks: ensure callbacks that assume grid-space `y_pred` per step still work (they should — decode produces grid-space output per step).
6. Unit tests mirroring `test_methods.py` rollout-threading tests: verify `process` is called N times, `encode` exactly once, forcings refreshed per step.

### Phase 3 — Parallelism & sharding audit ~1–2 weeks

- `LatentState` persists across steps: verify `shard_tensor`/`gather_tensor`/`grid_shard_sizes` handling when the hidden latent never returns to grid space between steps (hidden shard sizes are already computed; they must be carried in `LatentState`).
- Dropout + model sharding constraint (see `BaseProcessor.forward` assertion) applies per step — document.
- Activation memory: N latent steps × processor depth; measure vs. current pipeline, tune checkpoint granularity.

### Phase 4 — Inference (`anemoi-inference`, separate repo/plan)

- New runner keeping `z` GPU-resident; decode only at requested output steps; per-step forcings from checkpoint metadata. See `NOAA-PSL/anemoi-inference/plan_latent_rollout.md`.

### Phase 5 — Extensions (deferred)

- LAM/stretched-grid boundary forcing: encode boundary strip each step and overwrite boundary hidden nodes (latent-space injection).
- Latent history window (D1 option B), variable-Δt stepper (D5), ensemble & diffusion variants, multi-dataset support.

---

## 4. Risks & open questions

| Risk | Mitigation |
| --- | --- |
| Latent drift / instability over long rollouts without grid-space renormalization | Latent residual stepping + longer rollout curriculum; optionally LayerNorm on `z` between steps |
| Decoder quality without `x_skip` grid residual | Ablate; consider learned static skip from encoder-time latent |
| Forcing conditioning insufficient (diurnal cycle degradation) | FiLM conditioning in processor layers as fallback to additive conditioning |
| Sharding bugs with persistent latent | Phase 3 dedicated audit + multi-GPU integration test |
| Upstream divergence (fork drift) | Keep changes additive (new classes/configs), no edits to existing model/task classes where avoidable; aim for upstreamable PRs |

## 5. Success criteria

1. Single-step skill parity (within noise) with the equivalent enc-proc-dec baseline on a small config (e.g., o96 test dataset).
2. Multi-step rollout: latent model ≥ baseline RMSE at 5–10 day lead times after rollout fine-tuning.
3. Inference: measurable wall-clock reduction for long rollouts decoded sparsely.
4. All new code covered by unit tests; existing test suite untouched/passing.
