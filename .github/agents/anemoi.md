---
name: anemoi
description: Coding agent profile for Anemoi latent-rollout development
model: claude-sonnet-5.5
---

You are working on the Anemoi machine-learning weather forecasting framework
(fork of ecmwf/anemoi-core and ecmwf/anemoi-inference).

## Style and verbosity

Match the existing Anemoi code base exactly:

- Apache 2.0 / ECMWF copyright header block at the top of every new Python file
  (copy verbatim from an existing file, updating the year).
- NumPy-style docstrings with Parameters/Returns sections, as in
  `models/src/anemoi/models/layers/processor.py`.
- Full type annotations (`from __future__ import annotations` where used elsewhere);
  `Optional[ProcessGroup]`, `dict[str, torch.Tensor]` style matching neighboring code.
- Concise inline comments only where intent is non-obvious; no narrative comments.
- Hydra/OmegaConf-driven configuration: new components get config YAML entries with
  `_target_`, never hard-coded constructor wiring.
- Keep changes additive: new classes/modules/configs rather than modifying existing
  model or task classes, to remain upstreamable to ecmwf.
- Tests mirror existing patterns in `training/tests/unit/` and `models/tests/`
  (pytest, monkeypatch-based threading tests, shape assertions).
- Use existing utilities (`shard_tensor`, `gather_tensor`, `maybe_checkpoint`,
  `node_attributes`, `load_layer_kernels`) instead of reimplementing.
- Logging via module-level `LOGGER = logging.getLogger(__name__)`.
- Follow the repo's pre-commit config (ruff, line length, import order); run it
  before committing.

## Context

The current work implements latent-space autoregressive rollout
(encode-once → latent AR steps → decode-on-demand). The authoritative plan is
`plan_latent_rollout.md` on the `feature/latent-rollout-plan` branch; consult it
before making design decisions.
