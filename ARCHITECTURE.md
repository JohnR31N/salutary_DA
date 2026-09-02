# AllTheMix Architecture

One-screen map of what lives where. Details in each package's `__init__.py`
docstring; the import layering below is enforced by
`tests/architecture_tests/test_layering.py`.

```
allthemix/                  the library (pure JAX/flax training line)
├── data/                   pipelines + split protocols; helpers in data/utils/,
│                           loaders in data/datasets/, saliency cache in data/saliency/
├── networks/               flax backbones/heads; PyTorch numeric-compat constants
│                           (deliberate: they reproduce PyTorch-paper baselines)
├── methods/                one mixing method per module; selector.py = registry;
│                           shared validation in methods/utils/
├── training/
│   ├── engine/single/      one-device engine (train step + epoch loop)
│   ├── engine/parallel/    pmap engine (train/eval steps, loop, cross-device utils)
│   ├── losses/             CE variants, mixup soft targets, sumix
│   ├── utils/              early_stop, lr_scheduler, metric plumbing
│   └── strategy.py         Protocol interfaces methods implement to join training
├── cli/                    argument surface + train / run_suite entry points
├── competitors/            paper reproductions; torch-based generative pipelines
│                           (alia/saspa/diffusemix) quarantined behind
│                           utils/backend_environment.py and a separate venv
├── utils/                  dependency-light shared primitives (sharding, checkpoint,
│                           backend guard, reproducibility)
└── visualize/ debug/ diagnostics/   leaf tooling (import core, never imported by it)

salutary_da/ + scripts/analysis/     research instruments (experiment code, not library)
scripts/experiment_run/              node launch shells (flock, fail-closed backend)
scripts/agents/ + .agents/           multi-agent collaboration protocol and records
configs/                             per-dataset/model training configs
tests/                               mirrors the package tree; architecture_tests/
                                     locks the import graph
```

## Import layering

Allowed direction: `utils → data → networks/methods → training → cli`, with
`competitors` parallel to the core (it imports core; core does not import it)
and `visualize/debug/diagnostics` as leaves. The current graph carries two
recorded legacy inversions, frozen (no new ones allowed) and scheduled for
cleanup:

- `data/pipeline.py → competitors` (diffusemix manifest validation)
- `utils/saliency_preprocessor/saliency_io.py → data` (cache builder)

## Conventions

- Entry modules hold one public thing; private helpers live in a sibling
  `*_utils.py`; ~500 lines triggers a split.
- PyTorch-compat numeric constants (`PYTORCH_*` init/padding, torchvision-
  semantics augs, torchbearer cutmix variants) are a **reproduction
  contract**, not debt: the values match PyTorch defaults so JAX runs
  reproduce PyTorch-trained baselines. Do not "clean" them away.
- `salutary_da/` is deliberately not library-ified; instruments churn.
```
