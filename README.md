# AllTheMix

AllTheMix is a JAX/Flax image-classification training codebase for comparing
data-mixing augmentation methods under a shared training pipeline.

Implemented methods include ERM, MixUp, CutMix, CutMix+SUMix, FMix, ResizeMix,
SaliencyMix, Guided-SR, CatchUpMix, offline DiffuseMix, offline ALIA, and
offline SaSPA generation with a shared JAX training path.

The validation-aware methods include MetaAugment, IF-AugNet, and instantaneous
gradient alignment. They reuse the AllTheMix datasets, classifiers,
evaluation, checkpoint, CSV, and W&B paths; method-specific logic stays at the
training-strategy boundary.
See [MetaAugment details](docs/metaaugment.md),
[IF-AugNet integration details](docs/ifaugnet.md), and
[instantaneous GA details](docs/salutaryda.md) for method-equivalence
caveats.

Supported datasets:

- CIFAR-10 (`cifar10`)
- CIFAR-100 (`cifar100`)
- SVHN cropped (`svhn_cropped`)
- STL-10 (`stl10`)
- Tiny ImageNet (`tiny_imagenet`)
- Oxford-IIIT Pet (`oxford_iiit_pet`)
- Stanford Cars (`cars196`)
- CUB-200-2011 (`caltech_birds2011`, configs under `cub200`)
- CMC ImageNet-100 (`imagenet100`, local licensed ImageNet-1K subset)

## Layout

- `allthemix/data`: dataset loading, preprocessing, train/validation/test pipelines.
- `allthemix/methods`: baseline, MixUp, CutMix, FMix, ResizeMix, SaliencyMix, and GuidedMixup methods.
- `allthemix/competitors`: validation-aware and offline generative competitors.
- `allthemix/networks`: backbones, heads, classifiers, and model builder.
- `allthemix/training`: training/eval engines, losses, and metrics.
- `allthemix/cli`: command-line entry points.
- `allthemix/visualize`: unified visualization tool for mixed samples.
- `configs`: experiment configurations grouped by dataset and backbone.
- `scripts/experiment_run`: batch experiment launchers.
- `tests`: unit tests grouped by package area.

## Installation

On a TPU VM, create and validate the dedicated JAX and PyTorch/XLA
environments once:

```bash
bash scripts/environment/setup_backend_envs.sh all
```

`requirements.txt` remains a backward-compatible alias for this JAX entry.
For tests, use `requirements-dev.txt`.

Switch the current shell before starting each process:

```bash
source scripts/environment/use_backend.sh xla  # offline generation
source scripts/environment/use_backend.sh jax  # filtering and training
```

Both backend files reuse `requirements-common.txt`, but neither includes the
other backend. Never install both backend files into one environment: JAX and
PyTorch/XLA require different `libtpu` releases. They exchange only validated
image manifests and checkpoints on disk, never live TPU arrays.

See [Backend environments](docs/environments.md) for clean virtual-environment
commands and smoke checks.

For a verified, data-free move to another workstation, see
[Workstation migration](docs/migration/README.md).

## Training

Run one configuration:

```bash
python -m allthemix.cli.train --config configs/cifar10/preact_resnet18/cutmix.yaml
```

Run integrated MetaAugment with the same classifier and data stack:

```bash
python -m allthemix.cli.train --config configs/cifar10/preact_resnet18/metaaugment.yaml
```

Run integrated IF-AugNet through its four stages:

```bash
python -u -m allthemix.cli.train --config configs/cifar10/preact_resnet18/ifaugnet.yaml
```

Run the remaining formal IF-AugNet jobs together with the optimized
MetaAugment configurations. The default matrix skips the already completed
CIFAR-100 and STL-10 IF-AugNet jobs, while rerunning MetaAugment on all five
main datasets:

```bash
bash scripts/experiment_run/run_remaining_validation_aware.sh
```

The matrix is configurable, for example:

```bash
IF_DATASETS="cifar10 cub200 cars196" \
META_DATASETS="cifar10 cifar100 stl10 cub200 cars196" \
  bash scripts/experiment_run/run_remaining_validation_aware.sh
```

Run the CIFAR-100 instantaneous GA pipeline:

```bash
SEED=0 bash scripts/experiment_run/run_cifar100_salda_ga.sh smoke last_score
```

Generate DiffuseMix images with PyTorch/XLA and train them with JAX:

```bash
source scripts/environment/use_backend.sh xla
PJRT_DEVICE=TPU python -m allthemix.competitors.diffusemix \
  --dataset tiny_imagenet --data-dir ./data --validation-split 0 \
  --fractal-dir /path/to/fractals \
  --output-dir ./data/diffusemix/tiny_imagenet/official_release \
  --preset official_release --compact-output --device xla --xla-launch

source scripts/environment/use_backend.sh jax
python -m allthemix.cli.train \
  --config configs/tiny_imagenet/preact_resnet18_xla4/diffusemix.yaml
```

`--compact-output` keeps generation/composition at 512px but stores the final
PNG at the classifier input size, substantially reducing artifact storage.
Omit it when a 512px output artifact is required.

See [DiffuseMix generation and integration details](docs/diffusemix.md) for
paper/released-code presets, manifest semantics, split safety, and licensing.

Run the official-style ALIA stages with pretrained BLIP, Stable Diffusion,
CLIP, and the shared JAX baseline classifier:

```bash
python -m allthemix.competitors.alia prompts --dataset caltech_birds2011 --mode release --output data/alia/cub200/prompts.json
```

See [ALIA generation, filtering, and JAX integration](docs/alia.md) for the
complete TPU workflow and the boundary between official method semantics and
the shared AllTheMix classifier protocol.

Generate and filter official-style SaSPA images, then use source-aligned random
replacement in the common JAX trainer:

```bash
source scripts/environment/use_backend.sh xla
bash scripts/experiment_run/generate_dataset_saspa.sh cub200
CHECKPOINT=/path/to/matched_erm/best \
  bash scripts/experiment_run/filter_dataset_saspa.sh cub200
source scripts/environment/use_backend.sh jax
python -u -m allthemix.cli.train \
  --config configs/cub200/preact_resnet18/saspa.yaml
```

See [SaSPA generation, filtering, and training details](docs/saspa.md).

Prepare all three offline GenDA methods for STL-10 with the shared 10%
class-stratified split (ALIA and SaSPA require the matched ERM checkpoint for
their classifier-aware filters):

```bash
FRACTAL_DIR=/path/to/fractals \
  bash scripts/experiment_run/generate_dataset_diffusemix.sh stl10
bash scripts/experiment_run/generate_dataset_alia.sh stl10
bash scripts/experiment_run/generate_dataset_saspa.sh stl10

CHECKPOINT=/path/to/stl10_baseline_syncdist/best \
  bash scripts/experiment_run/filter_dataset_alia.sh stl10
CHECKPOINT=/path/to/stl10_baseline_syncdist/best \
  bash scripts/experiment_run/filter_dataset_saspa.sh stl10

bash scripts/experiment_run/run_stl10_genda.sh
```

Run the CIFAR-10 PreActResNet-18 experiment set:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18.sh
```

Run any supported PreActResNet-18 dataset config set:

```bash
DATASET=stl10 bash scripts/experiment_run/run_dataset_preact_resnet18.sh
```

### CMC ImageNet-100

ImageNet-100 uses the fixed 100-synset split published by the CMC authors,
not an arbitrary collection of 100 folders. AllTheMix does not download or
redistribute ImageNet. Prepare the subset from a licensed local ImageNet-1K
class-folder copy, then verify it before training:

```bash
python -m allthemix.cli.prepare_imagenet100 --source_dir /path/to/imagenet1k --output_dir ./data/imagenet100 --link_mode symlink
python -m allthemix.cli.prepare_imagenet100 --output_dir ./data/imagenet100 --verify_only
bash scripts/experiment_run/run_imagenet100_preact_resnet18.sh
```

See [CMC ImageNet-100 setup](docs/imagenet100.md) for the exact layout,
cardinality, label ordering, and storage tradeoffs for saliency maps.

## Evaluation Protocol

For paper-style experiments, train on the complete official training split.
Build the checkpoint-selection validation set from a deterministic,
class-stratified fraction of the official evaluation split, then evaluate the
sealed complement exactly once:

```yaml
validation_split: 0.1
val_source: test
eval_on_test_each_epoch: false
final_test: true
final_test_checkpoint: best
```

Here `test` means the dataset's official evaluation source: the official test
split for CIFAR/STL/Cars/CUB, and the official validation split for datasets
such as Tiny-ImageNet and CMC ImageNet-100. The main CSV stores checkpoint
validation metrics; `*_final_test.csv` stores metrics on the disjoint
official-evaluation complement. Validation examples never receive optimizer
updates. Results produced with the legacy train-sourced validation protocol
are a different protocol and must not be combined in one comparison table.
At the CLI, omitting `val_source` now selects this formal `test` source whenever
`validation_split > 0`; use `val_source: train` only for an explicit legacy or
low-data ablation.

## Reproducibility

`seed` controls model initialization, JAX method randomness, and the default
data stream. The training entry point also seeds Python, NumPy, and TensorFlow.
Train pipelines use explicitly seeded shuffling and stateless per-example
augmentation, so parallel `tf.data` mapping remains reproducible while each
epoch still receives a new shuffle and augmentation stream.

```yaml
seed: 0
data_seed: -1             # -1 reuses seed; set explicitly to decouple data RNG
deterministic_data: true  # default for formal experiments
strict_determinism: false # optional, slower TensorFlow kernel determinism
```

The class-stratified official-evaluation validation/final membership is fixed
across experiment seeds, so every method and seed uses the same checkpoint
validation examples and the same sealed final examples. Exact reruns still
require the same code, dependency versions, hardware type, and device
topology. See [reproducibility details](docs/reproducibility.md).

## Visualization

Visualize generated mixed samples:

```bash
python -m allthemix.visualize --config configs/cifar10/preact_resnet18/cutmix.yaml --output_dir outputs/visualize
```
