"""AllTheMix: JAX/flax mixing-augmentation benchmark library.

Package map (one-way layering; see ARCHITECTURE.md and
tests/architecture_tests/test_layering.py):

- ``data``        datasets, split protocols, preprocessing, saliency pipeline
- ``networks``    flax backbones/heads; PyTorch numeric-compat constants
- ``methods``     one mixing method per module; ``selector`` is the registry
- ``training``    engines (single/parallel), losses, strategy interfaces
- ``cli``         argument parsing and train/suite entry points
- ``competitors`` paper reproductions (torch-based generative pipelines are
                  quarantined here behind ``utils.backend_environment``)
- ``utils``       dependency-light shared utilities
- ``visualize`` / ``debug`` / ``diagnostics``  leaf tooling
"""
