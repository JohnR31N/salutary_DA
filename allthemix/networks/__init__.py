"""Flax network definitions.

- ``backbones``  PreActResNet and friends (with PyTorch numeric-compat
                 initializers/padding used to reproduce paper baselines)
- ``heads``      classifier heads
- ``classifiers`` assembled models
- ``builder``    name -> model construction
- ``utils``      feature-hook plumbing
"""
