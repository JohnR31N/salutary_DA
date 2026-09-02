"""Datasets, split protocols, and input pipelines.

- ``pipeline``       canonical train/val/test tf.data pipelines
- ``splits``         validation-split protocols and guards
- ``datasets``       per-dataset loaders (TFDS, class folders, Cars196, IN100)
- ``preprocessors``  augmentation and saliency-aware preprocessing
- ``salmix_pipeline`` saliency-channel pipeline variant
- ``saliency``       train-set saliency cache build/load
- ``utils``          data-domain helpers (cardinality/normalization/random/...)
"""
