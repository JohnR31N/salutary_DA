"""Mixing-augmentation methods: one method per module.

``selector`` maps method names to mixer functions; each module implements a
single published method (mixup, cutmix, resizemix, fmix, saliencymix,
guidedmixup, catchupmix, ...) with matching semantics; shared validation
lives in ``utils``.
"""
