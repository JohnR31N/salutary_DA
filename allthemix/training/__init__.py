"""Training engines, losses, and strategy interfaces.

- ``engine.single`` / ``engine.parallel``  one-device and pmap engines
- ``losses``        loss family (CE, mixup soft targets, sumix)
- ``strategy``      Protocol interfaces methods implement to join training
- ``utils``         training helpers (early_stop/lr_scheduler/metrics plumbing)
"""
