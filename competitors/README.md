# Competitor Provenance

Competitors run through the shared AllTheMix stack. Runtime code belongs under
`allthemix/competitors`; this top-level directory records provenance and must
not contain a second data pipeline, classifier implementation, training loop,
or CLI.

MetaAugment is integrated in `allthemix/competitors/metaaugment`. Its policy
MLP, 14 learned operations, sampler, and differentiable bilevel update were
adapted from the Apache-2.0 implementation at
`JohnR31N/MetaAugment@c00d2e1a2c341ba08c69e78dc7e49a2410678789`. All
task-model, dataset, split, evaluation, logging, and checkpoint
responsibilities are owned by AllTheMix.

DiffuseMix is implemented independently from the CC-BY-4.0 paper equations as
an offline image producer under `allthemix/competitors/diffusemix`. It records
the paper/released-code semantic differences and hands RGB PNGs to the shared
JAX pipeline through a validated manifest. No upstream code or fractal images
are redistributed: the authors' repositories have no standard license and
their license issue restricts those materials to research use. See
`docs/diffusemix.md` for the exact reproduction boundary.
