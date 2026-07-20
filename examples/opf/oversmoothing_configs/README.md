# OPF Oversmoothing Configs

This directory contains 21 standalone configurations:

- Seven heterogeneous stack types, each with and without GPS, at 16 layers.
- One HeteroSAGE attention-only GPS configuration at 16 layers.
- HeteroHEAT without GPS at depths 1, 2, 4, 8, 12, and 16.

Every configuration enables oversmoothing diagnostics for bus nodes and the
`ac_line` and `transformer` bus-to-bus edge types. The enabled metrics are
`feature_variance` and `dirichlet_energy`; the cosine metric is omitted.

`configs.txt` lists the configurations in experiment order for use by a Slurm
job array.
