# Heterogeneous HGT+GPS with OPF positional encodings and physics constraints

The reference configuration is
`configs/opf_hgt_gps_pe_physics.json`. It predicts bus `[Va, Vm]` and generator
`[Pg, Qg]` jointly. The OPF case and number of groups remain command-line
choices, so the same config can be used for every supported case.

## Positional encodings

`Architecture.positional_encodings.precompute` controls which artifacts are
created during preprocessing. `use` controls which of those artifacts enter
the model. An active encoding must also be in `precompute`.

- `laplacian`: the `k` smallest non-zero eigenvectors and eigenvalues of the
  unweighted bus Laplacian. Each bus receives its eigenvector coordinates and
  the graph eigenvalues. Optional random signs are sampled once per graph and
  eigenmode during training.
- `effective_resistance`: build the susceptance-weighted bus Laplacian using
  `1 / abs(x)` for AC lines and `1 / (abs(x) * abs(tap))` for transformers.
  For each bus, discard its zero self-resistance and store five values in this
  exact order: minimum, maximum, population standard deviation, median, mean.

Active bus PE values are concatenated with the learned bus embedding and
projected back to `hidden_dim`. The reference config precomputes both choices
but activates only the five-value effective-resistance summary. Set `use` to
`["laplacian"]`, `["effective_resistance"]`, both names, or an empty list to
select the input.

PE preprocessing supports pickle and HDF5. ADIOS is rejected when
`precompute` is non-empty because the heterogeneous PE attributes would not be
preserved by that route. Artifacts are cached under
`examples/opf/dataset/positional_encoding_cache` by topology fingerprint.

## Augmented-Lagrangian physics loss

`Training.DomainLoss.constraints` is the explicit list of families added to
the objective. The supported loss families are:

- `power_balance_p` and `power_balance_q`
- `voltage_bounds`
- `ac_line_angle_bounds` and `transformer_angle_bounds`
- `ac_line_apparent_power_limit` and
  `transformer_apparent_power_limit`
- `generator_active_power_bounds` and
  `generator_reactive_power_bounds`

Full complex AC branch equations are used for nodal balance and apparent
power at both ends of every rated branch. The diagnostic-only families
`ac_line_dc_flow_proxy` and `transformer_dc_flow_proxy` are logged but cannot
be selected for the loss.

For one equality family with residual vector `r`, the implemented contribution
is:

`lambda * mean(r) + rho / 2 * mean(r^2)`

For one inequality family, define `v = relu(h)`. Its contribution is:

`mu * mean(v) + rho / 2 * mean(v^2)`

This is the violation-consistent interpretation of the supplied formulation.
A literal quadratic term `mean(h^2)` would penalize satisfied inequalities
where `h < 0`; using `mean(relu(h)^2)` avoids that contradiction.

Each family has one scalar multiplier. After every training epoch, residual
sums and counts are globally reduced across DDP ranks, then the multipliers are
updated once:

`lambda <- lambda + rho * global_mean(r)`

`mu <- max(0, mu + rho * global_mean(relu(h)))`

Validation and test batches never update state. Multipliers are registered
buffers, so normal HydraGNN checkpoints save and restore them. Omitting
`DomainLoss.mode` retains the existing static fixed-weight/EMA loss behavior.

Every epoch logs train, validation, and test mean/max/MSE violations, signed
means, selected loss contributions, and dual values under `physics/...` in
TensorBoard and as `PhysicsBreakdown` lines in the run log.

## Example commands

Preprocess one case to HDF5:

```bash
python examples/opf/train_opf_solution_heterogeneous.py \
  --inputfile=configs/opf_hgt_gps_pe_physics.json \
  --case_name pglib_opf_case14_ieee \
  --num_groups 1 --modelname OPF_HGT_GPS_PE --preonly --hdf5
```

Train from that HDF5 dataset:

```bash
python examples/opf/train_opf_solution_heterogeneous.py \
  --inputfile=configs/opf_hgt_gps_pe_physics.json \
  --case_name pglib_opf_case14_ieee \
  --num_groups 1 --modelname OPF_HGT_GPS_PE --hdf5
```

## Frontier preprocessing for case 2000

The dedicated Frontier job uses
`configs/opf_hgt_gps_pe_case2000.json`, processes all locally available
`pglib_opf_case2000_goc` groups, and writes
`dataset/OPF_HGT_GPS_PE_case2000.h5`:

```bash
sbatch examples/opf/job-frontier-preprocess-case2000-hgt-gps-pe.sh
```

For a small smoke run, override the group and sample limits:

```bash
sbatch --export=ALL,OPF_NUM_GROUPS=1,OPF_MAX_SAMPLES=100 \
  examples/opf/job-frontier-preprocess-case2000-hgt-gps-pe.sh
```

The job refuses to replace an existing output directory. Set
`OPF_OVERWRITE=1` explicitly when replacement is intended.
