# Formal verification

## What this is

Lean 4 models of single Rust functions, with machine-checked theorems. A model is written by hand. The proofs are about the model, not about the Rust code.

## Layout

```
formal/
  .gitignore          ignores .lake/
  README.md           build commands
  THEOREMS.txt        theorem inventory
  core/               Lake package rcsr_core, no Mathlib
    lean-toolchain    pinned Lean release
    lakefile.toml
    lake-manifest.json
    RcsrCore.lean     root module, imports every module
    RcsrCore/Prim.lean  shared bit-pattern primitives
    RcsrCore/ErrorOrder.lean  model of error_order.rs
    Audit.lean        trust audit program
```

## Trust base

A theorem under the namespace `Rcsr` may depend only on:

- `propext`, `Classical.choice`, `Quot.sound`
- names of the form `<declaration>._native.bv_decide.ax_<n>_<m>`, one per use of `bv_decide`

`bv_decide` trusts an LRAT certificate checked by a verified checker that runs as compiled code. Assumptions of a proof are hypotheses or structure fields, never declared facts.

## Source rules

- Refused words in `formal/**/*.lean`, comments included: `sorry`, `axiom`, `native_decide`, `implemented_by`, `extern`, `unsafe`, `csimp`, `opaque`. Check with `grep -w`.
- `bv_decide` appears only in the macro `rcsr_bv` of `Prim.lean` (timeout 600 s) and in `Audit.lean`. Every proof calls `rcsr_bv`.
- Models that feed the SAT tactic use `bif` or Boolean operators, not nested `if` on a `Bool` used as a `Prop`.
- One module per Rust module. `RcsrCore.lean` imports every module; an unimported file is neither built nor audited.
- Each module starts with a doc comment that has a `Limits` paragraph.
- Names: `Rcsr.<Module>`, lower camel case functions.

## Theorem inventory

`formal/THEOREMS.txt` is the exact stdout of the audit: one line `<full name> <kernel|bv_decide>` per theorem, sorted. A deleted or weakened theorem changes the file. Regenerate with the audit command below and commit the result.

## Build and audit

```sh
cd formal/core
lake build --wfail
lake env lean --run Audit.lean > /tmp/audit.out
diff /tmp/audit.out ../THEOREMS.txt
```

The audit is a program, not an elaboration-time command, because Lake replays a cached build without running commands. It exits 1 and names every dependency outside the trust base.

## Toolchain

`formal/core/lean-toolchain` pins `leanprover/lean4:v4.36.0`. elan installs it. A toolchain change is its own issue and re-runs every proof.

## What is proven

| Rust function | Lean module | Theorems | Limit |
|---|---|---|---|
| `cmp_error`, `class` in `rcsr-engine/src/error_order.rs` | `RcsrCore/ErrorOrder.lean` | `le_refl`, `le_total`, `le_trans`, `lt_trans`, `eqv_symm`, `eqv_trans`, `lt_eqv_congr`, `eqv_lt_congr`, `trichotomy`, `cmpError_swap`, `cmpError_eq_iff`, `neg_inf_least`, `nan_greatest`, `signed_zeros_equal` | the model is flattened by hand |
| row-check bit planes: `settle_slot` in `rcsr-engine/src/flat_bound/rows.rs`, `CheckRows::settle` in `rcsr-engine/src/check_rows_gpu.rs`, `check_rows_kernel` in `rcsr-engine/src/cuda/check_rows.h` | `RcsrCore/FlatBound/Planes.lean` | `setbit_test`, `row_split`, `word_in_bounds`, `lane_lt`, `row_inj`, `int_words_no_overflow`, `shift_mask_nat`, `word_lt`, `bitAt_zeros`, `getD_setRow_same`, `getD_setRow_other`, `bitAt_setRow`, `bitAt_of_size_le`, `bitAt_setRows`, `setRows_cons`, `setRows_nil`, `size_setRow`, `size_setRows`, `size_zeros` | out-of-bounds `setRow` is a no-op in the model and a panic in Rust; every theorem that uses it carries the bound as a hypothesis |
| `reaches`, `interval_reaches`, `inside`, `IntervalSplit::new`, `IntervalSplit::role`, `IntervalSplit::effective`, `window_in_sample` in `rcsr-engine/src/intervals.rs` | `RcsrCore/Intervals.lean`, `RcsrCore/Intervals/Effective.lean` | `role_no_leak`, `role_train_clear_of_validation`, `mem_effective_train`, `mem_effective_validation`, `inside_iff_mem`, `intervalReaches_sound`, `intervalReaches_complete`, `effective_inside_iff_role`, `effective_chain`, `effective_roles_disjoint`, `effective_clear_of_test`, `wellFormed_of_table_check` | model on unbounded integers; `inside` only on lists sorted by start; not tied to the Rust text before #1512; `WellFormed` is three conditions: each of the three lists is a chain (every interval ends at or before the start of every later one, none is empty), no stamp is in both a training and a validation interval, and purge and embargo are not negative; a split that is not built by `parse_table` is not proven well-formed here |
| `Lockbox::check`, `contains`, `reaches`, `first_reaching`, `Guard::check`, `Guard::check_times`, `Declared::check_times` in `rcsr-engine/src/lockbox.rs` | `RcsrCore/Lockbox.lean` | `check_no_row_inside`, `guard_checkTimes_ok_iff`, `declared_checkTimes_ok_iff`, `reaches_iff_firstReaching`, `guard_open` | armed guards only; the construction of guards and the call sites are not proven; model on unbounded integers |
