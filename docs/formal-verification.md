# Formal verification

## What this is

Lean 4 models of single Rust functions, with machine-checked theorems. A model is written by hand. The proofs are about the model, not about the Rust code.

## Layout

```
formal/
  .gitignore          ignores .lake/
  README.md           build commands
  THEOREMS.txt        theorem inventory
  models.toml         source-span manifest of the models
  vectors/golden/     tracked golden vector files, written by rcsr-formal-vectors
  core/               Lake package rcsr_core, no Mathlib
    lean-toolchain    pinned Lean release
    lakefile.toml
    lake-manifest.json
    RcsrCore.lean     root module, imports every module
    RcsrCore/Prim.lean  shared bit-pattern primitives
    RcsrCore/ErrorOrder.lean  model of error_order.rs
    Audit.lean        trust audit program
    Check.lean        rcsr_model_check, replays vector files against the models
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

## Source-span manifest

`formal/models.toml` lists for each model the source spans it was written from, with the SHA-256 of each span. `rcsr-engine/tests/formal_sync.rs` recomputes the hashes. It is in the fast lint set and needs no Lean toolchain. The hash ties text to a review. It does not show that the model equals the code. A change of a std function that a model mirrors (`partial_cmp`, `saturating_add`) changes no span; only the golden vectors guard that.

Accepted lines, after trimming: an empty line; a line that starts with `#`; `version = 1` as the first entry; the headers `[[model]]` and `[[model.span]]`; `key = "value"`. A value has no escapes: a `"` or a backslash inside a value is an error. Anything else is an error.

Model keys: `id` (unique), `project` (`P0` to `P9`), `lean` (path of the Lean module from the repository root; the file must exist), `vectors` (optional; the name of the vector file without `.vec`). Span keys: `role`, `file` (path from the repository root), `anchor`, `close` (optional), `sha256` (64 hex digits).

| Role | Meaning |
|---|---|
| `modelled` | the Lean definition is a model of this text |
| `mirror` | a second implementation that must equal the modelled one (a CUDA copy) |
| `assumption` | text a proof relies on without modelling it (a constant, a compiler flag line) |

Every crate-local function that a modelled function calls is itself a span of the same model.

Span rule:

1. The start is the one line of `file` that equals `anchor` after leading and trailing white space is removed. Zero or several matches are an error.
2. With `close`, the end is the first later line that equals the start line's indentation followed by `close` (trailing white space ignored). Without `close` the span is the start line alone.
3. The hashed text is the lines from start to end, each with trailing white space removed and the start line's indentation removed, joined by `\n`, with a final `\n`.
4. The hash is the SHA-256 of that text, lowercase hex.

Moving a function or changing its indentation changes no hash. A doc comment above the function is outside the span. A comment inside the function is part of it. Renaming the function is "anchor matches 0 lines", a failure. A missing file, a missing anchor or a repeated anchor is a failure, never a skip.

## Vectors

The Lean model of a function is compared with the Rust function on vectors. `rcsr-engine/src/formal_vectors.rs` calls the real library function and writes inputs and outputs. The Lean program `rcsr_model_check` (`formal/core/Check.lean`) replays them against the compiled model. The models with vectors are the `vectors` keys of `formal/models.toml`, and `formal_vectors::MODELS` names the same set.

Format `rcsr-vec 1`: UTF-8 text, one file per model, `<model>.vec`.

```
rcsr-vec 1 <model> sig=<inputs>-><outputs>
<field> <field> ...
end <number of vector lines>
```

- A word field `w` is 16 lower-case hex digits of a `u64`. An `f64` is `to_bits`. A signed integer is its two's complement, sign-extended. A narrower integer is zero-extended. A `bool` is 0 or 1. An `Ordering` is 0 for `Less`, 1 for `Equal`, 2 for `Greater`.
- `sig` lists the field kinds, inputs before `->`, outputs after, separated by commas. `cmp_error` is `sig=w,w->w`.
- A list field `[w]` is the token `[`, the words, the token `]`. The checker does not accept list fields yet.
- Fields are separated by one space. The last line `end <n>` makes a cut-off file an error.

Two flows share one generator:

- Golden. `formal/vectors/golden/<model>.vec` is tracked. The test `golden_vectors_match_rust` regenerates it in memory and compares bytes. The checker replays the same file.
- Bulk. `rcsr-formal-vectors --bulk <dir> --count <n> --seed <s>` writes `<n>` vectors per model to a scratch directory. Nothing is tracked. The checker replays the files.

Directed set. Every file starts with directed rows, so a bulk file holds them too. For a float input: both zeros, both smallest subnormals, the largest subnormal, the smallest normal, plus and minus 1, plus and minus `MAX`, both infinities, a signalling NaN, a quiet NaN, a quiet NaN with payload and the all-ones pattern, each with both signs, and the 64 bit-pattern neighbours of zero and of `+inf` on each side. The rest of a file is random rows from SplitMix64 with a fixed seed, biased toward the directed set and toward one-bit differences.

```sh
cd rcsr-engine
cargo run --release --bin rcsr-formal-vectors -- --golden ../formal/vectors/golden
cargo run --release --bin rcsr-formal-vectors -- --bulk "$T" --count 100000 --seed 7
cd ../formal/core
lake build --wfail
.lake/build/bin/rcsr_model_check ../vectors/golden/cmp_error.vec "$T/cmp_error.vec"
```

`rcsr-formal-vectors` exit codes: 0 written, 1 a file could not be written, 2 usage or a count below the directed set. `rcsr_model_check` prints `<model>: <n> vectors, <m> mismatches` per file and the first 5 mismatching lines on stderr. Exit 0 when every file parses, has the signature of its model, has the right `end` count and every line is reproduced; 1 otherwise; 2 without arguments.

## When a modelled function changes

The pull request that changes a span sees `manifest_spans_match_sources` fail. The failure message prints the new hash. The author does one of two things in the same pull request and says which in the pull request text.

- Behaviour unchanged: update the `sha256` and state why the model still matches.
- Behaviour changed: update the Lean model and the proofs, update `formal/THEOREMS.txt` if statements changed, regenerate the golden file with `rcsr-formal-vectors --golden`, then update the `sha256`.

`golden_vectors_match_rust` also fails when the function changes behaviour with no span change (for example a std function it calls). The same two statements apply; the model check `rcsr_model_check` on the regenerated file must print 0 mismatches.

A hash update without one of the two statements is a review failure.

## Toolchain

`formal/core/lean-toolchain` pins `leanprover/lean4:v4.36.0`. elan installs it. A toolchain change is its own issue and re-runs every proof.

## What is proven

| Rust function | Lean module | Theorems | Limit |
|---|---|---|---|
| `cmp_error`, `class` in `rcsr-engine/src/error_order.rs` | `RcsrCore/ErrorOrder.lean` | `le_refl`, `le_total`, `le_trans`, `lt_trans`, `eqv_symm`, `eqv_trans`, `lt_eqv_congr`, `eqv_lt_congr`, `trichotomy`, `cmpError_swap`, `cmpError_eq_iff`, `neg_inf_least`, `nan_greatest`, `signed_zeros_equal` | the model is flattened by hand |
| row-check bit planes: `settle_slot` in `rcsr-engine/src/flat_bound/rows.rs`, `CheckRows::settle` in `rcsr-engine/src/check_rows_gpu.rs`, `check_rows_kernel` in `rcsr-engine/src/cuda/check_rows.h` | `RcsrCore/FlatBound/Planes.lean` | `setbit_test`, `row_split`, `word_in_bounds`, `lane_lt`, `row_inj`, `int_words_no_overflow`, `shift_mask_nat`, `word_lt`, `bitAt_zeros`, `getD_setRow_same`, `getD_setRow_other`, `bitAt_setRow`, `bitAt_of_size_le`, `bitAt_setRows`, `setRows_cons`, `setRows_nil`, `size_setRow`, `size_setRows`, `size_zeros` | out-of-bounds `setRow` is a no-op in the model and a panic in Rust; every theorem that uses it carries the bound as a hypothesis |
| `reaches`, `interval_reaches`, `inside`, `IntervalSplit::new`, `IntervalSplit::role`, `IntervalSplit::effective`, `window_in_sample` in `rcsr-engine/src/intervals.rs` | `RcsrCore/Intervals.lean`, `RcsrCore/Intervals/Effective.lean`, `RcsrCore/Intervals/Trim.lean` | `role_no_leak`, `role_train_clear_of_validation`, `mem_effective_train`, `mem_effective_validation`, `inside_iff_mem`, `intervalReaches_sound`, `intervalReaches_complete`, `effective_inside_iff_role`, `effective_chain`, `effective_roles_disjoint`, `effective_clear_of_test`, `wellFormed_of_table_check`, `trim_wellformed`, `effective_idempotent`, `role_trim_invariant` | model on unbounded integers; `inside` only on lists sorted by start; not tied to the Rust text before #1512; `WellFormed` is three conditions: each of the three lists is a chain (every interval ends at or before the start of every later one, none is empty), no stamp is in both a training and a validation interval, and purge and embargo are not negative; a split that is not built by `parse_table` is not proven well-formed here; trim invariance (`effective_idempotent`, `role_trim_invariant`, `trim_wellformed`): same purge and embargo in both trims; the split before the trim must be well-formed |
| `floor_day`, `ceil_day`, `hole`, `subtract`, `usable`, `slice`, group bounds of `calendar_groups` in `rcsr-engine/src/cpcv.rs` | `RcsrCore/Cpcv.lean` | `hole_covers_reach`, `usable_clear_of_locked`, `usable_chain`, `mem_slice`, `slice_disjoint`, `slice_cover`, `groupBound_mono` | interval functions only; `generate`, the partition into `n` groups and the combinatorics are not proven; model on unbounded integers; not tied to the Rust text before #1512 |
| `Lockbox::check`, `contains`, `reaches`, `first_reaching`, `Guard::check`, `Guard::check_times`, `Declared::check_times` in `rcsr-engine/src/lockbox.rs` | `RcsrCore/Lockbox.lean` | `check_no_row_inside`, `guard_checkTimes_ok_iff`, `declared_checkTimes_ok_iff`, `reaches_iff_firstReaching`, `guard_open` | armed guards only; the construction of guards and the call sites are not proven; model on unbounded integers |
| `pareto_fronts`, `nan_low` in `rcsr-engine/src/trials_table.rs` | `RcsrCore/TrialsTable/Fronts.lean` | `paretoFrontsK_total`, `paretoFrontsK_first`, `paretoFrontsK_equal_rows`, `rowDom_lex` | values are read through the order key (#1498 ties the key to the float order); fronts after the first are characterised only by the loop; no permutation law |
