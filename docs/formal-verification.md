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

## When a modelled function changes

The pull request that changes a span sees `manifest_spans_match_sources` fail. The failure message prints the new hash. The author does one of two things in the same pull request and says which in the pull request text.

- Behaviour unchanged: update the `sha256` and state why the model still matches.
- Behaviour changed: update the Lean model and the proofs, update `formal/THEOREMS.txt` if statements changed, then update the `sha256`.

A hash update without one of the two statements is a review failure.

## Toolchain

`formal/core/lean-toolchain` pins `leanprover/lean4:v4.36.0`. elan installs it. A toolchain change is its own issue and re-runs every proof.

## What is proven

| Rust function | Lean module | Theorems | Limit |
|---|---|---|---|
| `cmp_error`, `class` in `rcsr-engine/src/error_order.rs` | `RcsrCore/ErrorOrder.lean` | `le_refl`, `le_total`, `le_trans`, `lt_trans`, `eqv_symm`, `eqv_trans`, `lt_eqv_congr`, `eqv_lt_congr`, `trichotomy`, `cmpError_swap`, `cmpError_eq_iff`, `neg_inf_least`, `nan_greatest`, `signed_zeros_equal` | the model is flattened by hand |
