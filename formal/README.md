# formal

Lean 4 models of rcsr functions with machine-checked theorems. The toolchain comes from `formal/core/lean-toolchain` through elan. Details: [../docs/formal-verification.md](../docs/formal-verification.md).

```sh
cd formal/core
lake build --wfail
lake env lean --run Audit.lean > /tmp/audit.out
diff /tmp/audit.out ../THEOREMS.txt
```
