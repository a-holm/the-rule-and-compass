import RcsrCore.Prim

/-!
Binary64 `+` and `-` on bit patterns, round to nearest, ties to even.

These are the Rust operators `a + b` and `a - b` on `f64`. They are computable definitions for the executable
models; no theorem of the package states a property of them. A NaN result is the pattern `QNAN`: Rust does not fix
the sign or the payload of a NaN result, so the vector generator writes every NaN result as `QNAN`.

Limits: the tie to the Rust operators is the vector check of the models `f64_add` and `f64_sub`.
-/

namespace Rcsr.F64
open Rcsr.Prim

/-- The NaN pattern every NaN result is written as. -/
def QNAN : W := 0x7ff8000000000000#64

/-- Magnitude bits: the sign bit cleared. -/
def magNat (b : W) : Nat := b.toNat % 2^63

/-- Integer significand of a finite double: its value is `mant * 2^(expo - 1074)`, up to the sign. -/
def mant (b : W) : Nat := if magNat b / 2^52 = 0 then magNat b else 2^52 + magNat b % 2^52
def expo (b : W) : Nat := if magNat b / 2^52 = 0 then 0 else magNat b / 2^52 - 1

/-- The double nearest to `a * 2^(e - 1074)` for `a > 0`, ties to even, overflow to infinity; `neg` sets the sign. -/
def roundPack (neg : Bool) (a e : Nat) : W :=
  let l := a.log2
  let me : Nat × Nat :=
    if l ≤ 52 then
      let sh := min (52 - l) e
      (a * 2^sh, e - sh)
    else
      let d := l - 52
      let q := a / 2^d
      let rem := a % 2^d
      let half := 2^(d - 1)
      (if rem > half || (rem == half && q % 2 == 1) then q + 1 else q, e + d)
  let bits := me.2 * 2^52 + me.1
  let bits := if 2047 * 2^52 ≤ bits then 2047 * 2^52 else bits
  BitVec.ofNat 64 (bits + (if neg then 2^63 else 0))

/-- `a + b` on `f64`. -/
def add (x y : W) : W :=
  if isNaN x || isNaN y then QNAN
  else if !isFinite x then (if !isFinite y && x.msb != y.msb then QNAN else x)
  else if !isFinite y then y
  else
    let e := min (expo x) (expo y)
    let mx : Int := Int.ofNat (mant x * 2^(expo x - e))
    let my : Int := Int.ofNat (mant y * 2^(expo y - e))
    let s : Int := (if x.msb then -mx else mx) + (if y.msb then -my else my)
    if s = 0 then (if x.msb && y.msb then SIGN else 0#64)
    else roundPack (decide (s < 0)) s.natAbs e

/-- `a - b` on `f64`: IEEE subtraction is addition of the negated operand. -/
def sub (x y : W) : W := add x (y ^^^ SIGN)

end Rcsr.F64
