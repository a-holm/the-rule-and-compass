import Std.Tactic.BVDecide

/-!
Shared primitives of the rcsr models: Rust std operations on bit patterns.

An `f64` is the `BitVec 64` that `to_bits` returns. An `i64` or `u64` is a `BitVec 64` in two's complement.
Lean `Float` is not used.

Limits: each definition restates a Rust std operation by hand. The tie to the Rust code is the vector check.
-/

/-- The one place that calls the SAT tactic. The timeout is far above the measured need, so machine load cannot fail a proof. -/
macro "rcsr_bv" : tactic => `(tactic| bv_decide (config := { timeout := 600 }))

namespace Rcsr.Prim

abbrev W := BitVec 64

/-- Bits of `f64::INFINITY`; also the exponent mask. -/
def TOP : W := 0x7ff0000000000000#64
/-- The sign bit; also `i64::MIN`. -/
def SIGN : W := 0x8000000000000000#64

/-- `f64::is_finite`: the exponent field is not all ones. -/
def isFinite (b : W) : Bool := (b &&& TOP) != TOP
/-- `f64::is_nan`: the exponent field is all ones and the fraction is not zero. -/
def isNaN (b : W) : Bool := (b &&& TOP) == TOP && (b &&& 0x000fffffffffffff#64) != 0#64

/-- `if i < 0 { i = i64::MIN.wrapping_sub(i) }`: maps `to_bits() as i64` to an integer that is ordered like the float. -/
def fold (i : W) : W := if i.slt 0#64 then SIGN - i else i

/-- `i64::saturating_add`. -/
def satAdd (a k : W) : W :=
  let s := a + k
  if !a.msb && !k.msb && s.msb then 0x7fffffffffffffff#64
  else if a.msb && k.msb && !s.msb then SIGN
  else s

/-- `i64::clamp(-TOP, TOP)`. -/
def clampTop (i : W) : W := if i.slt (-TOP) then -TOP else if TOP.slt i then TOP else i

/-- IEEE `<` on two non-NaN doubles, through the folded key. -/
def fltLt (a b : W) : Bool := (fold a).slt (fold b)
/-- IEEE `==` on two non-NaN doubles, through the folded key. -/
def fltEq (a b : W) : Bool := fold a == fold b

/-- Order key of a double as an integer: `+0.0` and `-0.0` give 0, `+inf` gives `TOPi`. -/
def key (b : W) : Int := (fold b).toInt
def TOPi : Int := 9218868437227405312

/-- The order key in 66 bits, so that sums with a 64-bit step cannot wrap. -/
def key66 (b : W) : BitVec 66 := (fold b).signExtend 66

theorem satAdd_spec (a k : W) :
    (satAdd a k).signExtend 66 =
      (let s : BitVec 66 := a.signExtend 66 + k.signExtend 66
       let hi : BitVec 66 := (0x7fffffffffffffff#64).signExtend 66
       let lo : BitVec 66 := SIGN.signExtend 66
       if hi.slt s then hi else if s.slt lo then lo else s) := by
  simp only [satAdd, SIGN]
  rcsr_bv

theorem nan_not_finite (b : W) (h : isNaN b = true) : isFinite b = false := by
  simp only [isNaN, isFinite, TOP] at *
  rcsr_bv

theorem finite_not_nan (v : W) (h : isFinite v = true) : isNaN v = false := by
  simp only [isNaN, isFinite, TOP] at *
  rcsr_bv

theorem fold_signed_zero : fold SIGN = 0#64 := by decide

theorem key66_toInt (b : W) : (key66 b).toInt = key b := by
  simp [key66, key, BitVec.toInt_signExtend_of_le]

end Rcsr.Prim
