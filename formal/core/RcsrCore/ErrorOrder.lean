import RcsrCore.Prim

/-!
Model of `rcsr-engine/src/error_order.rs`: `class` and `cmp_error`.

Limits: the `match` of the Rust text is flattened into `lt` and `eqv`. That step is covered by the vector check, not by a theorem.
-/

namespace Rcsr.ErrorOrder
open Rcsr.Prim

/-- `class`: 2 NaN, 1 `+inf`, 0 every other value (finite and `-inf`). -/
def cls (b : W) : BitVec 2 := bif isNaN b then 2#2 else bif b == TOP then 1#2 else 0#2

def lt (a b : W) : Bool := cls a < cls b || (cls a == cls b && cls a == 0#2 && fltLt a b)
def eqv (a b : W) : Bool := cls a == cls b && (cls a != 0#2 || fltEq a b)

/-- `cmp_error`. -/
def cmpError (a b : W) : Ordering := if lt a b then .lt else if eqv a b then .eq else .gt

def le (a b : W) : Bool := lt a b || eqv a b

theorem le_total (a b : W) : le a b = true ∨ le b a = true := by
  simp only [le, lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN]; rcsr_bv
theorem le_trans (a b c : W) (h1 : le a b = true) (h2 : le b c = true) : le a c = true := by
  simp only [le, lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN] at *; rcsr_bv
theorem le_refl (a : W) : le a a = true := by
  simp only [le, lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN]; rcsr_bv
theorem lt_trans (a b c : W) (h1 : lt a b = true) (h2 : lt b c = true) : lt a c = true := by
  simp only [lt, cls, fltLt, fold, isNaN, TOP, SIGN] at *; rcsr_bv
theorem eqv_trans (a b c : W) (h1 : eqv a b = true) (h2 : eqv b c = true) : eqv a c = true := by
  simp only [eqv, cls, fltEq, fold, isNaN, TOP, SIGN] at *; rcsr_bv
theorem eqv_symm (a b : W) : eqv a b = eqv b a := by
  simp only [eqv, cls, fltEq, fold, isNaN, TOP, SIGN]; rcsr_bv
theorem lt_eqv_congr (a b c : W) (h1 : lt a b = true) (h2 : eqv b c = true) : lt a c = true := by
  simp only [lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN] at *; rcsr_bv
theorem eqv_lt_congr (a b c : W) (h1 : eqv a b = true) (h2 : lt b c = true) : lt a c = true := by
  simp only [lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN] at *; rcsr_bv
/-- Exactly one of `lt a b`, `eqv a b`, `lt b a` holds. -/
theorem trichotomy (a b : W) :
    (lt a b && !eqv a b && !lt b a) || (!lt a b && eqv a b && !lt b a) || (!lt a b && !eqv a b && lt b a) = true := by
  simp only [lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN]; rcsr_bv

/-- Rust `sort_by` contract: swapping the arguments reverses the result. -/
theorem cmpError_swap (a b : W) : cmpError b a = (cmpError a b).swap := by
  have h := trichotomy a b
  have hs := eqv_symm a b
  unfold cmpError
  cases h1 : lt a b <;> cases h2 : eqv a b <;> cases h3 : lt b a <;> simp_all [Ordering.swap]

theorem cmpError_eq_iff (a b : W) : cmpError a b = .eq ↔ eqv a b = true := by
  have h := trichotomy a b
  unfold cmpError
  cases h1 : lt a b <;> cases h2 : eqv a b <;> simp_all

/-- `-inf` is the least element. -/
theorem neg_inf_least (a : W) : le (TOP ||| SIGN) a = true := by
  simp only [le, lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN]; rcsr_bv
/-- Every NaN is a greatest element. -/
theorem nan_greatest (a n : W) : isNaN n = true → le a n = true := by
  simp only [le, lt, eqv, cls, fltLt, fltEq, fold, isNaN, TOP, SIGN]; rcsr_bv
theorem signed_zeros_equal : cmpError SIGN 0#64 = .eq := by decide

end Rcsr.ErrorOrder
