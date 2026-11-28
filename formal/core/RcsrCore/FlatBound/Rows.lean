import RcsrCore.Prim

/-!
Model of `ulps_step` in `rcsr-engine/src/flat_bound/rows.rs` and of its device copy `rc_ulps` in
`rcsr-engine/src/cuda/check_rows.h`.

Limits: the theorems are about these definitions. That the Rust function equals `ulpsStep` rests on the vector
check and the source hash. That the compiled CUDA code equals `rcUlps` rests on the source hash and the device tests.
-/

namespace Rcsr.FlatBound.Rows
open Rcsr.Prim

/-- `ulps_step(v: f64, k: i64) -> f64` on bit patterns. -/
def ulpsStep (v k : W) : W :=
  if !isFinite v then v else fold (clampTop (satAdd (fold v) k))

/-- `rc_ulps(double v, int k)`: plain `i += k`, upper clamp first, as in the C text. -/
def rcUlps (v : W) (k : BitVec 32) : W :=
  if !isFinite v then v else
    let i := fold v
    let i := i + k.signExtend 64
    let i := if TOP.slt i then TOP else i
    let i := if i.slt (-TOP) then -TOP else i
    fold i

/-- `f64::next_up` of the Rust core library, on bits. -/
def nextUp (b : W) : W :=
  if isNaN b || b == TOP then b else
    let abs := b &&& 0x7fffffffffffffff#64
    if abs == 0#64 then 1#64 else if b == abs then b + 1#64 else b - 1#64

/-- `f64::next_down` of the Rust core library, on bits. -/
def nextDown (b : W) : W :=
  if isNaN b || b == (TOP ||| SIGN) then b else
    let abs := b &&& 0x7fffffffffffffff#64
    if abs == 0#64 then (1#64 ||| SIGN) else if b == abs then b - 1#64 else b + 1#64

/-- The clamp of the order key to `[-TOP, TOP]`, in 66 bits. -/
def clamp66 (i : BitVec 66) : BitVec 66 :=
  let t : BitVec 66 := TOP.signExtend 66
  if i.slt (-t) then -t else if t.slt i then t else i

/-- Infinities and NaN are returned unchanged, bit for bit. -/
theorem inf_fixed (v k : W) (h : isFinite v = false) : ulpsStep v k = v := by
  simp [ulpsStep, h]

/-- The key of the result is the clamped exact sum, for every `i64` step. -/
theorem key_spec (v k : W) (h : isFinite v = true) :
    key66 (ulpsStep v k) = clamp66 (key66 v + k.signExtend 66) := by
  simp only [ulpsStep, key66, clamp66, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- A finite input never gives NaN. -/
theorem not_nan (v k : W) (h : isFinite v = true) : isNaN (ulpsStep v k) = false := by
  simp only [ulpsStep, isNaN, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- Monotone in the key order, for every input. -/
theorem mono (a b k : W) (h : (key66 a).sle (key66 b) = true) :
    (key66 (ulpsStep a k)).sle (key66 (ulpsStep b k)) = true := by
  simp only [ulpsStep, key66, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

theorem strict_up (v k : W) (h : isFinite v = true) (hk : (0#64).slt k = true) :
    (key66 v).slt (key66 (ulpsStep v k)) = true := by
  simp only [ulpsStep, key66, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

theorem strict_down (v k : W) (h : isFinite v = true) (hk : k.slt 0#64 = true) :
    (key66 (ulpsStep v k)).slt (key66 v) = true := by
  simp only [ulpsStep, key66, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- The device copy equals the host function for every `int` step. -/
theorem device_eq_host (v : W) (k : BitVec 32) : rcUlps v k = ulpsStep v (k.signExtend 64) := by
  simp only [rcUlps, ulpsStep, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- The C addition `i += k` cannot overflow a signed 64-bit integer. -/
theorem rc_add_no_overflow (v : W) (k : BitVec 32) (h : isFinite v = true) :
    (fold v).signExtend 66 + k.signExtend 66 = (fold v + k.signExtend 64).signExtend 66 := by
  simp only [fold, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- One step up is `next_up`, except that the smallest negative subnormal steps to `+0.0` where `next_up` gives `-0.0`. -/
theorem step_one_next_up (v : W) (h : isFinite v = true) :
    ulpsStep v 1#64 = nextUp v ∨ (ulpsStep v 1#64 = 0#64 ∧ nextUp v = SIGN) := by
  simp only [ulpsStep, nextUp, isNaN, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

theorem step_minus_one_next_down (v : W) (h : isFinite v = true) :
    ulpsStep v (-1#64) = nextDown v ∨ (ulpsStep v (-1#64) = 0#64 ∧ nextDown v = SIGN)
      ∨ (ulpsStep v (-1#64) = SIGN ∧ nextDown v = 0#64) := by
  simp only [ulpsStep, nextDown, isNaN, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- Steps of one sign compose. -/
theorem compose_up (v a b : W) (ha : (0#64).sle a = true) (hb : (0#64).sle b = true)
    (hs : (0#64).sle (a + b) = true) :
    ulpsStep (ulpsStep v a) b = ulpsStep v (a + b) := by
  simp only [ulpsStep, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

/-- A zero step returns the input, except that `-0.0` becomes `+0.0`. -/
theorem zero_step (v : W) : ulpsStep v 0#64 = v ∨ (v = SIGN ∧ ulpsStep v 0#64 = 0#64) := by
  simp only [ulpsStep, fold, clampTop, satAdd, isFinite, TOP, SIGN] at *
  rcsr_bv

theorem key_int_raw (v k : W) (h : isFinite v = true) :
    (fold (ulpsStep v k)).toInt
      = max (-9218868437227405312) (min 9218868437227405312 ((fold v).toInt + k.toInt)) := by
  have h66 := congrArg BitVec.toInt (key_spec v k h)
  rw [key66_toInt] at h66
  simp only [key] at h66
  rw [h66]
  have hk : (k.signExtend 66).toInt = k.toInt := by simp [BitVec.toInt_signExtend_of_le]
  have hv : (key66 v).toInt = (fold v).toInt := key66_toInt v
  have hb1 := BitVec.toInt_lt (x := k)
  have hb2 := BitVec.le_toInt (x := k)
  have hb3 := BitVec.toInt_lt (x := fold v)
  have hb4 := BitVec.le_toInt (x := fold v)
  have hT : (TOP.signExtend 66).toInt = 9218868437227405312 := by decide
  have hnT : (-(TOP.signExtend 66)).toInt = -9218868437227405312 := by decide
  have hsum : (key66 v + k.signExtend 66).toInt = (fold v).toInt + k.toInt := by
    rw [BitVec.toInt_add, hv, hk]
    apply Int.bmod_eq_of_le <;> omega
  unfold clamp66
  simp only [BitVec.slt_iff_toInt_lt, hT, hnT, hsum]
  by_cases h1 : (fold v).toInt + k.toInt < -9218868437227405312
  · simp only [h1, ↓reduceIte, hnT]; omega
  · simp only [h1, ↓reduceIte]
    by_cases h2 : 9218868437227405312 < (fold v).toInt + k.toInt
    · simp only [h2, ↓reduceIte, hT]; omega
    · simp only [h2, ↓reduceIte, hsum]; omega

/-- The key of the result is the clamped exact sum. Holds for every `i64` step, with no bound on `|k|`. -/
theorem key_int (v k : W) (h : isFinite v = true) :
    key (ulpsStep v k) = max (-TOPi) (min TOPi (key v + k.toInt)) := by
  simpa [key, TOPi] using key_int_raw v k h

theorem key_bounds (v k : W) (h : isFinite v = true) :
    -TOPi ≤ key (ulpsStep v k) ∧ key (ulpsStep v k) ≤ TOPi := by
  have := key_int_raw v k h
  simp only [key, TOPi]; omega

end Rcsr.FlatBound.Rows
