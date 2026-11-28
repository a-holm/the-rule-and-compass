import RcsrCore.Prim
import RcsrCore.Intervals

/-!
The `i64` arithmetic of `rcsr-engine/src/intervals.rs` against the `Int` model of
`RcsrCore/Intervals.lean`.

An `i64` is a `BitVec 64` in two's complement; its value is `toInt`. Three forms of each sum are
modelled: the plain form (`+` and `-`, which wrap in a release build), the wide form (the sum in 128
bits) and the saturating form. Comparisons of two `i64` values are exact and need no theorem.

Limits: one theorem per expression. No theorem here composes them into a whole function on `i64`.
-/

namespace Rcsr.Intervals
open Rcsr.Prim

/-! ### The magnitude bound -/

/-- Every stamp of the split lies in `[-B, B]`, and purge and embargo lie in `[0, B]`. -/
def Bounded (B : Int) (sp : Split) : Prop :=
  (∀ p ∈ sp.train ++ sp.validation ++ sp.test, -B ≤ p.1 ∧ p.1 ≤ B ∧ -B ≤ p.2 ∧ p.2 ≤ B) ∧
  0 ≤ sp.purge ∧ sp.purge ≤ B ∧ 0 ≤ sp.embargo ∧ sp.embargo ≤ B

/-- The bound of the Rust code: the constant `MAX_SECONDS` of `intervals.rs`. -/
def maxSeconds : Int := 2 ^ 40

theorem maxSeconds_le : maxSeconds ≤ 2 ^ 61 := by decide

/-! ### `i64` helpers -/

/-- `i64::max`. -/
def smax (a b : W) : W := if a.slt b then b else a

/-- `i64::saturating_sub`. -/
def satSub (a k : W) : W :=
  let s := a - k
  if !a.msb && k.msb && s.msb then 0x7fffffffffffffff#64
  else if a.msb && !k.msb && !s.msb then SIGN
  else s

theorem smax_toInt (a b : W) : (smax a b).toInt = max a.toInt b.toInt := by
  unfold smax
  by_cases h : a.slt b = true
  · have := BitVec.slt_iff_toInt_lt.mp h; simp only [h, ↓reduceIte]; omega
  · have : ¬ a.toInt < b.toInt := fun h' => h (BitVec.slt_iff_toInt_lt.mpr h')
    have h' : a.slt b = false := by simpa using h
    simp only [h', Bool.false_eq_true, ↓reduceIte]; omega

theorem toInt_one : (1#64 : W).toInt = 1 := by decide
theorem toInt_zero : (0#64 : W).toInt = 0 := by decide

theorem toInt_range (a : W) : -2 ^ 63 ≤ a.toInt ∧ a.toInt < 2 ^ 63 :=
  ⟨BitVec.le_toInt a, BitVec.toInt_lt⟩

theorem toInt_add_of_range (a b : W) (h : -2 ^ 63 ≤ a.toInt + b.toInt ∧ a.toInt + b.toInt < 2 ^ 63) :
    (a + b).toInt = a.toInt + b.toInt := by
  rw [BitVec.toInt_add]; exact Int.bmod_eq_of_le (by omega) (by omega)

theorem toInt_sub_of_range (a b : W) (h : -2 ^ 63 ≤ a.toInt - b.toInt ∧ a.toInt - b.toInt < 2 ^ 63) :
    (a - b).toInt = a.toInt - b.toInt := by
  rw [BitVec.toInt_sub]; exact Int.bmod_eq_of_le (by omega) (by omega)

/-- The value of an `i64` cast to `i128`. -/
theorem toInt_wide (a : W) : (a.signExtend 128).toInt = a.toInt :=
  BitVec.toInt_signExtend_of_le (by decide)

theorem toInt_wide_add (a b : W) :
    (a.signExtend 128 + b.signExtend 128).toInt = a.toInt + b.toInt := by
  have ha := toInt_range a
  have hb := toInt_range b
  rw [BitVec.toInt_add, toInt_wide, toInt_wide]; exact Int.bmod_eq_of_le (by omega) (by omega)

/-! ### `reaches` -/

/-- The plain form: `t + purge.max(1) > start && t < end + embargo.max(0)` with wrapping `+`. -/
def reachesWrap (s e t purge emb : W) : Bool :=
  s.slt (t + smax purge 1#64) && t.slt (e + smax emb 0#64)

/-- The wide form of `reaches` in `intervals.rs`: `t as i128 + purge.max(1) as i128 > start as i128 && t < end as i128 + embargo.max(0) as i128`. -/
def reachesWide (s e t purge emb : W) : Bool :=
  (s.signExtend 128).slt (t.signExtend 128 + (smax purge 1#64).signExtend 128) &&
    (t.signExtend 128).slt (e.signExtend 128 + (smax emb 0#64).signExtend 128)

/-- The wide form is the `Int` predicate on every `i64` input. -/
theorem reachesWide_exact (s e t purge emb : W) :
    reachesWide s e t purge emb = decide (reaches s.toInt e.toInt t.toInt purge.toInt emb.toInt) := by
  rw [Bool.eq_iff_iff]
  simp only [reachesWide, reaches, Bool.and_eq_true, BitVec.slt_iff_toInt_lt, decide_eq_true_eq,
    toInt_wide_add, toInt_wide, smax_toInt, toInt_one, toInt_zero]

/-- The plain form is the `Int` predicate when the stamps `t` and `e` lie in `[-B, B]` and purge and
embargo are at most `B`, for any `B` up to `2^61`. The start `s` needs no bound. -/
theorem reachesWrap_refines (B : Int) (hB : B ≤ 2 ^ 61) (s e t purge emb : W)
    (he : -B ≤ e.toInt ∧ e.toInt ≤ B) (ht : -B ≤ t.toInt ∧ t.toInt ≤ B)
    (hp : purge.toInt ≤ B) (hm : emb.toInt ≤ B) :
    reachesWrap s e t purge emb = decide (reaches s.toInt e.toInt t.toInt purge.toInt emb.toInt) := by
  have h1 : (t + smax purge 1#64).toInt = t.toInt + max purge.toInt 1 := by
    rw [toInt_add_of_range, smax_toInt, toInt_one]; rw [smax_toInt, toInt_one]; omega
  have h2 : (e + smax emb 0#64).toInt = e.toInt + max emb.toInt 0 := by
    rw [toInt_add_of_range, smax_toInt, toInt_zero]; rw [smax_toInt, toInt_zero]; omega
  rw [Bool.eq_iff_iff]
  simp only [reachesWrap, reaches, Bool.and_eq_true, BitVec.slt_iff_toInt_lt, decide_eq_true_eq, h1, h2]

/-- Without the bound the plain form is wrong: an embargo of `i64::MAX` wraps the sum, and the
predicate answers false for a stamp that reaches the range. -/
theorem reachesWrap_wraps :
    reachesWrap 0#64 100#64 (-5#64) 10#64 0x7fffffffffffffff#64 = false ∧
    reaches (0#64 : W).toInt (100#64 : W).toInt (-5#64 : W).toInt (10#64 : W).toInt
      (0x7fffffffffffffff#64 : W).toInt := by
  refine ⟨by decide, ?_⟩
  have h1 : (100#64 : W).toInt = 100 := by decide
  have h2 : (-5#64 : W).toInt = -5 := by decide
  have h3 : (10#64 : W).toInt = 10 := by decide
  have h4 : (0x7fffffffffffffff#64 : W).toInt = 9223372036854775807 := by decide
  rw [toInt_zero, h1, h2, h3, h4]; unfold reaches; omega

/-! ### `interval_reaches` -/

/-- The plain form: `a < b && a < end + embargo.max(0) && (b - 1) + purge.max(1) > start`. -/
def intervalReachesWrap (s e a b purge emb : W) : Bool :=
  a.slt b && a.slt (e + smax emb 0#64) && s.slt ((b - 1#64) + smax purge 1#64)

/-- The wide form of `interval_reaches` in `intervals.rs`: `a < b && (a as i128) < end as i128 + embargo.max(0) as i128 && b as i128 - 1 + purge.max(1) as i128 > start as i128`. -/
def intervalReachesWide (s e a b purge emb : W) : Bool :=
  a.slt b && (a.signExtend 128).slt (e.signExtend 128 + (smax emb 0#64).signExtend 128) &&
    (s.signExtend 128).slt (b.signExtend 128 - 1#128 + (smax purge 1#64).signExtend 128)

theorem intervalReachesWide_exact (s e a b purge emb : W) :
    intervalReachesWide s e a b purge emb =
      decide (intervalReaches s.toInt e.toInt a.toInt b.toInt purge.toInt emb.toInt) := by
  have hb := toInt_range b
  have hp := toInt_range (smax purge 1#64)
  have h1 : (1#128 : BitVec 128).toInt = 1 := by decide
  have h2 : (b.signExtend 128 - 1#128).toInt = b.toInt - 1 := by
    rw [BitVec.toInt_sub, toInt_wide, h1]; exact Int.bmod_eq_of_le (by omega) (by omega)
  have h3 : (b.signExtend 128 - 1#128 + (smax purge 1#64).signExtend 128).toInt
      = b.toInt - 1 + max purge.toInt 1 := by
    have hs := smax_toInt purge 1#64
    rw [toInt_one] at hs
    rw [BitVec.toInt_add, h2, toInt_wide, ← hs]
    exact Int.bmod_eq_of_le (by omega) (by omega)
  rw [Bool.eq_iff_iff]
  simp only [intervalReachesWide, intervalReaches, Bool.and_eq_true, BitVec.slt_iff_toInt_lt,
    decide_eq_true_eq, toInt_wide_add, toInt_wide, smax_toInt, toInt_zero, h3]
  omega

theorem intervalReachesWrap_refines (B : Int) (hB : B ≤ 2 ^ 61) (s e a b purge emb : W)
    (he : -B ≤ e.toInt ∧ e.toInt ≤ B) (hb : -B ≤ b.toInt ∧ b.toInt ≤ B)
    (hp : purge.toInt ≤ B) (hm : emb.toInt ≤ B) :
    intervalReachesWrap s e a b purge emb =
      decide (intervalReaches s.toInt e.toInt a.toInt b.toInt purge.toInt emb.toInt) := by
  have h0 : (b - 1#64).toInt = b.toInt - 1 := by
    rw [toInt_sub_of_range, toInt_one]; rw [toInt_one]; omega
  have h1 : ((b - 1#64) + smax purge 1#64).toInt = b.toInt - 1 + max purge.toInt 1 := by
    rw [toInt_add_of_range, h0, smax_toInt, toInt_one]; rw [h0, smax_toInt, toInt_one]; omega
  have h2 : (e + smax emb 0#64).toInt = e.toInt + max emb.toInt 0 := by
    rw [toInt_add_of_range, smax_toInt, toInt_zero]; rw [smax_toInt, toInt_zero]; omega
  rw [Bool.eq_iff_iff]
  simp only [intervalReachesWrap, intervalReaches, Bool.and_eq_true, BitVec.slt_iff_toInt_lt,
    decide_eq_true_eq, h1, h2]
  omega

/-! ### Zone edges and `first_reaching` -/

/-- The plain zone start `s - purge.max(1) + 1`. -/
def zoneStartWrap (s purge : W) : W := s - smax purge 1#64 + 1#64

/-- `purge.max(1) - 1` never wraps: the maximum is at least 1. -/
theorem p1m1_toInt (purge : W) : (smax purge 1#64 - 1#64).toInt = max purge.toInt 1 - 1 := by
  have h := toInt_range (smax purge 1#64)
  have hs := smax_toInt purge 1#64
  rw [toInt_one] at hs
  rw [toInt_sub_of_range, hs, toInt_one]; rw [hs, toInt_one]; omega

/-- The saturating zone start `s.saturating_sub(p1 - 1)` with `p1 = purge.max(1)`: `effective` in
`intervals.rs`, `Lockbox::first_reaching` in `lockbox.rs`, `hole` in `cpcv.rs`. -/
def zoneStartSat (s purge : W) : W := satSub s (smax purge 1#64 - 1#64)

/-- The plain zone end `e + embargo`. -/
def zoneEndWrap (e emb : W) : W := e + emb

/-- The saturating zone end `e.saturating_add(embargo)`: `effective` in `intervals.rs`; `hole` in `cpcv.rs` adds `embargo.max(purge).max(0)` instead. -/
def zoneEndSat (e emb : W) : W := satAdd e emb

theorem zoneStartWrap_refines (B : Int) (hB : B ≤ 2 ^ 61) (s purge : W)
    (hs : -B ≤ s.toInt ∧ s.toInt ≤ B) (hp : purge.toInt ≤ B) :
    (zoneStartWrap s purge).toInt = s.toInt - max purge.toInt 1 + 1 := by
  have h0 : (s - smax purge 1#64).toInt = s.toInt - max purge.toInt 1 := by
    rw [toInt_sub_of_range, smax_toInt, toInt_one]; rw [smax_toInt, toInt_one]; omega
  unfold zoneStartWrap
  rw [toInt_add_of_range, h0, toInt_one]; rw [h0, toInt_one]; omega

theorem zoneEndWrap_refines (B : Int) (hB : B ≤ 2 ^ 61) (e emb : W)
    (he : -B ≤ e.toInt ∧ e.toInt ≤ B) (hm : -B ≤ emb.toInt ∧ emb.toInt ≤ B) :
    (zoneEndWrap e emb).toInt = e.toInt + emb.toInt := by
  unfold zoneEndWrap; rw [toInt_add_of_range]; omega

/-- `i64::saturating_add` as a clamp of the exact sum. -/
theorem satAdd_toInt (a k : W) :
    (satAdd a k).toInt = max (-2 ^ 63) (min (2 ^ 63 - 1) (a.toInt + k.toInt)) := by
  have h := satAdd_spec a k
  have ha := toInt_range a
  have hk := toInt_range k
  have hr := toInt_range (satAdd a k)
  have e1 : ((satAdd a k).signExtend 66).toInt = (satAdd a k).toInt :=
    BitVec.toInt_signExtend_of_le (by decide)
  have e2 : (a.signExtend 66 + k.signExtend 66).toInt = a.toInt + k.toInt := by
    rw [BitVec.toInt_add, BitVec.toInt_signExtend_of_le (by decide),
      BitVec.toInt_signExtend_of_le (by decide)]
    exact Int.bmod_eq_of_le (by omega) (by omega)
  have e3 : ((0x7fffffffffffffff#64 : W).signExtend 66).toInt = 2 ^ 63 - 1 := by decide
  have e4 : (SIGN.signExtend 66).toInt = -2 ^ 63 := by decide
  rw [← e1, h]
  simp only []
  split
  · rename_i c; rw [BitVec.slt_iff_toInt_lt, e3, e2] at c; rw [e3]; omega
  · rename_i c
    rw [BitVec.slt_iff_toInt_lt, e3, e2] at c
    split
    · rename_i d; rw [BitVec.slt_iff_toInt_lt, e4, e2] at d; rw [e4]; omega
    · rename_i d; rw [BitVec.slt_iff_toInt_lt, e4, e2] at d; rw [e2]; omega

theorem satSub_spec (a k : W) :
    (satSub a k).signExtend 66 =
      (let s : BitVec 66 := a.signExtend 66 - k.signExtend 66
       let hi : BitVec 66 := (0x7fffffffffffffff#64).signExtend 66
       let lo : BitVec 66 := SIGN.signExtend 66
       if hi.slt s then hi else if s.slt lo then lo else s) := by
  simp only [satSub, SIGN]
  rcsr_bv

/-- `i64::saturating_sub` as a clamp of the exact difference. -/
theorem satSub_toInt (a k : W) :
    (satSub a k).toInt = max (-2 ^ 63) (min (2 ^ 63 - 1) (a.toInt - k.toInt)) := by
  have h := satSub_spec a k
  have ha := toInt_range a
  have hk := toInt_range k
  have e1 : ((satSub a k).signExtend 66).toInt = (satSub a k).toInt :=
    BitVec.toInt_signExtend_of_le (by decide)
  have e2 : (a.signExtend 66 - k.signExtend 66).toInt = a.toInt - k.toInt := by
    rw [BitVec.toInt_sub, BitVec.toInt_signExtend_of_le (by decide),
      BitVec.toInt_signExtend_of_le (by decide)]
    exact Int.bmod_eq_of_le (by omega) (by omega)
  have e3 : ((0x7fffffffffffffff#64 : W).signExtend 66).toInt = 2 ^ 63 - 1 := by decide
  have e4 : (SIGN.signExtend 66).toInt = -2 ^ 63 := by decide
  rw [← e1, h]
  simp only []
  split
  · rename_i c; rw [BitVec.slt_iff_toInt_lt, e3, e2] at c; rw [e3]; omega
  · rename_i c
    rw [BitVec.slt_iff_toInt_lt, e3, e2] at c
    split
    · rename_i d; rw [BitVec.slt_iff_toInt_lt, e4, e2] at d; rw [e4]; omega
    · rename_i d; rw [BitVec.slt_iff_toInt_lt, e4, e2] at d; rw [e2]; omega

/-- Inside the bound the saturating zone start is the exact one. -/
theorem zoneStartSat_refines (B : Int) (hB : B ≤ 2 ^ 61) (s purge : W)
    (hs : -B ≤ s.toInt ∧ s.toInt ≤ B) (hp : purge.toInt ≤ B) :
    (zoneStartSat s purge).toInt = s.toInt - max purge.toInt 1 + 1 := by
  unfold zoneStartSat
  rw [satSub_toInt, p1m1_toInt]; omega

/-- Inside the bound the saturating zone end is the exact one. -/
theorem zoneEndSat_refines (B : Int) (hB : B ≤ 2 ^ 61) (e emb : W)
    (he : -B ≤ e.toInt ∧ e.toInt ≤ B) (hm : -B ≤ emb.toInt ∧ emb.toInt ≤ B) :
    (zoneEndSat e emb).toInt = e.toInt + emb.toInt := by
  unfold zoneEndSat; rw [satAdd_toInt]; omega

/-- The saturating zone end is the exact end clamped to `i64::MAX`, for every input with a
non-negative embargo: no stamp below `i64::MAX` is lost. -/
theorem zoneEndSat_clamp (e emb : W) (hm : 0 ≤ emb.toInt) :
    (zoneEndSat e emb).toInt = min (2 ^ 63 - 1) (e.toInt + emb.toInt) := by
  have := toInt_range e
  unfold zoneEndSat; rw [satAdd_toInt]; omega

/-- The saturating zone start is the exact start clamped to `i64::MIN`, for every input: the
subtraction `s - (purge.max(1) - 1)` has a non-negative subtrahend, so it never exceeds `i64::MAX`. -/
theorem zoneStartSat_clamp (s purge : W) :
    (zoneStartSat s purge).toInt = max (-2 ^ 63) (s.toInt - max purge.toInt 1 + 1) := by
  have := toInt_range s
  unfold zoneStartSat
  rw [satSub_toInt, p1m1_toInt]; omega

end Rcsr.Intervals
