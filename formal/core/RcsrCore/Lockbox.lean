import RcsrCore.Intervals

/-!
Model of the locked-range predicates of `rcsr-engine/src/lockbox.rs`: `Lockbox::check`,
`Lockbox::contains`, `Lockbox::reaches`, `Lockbox::first_reaching`, the refusal of `Guard::check` and
`Guard::check_times`, and the refusal of `Declared::check_times`.

A check is modelled by the Boolean "returns `Ok`". Messages and the coverage record `Overlap` are not
modelled; of `overlap_reaching` only "no stamp is counted" is used.

Limits: the model is written by hand on `Int`. It says nothing about which code paths call a guard.
-/

namespace Rcsr.Lockbox
open Rcsr.Intervals

/-- `Lockbox`: the half-open range `[start, end)`. The Rust field `end` is `stop` here. -/
structure Box where
  start : Int
  stop : Int

/-- `Lockbox::check(start, end)` returns `Ok`. -/
def Box.check (l : Box) (s e : Int) : Bool := !(decide (s < l.stop) && decide (e > l.start))

/-- `Lockbox::contains`. -/
def Box.contains (l : Box) (ts : Int) : Bool := decide (l.start ≤ ts) && decide (ts < l.stop)

/-- `Lockbox::reaches`: the formula of `intervals::reaches` on the range of the box. -/
def Box.reaches (l : Box) (t purge emb : Int) : Prop := Intervals.reaches l.start l.stop t purge emb

instance (l : Box) (t purge emb : Int) : Decidable (l.reaches t purge emb) := by
  unfold Box.reaches; infer_instance

/-- `Lockbox::first_reaching`. -/
def Box.firstReaching (l : Box) (purge : Int) : Int := l.start - max purge 1 + 1

/-- `Guard`: `boxes` is the list `Guard::boxes()` yields (the primary range, then the second one). -/
structure Guard where
  boxes : List Box
  allow : Bool
  reportOnly : Bool

/-- `Guard::check` returns `Ok`. -/
def Guard.check (g : Guard) (s e : Int) : Bool :=
  g.allow || g.reportOnly || g.boxes.all (fun l => l.check s e)

/-- `Guard::check_times` returns `Ok`. -/
def Guard.checkTimes (g : Guard) (ts : List Int) : Bool :=
  g.allow || g.reportOnly || g.boxes.all (fun l => ts.all (fun t => !l.contains t))

/-- `Declared`: the declared range, the optional second range and `embargo.seconds`. -/
structure Declared where
  lockbox : Box
  also : Option Box
  embargo : Int

def Declared.ranges (d : Declared) : List Box := d.lockbox :: d.also.toList

/-- `Declared::check_times` returns `Ok`. -/
def Declared.checkTimes (d : Declared) (ts : List Int) (purge : Int) : Bool :=
  d.ranges.all (fun l => ts.all (fun t => !decide (l.reaches t (max purge 0) (max d.embargo 0))))

/-! ### One range -/

theorem check_ok_iff (l : Box) (s e : Int) : l.check s e = true ↔ ¬ (s < l.stop ∧ e > l.start) := by
  simp only [Box.check, Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not]; omega

theorem contains_iff (l : Box) (ts : Int) : l.contains ts = true ↔ memIv (l.start, l.stop) ts := by
  simp [Box.contains, memIv]

/-- Two non-empty half-open ranges pass the overlap test exactly when they share a stamp. -/
theorem overlap_iff (s e ls le : Int) (hw : s < e) (hl : ls < le) :
    (s < le ∧ e > ls) ↔ ∃ t, s ≤ t ∧ t < e ∧ ls ≤ t ∧ t < le := by
  constructor
  · intro h; exact ⟨max s ls, by omega⟩
  · rintro ⟨t, h⟩; omega

/-- An accepted window holds no stamp of the range. No condition on the shapes. -/
theorem check_no_row_inside (l : Box) (s e : Int) (h : l.check s e = true) :
    ∀ t, s ≤ t → t < e → l.contains t = false := by
  intro t h1 h2
  rw [check_ok_iff] at h
  cases hc : l.contains t
  · rfl
  · rw [contains_iff] at hc; unfold memIv at hc; simp only at hc; omega

/-- For a non-empty window and a non-empty range the check refuses exactly the windows that hold a
stamp of the range. -/
theorem check_ok_iff_no_row_inside (l : Box) (s e : Int) (hw : s < e) (hl : l.start < l.stop) :
    l.check s e = true ↔ ∀ t, s ≤ t → t < e → l.contains t = false := by
  refine ⟨check_no_row_inside l s e, fun h => ?_⟩
  rw [check_ok_iff]
  intro hov
  obtain ⟨t, h1, h2, h3, h4⟩ := (overlap_iff s e l.start l.stop hw hl).mp hov
  have := h t h1 h2
  rw [Bool.eq_false_iff, Ne, contains_iff] at this
  exact this ⟨h3, h4⟩

theorem reaches_eq (l : Box) (t purge emb : Int) :
    l.reaches t purge emb ↔ Intervals.reaches l.start l.stop t purge emb := Iff.rfl

/-- The stamps that reach the range are exactly `[first_reaching(purge), end + max(embargo, 0))`. -/
theorem reaches_iff_firstReaching (l : Box) (t purge emb : Int) :
    l.reaches t purge emb ↔ l.firstReaching purge ≤ t ∧ t < l.stop + max emb 0 := by
  unfold Box.reaches Box.firstReaching; exact reaches_iff_zone _ _ _ _ _

/-- The stamp one second before `first_reaching` is kept. -/
theorem firstReaching_pred_not_reaches (l : Box) (purge emb : Int) :
    ¬ l.reaches (l.firstReaching purge - 1) purge emb := by
  rw [reaches_iff_firstReaching]; omega

theorem contains_imp_reaches (l : Box) (t purge emb : Int) (h : l.contains t = true) :
    l.reaches t purge emb := by
  rw [contains_iff] at h; exact inside_imp_reaches _ _ _ _ _ h

theorem reaches_zero_iff_contains (l : Box) (t : Int) : l.reaches t 0 0 ↔ l.contains t = true := by
  rw [contains_iff]; exact Intervals.reaches_zero_iff_contains _ _ _

/-! ### `Guard` -/

theorem guard_check_ok_iff (g : Guard) (s e : Int) (hg : g.allow = false ∧ g.reportOnly = false) :
    g.check s e = true ↔ ∀ l ∈ g.boxes, ¬ (s < l.stop ∧ e > l.start) := by
  simp only [Guard.check, hg.1, hg.2, Bool.false_or, List.all_eq_true, check_ok_iff]

theorem guard_check_no_row_inside (g : Guard) (s e : Int)
    (hg : g.allow = false ∧ g.reportOnly = false) (h : g.check s e = true) :
    ∀ t, s ≤ t → t < e → ∀ l ∈ g.boxes, l.contains t = false := by
  intro t h1 h2 l hl
  simp only [Guard.check, hg.1, hg.2, Bool.false_or, List.all_eq_true] at h
  exact check_no_row_inside l s e (h l hl) t h1 h2

theorem guard_checkTimes_ok_iff (g : Guard) (ts : List Int)
    (hg : g.allow = false ∧ g.reportOnly = false) :
    g.checkTimes ts = true ↔ ∀ t ∈ ts, ∀ l ∈ g.boxes, l.contains t = false := by
  simp only [Guard.checkTimes, hg.1, hg.2, Bool.false_or, List.all_eq_true, Bool.not_eq_true']
  exact ⟨fun h t ht l hl => h l hl t ht, fun h l hl t ht => h t ht l hl⟩

/-- With an override or in report-only mode the guard refuses nothing. -/
theorem guard_open (g : Guard) (s e : Int) (ts : List Int) (hg : g.allow = true ∨ g.reportOnly = true) :
    g.check s e = true ∧ g.checkTimes ts = true := by
  rcases hg with h | h <;> simp [Guard.check, Guard.checkTimes, h]

/-! ### `Declared` -/

theorem declared_checkTimes_ok_iff (d : Declared) (ts : List Int) (purge : Int) :
    d.checkTimes ts purge = true ↔
      ∀ t ∈ ts, ∀ l ∈ d.ranges, ¬ l.reaches t (max purge 0) (max d.embargo 0) := by
  simp only [Declared.checkTimes, List.all_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
  exact ⟨fun h t ht l hl => h l hl t ht, fun h l hl t ht => h t ht l hl⟩

/-- A row set that passes the declared-range check holds no stamp of a declared range. -/
theorem declared_checkTimes_no_row_inside (d : Declared) (ts : List Int) (purge : Int)
    (h : d.checkTimes ts purge = true) : ∀ t ∈ ts, ∀ l ∈ d.ranges, l.contains t = false := by
  intro t ht l hl
  have := (declared_checkTimes_ok_iff d ts purge).mp h t ht l hl
  cases hc : l.contains t
  · rfl
  · exact absurd (contains_imp_reaches l t _ _ hc) this

end Rcsr.Lockbox
