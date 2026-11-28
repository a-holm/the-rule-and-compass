/-!
Model of the integer core of `rcsr-engine/src/intervals.rs`: `reaches`, `interval_reaches`, `inside`,
`IntervalSplit::new`, `IntervalSplit::role`, `IntervalSplit::effective` and `window_in_sample`.

Stamps, purge and embargo are `Int`. An interval is a pair `(start, end)`, half-open.

Limits: the model is written by hand and computes on unbounded integers. `inside` models
`partition_point` only on a list that is partitioned by the predicate, so every theorem about `inside`
carries `Chain`. The text parser `stamp`, `parse_table`, the lookback rule and every `f64` value are
outside the model.
-/

namespace Rcsr.Intervals

/-- A half-open interval `[start, end)` in whole seconds. -/
abbrev Iv := Int × Int

/-- `reaches(start, end, t, purge, embargo)`. -/
def reaches (s e t purge emb : Int) : Prop := t + max purge 1 > s ∧ t < e + max emb 0

instance (s e t purge emb : Int) : Decidable (reaches s e t purge emb) := by
  unfold reaches; infer_instance

/-- `interval_reaches(start, end, a, b, purge, embargo)`. -/
def intervalReaches (s e a b purge emb : Int) : Prop :=
  a < b ∧ a < e + max emb 0 ∧ (b - 1) + max purge 1 > s

instance (s e a b purge emb : Int) : Decidable (intervalReaches s e a b purge emb) := by
  unfold intervalReaches; infer_instance

/-- The stamp `t` lies in the interval `p`. -/
def memIv (p : Iv) (t : Int) : Prop := p.1 ≤ t ∧ t < p.2

/-- Some interval of the list holds `t`. -/
def mem (v : List Iv) (t : Int) : Prop := ∃ p ∈ v, memIv p t

/-- Each interval ends at or before the start of every later one, and no interval is empty. -/
def Chain (v : List Iv) : Prop := v.Pairwise (fun x y => x.2 ≤ y.1) ∧ ∀ p ∈ v, p.1 < p.2

/-- `inside`: `partition_point(|&(a, _)| a <= t)` is the length of the prefix whose starts are at or
before `t`; `v[p - 1]` is the last interval of that prefix. -/
def inside (v : List Iv) (t : Int) : Bool :=
  match (v.takeWhile (fun x => decide (x.1 ≤ t))).getLast? with
  | some x => decide (t < x.2)
  | none => false

/-- One piece against one zone: the inner loop body of `effective`, and `cpcv::subtract`. -/
def cut1 (z p : Iv) : List Iv :=
  if z.2 ≤ p.1 ∨ z.1 ≥ p.2 then [p]
  else (if p.1 < z.1 then [(p.1, z.1)] else []) ++ (if z.2 < p.2 then [(z.2, p.2)] else [])

/-- One zone against every piece. -/
def cut (z : Iv) (ps : List Iv) : List Iv := ps.flatMap (cut1 z)

/-- Every zone in turn against the pieces. -/
def cutAll (zs ps : List Iv) : List Iv := zs.foldl (fun acc z => cut z acc) ps

inductive Role where
  | train | validation | test
deriving DecidableEq, Repr

/-- `IntervalSplit`, without the fields that no modelled function reads. -/
structure Split where
  train : List Iv
  validation : List Iv
  test : List Iv
  purge : Int
  embargo : Int

/-- `IntervalSplit::new`: purge and embargo are clamped at 0. -/
def Split.new (train validation test : List Iv) (purge embargo : Int) : Split :=
  ⟨train, validation, test, max purge 0, max embargo 0⟩

/-- `IntervalSplit::role`. `ins` stands for `inside`, so that the no-leak theorems hold for any
membership test. -/
def role (ins : List Iv → Int → Bool) (sp : Split) (t : Int) : Option Role :=
  if ins sp.test t then some .test
  else if ins sp.train t then
    if sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) then none
    else if sp.validation.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.purge)) then none
    else some .train
  else if ins sp.validation t then
    if sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) then none
    else some .validation
  else none

/-- The order of `sort_unstable` on `(i64, i64)`. -/
def lexLe (x y : Iv) : Bool := decide (x.1 < y.1 ∨ (x.1 = y.1 ∧ x.2 ≤ y.2))

/-- The zones around the test intervals: `(s - p1 + 1, e + embargo)`. -/
def zonesTest (sp : Split) : List Iv :=
  sp.test.map (fun x => (x.1 - max sp.purge 1 + 1, x.2 + sp.embargo))

/-- The zones around the validation intervals: `(s - p1 + 1, e + purge)`. -/
def zonesVal (sp : Split) : List Iv :=
  sp.validation.map (fun x => (x.1 - max sp.purge 1 + 1, x.2 + sp.purge))

/-- The sorted zone list of `effective` for a role. -/
def zones (sp : Split) : Role → List Iv
  | .train => (zonesTest sp ++ zonesVal sp).mergeSort lexLe
  | .validation => (zonesTest sp).mergeSort lexLe
  | .test => []

/-- `IntervalSplit::effective`. -/
def effective (sp : Split) : Role → List Iv
  | .test => sp.test
  | .train => (sp.train.flatMap (fun p => cutAll (zones sp .train) [p])).mergeSort lexLe
  | .validation => (sp.validation.flatMap (fun p => cutAll (zones sp .validation) [p])).mergeSort lexLe

/-- `window_in_sample`. -/
def windowInSample (kept : List Iv) (s e purge emb : Int) : Bool :=
  kept.any (fun p => decide (intervalReaches s e p.1 p.2 purge emb))

/-! ### The predicate -/

theorem reaches_iff_zone (s e t purge emb : Int) :
    reaches s e t purge emb ↔ s - max purge 1 + 1 ≤ t ∧ t < e + max emb 0 := by
  unfold reaches; omega

theorem inside_imp_reaches (s e t purge emb : Int) (h : s ≤ t ∧ t < e) : reaches s e t purge emb := by
  unfold reaches; omega

theorem reaches_zero_iff_contains (s e t : Int) : reaches s e t 0 0 ↔ s ≤ t ∧ t < e := by
  unfold reaches; omega

theorem intervalReaches_sound (s e a b purge emb t : Int) (hab : a ≤ t ∧ t < b)
    (hr : reaches s e t purge emb) : intervalReaches s e a b purge emb := by
  unfold reaches at hr; unfold intervalReaches; omega

theorem intervalReaches_complete (s e a b purge emb : Int) (hse : s < e)
    (h : intervalReaches s e a b purge emb) : ∃ t, a ≤ t ∧ t < b ∧ reaches s e t purge emb := by
  unfold intervalReaches at h
  refine ⟨max a (s - max purge 1 + 1), ?_⟩
  unfold reaches; omega

/-- For an empty guarded window the formula can answer true although no stamp reaches it. -/
theorem intervalReaches_empty_window :
    intervalReaches 10 5 0 20 0 0 ∧ ¬ ∃ t : Int, 0 ≤ t ∧ t < 20 ∧ reaches 10 5 t 0 0 := by
  refine ⟨by unfold intervalReaches; omega, ?_⟩
  rintro ⟨t, _, _, h⟩; unfold reaches at h; omega

theorem windowInSample_sound (kept : List Iv) (s e purge emb : Int)
    (h : windowInSample kept s e purge emb = false) :
    ∀ p ∈ kept, ∀ t, memIv p t → ¬ reaches s e t purge emb := by
  intro p hp t ht hr
  have := intervalReaches_sound s e p.1 p.2 purge emb t ht hr
  simp only [windowInSample, List.any_eq_false, decide_eq_true_eq] at h
  exact h p hp this

theorem windowInSample_complete (kept : List Iv) (s e purge emb : Int) (hse : s < e)
    (h : windowInSample kept s e purge emb = true) :
    ∃ p ∈ kept, ∃ t, memIv p t ∧ reaches s e t purge emb := by
  simp only [windowInSample, List.any_eq_true, decide_eq_true_eq] at h
  obtain ⟨p, hp, hr⟩ := h
  obtain ⟨t, h1, h2, h3⟩ := intervalReaches_complete s e p.1 p.2 purge emb hse hr
  exact ⟨p, hp, t, ⟨h1, h2⟩, h3⟩

theorem new_nonneg (train validation test : List Iv) (purge embargo : Int) :
    0 ≤ (Split.new train validation test purge embargo).purge ∧
    0 ≤ (Split.new train validation test purge embargo).embargo := by
  simp only [Split.new]; omega

/-! ### `inside` -/

theorem inside_iff_mem (v : List Iv) (t : Int) (h : Chain v) : inside v t = true ↔ mem v t := by
  obtain ⟨hc, hne⟩ := h
  induction v with
  | nil => simp [inside, mem]
  | cons x rest ih =>
    have hc' := List.pairwise_cons.mp hc
    have ih' := ih hc'.2 (fun p hp => hne p (List.mem_cons_of_mem _ hp))
    have hx := hne x List.mem_cons_self
    have memc : mem (x :: rest) t ↔ memIv x t ∨ mem rest t := by simp [mem]
    have hrest : t < x.2 → ¬ mem rest t := by
      rintro h ⟨y, hy, hyt⟩; have := hc'.1 y hy; unfold memIv at hyt; omega
    rw [memc]
    by_cases h1 : x.1 ≤ t
    · by_cases h2 : t < x.2
      · have htw : rest.takeWhile (fun y => decide (y.1 ≤ t)) = [] := by
          cases rest with
          | nil => rfl
          | cons y ys =>
            have := hc'.1 y List.mem_cons_self
            simp [List.takeWhile_cons]; omega
        simp [inside, h1, htw, h2, memIv]
      · have hx' : ¬ memIv x t := by unfold memIv; omega
        simp only [hx', false_or]
        rw [← ih']
        simp only [inside, List.takeWhile_cons, h1, decide_true, ite_true, List.getLast?_cons]
        cases hl : (rest.takeWhile (fun y => decide (y.1 ≤ t))).getLast? with
        | none => simp [h2]
        | some y => simp
    · have hx' : ¬ memIv x t := by unfold memIv; omega
      have : ¬ mem rest t := hrest (by omega)
      simp [inside, h1, hx', this]

/-! ### The cut -/

theorem mem_cut1 (z p : Iv) (t : Int) : mem (cut1 z p) t ↔ memIv p t ∧ ¬ memIv z t := by
  unfold cut1 mem memIv
  by_cases h1 : z.2 ≤ p.1 ∨ z.1 ≥ p.2 <;> by_cases h2 : p.1 < z.1 <;> by_cases h3 : z.2 < p.2 <;>
    simp [h1, h2, h3] <;> omega

theorem mem_cut (z : Iv) (ps : List Iv) (t : Int) : mem (cut z ps) t ↔ mem ps t ∧ ¬ memIv z t := by
  constructor
  · rintro ⟨q, hq, hqt⟩
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.mp hq
    have := (mem_cut1 z p t).mp ⟨q, hqp, hqt⟩
    exact ⟨⟨p, hp, this.1⟩, this.2⟩
  · rintro ⟨⟨p, hp, hpt⟩, hz⟩
    obtain ⟨q, hq, hqt⟩ := (mem_cut1 z p t).mpr ⟨hpt, hz⟩
    exact ⟨q, List.mem_flatMap.mpr ⟨p, hp, hq⟩, hqt⟩

theorem mem_cutAll (zs ps : List Iv) (t : Int) :
    mem (cutAll zs ps) t ↔ mem ps t ∧ ∀ z ∈ zs, ¬ memIv z t := by
  induction zs generalizing ps with
  | nil => simp [cutAll]
  | cons z zs ih =>
    have := ih (cut z ps)
    simp only [cutAll, List.foldl_cons] at this ⊢
    rw [this, mem_cut]
    simp only [List.mem_cons, forall_eq_or_imp]
    constructor
    · rintro ⟨⟨a, b⟩, c⟩; exact ⟨a, b, c⟩
    · rintro ⟨a, b, c⟩; exact ⟨⟨a, b⟩, c⟩

theorem mem_perm {a b : List Iv} (h : a.Perm b) (t : Int) : mem a t ↔ mem b t := by
  unfold mem; constructor <;> rintro ⟨p, hp, hpt⟩
  · exact ⟨p, h.mem_iff.mp hp, hpt⟩
  · exact ⟨p, h.mem_iff.mpr hp, hpt⟩

theorem mem_flatMap_cutAll (zs own : List Iv) (t : Int) :
    mem (own.flatMap (fun p => cutAll zs [p])) t ↔ mem own t ∧ ∀ z ∈ zs, ¬ memIv z t := by
  constructor
  · rintro ⟨q, hq, hqt⟩
    obtain ⟨p, hp, hqp⟩ := List.mem_flatMap.mp hq
    have := (mem_cutAll zs [p] t).mp ⟨q, hqp, hqt⟩
    obtain ⟨⟨p', hp', hp't⟩, hz⟩ := this
    simp at hp'; subst hp'
    exact ⟨⟨p', hp, hp't⟩, hz⟩
  · rintro ⟨⟨p, hp, hpt⟩, hz⟩
    obtain ⟨q, hq, hqt⟩ := (mem_cutAll zs [p] t).mpr ⟨⟨p, by simp, hpt⟩, hz⟩
    exact ⟨q, List.mem_flatMap.mpr ⟨p, hp, hq⟩, hqt⟩

/-! ### No leak -/

/-- A row with the role train or validation reaches no test interval. No condition on the lists, on
purge or embargo, or on `ins`. -/
theorem role_no_leak (ins : List Iv → Int → Bool) (sp : Split) (t : Int)
    (h : role ins sp t = some .train ∨ role ins sp t = some .validation) :
    ∀ x ∈ sp.test, ¬ reaches x.1 x.2 t sp.purge sp.embargo := by
  intro x hx hr
  have hany : sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = true :=
    List.any_eq_true.mpr ⟨x, hx, by simpa using hr⟩
  unfold role at h
  simp [hany] at h

/-- A training row reaches no validation interval, with the purge on both sides. -/
theorem role_train_clear_of_validation (ins : List Iv → Int → Bool) (sp : Split) (t : Int)
    (h : role ins sp t = some .train) :
    ∀ x ∈ sp.validation, ¬ reaches x.1 x.2 t sp.purge sp.purge := by
  intro x hx hr
  have hany : sp.validation.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.purge)) = true :=
    List.any_eq_true.mpr ⟨x, hx, by simpa using hr⟩
  unfold role at h
  by_cases a : ins sp.test t = true <;> by_cases b : ins sp.train t = true <;>
    by_cases c : ins sp.validation t = true <;>
    by_cases d : sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = true <;>
    simp [a, b, c, d, hany] at h

/-! ### Membership in `effective` -/

theorem mem_zonesTest (sp : Split) (he : 0 ≤ sp.embargo) (t : Int) :
    (∀ z ∈ zonesTest sp, ¬ memIv z t) ↔ ∀ x ∈ sp.test, ¬ reaches x.1 x.2 t sp.purge sp.embargo := by
  simp only [zonesTest, List.mem_map]
  constructor
  · intro hz x hx hr
    exact hz _ ⟨x, hx, rfl⟩ (by unfold memIv; unfold reaches at hr; simp only; omega)
  · rintro h z ⟨x, hx, rfl⟩ hm
    exact h x hx (by unfold memIv at hm; unfold reaches; simp only at hm; omega)

theorem mem_zonesVal (sp : Split) (hp : 0 ≤ sp.purge) (t : Int) :
    (∀ z ∈ zonesVal sp, ¬ memIv z t) ↔ ∀ x ∈ sp.validation, ¬ reaches x.1 x.2 t sp.purge sp.purge := by
  simp only [zonesVal, List.mem_map]
  constructor
  · intro hz x hx hr
    exact hz _ ⟨x, hx, rfl⟩ (by unfold memIv; unfold reaches at hr; simp only; omega)
  · rintro h z ⟨x, hx, rfl⟩ hm
    exact h x hx (by unfold memIv at hm; unfold reaches; simp only at hm; omega)

/-- Membership in `effective(Train)`, for arbitrary lists. -/
theorem mem_effective_train (sp : Split) (hp : 0 ≤ sp.purge) (he : 0 ≤ sp.embargo) (t : Int) :
    mem (effective sp .train) t ↔
      mem sp.train t ∧ (∀ x ∈ sp.test, ¬ reaches x.1 x.2 t sp.purge sp.embargo) ∧
      (∀ x ∈ sp.validation, ¬ reaches x.1 x.2 t sp.purge sp.purge) := by
  unfold effective
  rw [mem_perm (List.mergeSort_perm _ _) t, mem_flatMap_cutAll, ← mem_zonesTest sp he t,
    ← mem_zonesVal sp hp t]
  simp only [zones, List.mem_mergeSort, List.mem_append]
  constructor
  · rintro ⟨h, hz⟩; exact ⟨h, fun z hz' => hz z (Or.inl hz'), fun z hz' => hz z (Or.inr hz')⟩
  · rintro ⟨h, h1, h2⟩; exact ⟨h, fun z hz => hz.elim (h1 z) (h2 z)⟩

/-- Membership in `effective(Validation)`, for arbitrary lists. -/
theorem mem_effective_validation (sp : Split) (he : 0 ≤ sp.embargo) (t : Int) :
    mem (effective sp .validation) t ↔
      mem sp.validation t ∧ ∀ x ∈ sp.test, ¬ reaches x.1 x.2 t sp.purge sp.embargo := by
  unfold effective
  rw [mem_perm (List.mergeSort_perm _ _) t, mem_flatMap_cutAll, ← mem_zonesTest sp he t]
  simp only [zones, List.mem_mergeSort]

theorem mem_effective_test (sp : Split) (t : Int) : mem (effective sp .test) t ↔ mem sp.test t := by
  simp [effective]

/-! ### The index of the interval that holds a row -/

/-- `partition_point(|&(a, _)| a <= t)` on a list sorted by start. -/
def ppoint (v : List Iv) (t : Int) : Nat := (v.takeWhile (fun x => decide (x.1 ≤ t))).length

theorem takeWhile_all (p : Iv → Bool) (l : List Iv) : ∀ x ∈ l.takeWhile p, p x = true := by
  induction l with
  | nil => intro x hx; simp at hx
  | cons a l ih =>
    intro x hx
    by_cases ha : p a = true
    · simp only [List.takeWhile_cons, ha, ite_true, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact ha
      · exact ih x hx
    · simp [ha] at hx

theorem getElem?_takeWhile (p : Iv → Bool) (l : List Iv) (i : Nat) (h : i < (l.takeWhile p).length) :
    l[i]? = (l.takeWhile p)[i]? := by
  have := List.getElem?_append_left (l₁ := l.takeWhile p) (l₂ := l.dropWhile p) h
  rwa [List.takeWhile_append_dropWhile] at this

/-- When `inside` answers true, `partition_point(..) - 1` does not underflow and names an interval
that holds the stamp: the closure `at` of `interval_split` in `data.rs`. -/
theorem at_index_valid (v : List Iv) (t : Int) (h : inside v t = true) :
    1 ≤ ppoint v t ∧ ∃ x, v[ppoint v t - 1]? = some x ∧ memIv x t := by
  unfold inside at h
  cases hl : (v.takeWhile (fun x => decide (x.1 ≤ t))).getLast? with
  | none => rw [hl] at h; exact absurd h (by simp)
  | some x =>
    rw [hl] at h
    have hx2 : t < x.2 := by simpa using h
    have hmem : x ∈ v.takeWhile (fun x => decide (x.1 ≤ t)) := List.mem_of_getLast? hl
    have hx1 : x.1 ≤ t := by simpa using takeWhile_all _ v x hmem
    have hne : v.takeWhile (fun x => decide (x.1 ≤ t)) ≠ [] := List.ne_nil_of_mem hmem
    have hpos : 0 < ppoint v t := List.length_pos_iff.mpr hne
    refine ⟨hpos, x, ?_, hx1, hx2⟩
    have hget : (v.takeWhile (fun x => decide (x.1 ≤ t)))[ppoint v t - 1]? = some x := by
      rw [← hl, List.getLast?_eq_getElem?]; rfl
    rw [getElem?_takeWhile (fun x => decide (x.1 ≤ t)) v (ppoint v t - 1)
      (by unfold ppoint at hpos ⊢; omega)]
    exact hget

end Rcsr.Intervals
