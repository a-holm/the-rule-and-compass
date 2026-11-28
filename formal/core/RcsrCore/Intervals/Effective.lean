import RcsrCore.Intervals

/-!
`IntervalSplit::role` and `IntervalSplit::effective` of `rcsr-engine/src/intervals.rs` are two
encodings of one assignment: a stamp has the role `r` exactly when it lies in `effective(r)`.

Limits: the theorems hold for the `Int` model of `RcsrCore/Intervals.lean`. They say nothing about
a list that breaks `WellFormed`, apart from the counterexample `validation_needs_disjoint`.
-/

namespace Rcsr.Intervals

/-- The shape a split must have for `role` and `effective` to agree. Test intervals may overlap the
other two roles. -/
def WellFormed (sp : Split) : Prop :=
  Chain sp.train ∧ Chain sp.validation ∧ Chain sp.test ∧
  (∀ t, ¬ (mem sp.train t ∧ mem sp.validation t)) ∧ 0 ≤ sp.purge ∧ 0 ≤ sp.embargo

/-! ### `role` as three Boolean conditions -/

theorem any_false_iff (l : List Iv) (t p m : Int) :
    l.any (fun x => decide (reaches x.1 x.2 t p m)) = false ↔ ∀ x ∈ l, ¬ reaches x.1 x.2 t p m := by
  simp [List.any_eq_false]

theorem role_test_iff (ins : List Iv → Int → Bool) (sp : Split) (t : Int) :
    role ins sp t = some .test ↔ ins sp.test t = true := by
  unfold role
  cases a : ins sp.test t <;> cases b : ins sp.train t <;> cases c : ins sp.validation t <;>
    cases d : sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) <;>
    cases e : sp.validation.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.purge)) <;>
    simp_all

theorem role_train_iff (ins : List Iv → Int → Bool) (sp : Split) (t : Int)
    (himp : ins sp.test t = true →
      sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = true) :
    role ins sp t = some .train ↔ ins sp.train t = true ∧
      sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = false ∧
      sp.validation.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.purge)) = false := by
  unfold role
  cases a : ins sp.test t <;> cases b : ins sp.train t <;> cases c : ins sp.validation t <;>
    cases d : sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) <;>
    cases e : sp.validation.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.purge)) <;>
    simp_all

theorem role_validation_iff (ins : List Iv → Int → Bool) (sp : Split) (t : Int)
    (himp : ins sp.test t = true →
      sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = true) :
    role ins sp t = some .validation ↔ ins sp.train t = false ∧ ins sp.validation t = true ∧
      sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = false := by
  unfold role
  cases a : ins sp.test t <;> cases b : ins sp.train t <;> cases c : ins sp.validation t <;>
    cases d : sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) <;>
    cases e : sp.validation.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.purge)) <;>
    simp_all

/-- A stamp inside a test interval reaches that interval. -/
theorem inside_test_imp_any (sp : Split) (hx : Chain sp.test) (t : Int) :
    inside sp.test t = true →
      sp.test.any (fun x => decide (reaches x.1 x.2 t sp.purge sp.embargo)) = true := by
  intro h
  obtain ⟨p, hp', hpt⟩ := (inside_iff_mem _ _ hx).mp h
  exact List.any_eq_true.mpr ⟨p, hp', by simpa using inside_imp_reaches _ _ _ _ _ hpt⟩

/-! ### `effective` and `role` are one predicate -/

theorem effective_train_iff_role (sp : Split) (hp : 0 ≤ sp.purge) (he : 0 ≤ sp.embargo)
    (ht : Chain sp.train) (hx : Chain sp.test) (t : Int) :
    mem (effective sp .train) t ↔ role inside sp t = some .train := by
  rw [mem_effective_train sp hp he, role_train_iff inside sp t (inside_test_imp_any sp hx t),
    inside_iff_mem _ _ ht, any_false_iff, any_false_iff]

theorem effective_validation_iff_role (sp : Split) (he : 0 ≤ sp.embargo)
    (hv : Chain sp.validation) (ht : Chain sp.train) (hx : Chain sp.test)
    (hd : ∀ t, ¬ (mem sp.train t ∧ mem sp.validation t)) (t : Int) :
    mem (effective sp .validation) t ↔ role inside sp t = some .validation := by
  rw [mem_effective_validation sp he, role_validation_iff inside sp t (inside_test_imp_any sp hx t),
    inside_iff_mem _ _ hv, any_false_iff, Bool.eq_false_iff, Ne, inside_iff_mem _ _ ht]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨fun h => hd t ⟨h, h1⟩, h1, h2⟩
  · rintro ⟨_, h1, h2⟩; exact ⟨h1, h2⟩

theorem effective_test_iff_role (sp : Split) (hx : Chain sp.test) (t : Int) :
    mem (effective sp .test) t ↔ role inside sp t = some .test := by
  rw [mem_effective_test, role_test_iff, inside_iff_mem _ _ hx]

/-- Without disjoint training and validation intervals the two encodings differ: stamp 60 is in
`effective(Validation)` and has no role. -/
def badSplit : Split :=
  { train := [(0, 100)], validation := [(50, 150)], test := [], purge := 0, embargo := 0 }

theorem validation_needs_disjoint :
    mem (effective badSplit .validation) 60 ∧ role inside badSplit 60 = none := by
  refine ⟨(mem_effective_validation badSplit (by decide) 60).mpr
    ⟨⟨(50, 150), by decide, by unfold memIv; omega⟩, by simp [badSplit]⟩, by decide⟩

/-- The kept training and validation lists share no stamp. No condition on the lists. -/
theorem effective_roles_disjoint (sp : Split) (hp : 0 ≤ sp.purge) (he : 0 ≤ sp.embargo) (t : Int) :
    ¬ (mem (effective sp .train) t ∧ mem (effective sp .validation) t) := by
  rintro ⟨h1, h2⟩
  obtain ⟨_, _, hv⟩ := (mem_effective_train sp hp he t).mp h1
  obtain ⟨⟨x, hx, hxt⟩, _⟩ := (mem_effective_validation sp he t).mp h2
  exact hv x hx (inside_imp_reaches _ _ _ _ _ hxt)

/-- No kept interval reaches a test interval: the guard `window_in_sample` answers false for every
non-empty test interval of the split. -/
theorem effective_clear_of_test (sp : Split) (hp : 0 ≤ sp.purge) (he : 0 ≤ sp.embargo) :
    ∀ x ∈ sp.test, x.1 < x.2 →
      windowInSample (effective sp .train ++ effective sp .validation) x.1 x.2 sp.purge sp.embargo
        = false := by
  intro x hx hne
  cases hw : windowInSample (effective sp .train ++ effective sp .validation) x.1 x.2 sp.purge sp.embargo
  · rfl
  · obtain ⟨p, hp', t, hpt, hr⟩ := windowInSample_complete _ _ _ _ _ hne hw
    rcases List.mem_append.mp hp' with h | h
    · exact absurd hr (((mem_effective_train sp hp he t).mp ⟨p, h, hpt⟩).2.1 x hx)
    · exact absurd hr (((mem_effective_validation sp he t).mp ⟨p, h, hpt⟩).2 x hx)

/-! ### `effective` returns a chain -/

/-- The pieces are in order, non-empty and inside `q`. -/
def Within (q : Iv) (ps : List Iv) : Prop :=
  ps.Pairwise (fun x y => x.2 ≤ y.1) ∧ ∀ r ∈ ps, q.1 ≤ r.1 ∧ r.1 < r.2 ∧ r.2 ≤ q.2

theorem cut1_within (z q : Iv) (hz : z.1 ≤ z.2) (hq : q.1 < q.2) : Within q (cut1 z q) := by
  unfold Within cut1
  by_cases h1 : z.2 ≤ q.1 ∨ z.1 ≥ q.2 <;> by_cases h2 : q.1 < z.1 <;> by_cases h3 : z.2 < q.2 <;>
    simp [h1, h2, h3] <;> omega

theorem cut_within (z q : Iv) (ps : List Iv) (hz : z.1 ≤ z.2) (h : Within q ps) :
    Within q (cut z ps) := by
  obtain ⟨hpw, hin⟩ := h
  refine ⟨List.pairwise_flatMap.mpr ⟨fun a ha => (cut1_within z a hz (hin a ha).2.1).1, ?_⟩, ?_⟩
  · refine hpw.imp_of_mem ?_
    intro a b ha hb hab x hx y hy
    have h1 := (cut1_within z a hz (hin a ha).2.1).2 x hx
    have h2 := (cut1_within z b hz (hin b hb).2.1).2 y hy
    omega
  · intro r hr
    obtain ⟨a, ha, hra⟩ := List.mem_flatMap.mp hr
    have h1 := (cut1_within z a hz (hin a ha).2.1).2 r hra
    have h2 := hin a ha
    omega

theorem cutAll_within (zs : List Iv) (q : Iv) (ps : List Iv) (hz : ∀ z ∈ zs, z.1 ≤ z.2)
    (h : Within q ps) : Within q (cutAll zs ps) := by
  induction zs generalizing ps with
  | nil => simpa [cutAll] using h
  | cons z zs ih =>
    have := ih (cut z ps) (fun y hy => hz y (List.mem_cons_of_mem _ hy))
      (cut_within z q ps (hz z List.mem_cons_self) h)
    simpa [cutAll] using this

theorem within_self (p : Iv) (hp : p.1 < p.2) : Within p [p] := by
  refine ⟨List.pairwise_singleton _ _, ?_⟩
  intro r hr; simp at hr; subst hr; omega

/-- Cutting every interval of a chain by zones that are not inverted gives a chain. -/
theorem chain_flatMap_cutAll (zs own : List Iv) (hz : ∀ z ∈ zs, z.1 ≤ z.2) (ho : Chain own) :
    Chain (own.flatMap (fun p => cutAll zs [p])) := by
  obtain ⟨hpw, hne⟩ := ho
  have hw : ∀ p ∈ own, Within p (cutAll zs [p]) :=
    fun p hp => cutAll_within zs p [p] hz (within_self p (hne p hp))
  refine ⟨List.pairwise_flatMap.mpr ⟨fun a ha => (hw a ha).1, ?_⟩, ?_⟩
  · refine hpw.imp_of_mem ?_
    intro a b ha hb hab x hx y hy
    have h1 := (hw a ha).2 x hx
    have h2 := (hw b hb).2 y hy
    omega
  · intro r hr
    obtain ⟨a, ha, hra⟩ := List.mem_flatMap.mp hr
    exact ((hw a ha).2 r hra).2.1

/-- A chain is sorted in the order of `sort_unstable`, so the final sort of `effective` returns it
unchanged. -/
theorem mergeSort_chain (v : List Iv) (h : Chain v) : v.mergeSort lexLe = v := by
  refine List.mergeSort_of_pairwise (h.1.imp_of_mem ?_)
  intro a b ha _ hab
  have := h.2 a ha
  simp only [lexLe, decide_eq_true_eq]; omega

theorem zonesTest_ordered (sp : Split) (he : 0 ≤ sp.embargo) (hne : ∀ x ∈ sp.test, x.1 < x.2) :
    ∀ z ∈ zonesTest sp, z.1 < z.2 := by
  simp only [zonesTest, List.mem_map]
  rintro z ⟨x, hx, rfl⟩
  have := hne x hx
  simp only; omega

theorem zonesVal_ordered (sp : Split) (hp : 0 ≤ sp.purge) (hne : ∀ x ∈ sp.validation, x.1 < x.2) :
    ∀ z ∈ zonesVal sp, z.1 < z.2 := by
  simp only [zonesVal, List.mem_map]
  rintro z ⟨x, hx, rfl⟩
  have := hne x hx
  simp only; omega

theorem zones_ordered (sp : Split) (hp : 0 ≤ sp.purge) (he : 0 ≤ sp.embargo)
    (hx : ∀ x ∈ sp.test, x.1 < x.2) (hv : ∀ x ∈ sp.validation, x.1 < x.2) (r : Role) :
    ∀ z ∈ zones sp r, z.1 < z.2 := by
  intro z hz
  cases r with
  | train =>
    simp only [zones, List.mem_mergeSort, List.mem_append] at hz
    exact hz.elim (zonesTest_ordered sp he hx z) (zonesVal_ordered sp hp hv z)
  | validation =>
    simp only [zones, List.mem_mergeSort] at hz
    exact zonesTest_ordered sp he hx z hz
  | test => simp [zones] at hz

/-- The list `effective` returns before its final sort. -/
def pieces (sp : Split) (r : Role) (own : List Iv) : List Iv :=
  own.flatMap (fun p => cutAll (zones sp r) [p])

theorem effective_train_eq (sp : Split) (h : WellFormed sp) :
    effective sp .train = pieces sp .train sp.train := by
  obtain ⟨ht, hv, hx, _, hp, he⟩ := h
  exact mergeSort_chain _ (chain_flatMap_cutAll _ _ (fun z hz => Int.le_of_lt (zones_ordered sp hp he hx.2 hv.2 .train z hz)) ht)

theorem effective_validation_eq (sp : Split) (h : WellFormed sp) :
    effective sp .validation = pieces sp .validation sp.validation := by
  obtain ⟨ht, hv, hx, _, hp, he⟩ := h
  exact mergeSort_chain _ (chain_flatMap_cutAll _ _ (fun z hz => Int.le_of_lt (zones_ordered sp hp he hx.2 hv.2 .validation z hz)) hv)

/-- `effective` returns sorted, pairwise disjoint, non-empty intervals. -/
theorem effective_chain (sp : Split) (h : WellFormed sp) (r : Role) : Chain (effective sp r) := by
  have h' := h
  obtain ⟨ht, hv, hx, _, hp, he⟩ := h
  cases r with
  | train =>
    rw [effective_train_eq sp h']
    exact chain_flatMap_cutAll _ _ (fun z hz => Int.le_of_lt (zones_ordered sp hp he hx.2 hv.2 .train z hz)) ht
  | validation =>
    rw [effective_validation_eq sp h']
    exact chain_flatMap_cutAll _ _ (fun z hz => Int.le_of_lt (zones_ordered sp hp he hx.2 hv.2 .validation z hz)) hv
  | test => exact hx

/-- The form the property test `interval_split_purges_and_embargoes_every_edge` samples: the binary
search on the kept list answers the role. -/
theorem effective_inside_iff_role (sp : Split) (h : WellFormed sp) (r : Role) (t : Int) :
    inside (effective sp r) t = true ↔ role inside sp t = some r := by
  rw [inside_iff_mem _ _ (effective_chain sp h r)]
  obtain ⟨ht, hv, hx, hd, hp, he⟩ := h
  cases r with
  | train => exact effective_train_iff_role sp hp he ht hx t
  | validation => exact effective_validation_iff_role sp he hv ht hx hd t
  | test => exact effective_test_iff_role sp hx t

/-! ### What the table check establishes -/

/-- Each interval ends at or before the start of the next one: the test of `parse_table` on the
merged, sorted list of all three roles. -/
def Adjacent : List Iv → Prop
  | [] => True
  | [_] => True
  | x :: y :: rest => x.2 ≤ y.1 ∧ Adjacent (y :: rest)

/-- The adjacent test implies the pairwise order only because no interval is empty. -/
theorem chain_of_adjacent (v : List Iv) (hne : ∀ p ∈ v, p.1 < p.2) (ha : Adjacent v) : Chain v := by
  refine ⟨?_, hne⟩
  induction v with
  | nil => exact List.Pairwise.nil
  | cons x rest ih =>
    cases rest with
    | nil => exact List.pairwise_singleton _ _
    | cons y ys =>
      obtain ⟨hxy, hrest⟩ := ha
      have ih' := ih (fun p hp => hne p (List.mem_cons_of_mem _ hp)) hrest
      refine List.pairwise_cons.mpr ⟨?_, ih'⟩
      intro z hz
      rcases List.mem_cons.mp hz with rfl | hz
      · exact hxy
      · have h1 := (List.pairwise_cons.mp ih').1 z hz
        have h2 := hne y (List.mem_cons_of_mem _ List.mem_cons_self)
        omega

/-- Sorted by start and pairwise disjoint is a chain. -/
theorem chain_of_sorted_disjoint (v : List Iv) (hne : ∀ p ∈ v, p.1 < p.2)
    (hs : v.Pairwise (fun x y => x.1 ≤ y.1))
    (hd : v.Pairwise (fun x y => x.2 ≤ y.1 ∨ y.2 ≤ x.1)) : Chain v := by
  refine ⟨?_, hne⟩
  have hboth := hs.and hd
  refine hboth.imp_of_mem ?_
  intro a b _ hb h
  have := hne b hb
  omega

/-- A table that passes the checks of `parse_table` is well formed: the merged list of all intervals
passes the adjacent test, no interval is empty, each role is sorted by start, and purge and embargo are
not negative. -/
theorem wellFormed_of_table_check (sp : Split) (all : List Iv)
    (hperm : (sp.train ++ sp.validation ++ sp.test).Perm all)
    (hne : ∀ p ∈ all, p.1 < p.2) (hadj : Adjacent all)
    (hst : sp.train.Pairwise (fun x y => x.1 ≤ y.1))
    (hsv : sp.validation.Pairwise (fun x y => x.1 ≤ y.1))
    (hsx : sp.test.Pairwise (fun x y => x.1 ≤ y.1))
    (hp : 0 ≤ sp.purge) (he : 0 ≤ sp.embargo) : WellFormed sp := by
  have hall := (chain_of_adjacent all hne hadj).1
  have hdis : (sp.train ++ sp.validation ++ sp.test).Pairwise (fun x y => x.2 ≤ y.1 ∨ y.2 ≤ x.1) :=
    (List.Perm.pairwise_iff (fun h => h.symm) hperm).mpr (hall.imp Or.inl)
  have hne' : ∀ p ∈ sp.train ++ sp.validation ++ sp.test, p.1 < p.2 :=
    fun p hp => hne p (hperm.mem_iff.mp hp)
  obtain ⟨htv, hx, _⟩ := List.pairwise_append.mp hdis
  obtain ⟨ht, hv, hcross⟩ := List.pairwise_append.mp htv
  refine ⟨chain_of_sorted_disjoint _ (fun p hp => hne' p (by simp [hp])) hst ht,
    chain_of_sorted_disjoint _ (fun p hp => hne' p (by simp [hp])) hsv hv,
    chain_of_sorted_disjoint _ (fun p hp => hne' p (by simp [hp])) hsx hx, ?_, hp, he⟩
  rintro t ⟨⟨a, ha, hat⟩, ⟨b, hb, hbt⟩⟩
  have := hcross a ha b hb
  unfold memIv at hat hbt; omega

end Rcsr.Intervals
