import RcsrCore.Intervals.Effective

/-!
Trim invariance of `IntervalSplit::effective` and `IntervalSplit::role` of `rcsr-engine/src/intervals.rs`.

`cpcv::generate` writes the output of `effective` as the declared training and validation intervals
of a split table, and the search trims the table again. `trim` is that step: a second trim changes
nothing, and no row changes its role.

Limits: the theorems hold for the `Int` model and need `WellFormed`.
-/

namespace Rcsr.Intervals

/-- The split whose declared training and validation intervals are the kept ones. -/
def trim (sp : Split) : Split :=
  { sp with train := effective sp .train, validation := effective sp .validation }

theorem trim_wellformed (sp : Split) (h : WellFormed sp) : WellFormed (trim sp) :=
  ⟨effective_chain sp h .train, effective_chain sp h .validation, h.2.2.1,
    effective_roles_disjoint sp h.2.2.2.2.1 h.2.2.2.2.2, h.2.2.2.2.1, h.2.2.2.2.2⟩

theorem cut1_fixed (z q : Iv) (hz : z.1 < z.2) (hq : q.1 < q.2)
    (hd : ∀ t, memIv q t → ¬ memIv z t) : cut1 z q = [q] := by
  unfold cut1
  by_cases h1 : z.2 ≤ q.1 ∨ z.1 ≥ q.2
  · simp [h1]
  · exact absurd (show memIv z (max q.1 z.1) by unfold memIv; omega)
      (hd (max q.1 z.1) (by unfold memIv; omega))

theorem cutAll_fixed (zs : List Iv) (q : Iv) (hq : q.1 < q.2) (hz : ∀ z ∈ zs, z.1 < z.2)
    (hd : ∀ z ∈ zs, ∀ t, memIv q t → ¬ memIv z t) : cutAll zs [q] = [q] := by
  induction zs with
  | nil => rfl
  | cons z zs ih =>
    have h1 : cut z [q] = [q] := by
      simp [cut, cut1_fixed z q (hz z List.mem_cons_self) hq (hd z List.mem_cons_self)]
    have := ih (fun y hy => hz y (List.mem_cons_of_mem _ hy)) (fun y hy => hd y (List.mem_cons_of_mem _ hy))
    simpa [cutAll, h1] using this

theorem flatMap_fixed (l : List Iv) (f : Iv → List Iv) (h : ∀ p ∈ l, f p = [p]) : l.flatMap f = l := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp [List.flatMap_cons, h a List.mem_cons_self, ih (fun p hp => h p (List.mem_cons_of_mem _ hp))]

/-- Every kept validation interval lies inside a declared one. -/
theorem effective_validation_sub (sp : Split) (h : WellFormed sp) :
    ∀ q ∈ effective sp .validation, ∃ v ∈ sp.validation, v.1 ≤ q.1 ∧ q.2 ≤ v.2 := by
  intro q hq
  rw [effective_validation_eq sp h] at hq
  obtain ⟨ht, hv, hx, _, hp, he⟩ := h
  obtain ⟨v, hv', hqv⟩ := List.mem_flatMap.mp hq
  have := (cutAll_within (zones sp .validation) v [v]
    (fun z hz => Int.le_of_lt (zones_ordered sp hp he hx.2 hv.2 .validation z hz))
    (within_self v (hv.2 v hv'))).2 q hqv
  exact ⟨v, hv', this.1, this.2.2⟩

theorem effective_validation_idempotent (sp : Split) (h : WellFormed sp) :
    effective (trim sp) .validation = effective sp .validation := by
  have hw := trim_wellformed sp h
  rw [effective_validation_eq (trim sp) hw]
  refine flatMap_fixed _ _ (fun q hq => ?_)
  have hq' : q ∈ effective sp .validation := hq
  refine cutAll_fixed _ q ((effective_chain sp h .validation).2 q hq')
    (zones_ordered (trim sp) hw.2.2.2.2.1 hw.2.2.2.2.2 hw.2.2.1.2 hw.2.1.2 .validation) ?_
  intro z hz t hqt
  have hm := (mem_effective_validation sp h.2.2.2.2.2 t).mp ⟨q, hq', hqt⟩
  have hz' : z ∈ zonesTest sp := by simpa [zones, trim, zonesTest] using hz
  exact (mem_zonesTest sp h.2.2.2.2.2 t).mpr hm.2 z hz'

theorem effective_train_idempotent (sp : Split) (h : WellFormed sp) :
    effective (trim sp) .train = effective sp .train := by
  have hw := trim_wellformed sp h
  rw [effective_train_eq (trim sp) hw]
  refine flatMap_fixed _ _ (fun q hq => ?_)
  have hq' : q ∈ effective sp .train := hq
  refine cutAll_fixed _ q ((effective_chain sp h .train).2 q hq')
    (zones_ordered (trim sp) hw.2.2.2.2.1 hw.2.2.2.2.2 hw.2.2.1.2 hw.2.1.2 .train) ?_
  intro z hz t hqt
  have hm := (mem_effective_train sp h.2.2.2.2.1 h.2.2.2.2.2 t).mp ⟨q, hq', hqt⟩
  have hp := h.2.2.2.2.1
  simp only [zones, List.mem_mergeSort, List.mem_append] at hz
  rcases hz with hz | hz
  · have hz' : z ∈ zonesTest sp := by simpa [trim, zonesTest] using hz
    exact (mem_zonesTest sp h.2.2.2.2.2 t).mpr hm.2.1 z hz'
  · simp only [zonesVal, List.mem_map] at hz
    obtain ⟨v', hv', rfl⟩ := hz
    obtain ⟨v, hv, h1, h2⟩ := effective_validation_sub sp h v' hv'
    intro hzt
    refine hm.2.2 v hv ?_
    have hpe : (trim sp).purge = sp.purge := rfl
    unfold memIv at hzt; unfold reaches; simp only [hpe] at hzt; omega

/-- A second trim returns the same lists. -/
theorem effective_idempotent (sp : Split) (h : WellFormed sp) (r : Role) :
    effective (trim sp) r = effective sp r := by
  cases r with
  | train => exact effective_train_idempotent sp h
  | validation => exact effective_validation_idempotent sp h
  | test => rfl

theorem option_ext {a b : Option Role} (h : ∀ r, a = some r ↔ b = some r) : a = b := by
  cases a with
  | none =>
    cases b with
    | none => rfl
    | some y => exact absurd ((h y).mpr rfl) (by simp)
  | some x => exact ((h x).mp rfl).symm

/-- No stamp changes its role when the declared intervals are replaced by the kept ones. -/
theorem role_trim_invariant (sp : Split) (h : WellFormed sp) (t : Int) :
    role inside (trim sp) t = role inside sp t := by
  refine option_ext (fun r => ?_)
  rw [← effective_inside_iff_role (trim sp) (trim_wellformed sp h) r t,
    ← effective_inside_iff_role sp h r t, effective_idempotent sp h r]

end Rcsr.Intervals
