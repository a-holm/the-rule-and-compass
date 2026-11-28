import RcsrCore.Pareto.Order

/-!
Model of `Archive::try_insert_view`, `Archive::prune` and `Archive::admits` in `rcsr-engine/src/pareto.rs`.

Epsilon-dominance (`Archive::dominates`) and the duplicate test are parameters `dom` and `near` with no
interpretation: `dom a b` stands for `dominates(a.error, a.cost, b.error, b.cost)` and `near it s` for
`it.cost == s.cost && (it.error - s.error).abs() < 1e-12`. The first test of the function is the parameter `reject`,
so that one set of theorems covers the code before and after #1466 (NaN cost).

Limits: `sort_by` is modelled by `List.mergeSort`, `dedup_by` by `dedupBy`, the `while` loop of `prune` by `dropMid`
with a fuel argument. The tree of a member is not modelled. The tie to the Rust code is the vector check.
-/

namespace Rcsr.Pareto
open Rcsr.Prim

section Generic
variable {α : Type}

/-- `Vec::dedup_by` after the first element. `prev` is the last element kept; `eq cur prev` removes `cur`. -/
def dedupGo (eq : α → α → Bool) (prev : α) : List α → List α
  | [] => []
  | s :: t => if eq s prev then dedupGo eq prev t else s :: dedupGo eq s t

/-- `Vec::dedup_by`: keeps the first element of each run. -/
def dedupBy (eq : α → α → Bool) : List α → List α
  | [] => []
  | s :: t => s :: dedupGo eq s t

/-- `while items.len() > cap { items.remove(items.len() / 2) }`, at most `fuel` rounds. -/
def dropMid (cap : Nat) : Nat → List α → List α
  | 0, xs => xs
  | fuel + 1, xs => if xs.length > cap then dropMid cap fuel (xs.eraseIdx (xs.length / 2)) else xs

theorem dedupGo_sublist (eq : α → α → Bool) (p : α) (xs : List α) : (dedupGo eq p xs).Sublist xs := by
  induction xs generalizing p with
  | nil => exact List.Sublist.slnil
  | cons s t ih =>
    simp only [dedupGo]
    split
    · exact (ih p).cons s
    · exact (ih s).cons_cons s

theorem dedupBy_sublist (eq : α → α → Bool) (xs : List α) : (dedupBy eq xs).Sublist xs := by
  cases xs with
  | nil => exact List.Sublist.slnil
  | cons s t => exact (dedupGo_sublist eq s t).cons_cons s

theorem dropMid_sublist (cap fuel : Nat) (xs : List α) : (dropMid cap fuel xs).Sublist xs := by
  induction fuel generalizing xs with
  | zero => exact List.Sublist.refl xs
  | succ f ih =>
    simp only [dropMid]
    split
    · exact (ih _).trans (List.eraseIdx_sublist xs _)
    · exact List.Sublist.refl xs

/-- The loop stops inside the fuel: `xs.length - cap` rounds are enough. -/
theorem dropMid_length (cap fuel : Nat) (xs : List α) (h : xs.length ≤ cap + fuel) :
    (dropMid cap fuel xs).length ≤ cap := by
  induction fuel generalizing xs with
  | zero => simpa [dropMid] using h
  | succ f ih =>
    simp only [dropMid]
    split
    · rename_i hlen
      refine ih _ ?_
      rw [List.length_eraseIdx_of_lt (by omega)]; omega
    · omega

end Generic

/-- `Archive::prune`: sort by cost then error, one member per cost, then drop middle members down to the cap. -/
def prune (cap : Nat) (items : List Mem) : List Mem :=
  let d := dedupBy (fun a b => feq a.cost b.cost) (items.mergeSort pruneLe)
  dropMid cap d.length d

/-- The insertion skeleton shared by `Archive::try_insert_view` and `QdCell::insert`: first test, dominated test,
eviction, duplicate test, push, and `shrink` when the list is longer than `cap`. -/
def insertWith (reject : Mem → Bool) (dom near : Mem → Mem → Bool) (cap : Nat) (shrink : List Mem → List Mem)
    (items : List Mem) (s : Mem) : List Mem :=
  if reject s then items
  else if items.any (fun it => dom it s) then items
  else
    let kept := items.filter (fun it => !dom s it)
    if kept.any (fun it => near it s) then kept
    else if (kept ++ [s]).length > cap then shrink (kept ++ [s]) else kept ++ [s]

/-- The first test of `try_insert_view` at `ad228547`: a non-finite error. -/
def rejectNonFinite (s : Mem) : Bool := !isFinite s.err
/-- The first test after #1466: a non-finite error or a NaN cost. -/
def rejectNonFiniteOrNanCost (s : Mem) : Bool := !isFinite s.err || isNaN s.cost

/-- `Archive::try_insert_view` on the member list. -/
def tryInsertBy (reject : Mem → Bool) (dom near : Mem → Mem → Bool) (cap : Nat) : List Mem → Mem → List Mem :=
  insertWith reject dom near cap (prune cap)

/-- `Archive::admits`. -/
def admitsBy (reject : Mem → Bool) (dom : Mem → Mem → Bool) (items : List Mem) (s : Mem) : Bool :=
  !reject s && !items.any (fun it => dom it s)

/-! ### `prune` -/

theorem prune_subset (cap : Nat) (items : List Mem) (m : Mem) (h : m ∈ prune cap items) : m ∈ items := by
  unfold prune at h
  exact List.mem_mergeSort.mp ((dedupBy_sublist _ _).subset ((dropMid_sublist _ _ _).subset h))

theorem prune_length (cap : Nat) (items : List Mem) : (prune cap items).length ≤ cap := by
  unfold prune
  exact dropMid_length cap _ _ (by omega)

/-- `prune` keeps every symmetric pairwise property of the list. -/
theorem prune_pairwise (R : Mem → Mem → Prop) (hsym : ∀ {x y : Mem}, R x y → R y x) (cap : Nat) (items : List Mem)
    (h : items.Pairwise R) : (prune cap items).Pairwise R := by
  unfold prune
  refine List.Pairwise.sublist ((dropMid_sublist _ _ _).trans (dedupBy_sublist _ _)) ?_
  exact (List.mergeSort_perm items pruneLe).symm.pairwise h hsym

/-! ### The insertion skeleton -/

section Insert
variable (reject : Mem → Bool) (dom near : Mem → Mem → Bool) (cap : Nat) (shrink : List Mem → List Mem)
  (items : List Mem) (s : Mem)

/-- A rejected candidate changes nothing. -/
theorem insertWith_reject (h : reject s = true) : insertWith reject dom near cap shrink items s = items := by
  simp [insertWith, h]

/-- `admits` is the first two tests of the insertion: a candidate that is not admitted changes nothing. -/
theorem insertWith_of_not_admits (h : admitsBy reject dom items s = false) :
    insertWith reject dom near cap shrink items s = items := by
  unfold admitsBy at h
  unfold insertWith
  cases hr : reject s
  · simp only [hr, Bool.not_false, Bool.true_and, Bool.not_eq_false'] at h
    simp [h]
  · simp

/-- Every member of the result was a member before or is the candidate. -/
theorem insertWith_subset (hsub : ∀ l m, m ∈ shrink l → m ∈ l) (m : Mem)
    (h : m ∈ insertWith reject dom near cap shrink items s) : m ∈ items ∨ m = s := by
  unfold insertWith at h
  split at h
  · exact Or.inl h
  · split at h
    · exact Or.inl h
    · simp only at h
      split at h
      · exact Or.inl (List.mem_filter.mp h).1
      · have hm : m ∈ items.filter (fun it => !dom s it) ++ [s] := by
          split at h
          · exact hsub _ _ h
          · exact h
        rcases List.mem_append.mp hm with h1 | h1
        · exact Or.inl (List.mem_filter.mp h1).1
        · exact Or.inr (by simpa using h1)

/-- A list that is within the cap stays within the cap. -/
theorem insertWith_length (hlen : ∀ l, (shrink l).length ≤ cap) (h : items.length ≤ cap) :
    (insertWith reject dom near cap shrink items s).length ≤ cap := by
  unfold insertWith
  have hf : (items.filter (fun it => !dom s it)).length ≤ items.length := List.filter_sublist.length_le
  split
  · exact h
  · split
    · exact h
    · simp only
      split
      · omega
      · split
        · exact hlen _
        · omega

/-- A property that holds for the members and for every candidate that is not rejected holds for the result. -/
theorem insertWith_forall (hsub : ∀ l m, m ∈ shrink l → m ∈ l) (P : Mem → Prop) (hi : ∀ m, m ∈ items → P m)
    (hs : reject s = false → P s) (m : Mem) (h : m ∈ insertWith reject dom near cap shrink items s) : P m := by
  cases hr : reject s
  · rcases insertWith_subset reject dom near cap shrink items s hsub m h with h1 | h1
    · exact hi m h1
    · exact h1 ▸ hs hr
  · rw [insertWith_reject reject dom near cap shrink items s hr] at h; exact hi m h

/-- No member dominates another, before and after, for every dominance predicate. -/
theorem insertWith_antichain
    (hpw : ∀ l, l.Pairwise (fun a b => dom a b = false ∧ dom b a = false) →
      (shrink l).Pairwise (fun a b => dom a b = false ∧ dom b a = false))
    (h : items.Pairwise (fun a b => dom a b = false ∧ dom b a = false)) :
    (insertWith reject dom near cap shrink items s).Pairwise (fun a b => dom a b = false ∧ dom b a = false) := by
  unfold insertWith
  split
  · exact h
  · split
    · exact h
    · rename_i hany
      have hk : (items.filter (fun it => !dom s it)).Pairwise (fun a b => dom a b = false ∧ dom b a = false) :=
        List.Pairwise.sublist List.filter_sublist h
      have hp : (items.filter (fun it => !dom s it) ++ [s]).Pairwise
          (fun a b => dom a b = false ∧ dom b a = false) := by
        refine List.pairwise_append.mpr ⟨hk, by simp, ?_⟩
        intro a ha b hb
        have hbs : b = s := by simpa using hb
        subst hbs
        have ha' := List.mem_filter.mp ha
        refine ⟨?_, by simpa using ha'.2⟩
        cases hd : dom a b
        · rfl
        · exact absurd (List.any_eq_true.mpr ⟨a, ha'.1, hd⟩) hany
      simp only
      split
      · exact hk
      · split
        · exact hpw _ hp
        · exact hp

end Insert

/-! ### `try_insert_view` -/

section TryInsert
variable (reject : Mem → Bool) (dom near : Mem → Mem → Bool) (cap : Nat) (items : List Mem) (s : Mem)

theorem tryInsertBy_reject (h : reject s = true) : tryInsertBy reject dom near cap items s = items :=
  insertWith_reject reject dom near cap _ items s h
theorem tryInsertBy_of_not_admits (h : admitsBy reject dom items s = false) :
    tryInsertBy reject dom near cap items s = items :=
  insertWith_of_not_admits reject dom near cap _ items s h
theorem tryInsertBy_subset (m : Mem) (h : m ∈ tryInsertBy reject dom near cap items s) : m ∈ items ∨ m = s :=
  insertWith_subset reject dom near cap _ items s (prune_subset cap) m h
theorem tryInsertBy_length (h : items.length ≤ cap) : (tryInsertBy reject dom near cap items s).length ≤ cap :=
  insertWith_length reject dom near cap _ items s (prune_length cap) h
theorem tryInsertBy_forall (P : Mem → Prop) (hi : ∀ m, m ∈ items → P m) (hs : reject s = false → P s)
    (m : Mem) (h : m ∈ tryInsertBy reject dom near cap items s) : P m :=
  insertWith_forall reject dom near cap _ items s (prune_subset cap) P hi hs m h
theorem tryInsertBy_antichain (h : items.Pairwise (fun a b => dom a b = false ∧ dom b a = false)) :
    (tryInsertBy reject dom near cap items s).Pairwise (fun a b => dom a b = false ∧ dom b a = false) :=
  insertWith_antichain reject dom near cap _ items s
    (fun l hl => prune_pairwise _ (fun hxy => ⟨hxy.2, hxy.1⟩) cap l hl) h

end TryInsert

/-- With either first test every member of an archive built by insertion has a finite error. -/
theorem tryInsertBy_finite (reject : Mem → Bool) (hrej : ∀ s, reject s = false → isFinite s.err = true)
    (dom near : Mem → Mem → Bool) (cap : Nat) (items : List Mem) (s : Mem)
    (h : ∀ m, m ∈ items → isFinite m.err = true) (m : Mem) (hm : m ∈ tryInsertBy reject dom near cap items s) :
    isFinite m.err = true :=
  tryInsertBy_forall reject dom near cap items s (fun m => isFinite m.err = true) h (hrej s) m hm

theorem rejectNonFinite_finite (s : Mem) (h : rejectNonFinite s = false) : isFinite s.err = true := by
  simpa [rejectNonFinite] using h
theorem rejectNonFiniteOrNanCost_finite (s : Mem) (h : rejectNonFiniteOrNanCost s = false) :
    isFinite s.err = true := by
  simp only [rejectNonFiniteOrNanCost, Bool.or_eq_false_iff] at h; simpa using h.1

/-- With the test of #1466 no member of an archive built by insertion has a NaN cost. -/
theorem tryInsertBy_no_nan_cost (dom near : Mem → Mem → Bool) (cap : Nat) (items : List Mem) (s : Mem)
    (h : ∀ m, m ∈ items → isNaN m.cost = false) (m : Mem)
    (hm : m ∈ tryInsertBy rejectNonFiniteOrNanCost dom near cap items s) : isNaN m.cost = false :=
  tryInsertBy_forall rejectNonFiniteOrNanCost dom near cap items s (fun m => isNaN m.cost = false) h
    (fun hr => by simp only [rejectNonFiniteOrNanCost, Bool.or_eq_false_iff] at hr; exact hr.2) m hm

/-- The test at `ad228547` lets a NaN cost in: the candidate enters an empty archive. -/
theorem rejectNonFinite_admits_nan_cost (dom near : Mem → Mem → Bool) :
    tryInsertBy rejectNonFinite dom near 4 [] ⟨0x7ff8000000000000#64, 0#64, 0#64⟩ = [⟨0x7ff8000000000000#64, 0#64, 0#64⟩] := by
  simp [tryInsertBy, insertWith, rejectNonFinite, isFinite, TOP]

/-! ### The instance that models the Rust text

The first test is `rejectNonFinite` until the NaN-cost guard of #1466 is in `try_insert_view`, then
`rejectNonFiniteOrNanCost`. -/

/-- `Archive::try_insert_view`. -/
def tryInsert := tryInsertBy rejectNonFinite
/-- `Archive::admits`. -/
def admits := admitsBy rejectNonFinite

theorem tryInsert_finite (dom near : Mem → Mem → Bool) (cap : Nat) (items : List Mem) (s : Mem)
    (h : ∀ m, m ∈ items → isFinite m.err = true) (m : Mem) (hm : m ∈ tryInsert dom near cap items s) :
    isFinite m.err = true :=
  tryInsertBy_finite rejectNonFinite rejectNonFinite_finite dom near cap items s h m hm

end Rcsr.Pareto
