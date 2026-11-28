import RcsrCore.Prim

/-!
Model of `pareto_fronts` and `nan_low` in `rcsr-engine/src/trials_table.rs`.

A value is compared through the integer key of `nan_low(value)`: `>=` and `>` on two non-NaN doubles are `≥` and `>`
on `Rcsr.Prim.key` (the bridge theorem is `Rcsr.FloatOrder.key_lt_iff_sval_lt`). Both zeros have key 0.

Limits: the `while` loop is modelled with a fuel argument equal to the number of rows; `mem_rounds_iff` shows the
fuel is enough. The front vector is rebuilt from the list of rounds. The tie to the Rust code is the vector check.
-/

namespace Rcsr.TrialsTable
open Rcsr.Prim

/-- `nan_low`: NaN becomes `-inf`. -/
def nanLow (b : W) : W := bif isNaN b then TOP ||| SIGN else b
/-- The order keys of one row after `nan_low`. -/
def rowKey (r : List W) : List Int := r.map (fun x => key (nanLow x))

/-- The closure `dominates(a, b)`: `a >= b` on every zipped pair and `a > b` on one. `zip` stops at the shorter row. -/
def rowDom (a b : List Int) : Bool :=
  (a.zip b).all (fun p => decide (p.2 ≤ p.1)) && (a.zip b).any (fun p => decide (p.2 < p.1))

/-- `rowDom` as a proposition. -/
def RowDom (a b : List Int) : Prop := (∀ p ∈ a.zip b, p.2 ≤ p.1) ∧ (∃ p ∈ a.zip b, p.2 < p.1)

/-- The rows of `left` that no other row of `left` dominates. -/
def now (v : List (List Int)) (left : List Nat) : List Nat :=
  left.filter (fun i => !left.any (fun j => j != i && rowDom (v.getD j []) (v.getD i [])))

/-- The rounds of the `while` loop: the list `now` of each round. -/
def rounds (v : List (List Int)) : Nat → List Nat → List (List Nat)
  | 0, _ => []
  | fuel + 1, left =>
    if left.isEmpty then []
    else (now v left) :: rounds v fuel (left.filter (fun i => !(now v left).contains i))

/-- The front number of row `i`: one plus the index of the round that holds it. -/
def frontOf (rs : List (List Nat)) (i : Nat) : Nat := rs.findIdx (fun r => r.contains i) + 1

/-- `pareto_fronts` on rows of order keys. -/
def paretoFrontsK (v : List (List Int)) : List Nat :=
  (List.range v.length).map (frontOf (rounds v v.length (List.range v.length)))

/-- `pareto_fronts`. -/
def paretoFronts (v : List (List W)) : List Nat := paretoFrontsK (v.map rowKey)

theorem rowDom_iff (a b : List Int) : rowDom a b = true ↔ RowDom a b := by
  simp [rowDom, RowDom, List.all_eq_true, List.any_eq_true]

/-- A dominated row is strictly below its dominator in the lexicographic order. This holds for ragged rows too. -/
theorem rowDom_lex : ∀ (a b : List Int), RowDom a b → b < a
  | [], _, ⟨_, p, hp, _⟩ => by simp at hp
  | _ :: _, [], ⟨_, p, hp, _⟩ => by simp at hp
  | x :: xs, y :: ys, ⟨hall, p, hp, hlt⟩ => by
    have hxy : y ≤ x := hall (x, y) (by simp)
    by_cases hne : y < x
    · exact List.cons_lt_cons_iff.mpr (Or.inl hne)
    · have hxe : y = x := by omega
      subst hxe
      have htail : RowDom xs ys := by
        refine ⟨fun q hq => hall q (by simp [hq]), ?_⟩
        simp only [List.zip_cons_cons, List.mem_cons] at hp
        rcases hp with rfl | hp
        · simp at hlt
        · exact ⟨p, hp, hlt⟩
      exact List.cons_lt_cons_iff.mpr (Or.inr ⟨rfl, rowDom_lex xs ys htail⟩)

theorem rowDom_irrefl (a : List Int) : rowDom a a = false := by
  cases h : rowDom a a
  · rfl
  · exact absurd (rowDom_lex a a ((rowDom_iff a a).mp h)) (List.lt_irrefl a)

/-- A non-empty list of rows has a row that is greatest in the lexicographic order. -/
theorem exists_lex_max (v : List (List Int)) (left : List Nat) (h : left ≠ []) :
    ∃ m, m ∈ left ∧ ∀ j, j ∈ left → ¬ (v.getD m [] < v.getD j []) := by
  induction left with
  | nil => exact absurd rfl h
  | cons x t ih =>
    by_cases ht : t = []
    · subst ht
      exact ⟨x, by simp, fun j hj => by
        have : j = x := by simpa using hj
        subst this; exact List.lt_irrefl _⟩
    · obtain ⟨m, hm, hmax⟩ := ih ht
      by_cases hmx : v.getD m [] < v.getD x []
      · refine ⟨x, by simp, fun j hj hlt => ?_⟩
        rcases List.mem_cons.mp hj with rfl | hj
        · exact List.lt_irrefl _ hlt
        · exact hmax j hj (List.lt_trans hmx hlt)
      · refine ⟨m, List.mem_cons_of_mem _ hm, fun j hj => ?_⟩
        rcases List.mem_cons.mp hj with rfl | hj
        · exact hmx
        · exact hmax j hj

/-- Every round of the loop finds at least one row: this is why the loop ends. -/
theorem now_ne_nil (v : List (List Int)) (left : List Nat) (h : left ≠ []) : now v left ≠ [] := by
  obtain ⟨m, hm, hmax⟩ := exists_lex_max v left h
  have hmem : m ∈ now v left := by
    unfold now
    refine List.mem_filter.mpr ⟨hm, ?_⟩
    simp only [Bool.not_eq_true', List.any_eq_false, Bool.and_eq_true, bne_iff_ne, ne_eq, not_and,
      Bool.not_eq_true]
    intro j hj _
    cases hd : rowDom (v.getD j []) (v.getD m [])
    · rfl
    · exact absurd (rowDom_lex _ _ ((rowDom_iff _ _).mp hd)) (hmax j hj)
  exact List.ne_nil_of_mem hmem

theorem now_subset (v : List (List Int)) (left : List Nat) (i : Nat) (h : i ∈ now v left) : i ∈ left :=
  (List.mem_filter.mp h).1

theorem rounds_length (v : List (List Int)) (fuel : Nat) (left : List Nat) : (rounds v fuel left).length ≤ fuel := by
  induction fuel generalizing left with
  | zero => simp [rounds]
  | succ f ih =>
    simp only [rounds]
    split
    · simp
    · simp only [List.length_cons]; have := ih (left.filter (fun i => !(now v left).contains i)); omega

/-- The fuel is enough: with `left.length` rounds allowed, every row of `left` is in one round, and nothing else is. -/
theorem mem_rounds_iff (v : List (List Int)) (fuel : Nat) (left : List Nat) (hf : left.length ≤ fuel) (i : Nat) :
    (∃ r, r ∈ rounds v fuel left ∧ i ∈ r) ↔ i ∈ left := by
  induction fuel generalizing left with
  | zero =>
    have : left = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this; simp [rounds]
  | succ f ih =>
    simp only [rounds]
    by_cases he : left = []
    · subst he; simp
    · have hne : left.isEmpty = false := by simpa using he
      simp only [hne, Bool.false_eq_true, ↓reduceIte, List.mem_cons, exists_eq_or_imp]
      obtain ⟨w, hw⟩ := List.exists_mem_of_ne_nil _ (now_ne_nil v left he)
      have hlt : (left.filter (fun i => !(now v left).contains i)).length < left.length := by
        refine List.length_filter_lt_length_iff_exists.mpr ⟨w, now_subset v left w hw, ?_⟩
        simp [hw]
      rw [ih _ (by omega)]
      constructor
      · rintro (h | h)
        · exact now_subset v left i h
        · exact (List.mem_filter.mp h).1
      · intro h
        by_cases hn : i ∈ now v left
        · exact Or.inl hn
        · exact Or.inr (List.mem_filter.mpr ⟨h, by simpa using hn⟩)

theorem paretoFrontsK_length (v : List (List Int)) : (paretoFrontsK v).length = v.length := by
  simp [paretoFrontsK]

/-- Every row gets a front number between 1 and the number of rows. The loop ends for every input. -/
theorem paretoFrontsK_total (v : List (List Int)) (i : Nat) (h : i < v.length) :
    1 ≤ (paretoFrontsK v).getD i 0 ∧ (paretoFrontsK v).getD i 0 ≤ v.length := by
  have hm := (mem_rounds_iff v v.length (List.range v.length) (by simp) i).mpr (List.mem_range.mpr h)
  have hidx : (rounds v v.length (List.range v.length)).findIdx (fun r => r.contains i) <
      (rounds v v.length (List.range v.length)).length := by
    refine List.findIdx_lt_length_of_exists ?_
    obtain ⟨r, hr, hir⟩ := hm
    exact ⟨r, hr, by simpa using hir⟩
  have hl := rounds_length v v.length (List.range v.length)
  simp only [paretoFrontsK, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range h, Option.map_some,
    Option.getD_some, frontOf]
  omega


theorem mem_now_iff (v : List (List Int)) (left : List Nat) (i : Nat) :
    i ∈ now v left ↔ i ∈ left ∧ ∀ j, j ∈ left → j ≠ i → rowDom (v.getD j []) (v.getD i []) = false := by
  unfold now
  simp only [List.mem_filter, Bool.not_eq_true', List.any_eq_false, Bool.and_eq_true, bne_iff_ne, ne_eq, not_and,
    Bool.not_eq_true]

theorem findIdx_rounds_zero (v : List (List Int)) (fuel : Nat) (left : List Nat) (i : Nat) (hf : 0 < fuel)
    (hl : left ≠ []) : (rounds v fuel left).findIdx (fun r => r.contains i) = 0 ↔ i ∈ now v left := by
  cases fuel with
  | zero => omega
  | succ f =>
    have hne : left.isEmpty = false := by simpa using hl
    simp only [rounds, hne, Bool.false_eq_true, ↓reduceIte, List.findIdx_cons]
    by_cases hc : i ∈ now v left
    · simp [hc]
    · simp [hc]

theorem paretoFrontsK_getD (v : List (List Int)) (i : Nat) (h : i < v.length) :
    (paretoFrontsK v).getD i 0 =
      (rounds v v.length (List.range v.length)).findIdx (fun r => r.contains i) + 1 := by
  simp only [paretoFrontsK, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range h, Option.map_some,
    Option.getD_some, frontOf]

/-- Front 1 is the set of rows that no other row dominates. -/
theorem paretoFrontsK_first (v : List (List Int)) (i : Nat) (h : i < v.length) :
    (paretoFrontsK v).getD i 0 = 1 ↔
      ∀ j, j < v.length → j ≠ i → rowDom (v.getD j []) (v.getD i []) = false := by
  rw [paretoFrontsK_getD v i h]
  have hl : List.range v.length ≠ [] := by
    intro he; have := List.range_eq_nil.mp he; omega
  have hz := findIdx_rounds_zero v v.length (List.range v.length) i (by omega) hl
  rw [mem_now_iff] at hz
  constructor
  · intro h1 j hj hji
    exact (hz.mp (by omega)).2 j (List.mem_range.mpr hj) hji
  · intro h2
    have := hz.mpr ⟨List.mem_range.mpr h, fun j hj hji => h2 j (List.mem_range.mp hj) hji⟩
    omega
theorem now_equal_rows (v : List (List Int)) (left : List Nat) (i j : Nat) (h : v.getD i [] = v.getD j [])
    (hl : i ∈ left ↔ j ∈ left) : i ∈ now v left ↔ j ∈ now v left := by
  have key : ∀ a b : Nat, v.getD a [] = v.getD b [] → a ∈ left → b ∈ left → a ∈ now v left → b ∈ now v left := by
    intro a b hab _ hb ha
    rw [mem_now_iff] at ha ⊢
    refine ⟨hb, fun k hk _ => ?_⟩
    by_cases hka : k = a
    · subst hka; rw [hab]; exact rowDom_irrefl _
    · rw [← hab]; exact ha.2 k hk hka
  constructor
  · intro hi
    have := now_subset v left i hi
    exact key i j h this (hl.mp this) hi
  · intro hj
    have := now_subset v left j hj
    exact key j i h.symm this (hl.mpr this) hj

theorem rounds_equal_rows (v : List (List Int)) (i j : Nat) (h : v.getD i [] = v.getD j []) :
    ∀ (fuel : Nat) (left : List Nat), (i ∈ left ↔ j ∈ left) →
      (rounds v fuel left).findIdx (fun r => r.contains i) = (rounds v fuel left).findIdx (fun r => r.contains j)
  | 0, _, _ => by simp [rounds]
  | fuel + 1, left, hl => by
    simp only [rounds]
    split
    · rfl
    · have hn := now_equal_rows v left i j h hl
      have hc : (now v left).contains i = (now v left).contains j := by
        rw [Bool.eq_iff_iff]; simpa using hn
      simp only [List.findIdx_cons, hc]
      rw [rounds_equal_rows v i j h fuel _ (by
        simp only [List.mem_filter, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not]
        rw [hl, hn])]

/-- Rows equal on every metric share a front (the last sentence of the Rust doc comment). -/
theorem paretoFrontsK_equal_rows (v : List (List Int)) (i j : Nat) (hi : i < v.length) (hj : j < v.length)
    (h : v.getD i [] = v.getD j []) : (paretoFrontsK v).getD i 0 = (paretoFrontsK v).getD j 0 := by
  rw [paretoFrontsK_getD v i hi, paretoFrontsK_getD v j hj]
  rw [rounds_equal_rows v i j h _ _ (by simp [hi, hj])]

/-- The model reads a value through its order key only, so the sign of a zero and the payload of a NaN are not seen. -/
theorem paretoFronts_congr (v w : List (List W)) (h : v.map rowKey = w.map rowKey) :
    paretoFronts v = paretoFronts w := by
  unfold paretoFronts; rw [h]

theorem rowKey_zero_sign : rowKey [SIGN] = rowKey [0#64] := by decide
theorem rowKey_nan_is_neg_inf : rowKey [0x7ff8000000000000#64] = rowKey [TOP ||| SIGN] := by decide

/-- Two incomparable rows, one row both dominate, one repeated row. -/
example : paretoFrontsK [[3, 1], [1, 3], [0, 0], [3, 1]] = [1, 1, 2, 1] := by decide
/-- Ragged rows: `zip` stops at the shorter row; an empty row is in the first front. -/
example : paretoFrontsK [[1], [2, 0], []] = [2, 1, 1] := by decide

end Rcsr.TrialsTable
