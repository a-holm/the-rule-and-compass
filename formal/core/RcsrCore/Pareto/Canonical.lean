import RcsrCore.Pareto.Order

/-!
Model of `Archive::canonicalize` and `Archive::hash` in `rcsr-engine/src/pareto.rs`.

`canonicalizeBy le idx cap xs` is the function with its sort comparator `le` and its trim index rule `idx` as
parameters, so that one set of theorems covers the code before and after #1463 (index rule) and #1465 (fourth sort key).
`canonicalize` and `archiveHash` name the instance that models the Rust text.

Limits: `sort_by` is modelled by `List.mergeSort` (both are stable). The staircase loop and the trim loop are restated
by hand. The tree of a member is not modelled. The tie to the Rust code is the vector check.
-/

namespace Rcsr.Pareto
open Rcsr.Prim

section Generic
variable {α β : Type}

/-- The staircase loop after the first kept member. `prev` is `staircase.last()`. -/
def stairGo (costEq errLt : α → α → Bool) (prev : α) : List α → List α
  | [] => []
  | s :: t =>
    if costEq prev s then stairGo costEq errLt prev t
    else if errLt s prev then s :: stairGo costEq errLt s t
    else stairGo costEq errLt prev t

/-- The staircase loop: keep the first member, then every member with a new cost and a strictly lower error. -/
def staircase (costEq errLt : α → α → Bool) : List α → List α
  | [] => []
  | s :: t => s :: stairGo costEq errLt s t

/-- The cap trim with the index rule `idx k n cap` as a parameter. `cap` is `self.cap`; the loop uses `cap.max(1)`. -/
def trimBy (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs : List α) : List α :=
  if xs.length > cap then
    (List.range (max cap 1)).filterMap (fun k => xs[idx k xs.length (max cap 1)]?)
  else xs

theorem stairGo_map (f : α → β) (ce el : α → α → Bool) (ce' el' : β → β → Bool)
    (hce : ∀ a b, ce a b = ce' (f a) (f b)) (hel : ∀ a b, el a b = el' (f a) (f b))
    (p : α) (xs : List α) :
    (stairGo ce el p xs).map f = stairGo ce' el' (f p) (xs.map f) := by
  induction xs generalizing p with
  | nil => rfl
  | cons s t ih =>
    simp only [stairGo, List.map_cons, ← hce, ← hel]
    split
    · exact ih p
    · split
      · simp [ih s]
      · exact ih p

/-- The staircase pass commutes with every map that keeps the two predicates it reads. -/
theorem staircase_map (f : α → β) (ce el : α → α → Bool) (ce' el' : β → β → Bool)
    (hce : ∀ a b, ce a b = ce' (f a) (f b)) (hel : ∀ a b, el a b = el' (f a) (f b))
    (xs : List α) :
    (staircase ce el xs).map f = staircase ce' el' (xs.map f) := by
  cases xs with
  | nil => rfl
  | cons s t => simp [staircase, stairGo_map f ce el ce' el' hce hel]

/-- The trim commutes with every map: it selects by position only. -/
theorem trimBy_map (f : α → β) (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs : List α) :
    (trimBy idx cap xs).map f = trimBy idx cap (xs.map f) := by
  unfold trimBy
  simp only [List.length_map]
  split
  · simp [List.map_filterMap, List.getElem?_map]
  · rfl

end Generic

/-- The index rule at `ad228547`: `k * n / cap`. -/
def trimIdx (k n cap : Nat) : Nat := k * n / cap
/-- The index rule of #1463: `k * (n - 1) / (cap - 1)`, and index 0 for a cap of 1. -/
def trimIdxEnds (k n cap : Nat) : Nat := if cap ≤ 1 then 0 else k * (n - 1) / (cap - 1)

/-- `Archive::canonicalize` with the comparator and the index rule as parameters. -/
def canonicalizeBy (le : Mem → Mem → Bool) (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs : List Mem) : List Mem :=
  trimBy idx cap (staircase (fun a b => feq a.cost b.cost) (fun a b => flt a.err b.err) (xs.mergeSort le))

/-- One step of FNV-1a over a 64-bit word. -/
def fnvStep (h w : W) : W := (h ^^^ w) * 0x00000100000001b3#64
/-- `Archive::hash` with the comparator as a parameter. -/
def archiveHashBy (le : Mem → Mem → Bool) (xs : List Mem) : W :=
  ((xs.mergeSort le).flatMap (fun m => [m.cost, m.err, m.hash])).foldl fnvStep 0xcbf29ce484222325#64
/-- No two different members of the list are tied under `le`. -/
def NoTie (le : Mem → Mem → Bool) (xs : List Mem) : Prop :=
  ∀ a, a ∈ xs → ∀ b, b ∈ xs → le a b = true → le b a = true → a = b

/-! ### Determinism on bits, for every comparator that is a total preorder -/

theorem canonicalizeBy_perm_of_noTie (le : Mem → Mem → Bool)
    (htot : ∀ a b, (le a b || le b a) = true) (htr : ∀ a b c, le a b = true → le b c = true → le a c = true)
    (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs ys : List Mem) (h : xs.Perm ys) (hx : NoTie le xs) :
    canonicalizeBy le idx cap xs = canonicalizeBy le idx cap ys := by
  unfold canonicalizeBy
  rw [mergeSort_eq_of_perm le htot htr xs ys h hx]

theorem archiveHashBy_perm_of_noTie (le : Mem → Mem → Bool)
    (htot : ∀ a b, (le a b || le b a) = true) (htr : ∀ a b c, le a b = true → le b c = true → le a c = true)
    (xs ys : List Mem) (h : xs.Perm ys) (hx : NoTie le xs) :
    archiveHashBy le xs = archiveHashBy le ys := by
  unfold archiveHashBy
  rw [mergeSort_eq_of_perm le htot htr xs ys h hx]

/-- The hash reads the sorted list through the three words of each member only. -/
theorem archiveHashBy_congr (le : Mem → Mem → Bool) (xs ys : List Mem)
    (h : xs.mergeSort le = ys.mergeSort le) : archiveHashBy le xs = archiveHashBy le ys := by
  unfold archiveHashBy; rw [h]

/-- The four-key order of #1465 has no ties at all. -/
theorem noTie_canonLe (xs : List Mem) : NoTie canonLe xs :=
  fun a _ b _ h1 h2 => canonLe_antisymm a b h1 h2

/-- With the four-key order of #1465 the result is a function of the multiset of members, for every bit pattern. -/
theorem canonicalizeBy_canonLe_perm (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs ys : List Mem) (h : xs.Perm ys) :
    canonicalizeBy canonLe idx cap xs = canonicalizeBy canonLe idx cap ys :=
  canonicalizeBy_perm_of_noTie canonLe canonLe_total canonLe_trans idx cap xs ys h (noTie_canonLe xs)

theorem archiveHashBy_canonLe_perm (xs ys : List Mem) (h : xs.Perm ys) :
    archiveHashBy canonLe xs = archiveHashBy canonLe ys :=
  archiveHashBy_perm_of_noTie canonLe canonLe_total canonLe_trans xs ys h (noTie_canonLe xs)

/-! ### Determinism up to the zero sign of the error, for the three-key order -/

/-- `canonicalize` with the three-key order commutes with folding the zero sign of the error. -/
theorem canonicalizeBy_sortLe_qkey (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs : List Mem) :
    (canonicalizeBy sortLe idx cap xs).map qkey = canonicalizeBy sortLe idx cap (xs.map qkey) := by
  unfold canonicalizeBy
  rw [trimBy_map, staircase_map qkey (fun a b => feq a.cost b.cost) (fun a b => flt a.err b.err)
    (fun a b => feq a.cost b.cost) (fun a b => flt a.err b.err) (fun a b => rfl) (fun a b => by simp [qkey, flt_zeroFold]),
    List.map_mergeSort (fun a _ b _ => (sortLe_qkey a b).symm)]

/-- Two inputs with the same multiset of folded members give the same folded result, when no error is NaN. -/
theorem canonicalizeBy_sortLe_perm_quot (idx : Nat → Nat → Nat → Nat) (cap : Nat) (xs ys : List Mem)
    (h : (xs.map qkey).Perm (ys.map qkey)) (hn : ∀ m, m ∈ xs → isNaN m.err = false) :
    (canonicalizeBy sortLe idx cap xs).map qkey = (canonicalizeBy sortLe idx cap ys).map qkey := by
  rw [canonicalizeBy_sortLe_qkey, canonicalizeBy_sortLe_qkey]
  refine canonicalizeBy_perm_of_noTie sortLe sortLe_total sortLe_trans idx cap _ _ h ?_
  intro a ha b hb h1 h2
  obtain ⟨a0, ha0, rfl⟩ := List.mem_map.mp ha
  obtain ⟨b0, hb0, rfl⟩ := List.mem_map.mp hb
  have hqa : isNaN (qkey a0).err = false := by simp [qkey, isNaN_zeroFold, hn a0 ha0]
  have hqb : isNaN (qkey b0).err = false := by simp [qkey, isNaN_zeroFold, hn b0 hb0]
  have := sortLe_tie_qkey (qkey a0) (qkey b0) hqa hqb h1 h2
  simpa [qkey, zeroFold_idem] using this

/-! ### The bit-level statement without `NoTie` is false for the three-key order -/

/-- Cost `1.0`, error `+0.0`, hash 7. -/
def tiePos : Mem := ⟨0x3ff0000000000000#64, 0#64, 7#64⟩
/-- Cost `1.0`, error `-0.0`, hash 7. -/
def tieNeg : Mem := ⟨0x3ff0000000000000#64, SIGN, 7#64⟩

theorem zero_tie_is_a_tie : sortLe tiePos tieNeg = true ∧ sortLe tieNeg tiePos = true ∧ tiePos ≠ tieNeg := by decide

theorem zero_tie_fwd : canonicalizeBy sortLe trimIdx 4 [tiePos, tieNeg] = [tiePos] := by
  unfold canonicalizeBy; rw [List.mergeSort_of_pairwise (by decide)]; decide
theorem zero_tie_rev : canonicalizeBy sortLe trimIdx 4 [tieNeg, tiePos] = [tieNeg] := by
  unfold canonicalizeBy; rw [List.mergeSort_of_pairwise (by decide)]; decide

/-- Two orders of one member set give different error bits, and so different archive hashes. -/
theorem zero_tie_bits_differ :
    (canonicalizeBy sortLe trimIdx 4 [tiePos, tieNeg]).map (·.err) ≠ (canonicalizeBy sortLe trimIdx 4 [tieNeg, tiePos]).map (·.err) := by
  rw [zero_tie_fwd, zero_tie_rev]; decide

theorem zero_tie_hash_differs :
    archiveHashBy sortLe (canonicalizeBy sortLe trimIdx 4 [tiePos, tieNeg]) ≠ archiveHashBy sortLe (canonicalizeBy sortLe trimIdx 4 [tieNeg, tiePos]) := by
  rw [zero_tie_fwd, zero_tie_rev]
  unfold archiveHashBy
  rw [List.mergeSort_singleton, List.mergeSort_singleton]; decide

/-- Before the staircase pass the two orders hash alike: FNV-1a carries a flip of the top bit of one word to the
top bit of the result, wherever the word stands. -/
theorem zero_tie_hash_blind_before_canonicalize : archiveHashBy sortLe [tiePos, tieNeg] = archiveHashBy sortLe [tieNeg, tiePos] := by
  unfold archiveHashBy
  rw [List.mergeSort_of_pairwise (by decide), List.mergeSort_of_pairwise (by decide)]; decide

/-- The four-key order of #1465 removes the counterexample. -/
theorem zero_tie_closed_by_canonLe (idx : Nat → Nat → Nat → Nat) (cap : Nat) :
    canonicalizeBy canonLe idx cap [tiePos, tieNeg] = canonicalizeBy canonLe idx cap [tieNeg, tiePos] :=
  canonicalizeBy_canonLe_perm idx cap _ _ (List.Perm.swap _ _ _)

/-! ### The instance that models the Rust text

The comparator is `sortLe` until `fn canonical_order` of #1465 exists in `pareto.rs`, then `canonLe`.
The index rule is `trimIdx` until the rule of #1463 is in `pareto.rs`, then `trimIdxEnds`. -/

/-- `Archive::canonicalize`. -/
def canonicalize (cap : Nat) (xs : List Mem) : List Mem := canonicalizeBy sortLe trimIdx cap xs
/-- `Archive::hash`. -/
def archiveHash (xs : List Mem) : W := archiveHashBy sortLe xs

/-- A permutation changes nothing when no two different members are tied. -/
theorem canonicalize_perm_of_noTie (cap : Nat) (xs ys : List Mem) (h : xs.Perm ys) (hx : NoTie sortLe xs) :
    canonicalize cap xs = canonicalize cap ys :=
  canonicalizeBy_perm_of_noTie sortLe sortLe_total sortLe_trans trimIdx cap xs ys h hx

theorem archiveHash_perm_of_noTie (xs ys : List Mem) (h : xs.Perm ys) (hx : NoTie sortLe xs) :
    archiveHash xs = archiveHash ys :=
  archiveHashBy_perm_of_noTie sortLe sortLe_total sortLe_trans xs ys h hx

theorem canonicalize_perm_quot (cap : Nat) (xs ys : List Mem)
    (h : (xs.map qkey).Perm (ys.map qkey)) (hn : ∀ m, m ∈ xs → isNaN m.err = false) :
    (canonicalize cap xs).map qkey = (canonicalize cap ys).map qkey :=
  canonicalizeBy_sortLe_perm_quot trimIdx cap xs ys h hn

end Rcsr.Pareto
