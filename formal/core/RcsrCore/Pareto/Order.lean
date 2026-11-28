import RcsrCore.Prim
import RcsrCore.ErrorOrder

/-!
Comparators of the front code: `f64::total_cmp`, the canonical member order of `rcsr-engine/src/pareto.rs`
(`Archive::canonicalize`, `Archive::hash`) and `rcsr-engine/src/search.rs` (`canonical_front`), the two-key order of
`Archive::prune` and `Archive::sorted`, and the order of `QdCell::best` in `rcsr-engine/src/qd.rs`.

A front member is read through three words only: cost bits, error bits and the structural hash of its tree.

Limits: `total_cmp` is restated from the Rust std source. A `then` chain of comparators is restated as `lexLe`.
The tie to the Rust code is the vector check.
-/

namespace Rcsr.Pareto
open Rcsr.Prim

/-- The words of a front member that the front code reads. -/
structure Mem where
  cost : W
  err : W
  hash : W
deriving DecidableEq, Repr

/-- The key of `f64::total_cmp`: `bits ^ (((bits >> 63) as u64) >> 1)`, compared as `i64`. -/
def tcBits (b : W) : W := b ^^^ ((b.sshiftRight 63) >>> 1)
/-- `a.total_cmp(&b) != Greater`. -/
def totalLe (a b : W) : Bool := (tcBits a).sle (tcBits b)
/-- The `total_cmp` key as an integer. -/
def totalKey (b : W) : Int := (tcBits b).toInt

/-- Rust `==` on `f64`: false when an operand is NaN, `-0.0` equals `+0.0`. -/
def feq (a b : W) : Bool := !isNaN a && !isNaN b && fltEq a b
/-- Rust `<` on `f64`: false when an operand is NaN. -/
def flt (a b : W) : Bool := !isNaN a && !isNaN b && fltLt a b

/-- Both bit patterns of zero. -/
def isZero (b : W) : Bool := b == 0#64 || b == SIGN
/-- Maps `-0.0` to `+0.0` and keeps every other bit pattern. -/
def zeroFold (b : W) : W := bif b == SIGN then 0#64 else b
/-- A member with the zero sign of its error folded away. -/
def qkey (m : Mem) : Mem := ⟨m.cost, zeroFold m.err, m.hash⟩

/-- `c1.then(c2) != Greater`, for comparators given as `le` relations. -/
def lexLe {α : Type} (le1 le2 : α → α → Bool) (a b : α) : Bool := le1 a b && (!le1 b a || le2 a b)

def costLe (a b : Mem) : Bool := totalLe a.cost b.cost
def errLe (a b : Mem) : Bool := ErrorOrder.le a.err b.err
def hashLe (a b : Mem) : Bool := a.hash.ule b.hash
def errBitsLe (a b : Mem) : Bool := a.err.ule b.err

/-- The three-key order: `cost.total_cmp`, then `cmp_error`, then the structural hash. -/
def sortLe : Mem → Mem → Bool := lexLe costLe (lexLe errLe hashLe)
/-- The four-key order `canonical_order` of #1465: the three keys, then the error bits. -/
def canonLe : Mem → Mem → Bool := lexLe costLe (lexLe errLe (lexLe hashLe errBitsLe))
/-- The order of `Archive::prune` and `Archive::sorted`: cost, then error. -/
def pruneLe : Mem → Mem → Bool := lexLe costLe errLe
/-- The order of `QdCell::best`: error, then cost. -/
def bestLe : Mem → Mem → Bool := lexLe errLe costLe

/-! ### `total_cmp` -/

theorem tcBits_injective (a b : W) (h : tcBits a = tcBits b) : a = b := by
  simp only [tcBits] at h; rcsr_bv
theorem totalKey_injective (a b : W) (h : totalKey a = totalKey b) : a = b :=
  tcBits_injective a b (BitVec.toInt_inj.mp h)
theorem totalLe_total (a b : W) : (totalLe a b || totalLe b a) = true := by
  simp only [totalLe, tcBits]; rcsr_bv
theorem totalLe_trans (a b c : W) (h1 : totalLe a b = true) (h2 : totalLe b c = true) : totalLe a c = true := by
  simp only [totalLe, tcBits] at *; rcsr_bv
theorem totalLe_antisymm (a b : W) (h1 : totalLe a b = true) (h2 : totalLe b a = true) : a = b := by
  simp only [totalLe, tcBits] at *; rcsr_bv
/-- The `total_cmp` order refines the float order `<`. -/
theorem totalLe_of_flt (a b : W) (h : flt a b = true) : totalLe a b = true := by
  simp only [totalLe, tcBits, flt, fltLt, fold, isNaN, TOP, SIGN] at *; rcsr_bv
theorem totalLe_iff_key (a b : W) : totalLe a b = true ↔ totalKey a ≤ totalKey b := by
  simp [totalLe, totalKey, BitVec.sle_iff_toInt_le]
/-- `total_cmp` puts `-0.0` strictly before `+0.0`. -/
theorem totalLe_neg_zero : totalLe SIGN 0#64 = true ∧ totalLe 0#64 SIGN = false := by decide

/-! ### Which `cmp_error` ties are not bit ties -/

theorem eqv_iff_bits (a b : W) :
    ErrorOrder.eqv a b = (a == b || (isZero a && isZero b) || (isNaN a && isNaN b)) := by
  simp only [ErrorOrder.eqv, ErrorOrder.cls, fltEq, fold, isNaN, isZero, TOP, SIGN]; rcsr_bv
theorem eqv_zeroFold (a b : W) (ha : isNaN a = false) (hb : isNaN b = false) :
    ErrorOrder.eqv a b = (zeroFold a == zeroFold b) := by
  simp only [ErrorOrder.eqv, ErrorOrder.cls, fltEq, fold, isNaN, zeroFold, TOP, SIGN] at *; rcsr_bv
theorem errLe_zeroFold (a b : W) : ErrorOrder.le (zeroFold a) (zeroFold b) = ErrorOrder.le a b := by
  simp only [ErrorOrder.le, ErrorOrder.lt, ErrorOrder.eqv, ErrorOrder.cls, fltLt, fltEq, fold, isNaN, zeroFold, TOP, SIGN]
  rcsr_bv
theorem flt_zeroFold (a b : W) : flt (zeroFold a) (zeroFold b) = flt a b := by
  simp only [flt, fltLt, fold, isNaN, zeroFold, TOP, SIGN]; rcsr_bv
theorem isNaN_zeroFold (a : W) : isNaN (zeroFold a) = isNaN a := by
  simp only [isNaN, zeroFold, TOP, SIGN]; rcsr_bv
theorem zeroFold_idem (a : W) : zeroFold (zeroFold a) = zeroFold a := by
  simp only [zeroFold, SIGN]; rcsr_bv

/-! ### A `then` chain of total preorders is a total preorder -/

section Lex
variable {α : Type} {le1 le2 : α → α → Bool}

theorem lexLe_total (t1 : ∀ a b, (le1 a b || le1 b a) = true) (t2 : ∀ a b, (le2 a b || le2 b a) = true)
    (a b : α) : (lexLe le1 le2 a b || lexLe le1 le2 b a) = true := by
  have h1 := t1 a b; have h2 := t2 a b
  unfold lexLe
  cases h3 : le1 a b <;> cases h4 : le1 b a <;> cases h5 : le2 a b <;> cases h6 : le2 b a <;> simp_all

theorem lexLe_trans
    (r1 : ∀ a b c, le1 a b = true → le1 b c = true → le1 a c = true)
    (r2 : ∀ a b c, le2 a b = true → le2 b c = true → le2 a c = true)
    (a b c : α) (hab : lexLe le1 le2 a b = true) (hbc : lexLe le1 le2 b c = true) : lexLe le1 le2 a c = true := by
  unfold lexLe at *
  simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true'] at *
  obtain ⟨hab1, hab2⟩ := hab
  obtain ⟨hbc1, hbc2⟩ := hbc
  refine ⟨r1 a b c hab1 hbc1, ?_⟩
  cases hca : le1 c a
  · exact Or.inl rfl
  · right
    have hcb := r1 c a b hca hab1
    have hba := r1 b c a hbc1 hca
    rcases hab2 with h | h
    · rw [hba] at h; cases h
    · rcases hbc2 with h' | h'
      · rw [hcb] at h'; cases h'
      · exact r2 a b c h h'

/-- Two elements are tied in the chain exactly when they are tied in both parts. -/
theorem lexLe_tie (a b : α) :
    (lexLe le1 le2 a b = true ∧ lexLe le1 le2 b a = true) ↔
      ((le1 a b = true ∧ le1 b a = true) ∧ (le2 a b = true ∧ le2 b a = true)) := by
  unfold lexLe
  cases le1 a b <;> cases le1 b a <;> cases le2 a b <;> cases le2 b a <;> simp
end Lex

/-! ### The four comparators are total preorders -/

theorem costLe_total (a b : Mem) : (costLe a b || costLe b a) = true := totalLe_total _ _
theorem costLe_trans (a b c : Mem) : costLe a b = true → costLe b c = true → costLe a c = true := totalLe_trans _ _ _
theorem errLe_total (a b : Mem) : (errLe a b || errLe b a) = true := by
  have := ErrorOrder.le_total a.err b.err; simpa [errLe] using this
theorem errLe_trans (a b c : Mem) : errLe a b = true → errLe b c = true → errLe a c = true := ErrorOrder.le_trans _ _ _
theorem hashLe_total (a b : Mem) : (hashLe a b || hashLe b a) = true := by simp only [hashLe]; rcsr_bv
theorem hashLe_trans (a b c : Mem) (h1 : hashLe a b = true) (h2 : hashLe b c = true) : hashLe a c = true := by
  simp only [hashLe] at *; rcsr_bv
theorem errBitsLe_total (a b : Mem) : (errBitsLe a b || errBitsLe b a) = true := by simp only [errBitsLe]; rcsr_bv
theorem errBitsLe_trans (a b c : Mem) (h1 : errBitsLe a b = true) (h2 : errBitsLe b c = true) :
    errBitsLe a c = true := by
  simp only [errBitsLe] at *; rcsr_bv

theorem sortLe_total (a b : Mem) : (sortLe a b || sortLe b a) = true :=
  lexLe_total costLe_total (lexLe_total errLe_total hashLe_total) a b
theorem sortLe_trans (a b c : Mem) : sortLe a b = true → sortLe b c = true → sortLe a c = true :=
  lexLe_trans costLe_trans (lexLe_trans errLe_trans hashLe_trans) a b c
theorem canonLe_total (a b : Mem) : (canonLe a b || canonLe b a) = true :=
  lexLe_total costLe_total (lexLe_total errLe_total (lexLe_total hashLe_total errBitsLe_total)) a b
theorem canonLe_trans (a b c : Mem) : canonLe a b = true → canonLe b c = true → canonLe a c = true :=
  lexLe_trans costLe_trans
    (lexLe_trans errLe_trans (lexLe_trans hashLe_trans errBitsLe_trans)) a b c
theorem pruneLe_total (a b : Mem) : (pruneLe a b || pruneLe b a) = true :=
  lexLe_total costLe_total errLe_total a b
theorem pruneLe_trans (a b c : Mem) : pruneLe a b = true → pruneLe b c = true → pruneLe a c = true :=
  lexLe_trans costLe_trans errLe_trans a b c
theorem bestLe_total (a b : Mem) : (bestLe a b || bestLe b a) = true :=
  lexLe_total errLe_total costLe_total a b
theorem bestLe_trans (a b c : Mem) : bestLe a b = true → bestLe b c = true → bestLe a c = true :=
  lexLe_trans errLe_trans costLe_trans a b c

/-! ### Ties -/

theorem errLe_tie (a b : Mem) : (errLe a b = true ∧ errLe b a = true) ↔ ErrorOrder.eqv a.err b.err = true := by
  have h := ErrorOrder.trichotomy a.err b.err
  have hs := ErrorOrder.eqv_symm a.err b.err
  simp only [errLe, ErrorOrder.le]
  cases h1 : ErrorOrder.lt a.err b.err <;> cases h2 : ErrorOrder.eqv a.err b.err <;>
    cases h3 : ErrorOrder.lt b.err a.err <;> simp_all

/-- The comparator class of the three-key order: equal cost bits, `cmp_error`-equal errors, equal hash. -/
theorem sortLe_antisymm_iff (a b : Mem) :
    (sortLe a b = true ∧ sortLe b a = true) ↔
      (a.cost = b.cost ∧ ErrorOrder.eqv a.err b.err = true ∧ a.hash = b.hash) := by
  unfold sortLe
  rw [lexLe_tie, lexLe_tie, errLe_tie]
  constructor
  · rintro ⟨⟨h1, h2⟩, he, h3, h4⟩
    refine ⟨totalLe_antisymm _ _ h1 h2, he, ?_⟩
    simp only [hashLe] at h3 h4; rcsr_bv
  · rintro ⟨hc, he, hh⟩
    have t1 := totalLe_total a.cost a.cost
    have t2 : a.hash.ule a.hash = true := by rcsr_bv
    refine ⟨⟨?_, ?_⟩, he, ?_, ?_⟩ <;> simp_all [costLe, hashLe]

/-- The four-key order has no ties between different members. -/
theorem canonLe_antisymm (a b : Mem) (h1 : canonLe a b = true) (h2 : canonLe b a = true) : a = b := by
  unfold canonLe at h1 h2
  have h := (lexLe_tie a b).mp ⟨h1, h2⟩
  have h' := (lexLe_tie a b).mp h.2
  have h'' := (lexLe_tie a b).mp h'.2
  have hc : a.cost = b.cost := totalLe_antisymm _ _ h.1.1 h.1.2
  have hh : a.hash = b.hash := by have := h''.1; simp only [hashLe] at this; rcsr_bv
  have he : a.err = b.err := by have := h''.2; simp only [errBitsLe] at this; rcsr_bv
  cases a; cases b; simp_all

/-- A three-key tie between two members without a NaN error is an equality of the folded members. -/
theorem sortLe_tie_qkey (a b : Mem) (ha : isNaN a.err = false) (hb : isNaN b.err = false)
    (h1 : sortLe a b = true) (h2 : sortLe b a = true) : qkey a = qkey b := by
  obtain ⟨hc, he, hh⟩ := (sortLe_antisymm_iff a b).mp ⟨h1, h2⟩
  rw [eqv_zeroFold a.err b.err ha hb] at he
  simp only [qkey, Mem.mk.injEq]
  exact ⟨hc, by simpa using he, hh⟩

/-- The three-key order does not see the zero sign of the error. -/
theorem sortLe_qkey (a b : Mem) : sortLe (qkey a) (qkey b) = sortLe a b := by
  simp only [sortLe, lexLe, costLe, errLe, hashLe, qkey, errLe_zeroFold]

/-! ### A stable sort under a total preorder is a function of the multiset when there are no ties -/

theorem mergeSort_eq_of_perm {α : Type} (le : α → α → Bool)
    (htot : ∀ a b, (le a b || le b a) = true)
    (htr : ∀ a b c, le a b = true → le b c = true → le a c = true)
    (xs ys : List α) (h : xs.Perm ys)
    (hanti : ∀ a, a ∈ xs → ∀ b, b ∈ xs → le a b = true → le b a = true → a = b) :
    xs.mergeSort le = ys.mergeSort le := by
  refine List.Perm.eq_of_pairwise (le := fun a b => le a b = true) ?_
    (List.pairwise_mergeSort htr htot xs) (List.pairwise_mergeSort htr htot ys)
    ((List.mergeSort_perm xs le).trans (h.trans (List.mergeSort_perm ys le).symm))
  intro a b ha hb hab hba
  have ha' : a ∈ xs := (List.mergeSort_perm xs le).mem_iff.mp ha
  have hb' : b ∈ xs := h.mem_iff.mpr ((List.mergeSort_perm ys le).mem_iff.mp hb)
  exact hanti a ha' b hb' hab hba

end Rcsr.Pareto
