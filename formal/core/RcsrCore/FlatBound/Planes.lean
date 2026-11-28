import RcsrCore.Prim

/-!
Bit planes of the row check: row `r` is bit `r % 32` of word `r / 32`, as in `settle_slot`
(`rcsr-engine/src/flat_bound/rows.rs`), `CheckRows::settle` (`rcsr-engine/src/check_rows_gpu.rs`) and the kernel
`check_rows_kernel` (`rcsr-engine/src/cuda/check_rows.h`).

Two layers. The first five theorems are facts about the machine words the Rust and C code computes with
(`row >> 5`, `row & 31`, `1u32 << bit`, `(n + 31) / 32`). The rest is the abstraction the settle theorems use:
a plane is an array of 32-bit words, `setRow` sets one row and `bitAt` reads one.

Limits: `setRow` leaves the plane unchanged for a word index out of bounds, where the Rust code panics.
Every theorem that uses it carries the bound as a hypothesis.
-/

namespace Rcsr.FlatBound.Planes

/-- `w | (1 << i)` then `(w >> j) & 1`: the bit `j` is set exactly when `i = j` or it was set. -/
theorem setbit_test (w i j : BitVec 32) (hi : i < 32#32) (hj : j < 32#32) :
    (((w ||| (1#32 <<< i)) >>> j) &&& 1#32) = if i = j then 1#32 else ((w >>> j) &&& 1#32) := by
  rcsr_bv

/-- `row = (row >> 5) * 32 + (row & 31)`. -/
theorem row_split (row : BitVec 64) : (row >>> 5) * 32#64 + (row &&& 31#64) = row := by
  rcsr_bv

/-- A row below `n` has its word below `(n + 31) / 32`, for `n` below `2^63`. -/
theorem word_in_bounds (row n : BitVec 64) (h : row < n) (hn : n ≤ 0x7fffffffffffffff#64) :
    row >>> 5 < (n + 31#64) >>> 5 := by
  rcsr_bv

theorem lane_lt (row : BitVec 64) : row &&& 31#64 < 32#64 := by
  rcsr_bv

/-- Two rows with the same word and the same bit are the same row. -/
theorem row_inj (r s : BitVec 64) (h1 : r >>> 5 = s >>> 5) (h2 : r &&& 31#64 = s &&& 31#64) : r = s := by
  rcsr_bv

/-- The device computes the word count in a C `int`: `(n + 31) / 32` does not overflow for `0 <= n < 2^31 - 31`. -/
theorem int_words_no_overflow (n : BitVec 32) (h0 : (0#32).sle n = true) (h : n.slt 0x7fffffe1#32 = true) :
    (n.signExtend 34 + 31#34) = (n + 31#32).signExtend 34 := by
  rcsr_bv

/-- `row >> 5` and `row & 31` on a `usize` are `row / 32` and `row % 32`. -/
theorem shift_mask_nat (row : Nat) : row >>> 5 = row / 32 ∧ row &&& 31 = row % 32 := by
  refine ⟨Nat.shiftRight_eq_div_pow row 5, ?_⟩
  exact Nat.and_two_pow_sub_one_eq_mod row 5

/-- A plane: 32 rows per word. -/
abbrev Plane := Array (BitVec 32)

/-- `n.div_ceil(32)`. -/
def nWords (n : Nat) : Nat := (n + 31) / 32

/-- A plane of `nw` zero words. -/
def zeros (nw : Nat) : Plane := Array.replicate nw 0#32

/-- `plane[row >> 5] |= 1 << (row & 31)`. -/
def setRow (pl : Plane) (row : Nat) : Plane :=
  pl.modify (row / 32) (fun w => w ||| (1#32 <<< (row % 32)))

/-- `(plane[row >> 5] >> (row & 31)) & 1 == 1`; a word past the end reads as 0. -/
def bitAt (pl : Plane) (row : Nat) : Bool := (pl.getD (row / 32) 0#32).getLsbD (row % 32)

/-- The rows of a list, set one after the other. -/
def setRows (pl : Plane) (rows : List Nat) : Plane := rows.foldl setRow pl

theorem word_lt (r n : Nat) (h : r < n) : r / 32 < nWords n := by
  unfold nWords; omega

@[simp] theorem size_setRow (pl : Plane) (r : Nat) : (setRow pl r).size = pl.size := by
  simp [setRow]

@[simp] theorem size_setRows (pl : Plane) (rows : List Nat) : (setRows pl rows).size = pl.size := by
  unfold setRows
  induction rows generalizing pl with
  | nil => rfl
  | cons r rs ih => simp [List.foldl_cons, ih]

@[simp] theorem size_zeros (nw : Nat) : (zeros nw).size = nw := by simp [zeros]

theorem bitAt_zeros (nw s : Nat) : bitAt (zeros nw) s = false := by
  unfold bitAt zeros
  by_cases h : s / 32 < nw
  · simp [Array.getD, h]
  · simp [Array.getD, h]

theorem getD_setRow_same (pl : Plane) (r : Nat) (hr : r / 32 < pl.size) :
    (setRow pl r).getD (r / 32) 0#32 = (pl.getD (r / 32) 0#32 ||| (1#32 <<< (r % 32))) := by
  simp [setRow, Array.getD_eq_getD_getElem?, hr, Array.getElem_modify]

theorem getD_setRow_other (pl : Plane) (r i : Nat) (h : ¬ r / 32 = i) :
    (setRow pl r).getD i 0#32 = pl.getD i 0#32 := by
  simp [setRow, Array.getD_eq_getD_getElem?, Array.getElem?_modify, h]

/-- Setting row `r` sets bit `r` and no other bit, when the word of `r` exists. -/
theorem bitAt_setRow (pl : Plane) (r s : Nat) (hr : r / 32 < pl.size) :
    bitAt (setRow pl r) s = (decide (s = r) || bitAt pl s) := by
  unfold bitAt
  by_cases hw : r / 32 = s / 32
  · rw [← hw, getD_setRow_same pl r hr, BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_one]
    have h32 : s % 32 < 32 := Nat.mod_lt _ (by decide)
    by_cases hm : s = r
    · subst hm; simp [h32]
    · have hne : ¬ s % 32 = r % 32 := by omega
      have h1 : (decide (s = r)) = false := by simp [hm]
      rw [h1]
      by_cases hlt : s % 32 < r % 32
      · simp [hlt]
      · have h2 : ¬ (s % 32 - r % 32 = 0) := by omega
        simp [h2]
  · have hne : ¬ s = r := by intro h; exact hw (by rw [h])
    rw [getD_setRow_other pl r (s / 32) hw]
    simp [hne]

/-- A row past the last word reads as 0. -/
theorem bitAt_of_size_le (pl : Plane) (s : Nat) (h : pl.size ≤ s / 32) : bitAt pl s = false := by
  unfold bitAt
  simp [Array.getD, Nat.not_lt.mpr h]

/-- After setting a list of rows, a bit is set exactly when its row is in the list or it was set. -/
theorem bitAt_setRows (pl : Plane) (rows : List Nat) (s : Nat) (hr : ∀ r ∈ rows, r / 32 < pl.size) :
    bitAt (setRows pl rows) s = (decide (s ∈ rows) || bitAt pl s) := by
  unfold setRows
  induction rows generalizing pl with
  | nil => simp
  | cons r rs ih =>
    have h1 : r / 32 < pl.size := hr r (by simp)
    have h2 : ∀ x ∈ rs, x / 32 < (setRow pl r).size := by
      intro x hx; rw [size_setRow]; exact hr x (by simp [hx])
    rw [List.foldl_cons, ih (setRow pl r) h2, bitAt_setRow pl r s h1]
    by_cases hs : s = r
    · simp [hs]
    · simp [hs]

theorem setRows_cons (pl : Plane) (r : Nat) (rows : List Nat) :
    setRows pl (r :: rows) = setRows (setRow pl r) rows := rfl

@[simp] theorem setRows_nil (pl : Plane) : setRows pl [] = pl := rfl

end Rcsr.FlatBound.Planes
