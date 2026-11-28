import RcsrCore.Intervals.Effective

/-!
Model of the interval layer of `rcsr-engine/src/cpcv.rs`: `floor_day`, `ceil_day`, `hole`, `subtract`,
`usable`, `slice` and the group bounds of `calendar_groups`.

`subtract(pieces, za, zb)` is `Intervals.cut (za, zb) pieces`: the loop body is the one of
`IntervalSplit::effective`.

Limits: the model is written by hand on `Int`. The combinatorics (`combinations`, `assignment`), the
split construction of `generate` and every `f64` value are outside the model.
-/

namespace Rcsr.Cpcv
open Rcsr.Intervals

/-- `floor_day`: `t.div_euclid(DAY) * DAY`. -/
def floorDay (t : Int) : Int := t / 86400 * 86400

/-- `ceil_day`: `(t + DAY - 1).div_euclid(DAY) * DAY`. -/
def ceilDay (t : Int) : Int := (t + 86400 - 1) / 86400 * 86400

/-- `Locked`: a locked range and the embargo after its end. The Rust field `end` is `stop` here. -/
structure Locked where
  start : Int
  stop : Int
  embargo : Int

/-- `hole`. -/
def hole (l : Locked) (purge : Int) : Iv :=
  (floorDay (l.start - max purge 1 + 1), ceilDay (l.stop + max (max l.embargo purge) 0))

/-- `usable`: `[start, end)` without the holes; empty pieces are dropped. -/
def usable (s e : Int) (holes : List Iv) : List Iv :=
  (cutAll holes [(s, e)]).filter (fun p => decide (p.2 > p.1))

/-- The loop of `slice`, started at the usable offset `off`. -/
def sliceFrom (off : Int) : List Iv → Int → Int → List Iv
  | [], _, _ => []
  | p :: rest, lo, hi =>
    (if min hi (off + (p.2 - p.1)) > max lo off
      then [(p.1 + max lo off - off, p.1 + min hi (off + (p.2 - p.1)) - off)] else []) ++
      sliceFrom (off + (p.2 - p.1)) rest lo hi

/-- `slice`: the pieces of usable time between the usable offsets `[lo, hi)`. -/
def slice (pieces : List Iv) (lo hi : Int) : List Iv := sliceFrom 0 pieces lo hi

/-- The stamp `t` sits at the usable offset `k` of the pieces, counted from `off`. -/
def atOffset (off : Int) : List Iv → Int → Int → Prop
  | [], _, _ => False
  | p :: rest, t, k => (memIv p t ∧ k = off + (t - p.1)) ∨ atOffset (off + (p.2 - p.1)) rest t k

/-- `bound(g)` of `calendar_groups`: the usable offset where group `g` starts. -/
def groupBound (days n g : Int) : Int := (g * days) / n * 86400

/-! ### Days and holes -/

theorem floorDay_spec (t : Int) : floorDay t ≤ t ∧ t < floorDay t + 86400 ∧ floorDay t % 86400 = 0 := by
  unfold floorDay; omega

theorem ceilDay_spec (t : Int) : t ≤ ceilDay t ∧ ceilDay t < t + 86400 ∧ ceilDay t % 86400 = 0 := by
  unfold ceilDay; omega

/-- Every stamp that reaches the locked range, with the embargo raised to the purge, lies in the hole. -/
theorem hole_covers_reach (l : Locked) (purge t : Int)
    (h : reaches l.start l.stop t purge (max l.embargo purge)) : memIv (hole l purge) t := by
  unfold reaches at h
  have h1 := floorDay_spec (l.start - max purge 1 + 1)
  have h2 := ceilDay_spec (l.stop + max (max l.embargo purge) 0)
  unfold memIv hole; simp only; omega

theorem hole_ordered (l : Locked) (purge : Int) (hl : l.start < l.stop) :
    (hole l purge).1 < (hole l purge).2 := by
  have h1 := floorDay_spec (l.start - max purge 1 + 1)
  have h2 := ceilDay_spec (l.stop + max (max l.embargo purge) 0)
  unfold hole; simp only; omega

/-! ### `usable` -/

theorem mem_usable (s e : Int) (holes : List Iv) (t : Int) :
    mem (usable s e holes) t ↔ (s ≤ t ∧ t < e) ∧ ∀ h ∈ holes, ¬ memIv h t := by
  have key : mem (usable s e holes) t ↔ mem (cutAll holes [(s, e)]) t := by
    unfold usable mem
    constructor
    · rintro ⟨p, hp, hpt⟩; exact ⟨p, (List.mem_filter.mp hp).1, hpt⟩
    · rintro ⟨p, hp, hpt⟩
      exact ⟨p, List.mem_filter.mpr ⟨hp, by unfold memIv at hpt; simp only [decide_eq_true_eq]; omega⟩, hpt⟩
  rw [key, mem_cutAll]
  simp [mem, memIv]

/-- No usable stamp reaches a locked range whose hole was removed. -/
theorem usable_clear_of_locked (s e : Int) (locked : List Locked) (purge t : Int)
    (h : mem (usable s e (locked.map (fun l => hole l purge))) t) :
    ∀ l ∈ locked, ¬ reaches l.start l.stop t purge (max l.embargo purge) := by
  intro l hl hr
  exact ((mem_usable s e _ t).mp h).2 (hole l purge) (List.mem_map.mpr ⟨l, hl, rfl⟩)
    (hole_covers_reach l purge t hr)

theorem usable_chain (s e : Int) (holes : List Iv) (hse : s < e) (hh : ∀ h ∈ holes, h.1 ≤ h.2) :
    Chain (usable s e holes) := by
  have hw := cutAll_within holes (s, e) [(s, e)] hh (within_self (s, e) hse)
  exact ⟨hw.1.sublist List.filter_sublist, fun p hp => (hw.2 p (List.mem_filter.mp hp).1).2.1⟩

/-! ### `slice` -/

/-- Every piece of a slice is non-empty and lies inside one of the pieces. -/
theorem sliceFrom_sub (off : Int) (ps : List Iv) (lo hi : Int) :
    ∀ q ∈ sliceFrom off ps lo hi, ∃ p ∈ ps, p.1 ≤ q.1 ∧ q.1 < q.2 ∧ q.2 ≤ p.2 := by
  induction ps generalizing off with
  | nil => intro q hq; simp [sliceFrom] at hq
  | cons p rest ih =>
    intro q hq
    simp only [sliceFrom, List.mem_append] at hq
    rcases hq with hq | hq
    · by_cases c : min hi (off + (p.2 - p.1)) > max lo off
      · simp only [c, ite_true, List.mem_singleton] at hq
        subst hq; exact ⟨p, List.mem_cons_self, by simp only; omega⟩
      · simp [c] at hq
    · obtain ⟨p', hp', h⟩ := ih (off + (p.2 - p.1)) q hq
      exact ⟨p', List.mem_cons_of_mem _ hp', h⟩

theorem slice_subset (pieces : List Iv) (lo hi t : Int) (h : mem (slice pieces lo hi) t) :
    mem pieces t := by
  obtain ⟨q, hq, hqt⟩ := h
  obtain ⟨p, hp, h1⟩ := sliceFrom_sub 0 pieces lo hi q hq
  exact ⟨p, hp, by unfold memIv at hqt ⊢; omega⟩

theorem sliceFrom_pairwise (off : Int) (ps : List Iv) (lo hi : Int)
    (hc : ps.Pairwise (fun x y => x.2 ≤ y.1)) :
    (sliceFrom off ps lo hi).Pairwise (fun x y => x.2 ≤ y.1) := by
  induction ps generalizing off with
  | nil => simp [sliceFrom]
  | cons p rest ih =>
    have hc' := List.pairwise_cons.mp hc
    simp only [sliceFrom]
    refine List.pairwise_append.mpr ⟨?_, ih (off + (p.2 - p.1)) hc'.2, ?_⟩
    · by_cases c : min hi (off + (p.2 - p.1)) > max lo off <;> simp [c]
    · intro a ha b hb
      obtain ⟨r, hr, h1⟩ := sliceFrom_sub (off + (p.2 - p.1)) rest lo hi b hb
      have := hc'.1 r hr
      by_cases c : min hi (off + (p.2 - p.1)) > max lo off
      · simp only [c, ite_true, List.mem_singleton] at ha
        subst ha; simp only; omega
      · simp [c] at ha

theorem sliceFrom_chain (off : Int) (ps : List Iv) (lo hi : Int) (h : Chain ps) :
    Chain (sliceFrom off ps lo hi) := by
  refine ⟨sliceFrom_pairwise off ps lo hi h.1, fun q hq => ?_⟩
  obtain ⟨p, _, h1⟩ := sliceFrom_sub off ps lo hi q hq
  exact h1.2.1

theorem slice_chain (pieces : List Iv) (lo hi : Int) (h : Chain pieces) : Chain (slice pieces lo hi) :=
  sliceFrom_chain 0 pieces lo hi h

/-- A stamp is in a slice exactly when it sits at a usable offset in `[lo, hi)`. -/
theorem mem_sliceFrom (off : Int) (ps : List Iv) (lo hi t : Int) :
    mem (sliceFrom off ps lo hi) t ↔ ∃ k, atOffset off ps t k ∧ lo ≤ k ∧ k < hi := by
  induction ps generalizing off with
  | nil => simp [sliceFrom, atOffset, mem]
  | cons p rest ih =>
    have hsplit : mem (sliceFrom off (p :: rest) lo hi) t ↔
        (memIv p t ∧ lo ≤ off + (t - p.1) ∧ off + (t - p.1) < hi) ∨
          mem (sliceFrom (off + (p.2 - p.1)) rest lo hi) t := by
      simp only [sliceFrom, mem, List.mem_append]
      constructor
      · rintro ⟨q, hq | hq, hqt⟩
        · left
          by_cases c : min hi (off + (p.2 - p.1)) > max lo off
          · simp only [c, ite_true, List.mem_singleton] at hq
            subst hq; unfold memIv at hqt ⊢; simp only at hqt; omega
          · simp [c] at hq
        · right; exact ⟨q, hq, hqt⟩
      · rintro (⟨hpt, h1, h2⟩ | ⟨q, hq, hqt⟩)
        · unfold memIv at hpt
          have c : min hi (off + (p.2 - p.1)) > max lo off := by omega
          exact ⟨(p.1 + max lo off - off, p.1 + min hi (off + (p.2 - p.1)) - off),
            Or.inl (by simp only [c, ite_true, List.mem_singleton]), by unfold memIv; simp only; omega⟩
        · exact ⟨q, Or.inr hq, hqt⟩
    rw [hsplit, ih (off + (p.2 - p.1))]
    simp only [atOffset]
    constructor
    · rintro (⟨hpt, h1, h2⟩ | ⟨k, hk, h1, h2⟩)
      · exact ⟨off + (t - p.1), Or.inl ⟨hpt, rfl⟩, h1, h2⟩
      · exact ⟨k, Or.inr hk, h1, h2⟩
    · rintro ⟨k, (⟨hpt, rfl⟩ | hk), h1, h2⟩
      · exact Or.inl ⟨hpt, h1, h2⟩
      · exact Or.inr ⟨k, hk, h1, h2⟩

theorem mem_slice (pieces : List Iv) (lo hi t : Int) :
    mem (slice pieces lo hi) t ↔ ∃ k, atOffset 0 pieces t k ∧ lo ≤ k ∧ k < hi :=
  mem_sliceFrom 0 pieces lo hi t

/-- An offset found after `off` belongs to a stamp of the pieces. -/
theorem atOffset_mem (off : Int) (ps : List Iv) (t k : Int) (h : atOffset off ps t k) : mem ps t := by
  induction ps generalizing off with
  | nil => exact absurd h (by simp [atOffset])
  | cons p rest ih =>
    simp only [atOffset] at h
    rcases h with ⟨hpt, _⟩ | h
    · exact ⟨p, List.mem_cons_self, hpt⟩
    · obtain ⟨q, hq, hqt⟩ := ih _ h
      exact ⟨q, List.mem_cons_of_mem _ hq, hqt⟩

/-- In a chain a stamp has one usable offset. -/
theorem atOffset_unique (off : Int) (ps : List Iv) (t k k' : Int) (hc : Chain ps)
    (h : atOffset off ps t k) (h' : atOffset off ps t k') : k = k' := by
  induction ps generalizing off with
  | nil => exact absurd h (by simp [atOffset])
  | cons p rest ih =>
    have hc' := List.pairwise_cons.mp hc.1
    have hrest : memIv p t → ¬ mem rest t := by
      rintro hpt ⟨y, hy, hyt⟩; have := hc'.1 y hy; unfold memIv at hpt hyt; omega
    simp only [atOffset] at h h'
    rcases h with ⟨hpt, rfl⟩ | h <;> rcases h' with ⟨hpt', rfl⟩ | h'
    · rfl
    · exact absurd (atOffset_mem _ _ _ _ h') (hrest hpt)
    · exact absurd (atOffset_mem _ _ _ _ h) (hrest hpt')
    · exact ih _ ⟨hc'.2, fun q hq => hc.2 q (List.mem_cons_of_mem _ hq)⟩ h h'

/-- Two slices at a common cut cover the whole slice. No condition on the pieces. -/
theorem slice_cover (pieces : List Iv) (lo mid hi t : Int) (h : lo ≤ mid ∧ mid ≤ hi) :
    mem (slice pieces lo hi) t ↔ mem (slice pieces lo mid) t ∨ mem (slice pieces mid hi) t := by
  simp only [mem_slice]
  constructor
  · rintro ⟨k, hk, h1, h2⟩
    by_cases c : k < mid
    · exact Or.inl ⟨k, hk, h1, c⟩
    · exact Or.inr ⟨k, hk, by omega, h2⟩
  · rintro (⟨k, hk, h1, h2⟩ | ⟨k, hk, h1, h2⟩)
    · exact ⟨k, hk, h1, by omega⟩
    · exact ⟨k, hk, by omega, h2⟩

/-- Slices of a chain over disjoint offset ranges share no stamp. -/
theorem slice_disjoint (pieces : List Iv) (lo mid mid' hi t : Int) (hc : Chain pieces) (hm : mid ≤ mid') :
    ¬ (mem (slice pieces lo mid) t ∧ mem (slice pieces mid' hi) t) := by
  simp only [mem_slice]
  rintro ⟨⟨k, hk, _, h2⟩, ⟨k', hk', h1', _⟩⟩
  have := atOffset_unique 0 pieces t k k' hc hk hk'
  omega

/-! ### Group bounds of `calendar_groups` -/

theorem groupBound_zero (days n : Int) : groupBound days n 0 = 0 := by simp [groupBound]

theorem groupBound_last (days n : Int) (hn : 0 < n) : groupBound days n n = days * 86400 := by
  unfold groupBound
  rw [Int.mul_ediv_cancel_left _ (Int.ne_of_gt hn)]

theorem groupBound_mono (days n g : Int) (hn : 0 < n) (hd : 0 ≤ days) :
    groupBound days n g ≤ groupBound days n (g + 1) := by
  unfold groupBound
  have h1 : g * days ≤ (g + 1) * days := by
    rw [Int.add_mul]; omega
  have h2 := Int.ediv_le_ediv hn h1
  omega

end Rcsr.Cpcv
