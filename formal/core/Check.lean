import RcsrCore
open Rcsr Rcsr.Prim

def hexVal (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 'a'.toNat + 10) else none

def parseWord (s : String) : Option W :=
  if s.length != 16 then none else
    s.foldl (fun acc c => do let a ← acc; let d ← hexVal c; pure (a * 16 + d)) (some 0) |>.map (BitVec.ofNat 64)

def ordCode : Ordering → W
  | .lt => 0 | .eq => 1 | .gt => 2

structure Model where
  sig : String
  nin : Nat
  nout : Nat
  run : Array W → Array W

/-- The registry: one line per model. `sig` is the text after `sig=` in the vector header. -/
def model (name : String) : Option Model :=
  match name with
  | "cmp_error" => some ⟨"w,w->w", 2, 1, fun a => #[ordCode (ErrorOrder.cmpError a[0]! a[1]!)]⟩
  | "ulps_step" => some ⟨"w,w->w", 2, 1, fun a => #[FlatBound.Rows.ulpsStep a[0]! a[1]!]⟩
  | _ => none

def checkFile (path : String) : IO Bool := do
  let lines ← IO.FS.lines path
  if lines.size < 2 then IO.eprintln s!"{path}: too short"; return false
  let head := lines[0]!.splitOn " "
  if head.length != 4 || head[0]! != "rcsr-vec" || head[1]! != "1" then
    IO.eprintln s!"{path}: bad header"; return false
  let some m := model head[2]! | IO.eprintln s!"{path}: unknown model {head[2]!}"; return false
  if head[3]! != s!"sig={m.sig}" then
    IO.eprintln s!"{path}: signature differs from the model, want sig={m.sig}"; return false
  let n := lines.size - 2
  if lines[lines.size - 1]! != s!"end {n}" then IO.eprintln s!"{path}: missing or wrong end line"; return false
  let mut bad := 0
  for i in [1:lines.size - 1] do
    let toks := (lines[i]!.splitOn " ").toArray
    let ws := toks.filterMap parseWord
    if ws.size != toks.size || ws.size != m.nin + m.nout then
      IO.eprintln s!"{path}:{i+1}: malformed"; return false
    let got := m.run (ws.extract 0 m.nin)
    if got != ws.extract m.nin (m.nin + m.nout) then
      bad := bad + 1
      if bad ≤ 5 then IO.eprintln s!"{path}:{i+1}: model gives {got.map (·.toHex)}, file {lines[i]!}"
  IO.println s!"{head[2]!}: {n} vectors, {bad} mismatches"
  return bad == 0

def main (args : List String) : IO UInt32 := do
  if args.isEmpty then IO.eprintln "usage: rcsr_model_check <file.vec>..."; return 2
  let mut ok := true
  for a in args do
    if !(← checkFile a) then ok := false
  return if ok then 0 else 1
