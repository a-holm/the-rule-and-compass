---
type: reference
description: How to measure a change so the result is repeatable: pinned binaries, paired holds, tagged numbers, a verifier.
---

# Measurement method

A measurement answers one question with one number. The numbers of this machine are local records under `$VLOCAL` (see [local.md](local.md)); this file holds the method only.

1. Before the run, write what is compared and which number decides. Write the threshold for a decision at the same time.
2. Pin both binaries with [bin-pin.sh](../_tools/bin-pin.sh). Record the sha256 and the commit of each. A run card holds them: [run-record.sh](../_tools/run-record.sh).
3. Prove equal results first: the archive and finalist hashes of A and B are equal. If they differ, the change is not a pure speed change and the timing says nothing.
4. Run A and B inside one lock hold, in alternating order, with [ab-pair.sh](../_tools/ab-pair.sh). Never compare a run with an older timing: load, clocks and neighbours differ.
5. Report wall time, user CPU time, the load and the CPU mask of every arm. Under idle scheduling on shared cores wall time is noise. Repeat one arm and state the spread. A difference inside the spread is no difference.
6. Choose levers from the stage ledger shares of the run itself. Shares stay stable when wall time does not. State the share threshold in advance.
7. Tag every number `measured`, `derived` or `estimate`. A derived number names its inputs. An estimate says what would turn it into a measurement.
8. A second agent, the verifier, repeats the headline number with the same pinned binaries. The result stands only when the repeat agrees within the stated spread.
9. When a real-data result must not be seen, profile on a synthetic panel of the same shape (same groups, bars, columns and flags).
10. Keep other CPU work off the cores of the measured process. Name the cores of that other work in the record.
11. Write the record from [the measurement template](../_templates/local/measurement.md) into `$VLOCAL/measurements/<date>-<slug>.md`. Add the headline to `machine.md` with the record name in its last column.

Never in a record: raw data, a value from inside a locked range before the seal, secrets, personal data. No measured number of any machine goes into a tracked file.
