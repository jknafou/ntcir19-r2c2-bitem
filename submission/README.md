# submission/ — the runs as submitted

Three folders: the passage retrieval runs as submitted in April, the answer runs as submitted in May, and the three unofficial runs of Table 4, which the organisers never scored.

```
submission/
├── pr-2026-04-25/        # passage retrieval, submitted on 25 April 2026
│   ├── BITEM-PG-1
│   ├── BITEM-PG-2
│   ├── BITEM-PR.zip      # ← the file uploaded to the organisers
│   └── submission.SHA256
├── ac-2026-05-15/        # answer construction, submitted on 15 May 2026
│   ├── BITEM-AC-1
│   ├── BITEM-AC-2
│   ├── BITEM-AC-3
│   ├── BITEM-AC-4        # a byte-copy of BITEM-AC-3, see below
│   ├── BITEM-AC.zip      # ← the file uploaded to the organisers
│   ├── BITEM-PG-1
│   ├── BITEM-PG-2
│   ├── BITEM-PR.zip      # not uploaded, see below
│   └── submission.SHA256
└── unofficial/
    ├── model-as-judge.AC                  # the run meant to be submitted as BITEM-AC-4
    ├── revised-longest-deliberation.AC    # rebuilt by rule from the first pass
    ├── revised-independent-agreement.AC   # rebuilt by rule from all the passes
    └── unofficial.SHA256
```

In the two official folders, the loose files are the unzipped contents of the
zip beside them. Each folder's checksum file covers every file in it:

```
cd submission/pr-2026-04-25 && shasum -a 256 -c submission.SHA256
cd submission/ac-2026-05-15 && shasum -a 256 -c submission.SHA256
cd submission/unofficial && shasum -a 256 -c unofficial.SHA256
```

(`sha256sum -c` on Linux.)

## The runs

What each run is, under the name it was submitted with:

| Run | What it is |
|---|---|
| `BITEM-PG-1` | Passages from the first pass on each question |
| `BITEM-PG-2` | Passages pooled over all recorded passes |
| `BITEM-AC-1` | The answer of the single pass behind PG-1 |
| `BITEM-AC-2` | The most confident answer across passes |
| `BITEM-AC-3` | The answer with the largest summed agreement across passes |
| `BITEM-AC-4` (submitted) | A byte-copy of BITEM-AC-3 |
| `unofficial/model-as-judge.AC` | The answer chosen by the model acting as a judge over the passes |
| `unofficial/revised-longest-deliberation.AC` | The first pass's own last answer, with a confidence of 100 or 0 set by the length of its longest reasoning turn |
| `unofficial/revised-independent-agreement.AC` | The answer most passes reached, with a confidence of 100, 50 or 0 set by how many passes reached it on their own |

## Question 0025 changed between the two deadlines

At the retrieval deadline, question 0025 read *"In the movie May 18
(Hwaryeohan Hyooga), what is the occupation of the main character
Min-Woo?"*. By the answer deadline it had been reissued as *"Which movie
had a higher box office, May 18 (Hwaryeohan Hyooga) or 12.12: The Day
(Seourui bom)?"*. We ran the agent again on the new text and submitted its
answers in `BITEM-AC-*`. Their nuggets cite passages retrieved for the new
text, so `ac-2026-05-15/` carries its own `BITEM-PG-1`, `BITEM-PG-2` and
`BITEM-PR.zip`, in which only the rows of question 0025 differ from
`pr-2026-04-25/`. That PR copy was not uploaded: the organisers kept and
scored the April files, which are identical to `pr-2026-04-25/`
(`analysis/pr_ensemble.py` checks this).

The organisers then left question 0025 out of the official evaluation.
The official tables, and the analyses here, cover the other 64 questions.

## BITEM-AC-4 was submitted as a copy of BITEM-AC-3

The submitted `BITEM-AC-4` is byte-for-byte `BITEM-AC-3`, here and in the
organisers' copy, so the official AC table carries two identical BITEM
rows. The analyses count it once.

The run that should have been submitted under that name, the model acting as
a judge, is `unofficial/model-as-judge.AC`. The organisers never scored it;
`analysis/lost_ac4.py` scores it against the gold answers, so its numbers are
ours and not official. For question 0025 it still answers from the April
passages.

## What new runs write here

Two scripts write beside the submitted runs. Git ignores their folders.

- `bash src/r2c2_agent/slurm/launch.sh --demo` writes its traces to
  `submission/demo-<timestamp>/`.
- `bash scripts/pipeline/07_submission.sh` assembles a packet laid out
  like `ac-2026-05-15/` in `submission/produced-<timestamp>/`
  (`produced-<timestamp>_demo/` with `--demo`).
