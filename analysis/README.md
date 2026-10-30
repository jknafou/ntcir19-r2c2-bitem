# analysis

The scripts behind the paper's tables, figures and numbers. Each one prints its numbers and writes a report beside itself. They use only the Python standard library, our runs in `submission/`, the tables committed in this folder, and the organisers' result files under `results/` (see the [repository README](../README.md#reproduce)).

Commands run from the repository root. Section numbers are the paper's.

| In the paper | Command |
| --- | --- |
| Table 1; the list lengths and the share of highly relevant passages (§4.1) | `python3 analysis/pr_ensemble.py submission` |
| Table 2; relevant passages per question by type (§4.1) | `python3 analysis/by_question_type.py submission` |
| Distinct articles per list, 8.6 against 13.1 (§4.1) | `python3 analysis/roster_duplicates.py submission` |
| Searches left at the default of 10 results (§4.1) | `python3 analysis/search_behaviour.py` |
| Table 3 | `python3 analysis/score_ac.py submission/ac-2026-05-15/BITEM-AC-1 submission/ac-2026-05-15/BITEM-AC-2 submission/ac-2026-05-15/BITEM-AC-3` |
| Ranks among the 25 answer runs (§4.2, §5.2) | `python3 analysis/all_runs_rescored.py` |
| Answers shared by the three runs, AUC of the single-pass confidence (§4.2) | `python3 analysis/ac_ensemble.py submission` |
| Single-pass answers given a confidence below 50 (§4.2) | `python3 analysis/confidence_behaviour.py submission` |
| Figure 2 | `python3 analysis/confidence_figure.py --panels=ac2`<br>`bash docs/figures/build.sh confidence-ac2` |
| Labels of the unofficial runs (§5) | `python3 analysis/answer_label_transfer.py submission` |
| Model as judge (§5.1, Table 4) | `python3 analysis/lost_ac4.py`<br>`python3 analysis/score_ac.py submission/unofficial/model-as-judge.AC` |
| The two rebuilt runs (§5.2, Table 4) | `python3 analysis/score_ac.py submission/unofficial/revised-longest-deliberation.AC submission/unofficial/revised-independent-agreement.AC`<br>`python3 analysis/rebuild_revised_runs.py --check submission/unofficial` |
| AUC of the longest reasoning turn (§5.2) | `python3 analysis/longest_turn_auc.py` |
| Figure 3 (§1, §6) | `python3 analysis/composite_shapes.py`<br>`bash docs/figures/build.sh composite-shapes` |

`score_ac.py` is the scorer the other scripts share, and `results_cache.tsv`, the per-question table several of them read, is rebuilt by `python3 analysis/build_results_cache.py submission`. The rules of the two rebuilt runs are in `make_revised_runs.py`; `rebuild_revised_runs.py` applies them to the committed tables and checks the result against `submission/unofficial/`. Building a figure needs LaTeX and a one-time setup, given at the top of `docs/figures/build.sh`.
