# Flow-Aware Temporal Pattern Mining for Multi-Stage Network Intrusion Detection (NIDS)

An M.Tech research project implementing a 13-stage, session-aware, chronologically
split intrusion detection pipeline on the **CIC-IDS2017** dataset. The pipeline
fuses three parallel detectors (Random Forest, BiLSTM, XGBoost) with frequent/
sequential pattern mining, a temporal attack-state graph, and an adaptive
risk-decision layer, then evaluates the whole system under streaming conditions
and produces a per-alert, three-modality explainability report.

## Why this design

Most NIDS papers score individual flows in isolation and shuffle train/test
splits, which silently leaks future information into the past and hides
multi-stage attacks that only become visible when several flows between the
same peers are read *in order*. This project instead:

- **Splits strictly chronologically** (Train = Days 1–2, Val = Day 3,
  Test = Days 4–7) — no shuffling anywhere, ever.
- **Reconstructs bidirectional sessions** from raw flows using a symmetric
  5-tuple key, so a multi-stage attack (recon → brute force → exfiltration)
  is visible as one evolving object instead of independent rows.
- **Never lets SMOTE+ENN touch anything temporal.** Synthetic flows have no
  real timestamps or behavioral meaning, so the "temporal-stream firewall"
  keeps `smote_enn_augment` output out of the BiLSTM, pattern mining, and the
  attack-state graph — it is used only for the tabular RF/XGBoost branch.
- **Freezes every learned weight before test.** Fusion weights are grid-searched
  on validation only; the risk meta-learner is fit on validation only; nothing
  is ever re-estimated on test or streaming data.

## Architecture (13 stages)

| Stage | What it does |
|---|---|
| 1 | Load the 7 CIC-IDS2017 day CSVs, fix known data-quality issues, chronological split |
| 2 | Clean/clip/scale features, encode labels, compute class weights, SMOTE+ENN (Branch B) |
| 3 | Assign features into 4 semantic groups (A: flow-statistical, B: protocol, C: temporal, D: TCP flags) |
| 4 | Reconstruct bidirectional sessions (symmetric 5-tuple key, 60s timeout, day/train-test isolation) |
| 5 | Rule-based behavioral event tokenization (10-token vocabulary, no label leakage) |
| 6 | Train RF (Group A+B), BiLSTM (token sequences), XGBoost (Group B+D) in parallel |
| 7 | Grid-search fusion weights on validation, freeze them |
| 8 | FP-Growth + PrefixSpan pattern mining on training sessions only |
| 9–10 | Temporal attack-state graph (EMA edge weights) + sequence consistency score TC_t |
| 11 | Logistic-regression meta-learner turns fused evidence into a risk score + alert tier |
| 12 | Streaming replay (60s window / 10s stride) — models frozen, only the graph adapts |
| 13 | Per-alert evidence chain: TreeSHAP (RF/XGB), DeepSHAP (BiLSTM), attack-graph path |

## Project structure

```
NIDS_Project/
├── NIDS_Pipeline_Complete.ipynb   # the full 13-stage pipeline notebook
├── config.py                      # every hyperparameter, in one place
├── requirements.txt
├── src/nids/                      # importable package, one module per stage
│   ├── data_loading.py            # Stage 1
│   ├── preprocessing.py           # Stage 2
│   ├── feature_engineering.py     # Stage 3
│   ├── sessions.py                # Stage 4
│   ├── events.py                  # Stage 5
│   ├── models.py                  # Stage 6
│   ├── fusion.py                  # Stage 7
│   ├── pattern_mining.py          # Stage 8
│   ├── attack_graph.py            # Stages 9-10
│   ├── risk.py                    # Stage 11
│   ├── streaming.py               # Stage 12
│   └── explainability.py          # Stage 13
└── tests/                         # dataset-free unit tests (synthetic fixtures)
```

## Setup

```bash
cd NIDS_Project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

Download the CIC-IDS2017 CSVs (`MachineLearningCSV.zip` from the
[UNB CIC-IDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html)) and
place the 7 per-day CSVs in the folder configured in `config.py`:

```python
DATASET_DIR = Path(r"D:\IDSPROJECT2026\CIC-IDS2017")
```

Either edit that path directly, or override it at runtime without touching
`config.py` by setting an environment variable before launching Jupyter:

```bash
# Windows (PowerShell)
$env:NIDS_DATASET_DIR = "D:\IDSPROJECT2026\CIC-IDS2017"
# macOS / Linux
export NIDS_DATASET_DIR=/path/to/CIC-IDS2017
```

## Running

```bash
jupyter notebook NIDS_Pipeline_Complete.ipynb
```

Run all cells top to bottom — each `[STAGE N]` section is self-contained and
saves its figure(s) to `figures/` as it goes. The `[RESULTS]` section at the
end writes `results/results_table.csv` and `results/model_comparison.csv`.

**Expect a full run over the complete CIC-IDS2017 dataset (~2.8M flows) to take
anywhere from tens of minutes to a few hours** depending on hardware — RF/XGBoost
training, 30-epoch BiLSTM training, and PrefixSpan mining are the main costs.
For a quick prototyping pass, point `NIDS_DATASET_DIR` at a subsampled copy of
the CSVs first.

### Tests

The test suite is entirely dataset-free (synthetic fixtures in
`tests/conftest.py`), so it runs anywhere without CIC-IDS2017 present:

```bash
pytest
```

## Key design decisions worth knowing before you read the code

- **`smote_enn_augment` is the only augmentation function in the project.**
  It is called exactly once per branch build, only for RF/XGBoost's tabular
  data, and is skipped entirely for any class below `SMOTE_MIN_SAMPLES` (50) —
  Heartbleed and Infiltration are handled by class weighting only.
- **A session's `Label` string is never overwritten**, even when a validation
  or test session belongs to an attack type unseen during training — only its
  derived `LabelID` (the K-way classifier target) falls back to BENIGN there,
  because the classifier genuinely cannot name a class it never saw. This
  keeps binary risk scoring (Stage 11) able to still call a novel/zero-day
  attack type "an attack," which is the entire point of the pattern-mining
  and attack-graph layers.
- **RF and XGBoost are flow-level classifiers**; Stage 7 onward needs one
  feature vector per *session* to fuse with the session-level BiLSTM/pattern/
  graph scores, so a session's flow feature vectors are mean-pooled into one
  representative row (`build_session_features` in the notebook) before being
  scored by the already-trained flow-level RF/XGBoost models.
- **Stage 7's fusion grid search needs SP_t/TC_t**, which are Stage 8/9-10
  outputs — but those stages only depend on Stage 5's training-session token
  sequences, not on anything from Stage 6. The notebook fits pattern mining
  and the attack graph inside the Stage 7 cell (documented there) so the
  fusion search has what it needs; Stages 8 and 9-10 then present and reuse
  those exact fitted artifacts rather than recomputing them.

## Terminology (used precisely throughout)

- **Streaming evaluation**, not "online learning" — RF/BiLSTM/XGBoost weights
  are frozen after Stage 6 and are only ever called with `.predict`/
  `.predict_proba` during streaming.
- **Graph update** — the attack-state graph's EMA edge-weight refresh during
  streaming. This is adaptive, parameter-free incremental statistics, not
  model learning.
- **Model update: none.** Nothing about RF, BiLSTM, or XGBoost changes after
  Stage 6 training completes.
