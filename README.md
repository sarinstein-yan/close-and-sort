# Close and Sort

Code for the paper **Close and Sort: Exact Maximum-Likelihood Aggregation of Cyclic Rank-Two Preferences**
(Kehan Liu, Zhaoyun Chen, Xianquan Yan), WSDM 2027.

## Setup

Use Python 3.11 or later. All experiments run on a CPU; no GPU is required.
Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reference environment uses Linux x86-64 with OpenBLAS. JuStRank gain
estimates and bootstrap intervals differed in a macOS/Accelerate run even
when profile counts and mean rank-two energy agreed. Floating-point
factorization and near-tied projected rankings are sensitive to the numerical
stack; cross-platform equality of those aggregates is not guaranteed.

`code/rank2_kemeny.py` contains CloseSort, the sign-only circular-cut baseline,
rank-two projection, exact dynamic programming and the exact MILP comparator.
`code/justrank_methods.py` contains the additional aggregation baselines used
in the real-data experiment.

## Reproduce the paper experiments

| Experiment | Command |
|---|---|
| Exact optimum, runner-up gap and residual stability | `python experiments/exactness_and_stability.py` |
| Sign-only versus weighted aggregation | `python experiments/sign_weight_separation.py` |
| Finite-sample determinant-logit estimation | `python experiments/finite_sample.py` |
| Complete-coverage LLM-judge aggregation | `python experiments/justrank_complete_coverage.py --input data/justrank_judge_scores.parquet` |
| Supplied-factor ranking-stage runtime | `python experiments/runtime.py` |

The defaults retain the paper's seeds and experimental settings. The
finite-sample experiment uses five seeds, 1,024 latent profiles per seed,
14 items and observation budgets 4, 16 and 64. The sign-only study uses
2,000 instances at each of its six sizes. The runtime benchmark uses
25 instances at each even size from 6 through 18 and five timing repeats;
it fixes numerical-library threads to one. Its input is the two-dimensional
factors: dense-matrix fitting is not part of this ranking-stage benchmark.

Each command computes new measurements and writes JSON or CSV under
`reproduced/`. Finite-sample summaries report population regret and projection
energy. The real-data output separates full-matrix baselines, same-factor
baselines on the empirical objective, and comparisons on the projected
rank-two objective. Runtime measurements depend on the machine and numerical
stack. No saved paper results or figures are distributed.

## Obtain the JuStRank input

Use the public [JuStRank judge-score dataset](https://huggingface.co/datasets/ibm-research/justrank_judge_scores)
at revision `b2863dc58936b7c5d8380c802601c21f26d48d2e`:

```bash
mkdir -p data
curl --fail --location \
  'https://huggingface.co/datasets/ibm-research/justrank_judge_scores/resolve/b2863dc58936b7c5d8380c802601c21f26d48d2e/data/test-00000-of-00001.parquet?download=true' \
  --output data/justrank_judge_scores.parquet
```

The script selects TokenProbs judgments, uses 100 questions to form disjoint
seven-system cohorts, and evaluates them on 200 disjoint questions. It uses
five-question blocks and 10,000 block-bootstrap repetitions. The source
filename is the dataset publisher's shard name; the construction and
evaluation question sets are selected by the script's fixed seed.

The dataset remains external and is subject to its upstream terms.
