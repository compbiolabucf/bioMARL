# BioMARL

Official code for **"Biological Pathway Guided Gene Selection Through Collaborative Reinforcement Learning"** (KDD 2025).

📄 Paper: https://arxiv.org/abs/2505.24155

BioMARL selects a compact, biologically meaningful subset of genes from high-dimensional
gene-expression data in two stages: (1) a **pathway-guided pre-filter** that combines statistical
scorers with KEGG pathway performance to reduce ~20k genes to a candidate set, and (2) a
**multi-agent RL selector** in which each candidate gene is a DQN agent that learns whether to be
selected, using a shared gene-graph GNN state, a centralized critic, a pairwise synergy memory, and
a reward blending predictive impact with pathway centrality and coverage.

---

## Repository layout

```
biomarl/
├── run.py                 # CLI entry point
├── model.py               # MARL model: GNN state, DQN agents, critic, shared memory, reward
├── prefilter.py           # pathway-guided pre-filter
├── feature_env.py         # downstream AUC evaluator
├── record.py
├── scorers/               # base scorers used by the pre-filter (K-Best / SVM / random forest)
├── baselines/             # comparison methods (see "Baselines")
└── utils/                 # metrics + logging
requirements.txt
pathways_new.txt           # KEGG pathway -> gene membership
```

## Installation

Tested on **Python 3.10, CUDA 12.4, NVIDIA GPUs** (a CUDA GPU is required).

```bash
# install a CUDA-matched PyTorch first (example: CUDA 12.4)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> `tables` (PyTables) must be **>= 3.10** (older builds are binary-incompatible with numpy 2.x and
> break HDF5 loading).

## Data

Download the preprocessed `data/` folder and place it at the repository root:

- **Data:** https://mega.nz/folder/SnYSxABB#71G8EDKgVgT_vDsncezc5Q

```
<repo root>/
├── data/
│   ├── BRCA_ER.hdf  BRCA_HER2.hdf  BRCA_PR.hdf  BRCA_TN.hdf  LUAD_cls.hdf  OV_cls.hdf
│   └── paths_er/ paths_her2/ paths_pr/ paths_tn/ paths_luad_cls/ paths_ov_cls/   # gene_cls_*.h5
├── pathways_new.txt
└── biomarl/
```

Available `--task_name` values: `BRCA_ER`, `BRCA_HER2`, `BRCA_PR`, `BRCA_TN`, `LUAD_cls`, `OV_cls`.
Paths default to repo-relative locations; override with `BIOMARL_DATA_DIR`, `BIOMARL_RESULT_DIR`,
`BIOMARL_PATHWAY_FILE`, or `BIOMARL_ROOT` if your data lives elsewhere.

---

## Running an experiment

```bash
python -m biomarl.run \
  --task_name BRCA_TN \
  --prefilter_num_feats 1000 \
  --marlfs_max_feats 100 \
  --marlfs_explore_steps 2600 \
  --num_marlfs_runs 10 \
  --n_samples 1500 \
  --use_meta --shared_memory --critic
```

Results (per-run and mean ± SD ROC/PR AUC + the ranked gene list) are written to
`results/marlfs_results_<task>_<timestamp>.txt`.

| Flag | Meaning |
|------|---------|
| `--task_name` | dataset to run |
| `--prefilter_num_feats` | genes kept after pathway pre-filtering |
| `--marlfs_max_feats` | `k` — size of the final selected gene set |
| `--marlfs_explore_steps` | MARL exploration steps per run |
| `--num_marlfs_runs` | independent runs (distributed over visible GPUs) |
| `--n_samples` | training examples for the reward meta-learner |
| `--use_meta` / `--shared_memory` / `--critic` | enable the three model components (see Ablations) |

**GPU selection.** Runs are distributed over the GPUs visible to the process. Pin GPUs with
`CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=0 python -m biomarl.run ...`.

**Short test.** DQN learning starts once the replay buffer fills (default 1700 steps). To exercise
learning quickly, lower it with `BIOMARL_MEMORY_CAPACITY`, e.g.
`BIOMARL_MEMORY_CAPACITY=100 python -m biomarl.run ... --marlfs_explore_steps 300`.

## Ablations

Each of the three model components is toggled by a flag; **drop the flag to ablate it**. Keep the
other two enabled and everything else identical to the full run:

| Ablation | Command |
|----------|---------|
| Full model | `... --use_meta --shared_memory --critic` |
| − Personalized reward | `... --shared_memory --critic`  *(drop `--use_meta`)* |
| − Centralized critic | `... --use_meta --shared_memory`  *(drop `--critic`)* |
| − Shared memory | `... --use_meta --critic`  *(drop `--shared_memory`)* |

Example:

```bash
# ablation: no centralized critic
python -m biomarl.run --task_name BRCA_TN --prefilter_num_feats 1000 --marlfs_max_feats 100 \
  --marlfs_explore_steps 2600 --num_marlfs_runs 10 --n_samples 1500 --use_meta --shared_memory
```

## Baselines

The eight comparison methods from the paper are in `biomarl/baselines/`:
**K-Best, mRMR, LASSO, RFE, LASSONet, GFS (genetic), RRA (robust rank aggregation), MCDM.**

Each shares BioMARL's downstream evaluator and has the same turnkey CLI: `--k` selects the number of
features, and the run iterates all six gene tasks (10 runs each), writing per-task + summary results to
`results_v3/`:

```bash
pip install -r requirements-baselines.txt        # extra baseline deps (see notes below)

python -m biomarl.baselines.KBest    --k 100
python -m biomarl.baselines.mRMR     --k 100
python -m biomarl.baselines.LASSO    --k 100
python -m biomarl.baselines.RFE      --k 100
python -m biomarl.baselines.LASSONet --k 100
python -m biomarl.baselines.GFS      --k 100
python -m biomarl.baselines.RRA      --k 100
python -m biomarl.baselines.MCDM     --k 100
```

All eight build `FeatureEvaluator(task)` directly from `data/<task>.hdf` — no extra pre-computation.

**Dependency notes** (`biomarl/baselines/__init__.py` installs a small `sklearn.utils._joblib` shim so
the legacy packages import under modern scikit-learn):
- `mcdm`, `xgboost` — MCDM and RRA (RRA reuses MCDM's model stack via `rest()`).
- `lassonet` — LASSONet.
- `mrmr_selection` — mRMR (its `mrmr` module can require an older scikit-learn at runtime).
- `sklearn-genetic-opt` — GFS (`GASearchCV`); it pulls in TensorFlow, which must be compatible with your
  numpy build.
