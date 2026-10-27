# Decisions

Log of meaningful technical decisions in this repo. Newest first.

Entries marked inferred were reconstructed after the fact. A Why line
starting with 'Likely:' is a guess; any other Why line is the author's
stated reason.

---

### 2026-10-27 — Remove the legacy SQL files
- **Decision:** Delete `sql/` (`schema.sql` and `employee_attrition_analysis.sql`) and its references in `README.md` and `flow.md`.
- **Why:** Removed legacy SQL: queried the old HR_1/HR_2 tables, not the current dataset; redundant for the ML pipeline.
- **Alternatives considered:** Keeping them labeled as legacy; rewriting them against the single-CSV schema.
- **Tradeoff accepted:** The SQL work is only visible in git history.
- **Source:** made-now

### 2026-10-27 — Keep the legacy SQL files, labeled as not wired in
Superseded by Remove the legacy SQL files.
- **Decision:** Keep `sql/schema.sql` and `sql/employee_attrition_analysis.sql` (a two-table `hr_1`/`hr_2` MySQL design), with header comments marking them legacy/illustrative and not used by `src/`.
- **Why:** Likely: they show SQL work from the project's earlier BI-dashboard phase, and deleting them would lose that history from the visible repo.
- **Alternatives considered:** Deleting them; rewriting them against the single-CSV schema.
- **Tradeoff accepted:** The repo contains code that does not run against the current dataset.
- **Source:** inferred

### 2026-10-27 — Dark "copper & teal" matplotlib style for all plots
- **Decision:** All figures use a dark background (`#0B0F19`) with teal (`#14B8A6`) for retained and copper (`#F97316`) for attrition, set through `rcParams` in `scripts/generate_visuals.py` and through hard-coded constants in `src/evaluate.py`.
- **Why:** Likely: a consistent visual identity for README and portfolio display.
- **Alternatives considered:** Default matplotlib/seaborn themes.
- **Tradeoff accepted:** The style constants are duplicated in two files rather than shared from one module.
- **Source:** inferred

### 2026-10-27 — SHAP TreeExplainer for explainability
- **Decision:** Explain the XGBoost model with `shap.TreeExplainer(feature_perturbation="tree_path_dependent")` and a summary dot plot written to `assets/shap_summary.png`, subsampling to at most 500 rows.
- **Why:** Likely: TreeExplainer gives exact, fast SHAP values for tree ensembles, and per-feature direction is easier to communicate to non-technical readers than raw feature importances.
- **Alternatives considered:** Built-in tree feature importances; permutation importance; KernelExplainer (model-agnostic but slow).
- **Tradeoff accepted:** Only works for tree models; the subsample uses an unseeded `np.random.choice` (not hit at 441 test rows).
- **Source:** inferred

### 2026-10-27 — joblib serialization of fitted models, not tracked in git
- **Decision:** Save the fitted classifier with `joblib.dump` to `models/<model>_model.pkl`; `.gitignore` excludes `models/*.pkl` and `models/*.sav`.
- **Why:** Likely: joblib is the standard scikit-learn persistence format, and binaries are regenerable so they don't belong in git.
- **Alternatives considered:** Plain `pickle`; committing the model; ONNX or XGBoost's native JSON format.
- **Tradeoff accepted:** Only the classifier is saved, not the fitted encoders or scaler, so the artifact cannot score raw rows on its own. Anyone cloning must retrain to get a model.
- **Source:** inferred

### 2026-10-27 — Three fixed-hyperparameter classifiers, XGBoost as default
- **Decision:** `src/train.py` supports Logistic Regression (C=1.0, max_iter=200), Random Forest (150 trees, max_depth=15, min_samples_split=5, min_samples_leaf=10) and XGBoost (200 trees, learning_rate=0.1, max_depth=6). XGBoost is the default in `train_pipeline()` and `run_full_pipeline_evaluation()`.
- **Why:** Likely: LR as an interpretable baseline, RF and XGBoost as the usual strong choices for tabular data, with XGBoost assumed best.
- **Alternatives considered:** Hyperparameter search; LightGBM/CatBoost.
- **Tradeoff accepted:** No tuning and no cross-validation, so the hyperparameters are hand-picked. XGBoost is the default by assumption, not because a comparison showed it was best.
- **Source:** inferred

### 2026-10-27 — SMOTE oversampling on the training split only
- **Decision:** Apply `imblearn.over_sampling.SMOTE(random_state=42)` to the training split after scaling; a `comb_smoteenn` flag allows SMOTEENN instead (unused by default).
- **Why:** Likely: the target is imbalanced (about 16% attrition), and oversampling only after the split keeps synthetic rows out of the test set.
- **Alternatives considered:** Class weights (`class_weight`, `scale_pos_weight`); random undersampling; SMOTEENN (left as an option).
- **Tradeoff accepted:** SMOTE interpolates label-encoded categoricals into values that aren't real categories.
- **Source:** inferred

### 2026-10-27 — 70/30 stratified split with StandardScaler fitted on train
- **Decision:** `train_test_split(test_size=0.3, stratify=y, random_state=42)` gives 1,029 train / 441 test rows; `StandardScaler` is fitted on train and applied to test.
- **Why:** Likely: stratification preserves the 16% positive rate in both splits, and fitting the scaler on train only avoids leaking test statistics.
- **Alternatives considered:** 80/20 split; k-fold cross-validation.
- **Tradeoff accepted:** One split means the metrics come from a single 441-row sample with no variance estimate.
- **Source:** inferred

### 2026-10-27 — LabelEncoder for all categorical columns
- **Decision:** Every object column (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`) is integer-encoded with `LabelEncoder`, fitted on the full dataset before the split.
- **Why:** Default choice, not deliberately compared. Wrong for Logistic Regression and for nominal features.
- **Alternatives considered:** One-hot encoding.
- **Tradeoff accepted:** Imposes a false ordering on nominal categories (hurts Logistic Regression); encoders are fitted before the split.
- **Source:** inferred

### 2026-10-27 — Drop ID and constant columns
- **Decision:** Drop `EmployeeNumber`, `EmployeeCount`, `StandardHours` and `Over18` (and `Employee ID` if present) before modeling.
- **Why:** `EmployeeNumber` is an identifier with no predictive meaning, and the other three hold one value for every row in this dataset.
- **Alternatives considered:** Keeping them and relying on the models to ignore them.
- **Tradeoff accepted:** None significant.
- **Source:** inferred

### 2026-10-27 — Single flat CSV (IBM HR Attrition), tracked in git
- **Decision:** The pipeline reads one file, `data/actual/WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 rows × 35 columns), with `encoding="utf-8-sig"` to strip the BOM. The CSV is committed to the repo.
- **Why:** The previous HR_1/HR_2 data (50k rows, from the original BI-dashboard version) was synthetic and unrelated to the ML goal. Replaced with the IBM HR Attrition sample, which is also fictional (created by IBM data scientists) but is the standard public benchmark with clear licensing. CSV tracked in commit `8721910`; pipeline switched to it in commit `ec5b8a6`.
- **Alternatives considered:** The earlier two-table Excel source; downloading from Kaggle at runtime.
- **Tradeoff accepted:** A small, fictional dataset limits how far any finding generalizes. Redistributing it requires following the dataset's own license.
- **Source:** inferred

### 2026-10-27 — Plain-script `src/` modules with a `sys.path` shim
- **Decision:** Pipeline stages are separate modules in `src/` (`data_ingestion`, `preprocessing`, `train`, `evaluate`), each runnable as `python src/<module>.py`, each inserting the repo root into `sys.path` so `from src.x import y` works.
- **Why:** Likely: run each stage directly without packaging or installing the project.
- **Alternatives considered:** An installable package (`pyproject.toml`) run with `python -m`; a single notebook.
- **Tradeoff accepted:** `sys.path` manipulation in each file; no CLI or config layer.
- **Source:** inferred

### 2026-10-27 — MIT license
- **Decision:** The repository is licensed under MIT (`LICENSE`). Covers code written in this repo only. data/ is under ODbL 1.0 / DbCL 1.0.
- **Why:** Likely: permissive, common default for portfolio and open-source code.
- **Alternatives considered:** Apache-2.0; no license.
- **Tradeoff accepted:** The `LICENSE` file itself doesn't distinguish the code from the third-party dataset in `data/`; the split is stated in the README License section.
- **Source:** inferred
