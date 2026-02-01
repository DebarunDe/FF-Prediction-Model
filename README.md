# FF-Prediction-Model

Player pro statistics YOY given by FantasyPros


Goal

Implement a reproducible data pipeline that trains models on historical QB seasons (2020–2024) and uses 2025 pre-season projections (2025QBPROJ.csv) to produce adjusted 2025 fantasy-point forecasts, percentile intervals, and ranked outputs. Provide code, tests, and artifacts ready to run locally.
Repository & data layout (use these exact paths)

Repo root (assume current working dir)
Historical CSVs: data/pro/qb/ contains per-year files for 2020–2024 named e.g. 2020QB.csv and 2020QBADD.csv (one row per player-season).
2025 projections CSV: data/pro/qb/2025QBPROJ.csv (file content as provided).
Output dir: outputs/
Environment & libraries

Python 3.9+ (3.10 preferred)
Use these libraries: pandas, numpy, scikit-learn, xgboost (or lightgbm), scipy, joblib, shap, matplotlib, seaborn, pyyaml
Provide requirements.txt and an example virtualenv/venv setup instruction.
Key assumptions (must be explicit in code/config)

The 2025QBPROJ.csv has two sets of ATT/CMP/YDS/TDS columns: the first group = passing (pass_ATT, pass_CMP, pass_YDS, pass_TD, pass_INT); the second group = rushing (rush_ATT, rush_YDS, rush_TD). There is also FL (fumbles lost) and a precomputed FPTS field (baseline projection).
Default scoring (configurable): pass_yds = 0.04 pts/yd (1 pt / 25 yds), pass_TD = 4 pts, INT = -2 pts, rush_yds = 0.1 pts/yd (1 pt / 10 yds), rush_TD = 6 pts, fumble_lost = -2 pts.
Use season totals (not per-game) as primary target; include FPTS/G as alternate target if games column exists.
Use 17 games as default season games if not present; allow override via config.
Deliverables (code + artifacts)

scripts/
run_pipeline.py — main CLI (train, evaluate, predict).
features.py — feature engineering and preprocessing.
modeling.py — model training (ridge + xgboost baseline), TD shrinkage module, MC-simulator for uncertainty.
validate.py — rolling-origin CV and metrics (MAE, RMSE, Spearman rho, top-K accuracy).
utils.py — parsing, name canonicalization, config loader.
notebooks/analysis.ipynb — exploratory analysis plots and diagnostics.
requirements.txt
configs/pipeline_config.yaml — hyperparams, scoring, file paths, random seed.
outputs/
2025_adjusted_predictions.csv (cols: Player, Team, proj_FPTS, adjusted_mean, adjusted_std, p10, p50, p90, predicted_rank)
2025_rank_probabilities.csv (probability a player finishes top-1..top-24 using simulations)
model.joblib (trained ensemble)
feature_importance.csv
diagnostics.pdf (plots: residuals, calibration, SHAP summary, rank density)
tests/
test_feature_shapes.py
test_scoring_conversion.py
test_cv_no_leakage.py
Pipeline specification — exact steps to implement

Data ingestion & cleaning

Read 2020–2024 historical files (both <year>QB.csv and <year>QBADD.csv if present). Prefer QBADD rows for fantasy FPTS if those contain FPTS; otherwise compute FPTS from stats using scoring in config.
Read data/pro/qb/2025QBPROJ.csv. Parse columns carefully; remap duplicate header names to pass_* and rush_* explicitly. Remove stray empty rows.
Standardize player names (strip whitespace, normalize punctuation) for any join operations.
For any missing historical seasons for a player, fill historical features with NaN and provide IsRookie flag. Default-fill numeric missing values with league mean when training if needed.
Target construction

Use historical season FPTS (from QBADD or computed) as y for training.
Compute FPTS/G if games column exists; otherwise compute per 17 games when needed for per-game comparisons.
Feature engineering (implement exactly)

From projections (2025): proj_pass_att, proj_pass_cmp, proj_pass_yds, proj_pass_tds, proj_pass_int, proj_rush_att, proj_rush_yds, proj_rush_tds, proj_fumbles, proj_projFPTS (if present).
Historical features (lag-1 and decay averages): pass_yds_{t-1}, pass_tds_{t-1}, rush_yds_{t-1}, rush_tds_{t-1}, int_{t-1}, fumbles_{t-1}, games_{t-1}
Rolling features: weighted average over last up to 3 seasons with weights [0.5, 0.3, 0.2] (if data present).
Trend features: pct_change_pass_yds = (t-1 - t-2) / t-2 (clip extreme values).
Ratios & interactions: proj_pass_YPA = proj_pass_yds / proj_pass_att; proj_pass_TD_rate = proj_pass_tds / proj_pass_att; combined_TDs = proj_pass_tds + proj_rush_tds.
Volume features: proj_total_attempts = proj_pass_att + proj_rush_att.
Meta features: age (if available), IsRookie, IsNewTeam (if team changed vs t-1), qb_type_dual (binary if historical median rush_yds >= 400), roster_pct (if available).
Special handling: touchdown shrinkage & modeling TDs

Compute league_avg_pass_TD_rate from 2020–2024 historical: league_pass_TD_rate = total_pass_TDs / total_pass_ATT.
Implement empirical-Bayes shrinkage for projected pass_TDs:
expected_pass_TDs = proj_pass_att * league_pass_TD_rate
posterior_pass_TD = (alpha * proj_pass_tds) + ((1 - alpha) * expected_pass_TDs)
Set default alpha = 0.6 configurable via YAML; allow model to also accept both raw and shrinked TD as features.
Also implement optional submodel to predict TDs (Poisson or negative binomial) using volume; use if enabled in config.
Modeling

Baseline score: deterministic conversion of proj counts → proj_FPTS using scoring formula (store as baseline_proj_fpts).
Models to fit (train on 2020–2024):
Ridge regression on engineered features (include baseline_proj_fpts as a feature).
XGBoost regressor with hyperparams (n_estimators=500, eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8) with early stopping on validation folds.
Ensemble: simple average of Ridge + XGBoost predictions (or weighted average with weights tunable).
Save trained model to outputs/model.joblib.
Validation — rolling-origin (time-series-aware)

CV windows:
Train on 2020 → validate on 2021
Train on 2020–2021 → validate on 2022
Train on 2020–2022 → validate on 2023
Train on 2020–2023 → validate on 2024
For each fold compute MAE, RMSE, Spearman correlation, top-K recall (K = 5, 12, 24).
Report mean & std of metrics across folds in outputs/validation_metrics.csv and include plots in diagnostics.pdf.
Uncertainty estimation (Monte Carlo sims)

For each player projection run N sims (default N=2000):
Sample pass_yds ~ Normal(mu=proj_pass_yds, sigma = sigma_pass_yds), where sigma_pass_yds is estimated from historical residuals grouped by player-type/volume bins.
Sample pass_TDs ~ NegativeBinomial or Poisson with mean = posterior_pass_TD (use dispersion fit from historical TD variance).
Sample rush_yds similarly; sample rush_TDs with Poisson/NB.
Sample INTs and fumbles with appropriate small-dispersion Poisson.
Convert each sim to FPTS using scoring formula and (optionally) pass through ensemble model to get an adjusted FPTS per sim.
From simulations produce mean, std, p10, p50, p90, and ranking distribution (proportion of sims where player is top-N).
Save results to outputs/2025_adjusted_predictions.csv and outputs/2025_rank_probabilities.csv.
Diagnostics & explainability

Compute SHAP summary for XGBoost and per-player SHAP contributions for top 25 QBs.
Produce plots: predicted vs actual (validation folds), residual histograms, calibration plots (predicted vs realized quantiles), rank density plots, SHAP feature importance.
Save feature importances to outputs/feature_importance.csv.
CLI & reproducibility

run_pipeline.py arguments: --config configs/pipeline_config.yaml --mode train|validate|predict --seed 42 --n-sims 2000
Set random seeds globally for numpy, sklearn, xgboost, and any samplers.
Save configuration and exact versions used to outputs/run_metadata.json.
Tests (automated)

test_scoring_conversion.py: verifies scoring formula reproduces FPTS for several known rows (use a small fixture).
test_feature_shapes.py: ensures feature matrix X has no NaNs for model-required columns after preprocessing (or explicit imputation performed).
test_cv_no_leakage.py: ensures for each CV fold training set years < validation year.
Performance targets & evaluation

Target CV Spearman rank >= 0.85 (reasonable goal; actual may vary)
CV MAE target: maximize ranking ability; report MAE and RMSE - no strict cutoff required.
Model should improve baseline_proj_fpts MAE by at least 5% if possible (report baseline vs model).
Extra optional enhancements (nice-to-have)

Automatically detect rookies and set stronger TD shrinkage (lower alpha).
Incorporate simple team-level volume proxies: team_pass_attempts_per_game (computed from historical rosters) or Vegas totals if available.
Add calibration layer (isotonic regression) to correct predicted means if biased.
Option to export results as JSON and a simple HTML report.
Example minimal commands

Install: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
Train & validate: python scripts/run_pipeline.py --config configs/pipeline_config.yaml --mode train
Predict 2025 & run sims: python scripts/run_pipeline.py --config configs/pipeline_config.yaml --mode predict --n-sims 2000
What to return to me

A short README in repo root describing usage and describing outputs produced.
All scripts and config files listed above.
outputs/2025_adjusted_predictions.csv and outputs/2025_rank_probabilities.csv as produced by running the pipeline.
Unit tests and a short test run showing they pass.
Notes for the assistant

Be explicit in code comments about the pass vs rush duplicate headers in 2025QBPROJ.csv and how they are remapped.
Keep the scoring function configurable and isolated so it can be changed easily.
Make sure the pipeline runs end-to-end on a machine with only the repo and data/pro/qb files (i.e., don't require external APIs).
Keep runtime reasonable: XGBoost params above are fine; parallelize simulations if possible.

Goal

Build a reproducible data pipeline (ETL + analysis + reports) for running back (RB) season stats and projections. The pipeline must ingest the historical RB CSVs (2020–2025) and the 2025 projections (2025RBPROJ.csv), compute standardized metrics and composite “top performer” scores, produce ranked outputs and visualizations, and include tests, documentation, and reproducible execution (Docker and/or Make).
Primary inputs (relative paths)

data/pro/rb/2020RB.csv
data/pro/rb/2020RBADD.csv
data/pro/rb/2021RB.csv
data/pro/rb/2021RBADD.csv
data/pro/rb/2022RB.csv
data/pro/rb/2022RBADD.csv
data/pro/rb/2023RB.csv
data/pro/rb/2023RBADD.csv
data/pro/rb/2024RB.csv
data/pro/rb/2024RBADD.csv
data/pro/rb/2025RB.csv
data/pro/rb/2025RBADD.csv
data/pro/rb/2025RBPROJ.csv
Target outputs

outputs/rb_historical_clean.csv — cleaned, normalized historical rows (one row per player-season) with canonical column names.
outputs/2025_projections_clean.csv — cleaned projections.
outputs/rb_percentiles_historical.csv — percentiles computed from historical seasons for metrics used.
outputs/2025_projected_percentiles.csv — projections mapped to historical percentiles.
outputs/2025_composite_scores.csv — each projected RB with component percentiles, composite score(s), rank(s), archetype, and risk flags.
outputs/figures/* — key plots (scatter, histograms, time series) in PNG/SVG.
docs/methodology.md — explanation of scoring, formulas, and how weights can be tuned.
tests/ — unit tests validating parsing, key numeric computations, and end‑to‑end pipeline run.
Dockerfile and Makefile (or task runner) to run pipeline reproducibly with one command.
Implementation requirements & conventions

Language: Python 3.10+ (use virtualenv/venv). Use pandas for tabular ETL. numpy, scipy or scikit-learn permitted for stats. matplotlib/seaborn/plotly for visuals. joblib for caching optional.
Code layout: src/etl.py, src/metrics.py, src/scoring.py, src/visualize.py, src/cli.py, notebooks/analysis.ipynb.
Config: config.yml with:
input_paths: historical_dir, projections_file
output_dir
composite_weights: {att:0.30, rec:0.25, eff:0.20, tds:0.15, dur:0.10}
ranking_mode: ['season_total','weekly_floor'] (explain differences)
percentile_method: ['historical','cohort_zscore']
Logging and errors: robust logging (info/debug). Fail early with clear messages when required columns missing. Gracefully handle malformed rows.
Cleaning & parsing details

CSVs contain quoted fields, commas in thousands, empty rows at file end. Robustly parse:
Strip leading/trailing quotes.
Normalize thousands separators (e.g., "2,027" -> 2027).
Convert "(FA)" or team tags in Player name to separate Player / Team columns where available. If Team is in projections, prefer that.
Trim whitespace and drop completely empty rows.
Canonical columns expected (examples): Year, Player, Team, G, ATT, Rush_YDS, Rush_TDS, REC, Rec_YDS, Rec_TDS, Fumbles, TGT, RZ_TGT, FPTS (where available), RostPercent (if present). If a metric is missing in a source, fill with NaN and document.
Feature engineering

Compute derived metrics per row:
YPC = Rush_YDS / ATT (handle ATT==0)
Total_TDs = Rush_TDS + Rec_TDS
Total_Yds = Rush_YDS + Rec_YDS
FPTS_per_game = FPTS / G (if FPTS present)
YBCON/ATT and YACON/ATT if columns available; else fallback to available proxies.
Standardize names (normalize casing). De‑duplicate players with identical name + year (sum counts or keep the record with largest ATT; prefer documented rule).
Percentiles & standardization

Build historical distributions using one row per player-season across 2020–2024 (optionally include 2025 if desired).
For each metric used in scoring (ATT, REC, YPC, Total_TDs, Total_Yds, RZ_TGT if available, FPTS/G if available), compute percentile rank (0 to 1) for the historical distribution.
Save percentiles to outputs/rb_percentiles_historical.csv.
Composite scoring

Default composite (season-total oriented): Score = 0.30 * P_ATT + 0.25 * P_REC + 0.20 * P_EFF + 0.15 * P_TD + 0.10 * P_DUR
P_ATT = percentile(ATT)
P_REC = percentile(REC + normalized TGT if available) — choose REC primarily
P_EFF = percentile(YPC) or blended percentile(YPC, Total_Yds/ATT)
P_TD = percentile(Total_TDs)
P_DUR = percentile(G or ATT)
Weekly-floor mode: increase weight on P_REC and P_FPTS/G.
Implement ability to:
Accept a config with alternative weights
Include an optional interaction bonus: if P_ATT > 0.8 and P_EFF > 0.8 then +bonus 0.03 (tunable)
Output composite score [0..100] and normalized rank.
Archetype & risk tags

Assign an archetype label based on rules:
“Bell-Cow” if ATT percentile ≥ 0.75 and REC percentile ≤ 0.40
“Three-Down” if ATT P ≥ 0.60 and REC P ≥ 0.60
“Explosive” if YPC P ≥ 0.80 with ATT P between 0.30–0.60
“Committee” otherwise
Add risk flags:
Low-volume risk: ATT < threshold (configurable)
Injury risk: historical games played trend shows decline (if past data available)
Regression risk: projected YPC > historical 90th pct and ATT very high (possible regression)
Analysis & visualization

Plots to produce:
Top scatter: ATT vs YPC with point size by Total_TDs and color by archetype; label top 25.
Percentile heatmap for top N (e.g., top 50 projected) showing P_ATT, P_REC, P_EFF, P_TD.
Time series of a selected player showing ATT, YDS, REC, FPTS across years.
Correlation matrix (pearson) of key metrics across the historical dataset: ATT, REC, YPC, Total_TDs, FPTS.
Distribution plots for ATT, YPC, REC for historical vs 2025 projections.
Save all plots to outputs/figures with descriptive filenames.
Testing & validation

Unit tests must cover:
CSV parsing of a provided sample with quoted thousands separators and trailing empty rows.
Correct calculation of YPC and Total_TDs, including ATT==0 handling.
Percentile calculation correctness on a small synthetic dataset (include fixed expected percentiles).
Composite score calculation given a known set of percentiles and weights.
Integration / e2e test:
A pipeline smoke test: run pipeline on a small subset of files and assert outputs exist, top‑ranked player is expected based on sample data.
Provide a simple test dataset in tests/data/ to run in CI.
Reproducibility & packaging

Provide a Dockerfile (python slim) that installs requirements and runs the pipeline; or provide a Makefile with:
make venv
make install
make run (runs ETL -> scoring -> visuals)
make test
Pin dependencies in requirements.txt. Provide runtime instructions in README.md.
Performance & scale

Target: handle >10k rows easily on single machine. Use vectorized pandas operations. Use chunked CSV reading if memory constrained. Cache intermediate cleaned data to outputs/cache/.
Acceptance criteria

Running the pipeline (make run or docker run) processes input CSVs and writes the target outputs listed above.
The top 50 composite scores are saved in outputs/2025_composite_scores.csv with Player, Team, composite_score, rank, archetype, risk_flags, and component percentiles.
Tests pass (make test).
docs/methodology.md documents formulas and how to change weights and percentile source.
Code is modular, documented, and unit-tested.
Example CLI usage

python -m src.cli run --config config.yml
python -m src.cli score --projections data/pro/rb/2025RBPROJ.csv --out outputs/2025_composite_scores.csv
python -m src.cli visualize --players "Saquon Barkley, Bijan Robinson" --out outputs/figures/
Example expected final CSV header (outputs/2025_composite_scores.csv)

Player, Team, ATT, Rush_YDS, Rush_TDS, REC, Rec_YDS, Rec_TDS, Total_TDs, YPC, Total_YDS, P_ATT, P_REC, P_EFF, P_TD, P_DUR, composite_score, rank, archetype, risk_flags
Bonus tasks (optional, mark in config)

Add a small ML model (linear regression / random forest) trained on historical seasons to predict FPTS from components (ATT, REC, YPC, TDs), showing model feature importances and using cross‑validation — then compare projected FPTS to model prediction and flag outliers.
Add a simple dashboard (Streamlit/Voila) for interactive exploration.
Deliverables

src/ package with code files.
requirements.txt, Dockerfile, Makefile.
config.yml example.
README.md with run instructions.
outputs/ CSVs and figures after a pipeline run.
docs/methodology.md and notebooks/analysis.ipynb.
tests/ unit and integration tests.
Notes for the assistant

Ask a single clarifying question only if a critical input is missing (e.g., do you want to include 2025 historical rows in percentiles?). Otherwise implement defaults above and document them in docs/methodology.md.
Keep the code modular and well-documented. Add inline comments for any domain‑specific logic (e.g., archetype thresholds).
Provide a short usage README at the end of the PR and a summary of results (top 10 scored RBs and a couple of generated plots).


Task

Implement a reproducible Python pipeline that trains a supervised model to classify/score wide receivers as “top performers” using historical season CSVs (2020–2024) and applies the model to 2025 projected totals (2025WRPROJ.csv).
Use a classification target: season is a “top” WR if it finishes in the top 24 WRs by season FPTS (from <year>WRADD.csv). Also provide regression results predicting season FPTS as a secondary output.
Requirements — environment & libraries

Language: Python 3.10+
Required libraries: pandas, numpy, scikit-learn, xgboost or lightgbm (pick one), optuna (hyperparam tuning), shap, matplotlib / seaborn, joblib, pyarrow (for faster I/O).
Use random_state/seed consistently for reproducibility.
Provide a requirements.txt with pinned versions.
Input files (assume local paths)

data/pro/wr/{year}WR.csv and data/pro/wr/{year}WRADD.csv for years 2020,2021,2022,2023,2024
data/pro/wr/2025WRPROJ.csv (projections)
High-level pipeline steps

Ingest & normalize:

Read all WR.csv and WRADD.csv for 2020–2024 and concatenate into a season‑level table.
Read 2025WRPROJ.csv into a projections table.
Normalize player names: strip trailing team parentheses (e.g., "Davante Adams (LAR)" -> "Davante Adams"), unify spacing & accents, lower/strip.
Ensure consistent column types (ints/floats), parse quoted CSV correctly.
Feature engineering (historical seasons):

From WR.csv and WRADD.csv build features per player-season:
Direct columns to include: G, TGT, % TM, REC, YDS, Y/R, AIR, AIR/R, YAC, YAC/REC (or YACON/YACON/R), RZ TGT, 10+,20+,30+, CATCHABLE, DROP, BRKTKL, FPTS, FPTS/G, ROST.
Derived features:
catch_rate = REC / TGT (handle TGT==0)
yac_per_rec = YAC / REC
air_share_est = AIR / team_pass_air proxy (if team totals not available, normalize AIR by season cohort median or use AIR/R)
target_share = TGT (use % TM when present)
big_play_rate = (20+ + 30+ maybe) / REC or per game
redzone_share = RZ_TGT / TGT (if TGT>0)
per_game metrics: YDS_per_game = YDS / G, REC_per_game = REC / G, TGT_per_game = TGT / G
trailing features: lag1_TGT, lag1_AIR, lag1_FPTS (previous season values)
yoy change features: d_TGT = TGT - lag1_TGT, pct_d_TGT, etc.
Document imputation rules and create a boolean flag for imputed features.
Label creation:

For each season (2020–2024) sort players by season FPTS (from WRADD). Create binary label top24 = 1 if rank <= 24 else 0.
Also store top12 and top36 labels optionally for alternative thresholds.
Keep the season as an identifier for CV.
Training / validation strategy:

Use leave‑one‑season‑out cross‑validation: train on 4 seasons, validate on the left‑out season; rotate for all 5 folds. This emulates out‑of‑time test.
For each fold, train a classifier (XGBoost or LightGBM) to predict top24.
Also train a regression model to predict FPTS as a secondary target (same CV).
Use Optuna to tune hyperparameters (limit trials e.g., 50) on training folds only.
Metrics & evaluation:

Classification metrics: AUC-ROC, precision@k (k=12,24), recall@k, F1, calibration plot (reliability).
Regression metrics: RMSE, MAE, R2; precision@k using predicted FPTS ranking.
Produce a fold‑wise and aggregate report: mean and std for metrics.
Output confusion matrices and precision@k bar charts.
Model interpretation:

Provide global feature importance (gain/SHAP), and a SHAP summary plot.
For a few example players show SHAP force plots (or waterfall style) that explain predicted probability.
Apply to 2025 projections:

Map 2025 projection columns into the same feature set. Where projection file lacks required features (e.g., TGT, AIR or %TM), impute:
If TGT missing but REC present, impute TGT = REC / historical_catch_rate (use player's 2024 catch_rate, else league median).
If AIR missing, impute AIR using player's historical AIR or cohort median scaled by projected REC_per_game.
Flag imputed values with boolean columns.
For players missing lag1 historical data, use cohort medians for lag features and mark them as new players.
Use the trained classifier to compute:
P(top24) and P(top12) probabilities
predicted FPTS from regression model (and 95% prediction interval from CV residuals)
Output a predictions CSV with columns: player, team, proj_REC, proj_YDS, proj_TDS, proj_FPTS, P_top24, P_top12, predicted_FPTS_reg, TPS (composite score), imputed_flags.
Tiering & explainable output:

Define tiers based on P_top24 and predicted_FPTS:
Elite WR1: P_top24 >= 0.8 and predicted_FPTS >= 170
High WR1: 0.6 <= P_top24 < 0.8
WR2: 0.3 <= P_top24 < 0.6
WR3/Bust: P_top24 < 0.3
Output a JSON or CSV with player → tier, probabilities, and top contributing features (from SHAP) per player.
Robustness & edge cases

Handle name collisions and duplicates (if a player appears multiple times in the same season, merge and sum relevant stats).
If a player has 0 targets/TGT missing, avoid divide-by-zero.
Record and propagate uncertainty flags for imputed or projected-only players.
Code structure & deliverables

A modular package or scripts:
data_ingest.py — reads/normalizes CSVs
features.py — feature engineering & imputation
train.py — training, CV, hyperparam tuning, metrics
predict.py — load model + apply to 2025 projections, output predictions
explain.py — SHAP analyses and plots
notebooks/analysis.ipynb — EDA and sample run
tests/: unit tests for key functions (name normalization, imputation, label creation)
README.md — how to run, expected outputs, dependencies, and TL;DR of the logic
Save artifacts:
models/classifier.joblib (or model.bin), models/regressor.joblib
outputs/2025_predictions.csv (with probabilities and tiers)
outputs/metrics_report.md and visualizations (roc.png, precision_at_k.png, shap_summary.png)
Include a small sample CLI: python train.py --data-dir data/pro/wr --out-dir outputs --seed 42
Acceptance criteria
Code runs end‑to‑end on provided CSVs and produces outputs above.
CV results (leave‑one‑season‑out) are printed and saved; classification AUC should be >0.80 if features are strong (report actual).
predictions CSV contains P_top24 and tier for every player in 2025WRPROJ.csv.
Provide a short README paragraph explaining imputation choices and recommended follow‑up checks.
Optional advanced improvements (nice to have)

Use time‑aware features like rolling 2‑season trends and age curves.
Train a hierarchical model that first predicts target share then converts to yards/TDs.
Provide a calibration step (Platt/isotonic) for classifier probabilities.
Add an uncertainty ensemble: train several randomized seeds/models and provide mean & std of predicted probabilities.
End of prompt

Notes / Tips for the assistant you call

Be explicit about column names encountered in the CSVs (they use quoted headers). Normalize and map them into snake_case.
Use season as the CV unit to avoid target leakage.
Persist preprocessing pipeline (scaler, encoders, imputation rules) with joblib so the same transformations apply to 2025 projections.
If you want, I can:

produce an executable starter repository scaffold (file list + minimal code) you can drop into your project, or
generate the full Python scripts and a runnable notebook implementing this pipeline end‑to‑end. Which would you prefer?
