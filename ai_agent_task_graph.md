# 🤖 AI-Agent Task Graphs — Fantasy Football FULL-PPR Projection Pipeline

This document defines **discrete, executable AI-agent task graphs** for implementing the season-long FULL-PPR fantasy football projection system.

Each agent has:
- A **clear responsibility boundary**
- **Inputs / outputs** it owns
- **Dependencies** on other agents

Agents should be implemented as **idempotent steps** so the pipeline can be rerun safely.

---

## 🧭 Global System Flow

```
[Data Ingestion]
      ↓
[Data Normalization & Schema Enforcement]
      ↓
[Feature Engineering]
      ↓
[Usage Models]
      ↓
[Efficiency Models]
      ↓
[TD Expectation Models]
      ↓
[Rookie Projection System]
      ↓
[Fantasy Point Assembly]
      ↓
[Market Data Blending]
      ↓
[Final Rankings & Outputs]
```

---

## 🧠 Agent 0 — Configuration & Constants Agent

### Purpose
Own all **global constants, scoring rules, and blend weights** so they are not duplicated.

### Responsibilities
- Define FULL-PPR scoring constants
- Define season length (17 games)
- Define market blend weight (default 15%)
- Define model hyperparameter defaults

### Outputs
- `config/scoring.yaml`
- `config/model_defaults.yaml`

### Downstream Dependencies
ALL agents

---

## 📥 Agent 1 — Raw Data Ingestion Agent

### Purpose
Fetch and store **raw, immutable datasets** from free sources.

### Inputs
- nflfastR data repositories
- Pro Football Reference tables
- Sleeper API
- FantasyPros ADP pages

### Responsibilities
- Download raw data
- Do **NO transformations**
- Version datasets by season

### Outputs
```
raw_data/
  ├── nflfastR/play_by_play/
  ├── nflfastR/player_stats/
  ├── pfr/snap_counts/
  ├── sleeper/players.json
  └── fantasypros/adp.csv
```

### Downstream Dependencies
Agent 2

---

## 🧹 Agent 2 — Data Cleaning & Canonicalization Agent

### Purpose
Standardize IDs, names, teams, and seasons into a **single canonical schema**.

### Responsibilities
- Resolve player ID mismatches
- Normalize team abbreviations
- Enforce schema contracts
- Drop unusable records

### Outputs
```
processed_data/
  ├── dim_player.parquet
  ├── fact_player_usage_season.parquet
  ├── fact_player_efficiency_season.parquet
  ├── fact_team_context_season.parquet
  └── fact_market_expectation.parquet
```

### Downstream Dependencies
Agents 3–9

---

## 🏗️ Agent 3 — Feature Engineering Agent

### Purpose
Create **model-ready season-long features** from processed tables.

### Responsibilities
- Compute per-game metrics
- Compute rolling multi-year averages
- Compute shares (target share, snap share)
- Apply efficiency regression to mean

### Outputs
```
features/
  ├── features_qb.parquet
  ├── features_rb.parquet
  ├── features_wr.parquet
  └── features_te.parquet
```

### Downstream Dependencies
Agents 4–7

---

## 📊 Agent 4 — Usage Projection Model Agent

### Purpose
Predict **season-long opportunity metrics**.

### Models
- LightGBM / XGBoost (position-specific)

### Predicts
- targets_pg
- carries_pg
- pass_attempts_pg
- routes_pg

### Outputs
```
model_outputs/
  ├── usage_qb.parquet
  ├── usage_rb.parquet
  ├── usage_wr.parquet
  └── usage_te.parquet
```

### Downstream Dependencies
Agents 6, 7, 8

---

## ⚡ Agent 5 — Efficiency Projection Agent

### Purpose
Predict **per-opportunity efficiency**, heavily regressed.

### Predicts
- yards_per_target
- yards_per_carry
- catch_rate
- EPA per opportunity

### Outputs
```
model_outputs/
  ├── efficiency_qb.parquet
  ├── efficiency_rb.parquet
  ├── efficiency_wr.parquet
  └── efficiency_te.parquet
```

### Downstream Dependencies
Agent 8

---

## 🎯 Agent 6 — Touchdown Expectation Agent

### Purpose
Model **expected TDs**, never raw TDs.

### Models
- Poisson or Negative Binomial

### Predicts
- expected_passing_tds_pg
- expected_rushing_tds_pg
- expected_receiving_tds_pg

### Outputs
```
model_outputs/
  └── expected_tds.parquet
```

### Downstream Dependencies
Agent 8

---

## 🧒 Agent 7 — Rookie Projection Agent

### Purpose
Generate projections for players **without NFL history**.

### Responsibilities
- Identify rookies
- Assign usage via draft-capital buckets
- Cap snap & target shares
- Regress efficiency to positional mean

### Outputs
```
model_outputs/
  └── rookie_projections.parquet
```

### Downstream Dependencies
Agent 8

---

## 🧮 Agent 8 — Fantasy Point Assembly Agent

### Purpose
Assemble **FULL-PPR fantasy points** from all modeled components.

### Responsibilities
- Merge usage, efficiency, TDs, rookies
- Apply scoring rules
- Compute fantasy PPG

### Outputs
```
intermediate_outputs/
  └── fantasy_points_model_only.parquet
```

### Downstream Dependencies
Agent 9

---

## 📈 Agent 9 — Market Data Blending Agent

### Purpose
Blend model projections with **expert market expectations**.

### Formula
```
Final_PPG = 0.85 × Model_PPG + 0.15 × Market_Implied_PPG
```

### Outputs
```
outputs/
  └── fantasy_points_blended.parquet
```

### Downstream Dependencies
Agent 10

---

## 🏆 Agent 10 — Ranking & Output Agent

### Purpose
Generate final **position and overall rankings**.

### Responsibilities
- Rank by position
- Rank overall
- Validate sanity constraints

### Outputs
```
outputs/
  ├── final_projections.csv
  ├── final_projections.parquet
  └── rankings.json
```

---

## 🔁 Agent 11 — Backtesting & Evaluation Agent (Optional but Recommended)

### Purpose
Evaluate accuracy on historical seasons.

### Metrics
- RMSE (PPG)
- MAE
- Spearman rank correlation
- Hit rate (top-12 / top-24)

### Outputs
```
evaluation/
  └── backtest_results.json
```

---

## 🧱 Design Principles for All Agents

- Deterministic execution
- Idempotent writes
- Versioned outputs
- No hard-coded paths
- No manual overrides

---

## ✅ This Task Graph Enables

- Parallel agent execution
- Easy debugging
- Seasonal re-runs
- Model iteration without pipeline rewrite

---

**This file should live at the root of the repository.**

