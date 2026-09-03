# CSE437 — NYC Airbnb Price Prediction

## Problem Statement
Predicts nightly Airbnb listing price in New York City from structural, geographic,
and host-activity features, and tests whether standard random train/test splitting
overstates model performance due to spatial autocorrelation between nearby listings —
a methodological gap not addressed in prior published work on this dataset.

## Dataset
New York City Airbnb Open Data (2019) — Kaggle, `dgomonov/new-york-city-airbnb-open-data`.
See `data/README.md` for source link, size, and how to obtain it.

## Target Variable
`price` (continuous, USD) — regression task. Modeled as `log(price + 1)` to address skew,
back-transformed for reporting.

## Techniques Used (mapped to CSE437 syllabus)
| Stage | Technique | Syllabus week |
|---|---|---|
| Preprocessing | Missing values, outliers, scaling | Week 2 |
| Statistical analysis | Descriptive stats + inferential test (ANOVA/t-test across boroughs) | Week 10 |
| Feature engineering | PCA (dimensionality reduction), RFE (feature selection) | Week 4 |
| Unsupervised piece | K-Means geo-clustering feature | Week 9 |
| Modeling — family 1 | Linear/Ridge Regression | Week 5 |
| Modeling — family 2 | Decision Tree/CART or AdaBoost | Week 6 / Week 8 |
| Validation | Random split vs. spatially-grouped split (GroupKFold by neighbourhood) | — |
| Hyperparameter tuning | Grid search (tree depth / AdaBoost n_estimators, learning_rate) | — |

## Notebooks (run in order)
1. `01_data_audit_and_eda.ipynb` — load, audit missingness/outliers, descriptive + inferential stats
2. `02_preprocessing.ipynb` — cleaning, transformation, scaling, evidence of what changed
3. `03_feature_engineering.ipynb` — PCA, RFE, K-Means geo-cluster feature, final feature set justification
4. `04_modeling_and_tuning.ipynb` — two model families, random vs. grouped split, grid search
5. `05_evaluation_and_error_analysis.ipynb` — metrics, residual analysis, where/why the model fails

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Faculty Feedback Notes
> Record the exact column-to-exclude / split instruction from your proposal feedback here
> before starting Notebook 02, so it isn't lost.
