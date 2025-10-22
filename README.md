House Prices — End-to-End XGBoost Pipeline (Optuna TPE, Pruning, Reproducible CV)

🔥 🔥 🔥 Result: RMSE ≈ 13,000 (public leaderboard), Top ~50 / 5,000+ participants.*
Measured on the Kaggle “House Prices: Advanced Regression Techniques” competition. Leaderboards fluctuate; numbers are for the run dated 2025-10-20.

🔥 Why this repo

An opinionated, production-style ML pipeline for tabular regression:

Data hygiene: targeted NA handling, IQR outlier detection + hard thresholds for extreme area metrics

Ordinal encoding for quality/exposure fields with typed dtypes to stay memory-efficient

ColumnTransformer + Pipeline to prevent leakage

5-Fold KFold (shuffle, seed) for reproducible cross-validation

XGBoost (hist + lossguide + max_bin) tuned with Optuna TPE + Median Pruner

Single command to train → CV → tune → fit → predict → submission.csv

🧠 Design Choices (and why)

lossguide + hist + max_bin: high-dimensional OHE → lossguide grows only the most loss-reducing nodes; hist + max_bin balance accuracy vs speed.

L1/L2 (reg_alpha, reg_lambda): shrinkage + sparsity against OHE explosion.

Subsample/colsample: stochasticity helps generalization.

ColumnTransformer: keeps preprocessing inside the CV loop, avoiding leakage.

Optuna: modern search with smarter sampling + early pruning.

🧱 Approach (Architecture)
Raw CSV (train/test)
   └─ Clean missing-heavy columns (>80% NA)
   └─ Cast discrete numerics to string (MSSubClass, MoSold, YrSold)
   └─ Split num/cat
   └─ IQR outlier scan → summary table
   └─ Hard-threshold row drops (LotArea, GrLivArea, TotalBsmtSF, LotFrontage, GarageArea)
   └─ Ordinal maps (Ex/Gd/TA/Fa/Po, Bsmt types, Exposure)
   └─ ColumnTransformer:
        - num → SimpleImputer(mean)
        - cat → SimpleImputer(most_frequent) → OneHotEncoder(ignore unknown)
   └─ Optuna tune XGBoost (hist, lossguide, max_bin, reg_alpha/lambda, colsampling, etc.)
   └─ 5-fold CV (neg_root_mean_squared_error) → minimize RMSE
   └─ Fit full pipeline on all train
   └─ Predict test → submission_feature_engineering.csv

📊 Key Techniques

Outliers (two-stage)

IQR scan for diagnostics; 2) Hard thresholds for area-type columns:
LotArea > 100000, GrLivArea > 4000, TotalBsmtSF > 4000, LotFrontage > 200, GarageArea > 1400.

Ordinal mapping

Qualities: {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}

Basement types/exposure mapped similarly and cast to compact integer dtypes.

Modeling

XGBRegressor(tree_method='hist', grow_policy='lossguide', max_bin ∈ [180,256])

Regularization: reg_alpha (L1), reg_lambda (L2)

Subsample/colsample to fight overfit in high-dim OHE space

Tuning (Optuna)

TPESampler(seed=35) → smarter than random/grid

MedianPruner(warmup=10) → early-stop bad trials

Objective returns positive RMSE; direction='minimize'

