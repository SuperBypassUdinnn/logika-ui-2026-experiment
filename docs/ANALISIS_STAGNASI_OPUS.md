# Experiment Stagnation Analysis: Logika UI 2026

**Date**: 2026-10-11
**Analyst**: Antigravity (for Fadhlurrahman Alaudin)
**Current Best OOF KS**: 0.5061 (exp042, blend 70/30 farhan+exp041)
**Target OOF KS**: 0.53
**Gap**: 0.024 (approximately 4.5% relative improvement needed)
**Reference**: Farhan's notebook achieves Public LB 0.53221 with OOF KS 0.5050

---

## 1. Experiment Timeline and Plateau Pattern

```
OOF KS
0.51 |                                              ___exp042 (0.506)
     |                                         ____/   exp040 (0.506)
0.50 |                              exp037 ___/
     |                         ____/  (0.500)
0.49 |               exp027 __/
     |              / (0.497)
0.48 |             /
0.47 |    exp019 _/
     |   / (0.473)
0.46 |  /
0.45 | exp018
     | (0.454)
     |                          <-- DIMINISHING RETURNS ZONE -->
0.40 |
     | exp001-017 (0.38-0.40)   <-- INITIAL PLATEAU -->
     +------------------------------------------------------------
       exp001    exp018    exp027    exp037    exp042
```

There are three distinct phases:

| Phase | Experiments | OOF KS Range | Primary Driver |
|:---|:---|:---|:---|
| Phase 1: Foundation | exp001-exp017 | 0.382 - 0.396 | Basic aggregates, tuning, first blend |
| Phase 2: Breakthrough | exp018-exp026 | 0.454 - 0.476 | E-commerce spend categories, velocity, early stopping |
| Phase 3: Diminishing returns | exp027-exp042 | 0.497 - 0.506 | More e-commerce, z-score bureau, 3-model blend, Farhan blend |

The trajectory shows that each subsequent improvement requires disproportionately more effort. The last 15 experiments (exp027-exp042) collectively gained only +0.009 KS.

---

## 2. Root Cause Analysis: Why Experiments Are Stagnating

### 2.1 Feature Engineering Ceiling Under Current Paradigm

The workspace [build_features.py](file:///home/superbypassudin/.clone/Github/logika-ui-2026-experiment/scripts/build_features.py) follows a **single-pass aggregation** paradigm: read each auxiliary table, filter by leakage cutoff, compute groupby statistics, merge back. Every experiment from exp027 onward has been adding more aggregates within this same paradigm, but the marginal KS of each new aggregate is near zero.

The ablation from [TEMUAN_EKSPERIMEN.md](file:///home/superbypassudin/.clone/Github/logika-ui-2026-experiment/docs/TEMUAN_EKSPERIMEN.md) proves this saturation:

| Group dropped | Delta KS |
|:---|:---:|
| eco_* | -0.089 |
| repay_* | -0.043 |
| bureau_* | -0.014 |
| prev_* | -0.002 |
| acct_* | neutral |

The bottom three groups collectively contribute less than 0.016 KS. Adding more aggregates in these groups will not move the needle.

### 2.2 Critical Data Columns Left on the Table

Comparing the [DATA_DICTIONARY.md](file:///home/superbypassudin/.clone/Github/logika-ui-2026-experiment/data/DATA_DICTIONARY.md) against the `usecols` actually loaded in `build_features.py`:

**account_behavior -- MAJOR GAP:**

| Column | In Dictionary | Used in build_features.py | Used in Farhan's Notebook |
|:---|:---:|:---:|:---:|
| `opening_balance` | Yes | **NO** | via `average_balance` proxy |
| `income_like_credits` | Yes | **NO** | **YES** (`acc_income_cr`) |
| `average_balance` | Yes | **NO** | **YES** (`acc_avg_bal`) |
| `minimum_balance` | Yes | **NO** | **YES** (`acc_min_bal`) |

The `income_like_credits` column is especially critical -- Farhan uses it to build `x_income_vs_observed = declared_monthly_income / acc_income_cr`, which tests whether declared income is consistent with actual observed income. This is a classic credit-risk red flag for income inflation.

**bureau_observations -- MODERATE GAP:**

| Column | In Dictionary | Used in build_features.py | Used in Farhan's Notebook |
|:---|:---:|:---:|:---:|
| `active_accounts` | Yes | **NO** | Implicitly via pivot |
| `revolving_credit_limit` | Yes | **NO** | Not directly |
| `revolving_outstanding_balance` | Yes | **NO** | Not directly |

**previous_applications -- MAJOR GAP:**

| Column | In Dictionary | Used in build_features.py | Used in Farhan's Notebook |
|:---|:---:|:---:|:---:|
| `term_months` | Yes | **NO** | **YES** (compute active loans) |
| `annual_interest_rate` | Yes | **NO** | **YES** (`prev2_appr_rate_max`) |
| `scheduled_monthly_installment` | Yes | **NO** | **YES** (`prev2_active_inst_sum`) |

### 2.3 Farhan's v1 Features: What the Workspace Pipeline is Missing

Farhan's notebook has a **two-layer feature system** (FE + FE1) where the v1 layer adds semantically richer features. Here is a detailed gap analysis:

#### A. Active Loan Burden (NOT in workspace)

Farhan's `f_previous_v1` computes:
- `prev2_cnt_active`: number of old loans still running at cutoff
- `prev2_active_inst_sum`: total monthly installment burden from active old loans
- `prev2_active_amt_sum`: total outstanding principal from active old loans

These feed into the crucial cross-ratio:
- `x_active_inst_to_income`: active installment burden relative to income
- `x_total_inst_to_income`: (active + proposed) installment relative to income

> [!IMPORTANT]
> The workspace has ZERO concept of "active loan at cutoff". It only counts historical applications but never checks if old approved loans are still running. This is a fundamental gap in credit-risk modeling: the **debt-to-income ratio at decision time**.

#### B. Income Verification (NOT in workspace)

- `x_income_vs_observed = declared_monthly_income / acc_income_cr`

The workspace never loads `income_like_credits` from `account_behavior.csv`, so it has no way to cross-check declared income against observed bank credits. This single feature alone can separate income inflaters.

#### C. Bureau History Across Applications (NOT in workspace)

Farhan's `f_bureau_v1` does something the workspace doesn't: it joins bureau reports from **other applications of the same customer** (not just the current application's bureau). This gives:
- `bur2_hist_z_mean`: average z-score from historical bureau reports
- `bur2_hist_z_last`: most recent historical z-score
- `bur2_z_delta`: change in z-score from historical to current (improving/deteriorating)
- `bur2_util_delta`: change in utilization from historical to current

The workspace's `agg_bureau` only looks at bureau reports linked to `application_id` directly, missing the customer-level bureau trajectory.

#### D. Repayment Streak and Exponential Decay (PARTIALLY in workspace)

Farhan uses `streak = cumprod(late)` from most recent installment -- consecutive late streak counting. The workspace has `repay_trail_late_n` which is conceptually similar but implemented differently. Farhan also uses `w = exp(-days_since_due / 180)` for exponential decay, while workspace uses `rec_w = 1 / (1 + days/30)`. Both are valid, but the exponential decay has sharper signal discrimination for recent behavior.

#### E. E-commerce Granularity (PARTIALLY in workspace)

Farhan's `f_ecom_v1` adds:
- **Payment method ratio in 90d window**: `ec2_pm90_COD`, `ec2_pm90_e_wallet`, etc.
- **Cancel value ratio**: `ec2_cancel_value_ratio = cancelled_total / all_total`
- **Category share and HHI** (Herfindahl index for spend concentration)
- **Category share in 90d** (recent category shift)
- **Unit price mean** and **items per order**
- **Gap statistics** (std, max) between orders

The workspace has some category and velocity features, but misses the nuanced payment-method and concentration metrics.

#### F. Cross-Table Ratios (PARTIALLY in workspace)

The workspace has 6 basic income ratios. Farhan adds:
- `x_ec90_to_inst`: 90-day e-commerce spend relative to proposed installment
- `x_loan_to_prev_max`: current loan amount vs largest previously approved loan
- `x_rate_diff_prev`: interest rate difference vs previous max rate

### 2.4 CV Strategy: Potential Optimistic Bias

[make_folds.py](file:///home/superbypassudin/.clone/Github/logika-ui-2026-experiment/scripts/make_folds.py) uses plain `StratifiedKFold` on `application_id` without grouping by `customer_id`. If customers have multiple applications across train folds, the model may learn customer-specific patterns and leak them into validation, making OOF KS artificially optimistic.

Farhan's notebook mentions `StratifiedGroupKFold per customer_id` in the EDA recommendation (Cell 36) but the actual implementation also uses plain `StratifiedKFold` -- so both have the same potential issue. This is less of a gap between the two, but it means both are potentially overestimating local KS (the 0.505 OOF becoming 0.532 on public LB is suspicious -- normally LB should be close to or below OOF).

### 2.5 Model Architecture: Shallow Exploration

The workspace uses:
- LightGBM with `metric='auc'` (not KS-optimized)
- XGBoost mirror
- CatBoost
- Simple average blend
- Logistic stacker (failed, Cell 23)

What is NOT being explored:
- **Custom KS objective/eval_metric** for LightGBM (train to maximize KS directly)
- **Bayesian hyperparameter search** (only 12-config grid was tried)
- **Multi-seed averaging** for variance reduction (only tried bagging 3 seeds in exp016, deemed "negative" but the seed count was too low and gains were expected to be small)
- **Rank-probability calibration** (blend on rank-percentiles instead of raw probabilities)

### 2.6 Model Default: Training with AUC, Evaluating with KS

The [model.py](file:///home/superbypassudin/.clone/Github/logika-ui-2026-experiment/src/model.py) factory sets `metric='auc'` as the LightGBM default. While AUC and KS are correlated, they are not the same objective. AUC measures overall discrimination across all thresholds equally, while KS measures the **maximum separation** at the optimal threshold. A model optimized for AUC may spread its discriminative power across all thresholds, while a KS-optimal model would concentrate discrimination around the point of maximum CDF divergence.

---

## 3. Strategies That Should Be Used But Are Not

### 3.1 Active Loan Burden Features (Estimated Impact: +0.005-0.010 KS)

Compute which previous approved loans are still active at `decision_datetime` by estimating loan end date (`decision_datetime + term_months * 30.4 days`). Extract:
- Count of active loans
- Sum of active monthly installments
- Total active outstanding principal
- Ratio: (active installment + proposed installment) / declared income

This is the most standard credit-risk feature family and its absence is the single largest gap.

### 3.2 Income Verification via Bank Account (Estimated Impact: +0.003-0.007 KS)

Load `income_like_credits` and `average_balance` from `account_behavior.csv`. Compute:
- `income_vs_observed = declared_monthly_income / mean(income_like_credits)`
- `income_vs_avg_balance = declared_monthly_income / mean(average_balance)`

Values >> 1.0 indicate income inflation (declared much higher than observed), a strong default predictor.

### 3.3 Cross-Application Bureau History (Estimated Impact: +0.003-0.005 KS)

Currently, bureau features are only from reports linked to the scored `application_id`. But customers have reports from previous applications. Join bureau observations on `customer_id` (not just `application_id`) to get:
- Historical z-score trajectory
- Delta between current and historical bureau scores (improving vs deteriorating)
- Historical utilization and delinquency trends

### 3.4 Custom KS Eval Metric for LightGBM (Estimated Impact: +0.002-0.005 KS)

Replace `metric='auc'` with a custom `feval` function that computes KS. This ensures early stopping selects the iteration that maximizes KS, not AUC.

```python
def lgb_ks_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    ks = ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic
    return 'ks', ks, True  # higher is better
```

### 3.5 Bayesian Hyperparameter Optimization (Estimated Impact: +0.002-0.005 KS)

The workspace's 12-config grid search (exp002) found +0.003. A proper Optuna search over a wider space (including `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `max_depth`) with 50-100 trials would likely find a better configuration.

### 3.6 Adversarial Validation (Estimated Impact: diagnostic, no direct KS gain)

Train a classifier to distinguish train from test rows. If train/test are indistinguishable (AUC ~ 0.5), the standard CV is reliable. If AUC >> 0.5, features contributing to the distinction are distribution-shifted and should be handled (e.g., removed or normalized). This step is absent from the workspace.

### 3.7 Noise Feature Pruning via Ablation (Estimated Impact: +0.001-0.003 KS)

Farhan's notebook identifies `NOISE_GROUPS = ('account', 'account_3m', 'repayment_6m', 'flags')` and `NOISE_COLS = ['term_months', 'has_bur', 'has_ec']` as non-contributing features and drops them. The workspace's exp012 attempted to drop 15 low-gain features but lost KS (-0.004) -- because it used gain-based selection which is unreliable. Ablation-based selection (drop a group, re-train, measure KS delta) as Farhan does is more trustworthy.

### 3.8 Full Account Behavior Column Usage (Estimated Impact: +0.001-0.003 KS)

Beyond `income_like_credits`, also load:
- `average_balance`: more stable than `closing_balance` for monthly averages
- `minimum_balance`: stress indicator (how low did the account get)
- `opening_balance`: combined with `closing_balance` gives intra-month flow direction

The workspace currently uses only 4 of 8 available account columns.

---

## 4. Diagnosis: Why 42 Experiments Reached Only 0.506

| Root Cause | Evidence | Impact |
|:---|:---|:---|
| Unused data columns (4 from account, 3 from previous, 3 from bureau) | `usecols` in build_features.py vs DATA_DICTIONARY.md | HIGH: missing income verification, active loan burden, revolving limits |
| No active-loan-at-cutoff concept | `agg_prev` only counts historical apps, never checks if old loans are still running | HIGH: fundamental credit-risk blind spot |
| No cross-application bureau history | `agg_bureau` joins on `application_id`, never on `customer_id` for historical reports | MEDIUM: misses bureau trajectory across applications |
| Single-pass aggregation paradigm | All features are groupby().agg() without second-pass derived features | MEDIUM: limits interaction complexity |
| Training metric mismatch | `metric='auc'` but evaluation on KS | LOW-MEDIUM: early stopping may pick sub-optimal iteration |
| Shallow hyperparameter search | 12-config grid, one-shot | LOW: tuning ceiling is low for tabular GBDT, but 2-5 point improvement still available |
| Same-paradigm iterations | exp027-042 all add more aggregates of same type | HIGH: explains diminishing returns |

---

## 5. Recommended Action Plan (Priority Ordered)

### Phase A: Close the Data Gap (Expected: 0.506 -> 0.515-0.520)

| Step | Action | Estimated Delta |
|:---|:---|:---:|
| A1 | Add `income_like_credits`, `average_balance`, `minimum_balance` to `agg_account` | +0.002 |
| A2 | Compute active-loan burden in `agg_prev` (count, installment sum, outstanding) | +0.005 |
| A3 | Add cross-table ratios: `x_total_inst_to_income`, `x_income_vs_observed`, `x_loan_to_prev_max` | +0.003 |
| A4 | Load `scheduled_monthly_installment`, `annual_interest_rate`, `term_months` from previous_applications | +0.002 |

### Phase B: Richer Feature Semantics (Expected: 0.520 -> 0.525-0.530)

| Step | Action | Estimated Delta |
|:---|:---|:---:|
| B1 | Cross-application bureau history (join on customer_id, compute z-score delta) | +0.003 |
| B2 | E-commerce payment method ratio in 90d, cancel value ratio, category HHI | +0.002 |
| B3 | Repayment streak cumprod, last-3/last-6 installment features | +0.001 |
| B4 | Ablation-based noise pruning (drop non-contributing groups identified by Farhan) | +0.001 |

### Phase C: Model Optimization (Expected: 0.530 -> 0.533-0.537)

| Step | Action | Estimated Delta |
|:---|:---|:---:|
| C1 | Custom KS eval_metric for early stopping | +0.002 |
| C2 | Optuna hyperparameter search (50-100 trials) | +0.002 |
| C3 | Multi-seed averaging (5 seeds per model) | +0.001 |
| C4 | Improved blend: 4-5 diverse models with optimized weights | +0.002 |

### Phase D: Validation Integrity (No direct KS gain, prevents surprises)

| Step | Action |
|:---|:---|
| D1 | Adversarial validation: verify train/test indistinguishability |
| D2 | Consider GroupKFold by `customer_id` if overlap is significant |
| D3 | Multi-seed CV (3 seeds) for stable OOF estimates |

---

## 6. Specific Code Changes Required in build_features.py

### 6.1 Account Behavior: Load Missing Columns

Current (`agg_account`, line 295-306):
```python
usecols=[
    "customer_id", "month", "closing_balance",
    "inflows", "outflows", "low_balance_days",
]
```

Should be:
```python
usecols=[
    "customer_id", "month", "closing_balance", "opening_balance",
    "inflows", "outflows", "low_balance_days",
    "income_like_credits", "average_balance", "minimum_balance",
]
```

### 6.2 Previous Applications: Load Installment/Rate Columns

Current (`agg_prev`, line 69-80):
```python
usecols=[
    "application_id", "customer_id", "decision_datetime",
    "requested_amount", "decision", "approved_amount",
]
```

Should add:
```python
    "term_months", "annual_interest_rate", "scheduled_monthly_installment",
```

### 6.3 New Function: Active Loan Computation

```python
def compute_active_loans(m, apps):
    """Compute active loan count and burden at decision time."""
    ap = m[m["decision"] == "approved"].copy()
    ap["end_date"] = ap["decision_datetime"] + pd.to_timedelta(
        ap["term_months"] * 30.4, unit="D"
    )
    act = ap[ap["end_date"] > ap["cutoff"]]
    gact = act.groupby("application_id")
    return pd.DataFrame({
        "prev_cnt_active": gact.size(),
        "prev_active_inst_sum": gact["scheduled_monthly_installment"].sum(),
        "prev_active_amt_sum": gact["approved_amount"].sum(),
    }).reset_index()
```

### 6.4 Cross-Table Ratios to Add in main()

```python
inc = feat["declared_monthly_income"].clip(lower=1)
feat["x_total_inst_to_income"] = (
    feat["prev_active_inst_sum"].fillna(0) + feat["proposed_monthly_installment"]
) / inc
feat["x_income_vs_observed"] = feat["declared_monthly_income"] / (
    feat["acct_income_cr_mean"].replace(0, np.nan)
)
feat["x_loan_to_prev_max"] = feat["loan_amount"] / (
    feat["prev_approved_max"].replace(0, np.nan)
)
```

---

## 7. Summary

The experiment stagnation is not due to model tuning ceiling or blend saturation -- it is fundamentally a **feature engineering coverage gap**. The workspace pipeline uses only 60-70% of the available data columns and misses several key credit-risk concepts:

1. **Active loan burden at decision time** -- the most impactful missing concept
2. **Income verification via bank account credits** -- a classic fraud/risk signal
3. **Cross-application bureau trajectory** -- how the customer's credit score evolves
4. **Cross-table ratios that encode financial stress** -- installment-to-income, debt-to-observed-income

Closing these gaps should bring OOF KS from 0.506 to the 0.52-0.53 range. Combined with model-level optimizations (KS eval metric, Optuna, multi-seed), reaching and exceeding the 0.53 target is achievable within the remaining competition window.

> [!WARNING]
> All estimated deltas above are independent-marginal estimates. Due to feature correlation and model capacity constraints, the cumulative gain will be less than the sum. A realistic combined estimate is +0.015-0.025 KS from current 0.506, placing the expected range at 0.521-0.531.
