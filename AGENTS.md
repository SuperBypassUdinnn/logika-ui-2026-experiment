# Agent Governance: Logika UI 2026 (Tabular, KS Metric)

This workspace hosts the Loan Default tabular competition pipeline for team
Farhan / Fadhlurrahman Alaudin / Ilham. Metric: Kolmogorov-Smirnov (KS) statistic.

## 1. Submission Gatekeeper (MANDATORY)

1. The agent is STRICTLY FORBIDDEN from running `kaggle competitions submit`
   without EXPLICIT, UNAMBIGUOUS user approval in the current turn.
2. Before requesting approval, the agent must run
   `python scripts/audit_submission.py <path_to_csv>` and present:
   - target file and timestamp
   - OOF KS score and fold-wise KS
   - predicted probability distribution (mean, deciles)
   - row count match with test (30,000), header `application_id,default_probability`
   - leakage checklist pass (cutoff `decision_datetime`)
   - remaining daily quota

## 2. Decision Rationale (MANDATORY)

For every model, feature, or post-processing change, record in
`docs/SUBMISSIONS.md` and `docs/LEDGER_EKSPERIMEN.md`:

1. Specific delta
2. Motivation and hypothesis
3. Expected OOF KS impact (especially high-risk segment)
4. Risk and trade-off

## 3. Reproducibility

1. Fixed seed `42` for all randomness.
2. Stratified 5-fold by default (`train_folds.csv`); time-based split only as
   explicit robustness check, never silent replacement.
3. Local OOF KS is source of truth, never public leaderboard probing.
4. One experiment = one folder `outputs/experiments/expXXX/` with
   `oof.csv`, `test_pred.npy`, `submission.csv`, `log.md`.
5. No duplicate canonical artifacts outside `outputs/`.
6. No `__pycache__` or weight dumps in repo.

## 4. Leakage Prevention (TABULAR-SPECIFIC)

1. All features must use only rows with timestamp strictly before
   `decision_datetime` of the scored application.
2. Bureau reports only if `available_datetime <= decision_datetime`.
3. E-commerce / repayment / account only if event month < decision month.
4. Application IDs and customer IDs are join keys only, never numeric features.

## 5. Directory Map

- `src/`: reusable modules, imported by notebooks
- `scripts/`: automation, no 500-line notebook builders
- `notebooks/`: thin wrappers calling `src/`
- `outputs/`: canonical artifacts only
- `current/`: per-member submodules (`current/fadhlur`, `current/farhan`, `current/ilham`)
- `docs/`: Bahasa Indonesia operational docs
- `.agents/skills/`: project skills, single source of truth, auto-discovered
  by OpenCode (compatibility path) and Antigravity, do not mirror elsewhere
- `.agents/rules/`: agent rules; `.agents/knowledge/`: canonical competition
  spec (overview, rules); `.agents/MEMORY.md`: shared project memory
  (decisions, stack, conventions) readable by all agents

## 6. Git Constraints

1. No commits, pushes, or PRs without explicit user instruction in current turn.
2. Commit messages follow Conventional Commits in plain text, no Markdown.
3. PR descriptions use rich Markdown, focus on deliverables, never mention
   local indexing tooling.
