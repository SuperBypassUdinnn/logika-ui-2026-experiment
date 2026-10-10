---
name: submission-gatekeeper
description: Mandatory gatekeeper for Kaggle submissions in Logika UI 2026. Use for auditing, scorecards, approval, and logging KS submissions.
---

## Trigger Keywords
`submit`, `submission`, `kaggle submit`, `leaderboard`, `upload prediction`, `kirim submission`

## Strict Invariant
NEVER SUBMIT PREDICTIONS TO KAGGLE WITHOUT EXPLICIT USER CONFIRMATION.

## Sequence

### 1. Audit
```bash
python scripts/audit_submission.py outputs/submissions/subXX/submission.csv --oof outputs/experiments/expXXX/oof.csv
```
Stop on non-zero exit.

### 2. Scorecard
```markdown
### Pre-Submission Verification Scorecard
- Candidate CSV: `outputs/submissions/subXX/submission.csv`
- Rows: 30000, header `application_id,default_probability`
- OOF KS: `0.XXXX` (plus fold-wise KS)
- Mean prob and deciles: `0.XXXX`
- Leakage checklist: cutoff `decision_datetime` passed
- Daily quota remaining: X of 3
```

### 3. Approval
Ask user for explicit approval before any `kaggle competitions submit`.

### 4. Log
After LB score arrives, update `docs/SUBMISSIONS.md` and `docs/LEDGER_EKSPERIMEN.md` with ID, timestamp, OOF KS, public KS, notebook path.
