# Submission Guardrails

1. Never run `kaggle competitions submit` without explicit user approval.
2. Always run `scripts/audit_submission.py` first and show scorecard.
3. Verify 30,000 rows, header `application_id,default_probability`, zero nulls.
4. Track daily quota (max 3/day). Stop if quota exhausted.
