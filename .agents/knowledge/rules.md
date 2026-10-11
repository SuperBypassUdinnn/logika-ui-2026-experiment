# Competition Rules: DSC LOGIKA UI 2026 (Knowledge)

Source: Kaggle `data-science-competition-logika-ui-2026` Rules tab.
Scraped 2026-10-11. Binding constraints mapped to workspace enforcement.

## 1. Eligibility

- Active Indonesian undergraduate students (S1/D4/D3/D2/D1), odd semester 2026/2027.
- Team of 2-3 members, may span universities, one team per participant.
- Kaggle team name must match committee registration.

## 2. Participation

- Follow committee procedures; re-registration required for finalists.

## 3. Dataset and Modeling (enforced via `.agents/rules/`)

- ONLY committee-provided datasets. External data prohibited.
  Enforced: `competition_invariants.md`.
- Follow documented table relationships (`data/DATA_DICTIONARY.md`).
- Features only from info available at `decision_datetime`.
  Enforced: `leakage_prevention.md`, `docs/ANTI_LEAKAGE.md`.
- No future target-period info, no leakage.

## 4. Language

- Python required. Enforced: `requirements.txt`, `src/`.

## 5. Submission (enforced via `submission_guardrails.md`, `scripts/audit_submission.py`)

- Via Kaggle only. Columns `application_id,default_probability`.
- Exactly one prediction per test application, probs in [0, 1].
- No missing/duplicated/extra IDs. Row count must match test (30,000).

## 6. Evaluation

- KS statistic. Public 45%, Private 45%, Notebook reproducibility 10%.

## 7-8. Notebook

- Reproducible, executable, no unauthorized external data.
- Naming `ParticipantNumber_TeamName_Notebook`.
- Deadline 30 Oct 2026 23:59 WIB, -1 point per late minute.

## 9-10. Final

- Top 8 elimination teams advance; offline final at UI Depok.

## 11-12. Prohibited and Sanctions

- No multi-teaming, false info, external data, leakage, LB manipulation,
  plagiarism, cheating. Sanctions up to disqualification and prize revocation.

## 13. Privacy

- No de-anonymization attempts of individuals, orgs, or locations.

## Kaggle Foundational Rules (operationally relevant excerpts)

- One Kaggle account per person; one team per participant.
- Max team size per competition page; mergers allowed before deadline
  if combined size and submission counts stay within limits.
- No private code/data sharing outside the team. Public sharing allowed
  only on the competition forum/notebooks under an OSI-approved license.
- Winners decided by Private LB rank; ties broken by earlier submission.
- Submissions must not use hand labeling or human prediction of test data.
