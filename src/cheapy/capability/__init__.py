"""The fitted capability model behind `performance_score`.

Inference only: `capability_model.score_for_trajectory()` loads `artifacts/pair_model.json`
and scores a trajectory in milliseconds, with no network and no API keys. The offline
elicitation and training pipeline that *produced* that artifact lives outside the shipped
package, in `research/capability_fitting/`.

Design: docs/capability_model.md (read its REVISION header first).
"""
