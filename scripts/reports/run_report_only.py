"""Generate the analysis report (models already trained)."""
from pathlib import Path

from aoe4_predict.report import generate_report

generate_report(
    report_path=Path("reports/generated/analysis_report_s9s10_test_s11.md"),
    model_path=Path("models/aoe4_predict/lgbm_s9s10_test_s11.txt"),
    meta_path=Path("models/aoe4_predict/lgbm_s9s10_test_s11_meta.json"),
)
