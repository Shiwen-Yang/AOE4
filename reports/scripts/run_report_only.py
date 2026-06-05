"""Generate the analysis report (models already trained)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aoe4_predict.report import generate_report

generate_report(
    report_path=Path("reports/analysis_report_s9s10_test_s11.md"),
    model_path=Path("models/lgbm_s9s10_test_s11.txt"),
    meta_path=Path("models/lgbm_s9s10_test_s11_meta.json"),
)
