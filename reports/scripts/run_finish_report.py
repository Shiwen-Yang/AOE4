"""Resume: retrain XGB with tuned params, then generate report."""
import json, gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aoe4_predict.db import get_conn
from aoe4_predict.features_extra import extend_training_features, FAMILY_FEATURES, DISABLED_FAMILIES
from aoe4_predict.model import train_xgb, XGB_DEFAULT_PARAMS
from aoe4_predict.report import generate_report

TRAIN_SEASONS = [9, 10]
TEST_SEASONS  = [11]
XGB_MODEL  = Path("models/xgb_s9s10_test_s11.ubj")
XGB_META   = Path("models/xgb_s9s10_test_s11_meta.json")
LGBM_MODEL = Path("models/lgbm_s9s10_test_s11.txt")
LGBM_META  = Path("models/lgbm_s9s10_test_s11_meta.json")
REPORT     = Path("reports/analysis_report_s9s10_test_s11.md")

# Load tuned XGB best params
best_params = json.loads(Path("models/xgb_best_params.json").read_text())

print("Loading dataset from DuckDB extended tables (training_features already built)...")
conn = get_conn(None)
families = set(FAMILY_FEATURES.keys()) - DISABLED_FAMILIES
df = extend_training_features(conn, None, families)
conn.close()
print(f"  Dataset: {len(df):,} rows × {len(df.columns)} cols")

print("\nRe-training XGBoost with tuned params (S9+S10 → S11 holdout)...")
final_params = {**XGB_DEFAULT_PARAMS, **best_params}
_, meta = train_xgb(df, params=final_params, test_seasons=TEST_SEASONS,
                    model_path=XGB_MODEL, meta_path=XGB_META)
print("\n── XGBoost Tuned Metrics ──")
for split, m in meta["metrics"].items():
    print(f"  {split:<6}  AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")

del df; gc.collect()

print("\nGenerating analysis report...")
generate_report(report_path=REPORT, model_path=LGBM_MODEL, meta_path=LGBM_META)
