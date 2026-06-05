"""Direct training runner — avoids the -m module path that gets killed by sandbox."""
import gc, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aoe4_predict.db import get_conn
from aoe4_predict.features import build_civ_matchup_priors, build_player_stats, build_training_features
from aoe4_predict.features_extra import extend_training_features, FAMILY_FEATURES, DISABLED_FAMILIES
from aoe4_predict.model import train as train_model
import sys

def run(seasons, test_seasons):
    all_seasons = sorted(set(seasons) | set(test_seasons))
    families = set(FAMILY_FEATURES.keys()) - DISABLED_FAMILIES
    train_tag = "s" + "s".join(str(s) for s in seasons)
    test_tag  = "s" + "s".join(str(s) for s in test_seasons)
    model_path = Path(f"models/lgbm_{train_tag}_test_{test_tag}.txt")
    meta_path  = model_path.parent / (model_path.stem + "_meta.json")

    print(f"Training on seasons: {seasons}  (test holdout: {test_seasons})")
    print(f"Extra feature families: {sorted(families)}")

    conn = get_conn(None)
    t0 = time.time()

    print("\n1. Building player_stats...")
    build_player_stats(conn)
    print("\n2. Building civ matchup priors...")
    build_civ_matchup_priors(conn)
    print("\n3. Building training features...")
    df = build_training_features(conn, train_seasons=all_seasons)

    print("\n3b. Adding extended feature families...")
    del df
    gc.collect()
    conn.close()
    conn = get_conn(None)
    df = extend_training_features(conn, None, families)
    conn.close()

    n_feat = len([c for c in df.columns if c != 'target'])
    print(f"\n4. Training model ({len(df):,} rows, {n_feat} features)...")
    _, meta = train_model(df, model_path=model_path, meta_path=meta_path,
                          test_seasons=test_seasons)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print("\n── LightGBM Metrics ──")
    for split, m in meta["metrics"].items():
        print(f"  {split:<6}  AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  Brier={m['brier']:.4f}")
    return meta

if __name__ == "__main__":
    # Run 1: S7-S10 train, S11 test
    run(seasons=[7, 8, 9, 10], test_seasons=[11])
