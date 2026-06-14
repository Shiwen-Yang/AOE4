"""
Recompute pregame win probabilities for every usable replay match.

Fixes two bugs from the previous computation:
  1. Wrong argument order — slot2 was passed as player_a.
     Fix: always call predict_match(slot1_id, slot2_id).
  2. No timestamp filter — player stats included the replay match outcomes.
     Fix: pass before_timestamp=row['started_at'] to scope history.

Output: data/realtime_outcome_prediction/features/v1/pregame_probs.parquet
        columns: replay_id, pregame_win_prob_slot1
"""
import duckdb
import pandas as pd
from aoe4_predict.predict import predict_match
from aoe4_predict.model import load_model

MATCHES_PATH = "data/realtime_outcome_prediction/features/v1/matches.parquet"
OUT_PATH     = "data/realtime_outcome_prediction/features/v1/pregame_probs.parquet"
DB_PATH      = "aoe4_work.duckdb"

matches = pd.read_parquet(MATCHES_PATH)
usable  = matches[matches["usable"] == True].copy()
print(f"Usable matches: {len(usable)}")

conn   = duckdb.connect(DB_PATH, read_only=True)
model, meta = load_model()

rows = []
skip = 0
for i, (_, row) in enumerate(usable.iterrows()):
    slot1 = int(row["slot1_profile_id"])
    slot2 = int(row["slot2_profile_id"])
    ts    = row["started_at"]          # pandas Timestamp → datetime via predict_match

    try:
        result = predict_match(
            player_a_id=slot1,
            player_b_id=slot2,
            conn=conn,
            model=model,
            meta=meta,
            before_timestamp=ts,
        )
        prob = result["win_prob_a"]    # P(player_a wins) = P(slot1 wins)
    except Exception as exc:
        skip += 1
        print(f"  skip {row['replay_id']}: {exc}")
        continue

    rows.append({"replay_id": int(row["replay_id"]), "pregame_win_prob_slot1": prob})

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(usable)} done, {skip} skipped so far")

conn.close()
print(f"Computed: {len(rows)}, skipped: {skip}")

df = pd.DataFrame(rows)
print(df["pregame_win_prob_slot1"].describe())
df.to_parquet(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
