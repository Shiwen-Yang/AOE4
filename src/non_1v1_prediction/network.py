"""
Teammate co-occurrence network + premade features.

Two layers, kept strictly separate:

  A. WEEKLY-CUTOFF features (leakage-free, fed to the outcome model). The cumulative
     co-team graph is snapshotted at ISO-week boundaries; a match in week W sees only
     games from weeks strictly before W. A pair are an "established premade" once their
     cumulative co-team count reaches TEAMMATE_X (`establish_week`), and that only counts
     for matches in later weeks. All SQL window aggregation — no networkx.

  B. DESCRIPTIVE full-window snapshot (analysis only, NEVER a model feature). The end-of
     -window thresholded graph for "who teams with whom": degrees, components, top groups,
     and the same-team-vs-opposite-team validation of the threshold x.

Edges sum same-team games across ALL team modes (a party plays together across modes).
"""
import time

import pandas as pd

from .config import TEAM_MODES, TEAMMATE_X, WEEK_TRUNC
from .db import get_conn, table_exists


def _modes_sql(modes: list[str]) -> str:
    return "(" + ",".join(f"'{m}'" for m in modes) + ")"


# ── Layer A: weekly-cutoff co-team tables ────────────────────────────────────

def build_coteam_base(conn, modes: list[str] | None = None) -> None:
    """coteam_weekly → coteam_cum (cum_before_week, cum_incl) → pair_establish."""
    modes = modes or TEAM_MODES
    m = _modes_sql(modes)

    conn.execute(f"""
    CREATE OR REPLACE TABLE coteam_weekly AS
    SELECT a.profile_id AS p1, b.profile_id AS p2,
           date_trunc('{WEEK_TRUNC}', g.started_at) AS wk,
           count(*) AS g
    FROM participants a
    JOIN participants b
      ON a.game_id = b.game_id AND a.team_id = b.team_id AND a.profile_id < b.profile_id
    JOIN games g ON g.game_id = a.game_id
    WHERE g.kind IN {m} AND g.started_at IS NOT NULL
    GROUP BY 1, 2, 3
    """)

    conn.execute("""
    CREATE OR REPLACE TABLE coteam_cum AS
    SELECT p1, p2, wk, g,
        COALESCE(SUM(g) OVER (PARTITION BY p1, p2 ORDER BY wk
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS cum_before_week,
        SUM(g) OVER (PARTITION BY p1, p2 ORDER BY wk
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)      AS cum_incl
    FROM coteam_weekly
    """)

    conn.execute(f"""
    CREATE OR REPLACE TABLE pair_establish AS
    SELECT p1, p2, MIN(wk) AS establish_week
    FROM coteam_cum
    WHERE cum_incl >= {TEAMMATE_X}
    GROUP BY 1, 2
    """)


def build_team_premade_agg(conn, modes: list[str] | None = None, rebuild: bool = True) -> None:
    """
    One row per (game_id, team_id) with weekly-cutoff premade features:
      max_prior_coteam, mean_prior_coteam, n_premade_pairs, n_premade_players,
      team_is_premade, premade_partners_mean.
    """
    modes = modes or TEAM_MODES
    m = _modes_sql(modes)
    if rebuild or not table_exists(conn, "coteam_cum"):
        build_coteam_base(conn, modes)

    # team's internal pairs, tagged with the match week
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE _team_pairs AS
    SELECT a.game_id, a.team_id, a.profile_id AS p1, b.profile_id AS p2,
           date_trunc('{WEEK_TRUNC}', g.started_at) AS wk
    FROM participants a
    JOIN participants b
      ON a.game_id = b.game_id AND a.team_id = b.team_id AND a.profile_id < b.profile_id
    JOIN games g ON g.game_id = a.game_id
    WHERE g.kind IN {m} AND g.started_at IS NOT NULL
    """)

    # pair-level aggregates (cum_before_week available because the pair plays this very week)
    conn.execute("""
    CREATE OR REPLACE TEMP TABLE _pairagg AS
    SELECT tp.game_id, tp.team_id,
        COALESCE(MAX(cc.cum_before_week), 0) AS max_prior_coteam,
        COALESCE(AVG(cc.cum_before_week), 0) AS mean_prior_coteam,
        SUM(CASE WHEN pe.establish_week IS NOT NULL AND pe.establish_week < tp.wk
                 THEN 1 ELSE 0 END)          AS n_premade_pairs
    FROM _team_pairs tp
    LEFT JOIN coteam_cum cc ON cc.p1 = tp.p1 AND cc.p2 = tp.p2 AND cc.wk = tp.wk
    LEFT JOIN pair_establish pe ON pe.p1 = tp.p1 AND pe.p2 = tp.p2
    GROUP BY 1, 2
    """)

    # distinct players involved in an established premade pair, per (game, team)
    conn.execute("""
    CREATE OR REPLACE TEMP TABLE _premade_players AS
    WITH est AS (
        SELECT tp.game_id, tp.team_id, tp.p1, tp.p2
        FROM _team_pairs tp
        JOIN pair_establish pe ON pe.p1 = tp.p1 AND pe.p2 = tp.p2
        WHERE pe.establish_week < tp.wk
    ),
    pids AS (
        SELECT game_id, team_id, p1 AS pid FROM est
        UNION
        SELECT game_id, team_id, p2 AS pid FROM est
    )
    SELECT game_id, team_id, count(*) AS n_premade_players
    FROM pids GROUP BY 1, 2
    """)

    # node feature: established-partner count per player, cumulative through prior weeks
    conn.execute("""
    CREATE OR REPLACE TEMP TABLE _player_partner_cum AS
    WITH events AS (
        SELECT p1 AS player, establish_week AS wk FROM pair_establish
        UNION ALL
        SELECT p2 AS player, establish_week AS wk FROM pair_establish
    ),
    per_week AS (
        SELECT player, wk, count(*) AS new_partners FROM events GROUP BY 1, 2
    )
    SELECT player, wk,
        SUM(new_partners) OVER (PARTITION BY player ORDER BY wk
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_incl
    FROM per_week
    """)

    # ASOF: each participant's established-partner count as of the week BEFORE their match
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE _part_node AS
    SELECT pp.game_id, pp.team_id,
           COALESCE(pc.cum_incl, 0) AS premade_partner_count
    FROM (
        SELECT pa.game_id, pa.team_id, pa.profile_id,
               date_trunc('{WEEK_TRUNC}', g.started_at) AS wk
        FROM participants pa
        JOIN games g ON g.game_id = pa.game_id
        WHERE g.kind IN {m} AND g.started_at IS NOT NULL
    ) pp
    ASOF LEFT JOIN _player_partner_cum pc
        ON pc.player = pp.profile_id AND pp.wk > pc.wk
    """)

    conn.execute("""
    CREATE OR REPLACE TABLE team_premade_agg AS
    SELECT
        pa.game_id, pa.team_id,
        pa.max_prior_coteam,
        pa.mean_prior_coteam,
        pa.n_premade_pairs,
        COALESCE(pp.n_premade_players, 0)            AS n_premade_players,
        CASE WHEN pa.n_premade_pairs > 0 THEN 1 ELSE 0 END AS team_is_premade,
        COALESCE(nd.premade_partners_mean, 0)        AS premade_partners_mean
    FROM _pairagg pa
    LEFT JOIN _premade_players pp USING (game_id, team_id)
    LEFT JOIN (
        SELECT game_id, team_id, AVG(premade_partner_count) AS premade_partners_mean
        FROM _part_node GROUP BY 1, 2
    ) nd USING (game_id, team_id)
    """)


# Premade columns added to each team (become _a/_b) and the directional ones (become _diff).
PREMADE_TEAM_COLS = [
    "max_prior_coteam", "mean_prior_coteam", "n_premade_pairs", "n_premade_players",
    "team_is_premade", "premade_partners_mean",
]
PREMADE_DIFF_COLS = [
    "max_prior_coteam", "mean_prior_coteam", "n_premade_pairs", "n_premade_players",
    "premade_partners_mean",
]


def ensure_premade(conn, modes: list[str] | None = None) -> None:
    if not table_exists(conn, "team_premade_agg"):
        build_team_premade_agg(conn, modes)


# ── Layer B: descriptive full-window network (analysis only) ─────────────────

def build_teammate_edges(conn, modes: list[str] | None = None) -> None:
    """Full-window edge table: games_together (same team) + opponent_games (control)."""
    modes = modes or TEAM_MODES
    m = _modes_sql(modes)
    conn.execute(f"""
    CREATE OR REPLACE TABLE teammate_edges AS
    WITH same AS (
        SELECT a.profile_id p1, b.profile_id p2, count(*) games_together
        FROM participants a
        JOIN participants b ON a.game_id=b.game_id AND a.team_id=b.team_id AND a.profile_id<b.profile_id
        JOIN games g ON g.game_id=a.game_id
        WHERE g.kind IN {m}
        GROUP BY 1,2
    ),
    opp AS (
        SELECT a.profile_id p1, b.profile_id p2, count(*) opponent_games
        FROM participants a
        JOIN participants b ON a.game_id=b.game_id AND a.team_id<>b.team_id AND a.profile_id<b.profile_id
        JOIN games g ON g.game_id=a.game_id
        WHERE g.kind IN {m}
        GROUP BY 1,2
    )
    SELECT s.p1, s.p2, s.games_together, COALESCE(o.opponent_games, 0) AS opponent_games
    FROM same s LEFT JOIN opp o ON o.p1=s.p1 AND o.p2=s.p2
    """)


def threshold_distribution(conn, modes: list[str] | None = None,
                           xs=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20)) -> pd.DataFrame:
    """
    Same-team vs opposite-team pair counts at each candidate x — the x-validation table.

    The opposite-team population is computed INDEPENDENTLY (all pairs who ever faced each
    other), giving the true random-matchmaking baseline: you cannot queue to be *against* a
    chosen player, so opposite-team co-occurrence is pure chance.
    """
    modes = modes or TEAM_MODES
    m = _modes_sql(modes)
    if not table_exists(conn, "teammate_edges"):
        build_teammate_edges(conn, modes)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE _opp_pairs AS
    SELECT a.profile_id p1, b.profile_id p2, count(*) g
    FROM participants a
    JOIN participants b ON a.game_id=b.game_id AND a.team_id<>b.team_id AND a.profile_id<b.profile_id
    JOIN games gg ON gg.game_id=a.game_id
    WHERE gg.kind IN {m}
    GROUP BY 1,2
    """)
    rows = []
    for x in xs:
        same = conn.execute("SELECT count(*) FROM teammate_edges WHERE games_together>=?", [x]).fetchone()[0]
        opp = conn.execute("SELECT count(*) FROM _opp_pairs WHERE g>=?", [x]).fetchone()[0]
        rows.append({"x": x, "same_team_pairs": same, "random_opp_pairs": opp,
                     "enrichment_ratio": same / max(opp, 1)})
    return pd.DataFrame(rows)


def build_player_network_stats(conn, x: int | None = None) -> "pd.DataFrame":
    """
    Full-window node stats on the thresholded graph (degree, weighted degree, clustering,
    component). DESCRIPTIVE ONLY — not a model feature.
    """
    import networkx as nx

    x = x or TEAMMATE_X
    if not table_exists(conn, "teammate_edges"):
        build_teammate_edges(conn)
    edges = conn.execute(
        "SELECT p1, p2, games_together FROM teammate_edges WHERE games_together>=?", [x]
    ).df()

    G = nx.Graph()
    G.add_weighted_edges_from(edges[["p1", "p2", "games_together"]].itertuples(index=False, name=None))

    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    clus = nx.clustering(G)
    comp_id = {}
    comp_size = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_id[n] = cid
            comp_size[n] = len(comp)

    stats = pd.DataFrame({
        "profile_id": list(deg.keys()),
        "degree": [deg[n] for n in deg],
        "weighted_degree": [wdeg[n] for n in deg],
        "clustering": [clus[n] for n in deg],
        "component_id": [comp_id[n] for n in deg],
        "component_size": [comp_size[n] for n in deg],
    })
    conn.execute("CREATE OR REPLACE TABLE player_network_stats AS SELECT * FROM stats")
    return stats


def network_summary(conn, x: int | None = None) -> dict:
    """Structural summary of the thresholded graph for the report."""
    import networkx as nx

    x = x or TEAMMATE_X
    stats = (conn.execute("SELECT * FROM player_network_stats").df()
             if table_exists(conn, "player_network_stats")
             else build_player_network_stats(conn, x))
    n_edges = conn.execute("SELECT count(*) FROM teammate_edges WHERE games_together>=?", [x]).fetchone()[0]
    return {
        "x": x,
        "n_nodes": int(len(stats)),
        "n_edges": int(n_edges),
        "n_components": int(stats["component_id"].nunique()),
        "largest_component": int(stats["component_size"].max()),
        "mean_degree": float(stats["degree"].mean()),
        "median_degree": float(stats["degree"].median()),
        "max_degree": int(stats["degree"].max()),
    }


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))
    return head + "\n" + body


def write_network_report(conn, modes: list[str] | None = None, out_path=None) -> "Path":
    """Persist the descriptive 'who teams with whom' deliverable as markdown."""
    from pathlib import Path

    from .config import REPORT_DIR, TEAMMATE_X

    modes = modes or TEAM_MODES
    summ = network_summary(conn)
    summ["mean_degree"] = round(summ["mean_degree"], 2)
    dist = threshold_distribution(conn, modes).round({"enrichment_ratio": 1})
    top = conn.execute("""
        SELECT p1, p2, games_together, opponent_games
        FROM teammate_edges ORDER BY games_together DESC LIMIT 12
    """).df()
    comp = conn.execute("""
        SELECT component_size, count(*) AS n_components
        FROM (SELECT component_id, any_value(component_size) component_size
              FROM player_network_stats GROUP BY component_id)
        GROUP BY component_size ORDER BY component_size DESC LIMIT 8
    """).df()

    L = ["# AOE4 Teammate Co-occurrence Network\n",
         f"_Modes {modes} · threshold x = {TEAMMATE_X} · generated {time.strftime('%Y-%m-%d')}_\n",
         "Nodes = players; an edge means two players were teammates in ≥ x games (summed across "
         "all team modes). This is the **descriptive full-window** graph (who teams with whom). "
         "The model uses a separate **weekly-snapshot, leakage-free** version of these "
         "relationships — see the per-mode predictability reports.\n",
         "## Threshold validation (same-team vs random opposite-team)\n",
         "Two players cannot queue to be *against* each other, so opposite-team co-occurrence is "
         "the pure random-matchmaking baseline. `enrichment_ratio` = how many more same-team "
         f"pairs reach x than random. x = {TEAMMATE_X} keeps a strongly premade-dominated edge set.\n",
         _md(dist), "",
         "## Graph structure\n",
         _md(pd.DataFrame([summ])), "",
         "Largest connected components (by size):\n",
         _md(comp), "",
         "## Top teammate pairs (high games together, ~zero as opponents = clearly premade)\n",
         _md(top), ""]

    out_path = Path(out_path or (REPORT_DIR / "non_1v1_teammate_network_report.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    return out_path


def build_all(modes: list[str] | None = None, db_path=None) -> dict:
    """End-to-end: weekly-cutoff feature tables + descriptive network. Returns summary."""
    conn = get_conn(db_path)
    try:
        t0 = time.time()
        print("  Building weekly co-team tables ...", flush=True)
        build_coteam_base(conn, modes)
        print("  Building team premade aggregates ...", flush=True)
        build_team_premade_agg(conn, modes, rebuild=False)
        print("  Building descriptive edges + node stats ...", flush=True)
        build_teammate_edges(conn, modes)
        build_player_network_stats(conn)
        summ = network_summary(conn)
        print(f"  Network built in {time.time()-t0:.1f}s", flush=True)
        return summ
    finally:
        conn.close()
