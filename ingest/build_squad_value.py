"""Build a national-team squad-value table (as-of, leakage-safe) from Transfermarkt data.

For each nation and quarterly grid date g, sum the market value of its top-25 / top-11 most valuable
players (by citizenship), using each player's latest valuation strictly before g. A match on date D
then takes the most recent grid point g <= D, whose values used only valuations < g <= D.

Input : data/raw/transfermarkt/{players,player_valuations}.csv
Output: data/processed/squad_value.parquet  (nation, asof_date, sv_top25, sv_top11, n_players)
"""
import pandas as pd

BASE = "data/raw/transfermarkt"
OUT = "data/processed/squad_value.parquet"
TOP_N = 25

def main():
    pl = pd.read_csv(f"{BASE}/players.csv", usecols=["player_id", "country_of_citizenship"])
    pv = pd.read_csv(f"{BASE}/player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
    pv["date"] = pd.to_datetime(pv["date"])
    pv = pv.dropna(subset=["market_value_in_eur"])
    pv = pv.merge(pl, on="player_id", how="inner").dropna(subset=["country_of_citizenship"])
    pv = pv.sort_values("date").reset_index(drop=True)
    print(f"{len(pv):,} dated valuations across {pv['country_of_citizenship'].nunique()} citizenships")

    grid = pd.date_range("2005-01-01", pv["date"].max() + pd.offsets.QuarterBegin(), freq="QS")
    rows = []
    for g in grid:
        sub = pv[pv["date"] < g]
        if sub.empty:
            continue
        latest = sub.groupby("player_id").tail(1)              # most recent value before g, per player
        latest = latest.sort_values("market_value_in_eur", ascending=False)
        for nation, grp in latest.groupby("country_of_citizenship"):
            vals = grp["market_value_in_eur"].to_numpy()
            rows.append((nation, g, vals[:TOP_N].sum(), vals[:11].sum(), len(vals)))

    out = pd.DataFrame(rows, columns=["nation", "asof_date", "sv_top25", "sv_top11", "n_players"])
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(out):,}  nations={out['nation'].nunique()}  "
          f"grid {grid[0].date()}..{grid[-1].date()}")
    return out

if __name__ == "__main__":
    main()
