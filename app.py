"""CourtScope: NBA player-performance research (not wagering advice).

Version 2 adds published NBA box-score data, while retaining a clearly-labelled
synthetic fallback. Uses no AI model: it interprets straightforward player/stat
requests and only reports calculations from supplied or retrieved rows.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CourtScope | NBA Research", page_icon="🏀", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; color: #e6edf7; }
    [data-testid="stSidebar"] { background-color: #101b30; }
    [data-testid="stMetric"] { background-color: #14233a; border: 1px solid #263650;
        padding: 14px; border-radius: 14px; }
    h1,h2,h3 { letter-spacing: -.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
COLS = ["game_date", "player", "team", "opponent", "home_away", "position", "minutes", "pts", "reb", "ast", "threes"]
STATS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "threes": "Three-pointers"}
SEASONS = {"2026–27": 2027, "2025–26": 2026}
DATA_CREDIT = "SportsDataverse (compiled from ESPN box scores), CC BY 4.0"


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(COLS) - set(frame.columns))
    if missing:
        raise ValueError("Missing data columns: " + ", ".join(missing))
    df = frame.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for c in ("minutes", "pts", "reb", "ast", "threes"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["game_date", "player", "pts", "reb", "ast", "threes"])
    df = df[df["minutes"].fillna(0) > 0].copy()
    if df.empty:
        raise ValueError("No valid completed player-game rows found.")
    return df.sort_values("game_date", ascending=False).reset_index(drop=True)


def normalize_espn_boxscores(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert documented SportsDataverse ESPN NBA box-score fields."""
    required = [
        "game_date", "athlete_display_name", "team_abbreviation",
        "opponent_team_abbreviation", "home_away", "minutes", "points",
        "rebounds", "assists", "three_point_field_goals_made",
    ]
    missing = [field for field in required if field not in raw.columns]
    if missing:
        raise ValueError("Data provider changed its columns: " + ", ".join(missing))

    df = raw.copy()
    if "season_type" in df.columns:
        df = df[pd.to_numeric(df["season_type"], errors="coerce") == 2]
    if "did_not_play" in df.columns:
        did_not_play = df["did_not_play"].fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))
        df = df[~did_not_play]
    df = df.rename(columns={
        "athlete_display_name": "player", "team_abbreviation": "team",
        "opponent_team_abbreviation": "opponent", "points": "pts",
        "rebounds": "reb", "assists": "ast",
        "three_point_field_goals_made": "threes",
        "athlete_position_abbreviation": "position",
    })
    if "position" not in df.columns:
        df["position"] = "—"
    df["home_away"] = df["home_away"].fillna("Unknown").astype(str).str.capitalize()
    return prepare_data(df[COLS])


@st.cache_data(ttl=4 * 60 * 60, show_spinner=False)
def get_nba_logs(season_end: int) -> tuple[pd.DataFrame, str]:
    # Publicly documented release asset, refreshed by the community data pipeline.
    url = (
        "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
        f"espn_nba_player_boxscores/player_box_{season_end}.parquet"
    )
    response = requests.get(
        url,
        timeout=35,
        headers={"User-Agent": "CourtScope-NBA-Research/2.0", "Accept": "application/octet-stream"},
    )
    response.raise_for_status()
    if not response.content:
        raise ValueError("The NBA data download was empty.")
    raw = pd.read_parquet(BytesIO(response.content))
    return normalize_espn_boxscores(raw), url


def match_request(query: str, known_players: list[str], default_stat: str) -> tuple[str | None, str]:
    lower = query.casefold()
    chosen = next((p for p in sorted(known_players, key=len, reverse=True) if p.casefold() in lower), None)
    if chosen is None:
        # Permit unique first-name or surname searches, e.g. 'LeBron points'.
        for p in known_players:
            chunks = p.casefold().split()
            if any(len(part) >= 5 and re.search(r"\b" + re.escape(part) + r"\b", lower) for part in chunks):
                matches = [name for name in known_players if any(
                    len(part) >= 5 and re.search(r"\b" + re.escape(part) + r"\b", lower)
                    for part in name.casefold().split()
                )]
                if len(matches) == 1:
                    chosen = matches[0]
                break
    if re.search(r"\brebounds?\b|\bboards?\b", lower):
        stat = "reb"
    elif re.search(r"\bassists?\b", lower):
        stat = "ast"
    elif re.search(r"\b(?:threes?|3pm|three[- ]?pointers?|3[- ]?pointers?)\b", lower):
        stat = "threes"
    elif re.search(r"\bpoints?\b|\bscoring\b", lower):
        stat = "pts"
    else:
        stat = default_stat
    return chosen, stat


st.title("🏀 CourtScope")
st.caption("NBA performance research · verified game logs · recent trends · matchup history")
with st.sidebar:
    st.header("Research data")
    source = st.radio("Choose data", ["NBA season game logs", "Upload CSV", "Illustrative demo"], index=0)
    selected_season = st.selectbox("NBA season", list(SEASONS)) if source == "NBA season game logs" else None
    uploaded = st.file_uploader("Upload player-game CSV", type=["csv"]) if source == "Upload CSV" else None
    if source == "NBA season game logs":
        st.caption("Published player box scores; updates depend on the data provider and may lag completed games.")
        if st.button("Refresh downloaded data"):
            get_nba_logs.clear()
            st.rerun()
    elif source == "Illustrative demo":
        st.warning("Synthetic example players and statistics, NOT real NBA data.")
    with st.expander("CSV columns"):
        st.code(", ".join(COLS))

all_games = None
source_url = None
try:
    if source == "NBA season game logs":
        with st.spinner("Loading published NBA box scores…"):
            all_games, source_url = get_nba_logs(SEASONS[selected_season])
        st.success(f"Real historical NBA box scores · {selected_season}")
    elif source == "Upload CSV" and uploaded:
        all_games = prepare_data(pd.read_csv(uploaded))
        st.info("Showing your uploaded CSV; verify its accuracy and permission to reuse it.")
    elif source == "Illustrative demo":
        demo_path = ROOT / "sample_game_logs.csv"
        if not demo_path.exists():
            st.error("The demo CSV is missing. Keep sample_game_logs.csv in your GitHub repository.")
            st.stop()
        all_games = prepare_data(pd.read_csv(demo_path))
        st.warning("ILLUSTRATIVE DEMO DATA — fictional players and statistics")
except Exception as exc:
    st.error("Unable to load player game logs. This is a data connection or file issue, not a player result.")
    st.caption(f"Technical details: {type(exc).__name__}: {exc}")
    st.info("Use the sidebar to select the illustrative demo, or retry the NBA data source later.")
    st.stop()

if all_games is None:
    st.info("Upload a CSV to see player statistics, or select another data source.")
    st.stop()

if source_url:
    st.caption(f"Source: [{DATA_CREDIT}]({source_url}) · Latest recorded game: {all_games['game_date'].max().date()}")

players = sorted(all_games["player"].dropna().unique().tolist())
if st.session_state.get("current_source") != source + str(selected_season):
    st.session_state.current_source = source + str(selected_season)
    st.session_state.selected_player = players[0]
    st.session_state.selected_stat = "pts"
    st.session_state.messages = []

question = st.chat_input("Ask: LeBron James points, Stephen Curry threes, Jokić rebounds…")
if question:
    selected_player, selected_stat = match_request(question, players, st.session_state.selected_stat)
    st.session_state.messages.append({"role": "user", "content": question})
    if selected_player:
        st.session_state.selected_player = selected_player
        st.session_state.selected_stat = selected_stat
        player_games = all_games[all_games["player"] == selected_player]
        last_five = player_games.head(5)
        response = (
            f"**{selected_player} · {STATS[selected_stat]}** · "
            f"last {len(last_five)} games average: **{last_five[selected_stat].mean():.1f}**. "
            f"Showing verified game logs, splits and history below. "
            f"This is a statistics lookup, not a generative AI response."
        )
    else:
        response = "I couldn't match that name in this dataset. Select a player below or enter their full name."
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages[-8:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

left, right = st.columns([2, 1])
with left:
    st.selectbox("Player", players, key="selected_player")
with right:
    st.selectbox("Statistic", list(STATS), format_func=lambda c: STATS[c], key="selected_stat")

player = st.session_state.selected_player
stat = st.session_state.selected_stat
subset = all_games.loc[all_games["player"] == player].copy()
if subset.empty:
    st.info("No game logs were found for that player.")
    st.stop()

st.subheader(f"{player} · {STATS[stat]}")
st.caption(f"{subset.iloc[0]['team']} · {len(subset)} recorded games · Most recent: {subset.iloc[0]['game_date'].date()}")
metrics = st.columns(4)
for idx, number in enumerate((5, 10, 20)):
    values = subset.head(number)[stat]
    metrics[idx].metric(f"Last {len(values)} games · average", f"{values.mean():.1f}")
metrics[-1].metric("Season games in dataset", len(subset))

plot = subset.head(20).sort_values("game_date")
st.markdown("**Recent game-by-game performance**")
st.bar_chart(plot.set_index("game_date")[[stat]], x_label="Game date", y_label=STATS[stat])

col_home, col_opp = st.columns(2)
with col_home:
    st.markdown("**Home / away performance**")
    split = subset.groupby("home_away").agg(
        games=(stat, "size"), average=(stat, "mean"), avg_minutes=("minutes", "mean")
    ).round(1)
    st.dataframe(split.rename(columns={"games": "Games", "average": STATS[stat], "avg_minutes": "Avg minutes"}), use_container_width=True)
with col_opp:
    st.markdown("**Previous performances by opponent**")
    opp = subset.groupby("opponent")[stat].agg(games="size", average="mean").round(1).sort_values("games", ascending=False)
    st.dataframe(opp.rename(columns={"games": "Games", "average": STATS[stat]}), use_container_width=True)

st.markdown("**Recent game logs**")
show = subset.head(20)[COLS].copy()
show["game_date"] = show["game_date"].dt.strftime("%Y-%m-%d")
st.dataframe(show, hide_index=True, use_container_width=True)

with st.expander("About the research and limitations"):
    st.markdown(
        "- Game logs show historical results, not predictions.\n"
        "- The chat currently recognises names and stat categories; it does **not** use a language model.\n"
        "- Opponent history is not the same as a defensive ranking by player position.\n"
        "- Data may be delayed or temporarily unavailable; use the displayed latest game date.\n"
        "- Source attribution: [SportsDataverse](https://www.sportsdataverse.org/) "
        "(data release licence: CC BY 4.0)."
    )
