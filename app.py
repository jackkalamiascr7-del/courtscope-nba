"""NBA performance-research starter: chat-like interface, no wagering features.

This starter deliberately uses clearly labelled synthetic examples unless the user
uploads their own game logs. It does not fetch live statistics or call an LLM.
"""

from pathlib import Path
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CourtScope | NBA Research", page_icon="🏀", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color: #0b1220; color: #e6edf7; }
      [data-testid="stSidebar"] { background-color: #101b30; }
      [data-testid="stMetric"] { background-color:#14233a; border:1px solid #263650;
          padding:14px; border-radius:14px; }
      h1,h2,h3 { letter-spacing:-.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
REQUIRED = ["game_date", "player", "team", "opponent", "home_away", "position", "minutes", "pts", "reb", "ast", "threes"]
STAT_NAMES = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "threes": "Three-pointers"}


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    absent = [key for key in REQUIRED if key not in raw.columns]
    if absent:
        raise ValueError(f"Missing CSV columns: {', '.join(absent)}")
    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for field in ["minutes", "pts", "reb", "ast", "threes"]:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=["game_date", "player", "pts", "reb", "ast", "threes"])
    if df.empty:
        raise ValueError("No valid dated game logs were found in this file.")
    return df.sort_values("game_date", ascending=False).reset_index(drop=True)


def parse_request(query: str, players: list[str], current_stat: str):
    lower = query.casefold()
    chosen = next((p for p in sorted(players, key=len, reverse=True) if p.casefold() in lower), None)
    if re.search(r"rebounds?|boards?", lower):
        field = "reb"
    elif re.search(r"assists?", lower):
        field = "ast"
    elif re.search(r"three[ -]?pointers?|3[ -]?pointers?|threes?|3pm", lower):
        field = "threes"
    elif re.search(r"points?|scoring", lower):
        field = "pts"
    else:
        field = current_stat
    return chosen, field


st.title("🏀 CourtScope")
st.caption("NBA performance research • player game logs • matchup history • visual analysis")

with st.sidebar:
    st.header("Research data")
    uploaded = st.file_uploader("Upload NBA game logs (CSV)", type=["csv"])
    use_demo = st.toggle("Use illustrative demo data", value=True, disabled=uploaded is not None)
    st.caption("Only data you upload is real. The included demo is entirely synthetic, not live NBA statistics.")
    with st.expander("Expected CSV columns"):
        st.code(", ".join(REQUIRED))
        st.caption("Each row should represent one player in one completed game.")

try:
    if uploaded is not None:
        all_games = prepare_data(pd.read_csv(uploaded))
        data_label = "Your uploaded game logs"
    elif use_demo:
        all_games = prepare_data(pd.read_csv(ROOT / "sample_game_logs.csv"))
        data_label = "ILLUSTRATIVE DEMO DATA — NOT REAL NBA STATS"
    else:
        all_games = None
        data_label = "No data connected"
except Exception as exc:
    all_games = None
    data_label = "Unable to load game logs"
    st.error(str(exc))

if all_games is None:
    st.info("Upload a CSV of verified player game logs, or enable illustrative demo data to explore the interface.")
    st.stop()

st.warning(data_label, icon="ℹ️")
players = sorted(all_games["player"].dropna().unique().tolist())

left, right = st.columns([2.0, 1.0])
with left:
    selected_player = st.selectbox("Player", players)
with right:
    selected_stat = st.selectbox("Statistic", list(STAT_NAMES), format_func=lambda x: STAT_NAMES[x])

if "messages" not in st.session_state:
    st.session_state.messages = []

for entry in st.session_state.messages[-6:]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

question = st.chat_input("Ask: Demo Forward points, or Demo Guard rebounds")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    detected_player, detected_stat = parse_request(question, players, selected_stat)
    if detected_player:
        selected_player, selected_stat = detected_player, detected_stat
        response = (f"Showing **{STAT_NAMES[selected_stat].lower()}** and recent game logs for "
                    f"**{selected_player}** below. This is a rule-based prototype, not yet an AI model.")
    else:
        response = ("I couldn't identify a player in the current dataset. "
                    "Try a player name displayed in the selector above, followed by 'points', "
                    "'rebounds', 'assists', or 'threes'.")
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(response)

subset = all_games[all_games["player"] == selected_player].copy()
if subset.empty:
    st.info("No games available for that player.")
    st.stop()

st.subheader(f"{selected_player} · {STAT_NAMES[selected_stat]}")
metric_cols = st.columns(4)
for idx, n in enumerate([5, 10, 20]):
    data = subset.head(n)[selected_stat]
    metric_cols[idx].metric(f"Last {len(data)} games average", f"{data.mean():.1f}")
metric_cols[3].metric("Games in dataset", len(subset))

plot_data = subset.head(20).sort_values("game_date")
st.markdown("**Game-by-game performance (up to 20 most recent games)**")
st.bar_chart(plot_data.set_index("game_date")[[selected_stat]], x_label="Game date", y_label=STAT_NAMES[selected_stat])

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Home vs away averages**")
    ha = subset.groupby("home_away")[[selected_stat, "minutes"]].mean().round(1)
    st.dataframe(ha.rename(columns={selected_stat: STAT_NAMES[selected_stat], "minutes": "Minutes"}), use_container_width=True)
with c2:
    st.markdown("**Historical opponent splits**")
    opp = subset.groupby("opponent")[selected_stat].agg(["count", "mean"]).round(1)
    opp = opp.rename(columns={"count": "Games", "mean": "Average"}).sort_values("Games", ascending=False)
    st.dataframe(opp, use_container_width=True)

st.markdown("**Recent game logs**")
show = subset.head(20).copy()
show["game_date"] = show["game_date"].dt.strftime("%Y-%m-%d")
st.dataframe(show[REQUIRED], use_container_width=True, hide_index=True)

with st.expander("What still needs to be built?"):
    st.markdown(
        "- Connect a properly licensed NBA statistics provider and keep game logs updated.\n"
        "- Add verified defensive statistics by position, clearly defining how positions and ranks are assigned.\n"
        "- Add scheduled opponents, team availability, and cited injury updates.\n"
        "- Connect an AI language model to explain verified statistics, with source links and no invented numbers."
    )
