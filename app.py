"""CourtScope V3: NBA historical player research & conversational statistics.

Not a betting decision or prediction service. Numbers come from the downloaded
or uploaded game-log dataset only; chat is rule-based (no generative model yet).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import unicodedata

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CourtScope | NBA Research", page_icon="🏀", layout="wide")
st.markdown("""<style>
.stApp { background-color: #0b1220; color: #e6edf7; }
[data-testid="stSidebar"] { background-color: #101b30; }
[data-testid="stMetric"] {background-color:#14233a;border:1px solid #263650;padding:14px;border-radius:14px;}
[data-testid="stMetric"] label, [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] {color:#edf5ff !important; opacity:1 !important;}
.stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {color:#d5e2f3;}
h1,h2,h3 {letter-spacing:-.02em;}
</style>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
COLS = ["game_date", "player", "team", "opponent", "home_away", "position", "minutes", "pts", "reb", "ast", "threes"]
STATS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "threes": "Three-pointers made"}
SEASONS = {"2026–27": 2027, "2025–26": 2026}
# Full name/alias -> abbreviations that occur in game logs, including ESPN's GS/UTAH style.
TEAM_ALIASES = {
    "atlanta hawks": "ATL", "hawks": "ATL", "boston celtics": "BOS", "celtics": "BOS",
    "brooklyn nets": "BKN", "nets": "BKN", "charlotte hornets": "CHA", "hornets": "CHA",
    "chicago bulls": "CHI", "bulls": "CHI", "cleveland cavaliers": "CLE", "cavaliers": "CLE", "cavs": "CLE",
    "dallas mavericks": "DAL", "mavericks": "DAL", "mavs": "DAL", "denver nuggets": "DEN", "nuggets": "DEN",
    "detroit pistons": "DET", "pistons": "DET", "golden state warriors": "GS", "golden state": "GS", "warriors": "GS", "gsw": "GS",
    "houston rockets": "HOU", "rockets": "HOU", "indiana pacers": "IND", "pacers": "IND",
    "los angeles clippers": "LAC", "la clippers": "LAC", "clippers": "LAC",
    "los angeles lakers": "LAL", "la lakers": "LAL", "lakers": "LAL", "memphis grizzlies": "MEM", "grizzlies": "MEM",
    "miami heat": "MIA", "heat": "MIA", "milwaukee bucks": "MIL", "bucks": "MIL",
    "minnesota timberwolves": "MIN", "timberwolves": "MIN", "wolves": "MIN",
    "new orleans pelicans": "NO", "pelicans": "NO", "new york knicks": "NY", "knicks": "NY",
    "oklahoma city thunder": "OKC", "thunder": "OKC", "orlando magic": "ORL", "magic": "ORL",
    "philadelphia 76ers": "PHI", "76ers": "PHI", "sixers": "PHI", "phoenix suns": "PHX", "suns": "PHX",
    "portland trail blazers": "POR", "trail blazers": "POR", "blazers": "POR",
    "sacramento kings": "SAC", "kings": "SAC", "san antonio spurs": "SA", "spurs": "SA",
    "toronto raptors": "TOR", "raptors": "TOR", "utah jazz": "UTAH", "jazz": "UTAH",
    "washington wizards": "WSH", "wizards": "WSH",
}

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

def normalize_boxscores(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["game_date", "athlete_display_name", "team_abbreviation",
                "opponent_team_abbreviation", "home_away", "minutes", "points",
                "rebounds", "assists", "three_point_field_goals_made"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("Source format has changed. Missing columns: " + ", ".join(missing))
    df = raw.copy()
    if "season_type" in df.columns:
        df = df[pd.to_numeric(df["season_type"], errors="coerce") == 2]
    if "did_not_play" in df.columns:
        not_playing = df["did_not_play"].fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))
        df = df[~not_playing]
    df = df.rename(columns={
        "athlete_display_name": "player", "team_abbreviation": "team",
        "opponent_team_abbreviation": "opponent", "points": "pts", "rebounds": "reb",
        "assists": "ast", "three_point_field_goals_made": "threes",
        "athlete_position_abbreviation": "position"})
    if "position" not in df.columns:
        df["position"] = "—"
    df["home_away"] = df["home_away"].fillna("Unknown").astype(str).str.capitalize()
    return prepare_data(df[COLS])

@st.cache_data(ttl=4*60*60, show_spinner=False)
def get_nba_logs(season: int) -> tuple[pd.DataFrame, str]:
    url = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
           f"espn_nba_player_boxscores/player_box_{season}.parquet")
    response = requests.get(url, timeout=35, headers={
        "User-Agent":"CourtScope-NBA-Research/3.0", "Accept":"application/octet-stream"})
    response.raise_for_status()
    if not response.content:
        raise ValueError("The NBA dataset download is empty.")
    return normalize_boxscores(pd.read_parquet(BytesIO(response.content))), url

def normalized_name(text: str) -> str:
    """Ignore accents when matching a name, while retaining original display."""
    return "".join(char for char in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(char))


def pick_player(query: str, names: list[str]) -> str | None:
    lower = normalized_name(query)
    # Full names take priority over ambiguous first/surnames.
    full = [name for name in names if normalized_name(name) in lower]
    if full:
        return max(full, key=len)
    matches = [name for name in names if any(
        len(part) >= 5 and re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", lower)
        for part in normalized_name(name).split())]
    return matches[0] if len(matches) == 1 else None

def pick_stat(query: str, current: str) -> str:
    lower = query.casefold()
    if re.search(r"\brebounds?\b|\bboards?\b", lower): return "reb"
    if re.search(r"\bassists?\b", lower): return "ast"
    if re.search(r"\b(?:threes?|3pm|three[- ]?pointers?|3[- ]?pointers?)\b", lower): return "threes"
    if re.search(r"\bpoints?\b|\bscoring\b", lower): return "pts"
    return current

def pick_opponent(query: str, seen: set[str]) -> str | None:
    lower = query.casefold()
    # The dataset abbreviations take precedence when present after "vs" or "against".
    match = re.search(r"\b(?:vs\.?|versus|against)\s+([A-Za-z]{2,5})\b", query, re.I)
    if match:
        literal = match.group(1).upper()
        if literal in seen: return literal
    for alias, wanted in sorted(TEAM_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lower):
            equivalent = {wanted}
            # ESPN team abbreviations can be GS/GSW, NY/NYK, SA/SAS, NO/NOP, WSH/WAS, UTAH/UTA.
            for first, second in [("GS","GSW"),("NY","NYK"),("SA","SAS"),("NO","NOP"),("WSH","WAS"),("UTAH","UTA")]:
                if wanted == first: equivalent.add(second)
                if wanted == second: equivalent.add(first)
            possible = equivalent & seen
            if possible: return sorted(possible)[0]
    return None

def selected_window(q: str) -> int:
    m = re.search(r"\blast\s+(5|10|15|20|30)\s*(?:games?)?", q.casefold())
    return int(m.group(1)) if m else 10

def make_answer(player: str, stat: str, opponent: str | None, window: int,
                data: pd.DataFrame, season: str, url: str | None) -> str:
    logs = data.loc[data.player == player].sort_values("game_date", ascending=False)
    label = STATS[stat].lower()
    latest = logs.iloc[0]
    period = logs.head(window)
    last_five = logs.head(5)
    result = [f"### {player} — {STATS[stat]}",
              f"**Dataset:** {season} | Most recent recorded game: **{latest.game_date.date()}** | "
              f"{len(logs)} games recorded",
              f"- **Most recent game:** {int(latest[stat])} {label} vs {latest.opponent} "
              f"({latest.minutes:.1f} minutes; {int(latest.pts)} PTS, {int(latest.reb)} REB, "
              f"{int(latest.ast)} AST)",
              f"- **Last {len(period)} games:** {period[stat].mean():.1f} {label} per game",
              f"- **Season dataset:** {logs[stat].mean():.1f} {label} per game",
              f"- **Recent minutes:** {period.minutes.mean():.1f} minutes per game",
              "- **Last five games:** " + ", ".join(str(int(value)) for value in last_five[stat])
              + f" {label} (newest first)"]
    if opponent:
        head_to_head = logs.loc[logs.opponent == opponent]
        if len(head_to_head):
            result.append(f"- **Against {opponent}:** {head_to_head[stat].mean():.1f} "
                          f"{label} across {len(head_to_head)} games")
            brief = head_to_head.head(5)
            result.append("**Most recent meetings:** " + "; ".join(
                f"{row.game_date.date()}: {int(row[stat])}" for _, row in brief.iterrows()))
        else:
            result.append(f"- **Against {opponent}:** no games in the selected season dataset.")
    else:
        home = logs.loc[logs.home_away == "Home"]
        away = logs.loc[logs.home_away == "Away"]
        if len(home) and len(away):
            result.append(f"- **Home / away:** {home[stat].mean():.1f} home "
                          f"({len(home)} games), {away[stat].mean():.1f} away "
                          f"({len(away)} games)")
    result += ["**Context:** Historical observations, not predictions. "
               "No real-time injury feed, next-opponent feed, or defensive ranking by position.",
               f"**Source:** [SportsDataverse NBA player box scores]({url})"
               if url else "**Source:** uploaded/demo game logs; check their accuracy."]
    return "\n\n".join(result)

st.title("🏀 CourtScope")
st.caption("NBA historical player research · verified game logs · conversational statistics")
with st.sidebar:
    st.header("Research data")
    source = st.radio("Choose data", ["NBA season game logs", "Upload CSV", "Illustrative demo"], index=0)
    selected_season = st.selectbox("NBA season", list(SEASONS)) if source == "NBA season game logs" else "Uploaded/Demo"
    uploaded = st.file_uploader("Upload player-game CSV", type=["csv"]) if source == "Upload CSV" else None
    if source == "NBA season game logs":
        st.caption("Published season box scores. Dataset may lag recently completed NBA games.")
        if st.button("Refresh downloaded data"):
            get_nba_logs.clear()
            st.rerun()
    elif source == "Illustrative demo":
        st.warning("Synthetic example players. This is NOT live NBA data.")
    with st.expander("CSV column names"):
        st.code(", ".join(COLS))

try:
    if source == "NBA season game logs":
        with st.spinner("Loading published NBA box scores…"):
            all_games, source_url = get_nba_logs(SEASONS[selected_season])
        st.success(f"Historical NBA player game logs loaded · {selected_season}")
    elif source == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV to begin, or choose NBA season game logs.")
            st.stop()
        all_games, source_url = prepare_data(pd.read_csv(uploaded)), None
        st.info("Uploaded data: source, licence and accuracy not verified.")
    else:
        all_games = prepare_data(pd.read_csv(ROOT / "sample_game_logs.csv"))
        source_url = None
        st.warning("ILLUSTRATIVE DEMO — fictional statistics")
except Exception as e:
    st.error("Could not load NBA data. The provider may not have published this season yet.")
    st.caption(f"Details: {type(e).__name__}: {e}")
    st.info("Try 2025–26 for archived data or choose the illustrative demo. Never treat demo data as real.")
    st.stop()

latest = all_games.game_date.max().date()
if source_url:
    st.caption(f"Source: [SportsDataverse NBA box scores]({source_url}) · Latest available game: **{latest}**")
if source == "NBA season game logs" and selected_season == "2025–26":
    st.info("You're viewing the completed 2025–26 season, not the current 2026–27 season. "
            "Use the sidebar to switch seasons if the newer release is available.")

players = sorted(all_games.player.dropna().unique().tolist())
source_key = f"{source}:{selected_season}:{getattr(uploaded, 'name', '')}"
if st.session_state.get("source_key") != source_key:
    st.session_state.source_key = source_key
    st.session_state.selected_player = players[0]
    st.session_state.selected_stat = "pts"
    st.session_state.opponent_filter = "All opponents"
    st.session_state.messages = []

st.subheader("Ask CourtScope")
st.caption("Try: 'Jaylen Brown latest game', 'LeBron James points against Warriors', "
           "or 'Stephen Curry last 20 threes'. "
           "This is data-grounded chat, not a generative language model yet.")
question = st.chat_input("Ask about a player's performance or previous matchups…")
if question:
    st.session_state.messages.append({"role":"user", "content":question})
    if re.search(r"\b(?:bet(?:ting)?|wager|parlay|odds|prop|over|under)\b", question, re.I):
        answer = "I can explain historical NBA player and team statistics, but not evaluate bets or wagering lines. Try asking for recent game averages or previous matchups."
    else:
        chosen = pick_player(question, players)
        if chosen is None:
            answer = "I couldn't confidently match a player in this season. Use their full name, or select the player from the dropdown below."
        else:
            st.session_state.selected_player = chosen
            stat = pick_stat(question, st.session_state.selected_stat)
            st.session_state.selected_stat = stat
            opp = pick_opponent(question, set(all_games.opponent.dropna().astype(str)))
            st.session_state.opponent_filter = opp if opp and opp in set(all_games.loc[all_games.player==chosen, "opponent"]) else "All opponents"
            answer = make_answer(chosen, stat, opp, selected_window(question), all_games, selected_season, source_url)
    st.session_state.messages.append({"role":"assistant", "content":answer})

for message in st.session_state.messages[-10:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

left, right = st.columns([2, 1])
with left:
    st.selectbox("Player", players, key="selected_player")
with right:
    st.selectbox("Statistic", list(STATS), format_func=lambda s:STATS[s], key="selected_stat")

player, stat = st.session_state.selected_player, st.session_state.selected_stat
logs = all_games.loc[all_games.player == player].sort_values("game_date", ascending=False).copy()
st.subheader(f"{player} · {STATS[stat]}")
st.caption(f"{logs.iloc[0].team} · {len(logs)} games · Latest player game: {logs.iloc[0].game_date.date()}")
cols = st.columns(4)
for i, count in enumerate((5,10,20)):
    values = logs.head(count)[stat]
    cols[i].metric(f"Last {len(values)} games", f"{values.mean():.1f}")
cols[3].metric("Season games", str(len(logs)))

st.markdown("**Game-by-game performance (last 20)**")
st.bar_chart(logs.head(20).sort_values("game_date").set_index("game_date")[[stat]],
             x_label="Game date", y_label=STATS[stat])

home, opponent = st.columns(2)
with home:
    st.markdown("**Home / away performance**")
    comparison = logs.groupby("home_away").agg(games=(stat, "size"), average=(stat, "mean"), minutes=("minutes", "mean")).round(1)
    st.dataframe(comparison.rename(columns={"games":"Games", "average":STATS[stat], "minutes":"Average minutes"}), use_container_width=True)
with opponent:
    st.markdown("**Previous performances against opponents**")
    history = logs.groupby("opponent")[stat].agg(games="size", average="mean").round(1).sort_values("games", ascending=False)
    st.dataframe(history.rename(columns={"games":"Games", "average":STATS[stat]}), use_container_width=True)

st.markdown("**Player game logs**")
opp_options = ["All opponents"] + sorted(logs.opponent.dropna().unique().tolist())
if st.session_state.opponent_filter not in opp_options:
    st.session_state.opponent_filter = "All opponents"
st.selectbox("Filter by opponent", opp_options, key="opponent_filter")
display_logs = logs if st.session_state.opponent_filter == "All opponents" else logs.loc[logs.opponent == st.session_state.opponent_filter]
show = display_logs.head(30)[COLS].copy()
show["game_date"] = show["game_date"].dt.strftime("%Y-%m-%d")
st.dataframe(show, hide_index=True, use_container_width=True)

with st.expander("About CourtScope, data quality and limitations"):
    st.markdown("- This site currently provides rule-based conversational statistics, **not generative AI**.\n"
                "- Shows historical regular-season player box scores only. No live injuries, future opponent feed or predictions.\n"
                "- An opponent-history average is **not** a defensive ranking by position.\n"
                "- The latest available data date is always displayed. Choose a season deliberately.\n"
                "- Data source: [SportsDataverse](https://www.sportsdataverse.org/), "
                "whose NBA release assets are published under CC BY 4.0.\n"
                "- This is a basketball statistics research tool, not a wagering recommendation service.")
