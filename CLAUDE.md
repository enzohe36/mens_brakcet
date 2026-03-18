- Be brutally honest and straightforward in your response.
- Whenever the user asks you to identify an issue or make a suggestion, generate a well-thought-out answer first, and then reconsider each part of the answer, assuming it wrong until proven true.
- If you are unsure about how to answer a question, ask for additional information that you need to answer it.
- Do not give suggestions that "might work"; give suggestions that you are sure will work.
- If you plan to modify a file, outline your proposed changes and ask for user confirmation first. Do not implement anything without explicit user confirmation.
- If the user is wrong, point it out.

## Workflow

1. **Build `teams_<year>.csv`** from the official NCAA bracket PDF. Columns:
   - `team` — lowercase, spaces and symbols replaced by `_` (e.g. `st_johns`, `texas_a_m`)
   - `region` — one of `east`, `west`, `south`, `midwest`
   - `opp_region` — the opposing region in the Final Four
   - `seed` — seed within each region (1–16)
   - `first4` — `0` = no, `1` = yes (First Four play-in team)
2. **Run `get_winrate.py`** — extracts NetRtg from `stats_<year>.html`, applies sigmoid, and adds the `winrate` column to the CSV.
3. **Run `get_bracket.py`** — runs the GA and generates bracket draws.
