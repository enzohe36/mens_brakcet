"""Extract NetRtg from stats HTML, apply sigmoid, and add winrate column to teams CSV."""
import csv, re, math, os

BASE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE, "2026", "stats_2026.html")
CSV_PATH = os.path.join(BASE, "2026", "teams_2026.csv")

# Map CSV team name -> HTML display name
CSV_TO_HTML = {
    "akron": "Akron", "alabama": "Alabama", "arizona": "Arizona",
    "arkansas": "Arkansas", "byu": "BYU", "cal_baptist": "Cal Baptist",
    "clemson": "Clemson", "duke": "Duke", "florida": "Florida",
    "furman": "Furman", "georgia": "Georgia", "gonzaga": "Gonzaga",
    "hawaii": "Hawaii", "high_point": "High Point", "hofstra": "Hofstra",
    "houston": "Houston", "howard": "Howard", "idaho": "Idaho",
    "illinois": "Illinois", "iowa": "Iowa", "iowa_st": "Iowa St.",
    "kansas": "Kansas", "kennesaw_st": "Kennesaw St.", "kentucky": "Kentucky",
    "lehigh": "Lehigh", "long_island": "LIU", "louisville": "Louisville",
    "mcneese": "McNeese", "miami_fl": "Miami FL", "miami_oh": "Miami OH",
    "michigan": "Michigan", "michigan_st": "Michigan St.", "missouri": "Missouri",
    "nc_state": "N.C. State", "nebraska": "Nebraska",
    "north_carolina": "North Carolina", "north_dakota_st": "North Dakota St.",
    "northern_iowa": "Northern Iowa", "ohio_st": "Ohio St.", "penn": "Penn",
    "prairie_view_a_m": "Prairie View A&M", "purdue": "Purdue",
    "queens_nc": "Queens", "saint_louis": "Saint Louis",
    "saint_marys": "Saint Mary's", "santa_clara": "Santa Clara",
    "siena": "Siena", "smu": "SMU", "south_florida": "South Florida",
    "st_johns": "St. John's", "tcu": "TCU", "tennessee": "Tennessee",
    "tennessee_st": "Tennessee St.", "texas": "Texas", "texas_a_m": "Texas A&M",
    "texas_tech": "Texas Tech", "troy": "Troy", "ucf": "UCF", "ucla": "UCLA",
    "uconn": "Connecticut", "umbc": "UMBC", "utah_st": "Utah St.",
    "vanderbilt": "Vanderbilt", "vcu": "VCU", "villanova": "Villanova",
    "virginia": "Virginia", "wisconsin": "Wisconsin", "wright_st": "Wright St.",
}

# Reverse: HTML name -> CSV name
HTML_TO_CSV = {v: k for k, v in CSV_TO_HTML.items()}

# Parse HTML for NetRtg values
# Row pattern: <td>rank</td><td>...<a>Team Name</a>...</td><td>conf</td><td>W-L</td><td>NetRtg</td>
with open(HTML_PATH) as f:
    html = f.read()

# Extract team name (display text inside <a>) and NetRtg (5th <td>)
# td elements are on the same line with no whitespace between them;
# conf td contains a nested <a> tag, so use .*? to skip it.
row_re = re.compile(
    r'<tr[^>]*>\s*'
    r'<td[^>]*>\d+</td>'                               # rank
    r'<td[^>]*><a[^>]*>([^<]+)</a>.*?</td>'            # team name
    r'<td[^>]*>.*?</td>'                                # conf (has nested <a>)
    r'<td[^>]*>[^<]*</td>'                              # W-L
    r'<td[^>]*>([^<]+)</td>'                             # NetRtg
)

html_netrtg = {}  # HTML display name -> NetRtg float
for m in row_re.finditer(html):
    name = m.group(1).strip()
    try:
        netrtg = float(m.group(2).strip())
    except ValueError:
        continue
    html_netrtg[name] = netrtg

# Match to our 68 tournament teams
netrtg = {}  # csv_name -> NetRtg
missing = []
for csv_name, html_name in CSV_TO_HTML.items():
    if html_name in html_netrtg:
        netrtg[csv_name] = html_netrtg[html_name]
    else:
        missing.append((csv_name, html_name))

if missing:
    print(f"WARNING: Could not find NetRtg for: {missing}")
    raise SystemExit(1)

print(f"Found NetRtg for all {len(netrtg)} teams")

# Sigmoid: winrate = 1 / (1 + exp(-k * NetRtg))
# k ≈ 0.0317 fits KenPom's empirical relationship between NetRtg margin
# and game-outcome probability (~11-pt gap doubles the odds).
K = 0.0317
sigmoid = {t: 1.0 / (1.0 + math.exp(-K * v)) for t, v in netrtg.items()}

# Show examples
print(f"\nSigmoid (k={K}) examples:")
for t in ["duke", "miami_oh", "lehigh", "prairie_view_a_m"]:
    print(f"  {t}: NetRtg={netrtg[t]:+.2f}  winrate={sigmoid[t]:.6f}")
print(f"  Min: {min(sigmoid.values()):.6f}  Max: {max(sigmoid.values()):.6f}")

# Read existing CSV, add/update winrate column, preserve everything else
rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    for row in reader:
        rows.append(row)

if 'winrate' not in fieldnames:
    fieldnames.append('winrate')

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        row['winrate'] = f"{sigmoid[row['team']]:.10f}"
        writer.writerow(row)

print(f"\nUpdated {CSV_PATH} — added/updated 'winrate' column")
