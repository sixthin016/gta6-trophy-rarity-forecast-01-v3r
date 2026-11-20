"""Practice proxy for the v3-rebuild forecaster.

The real practice answers (the 04:30Z / 05:00Z reads) are sealed on the
make.fun server -- grade-local refuses this competition ("graded on the
server only") -- so this script scores our actual p00/p01 forecasts
against synthetic refresh truths built from the practice cases
themselves: the history is perfectly flat at 638 players, one refresh
lands at the target, new players earn at phi x the incumbent rate.

Scenarios (mult = player-count jump of the refresh, phi = new-cohort
earn rate relative to incumbents):
  A  mult 5,  phi 0.5  -- our own prior (PHI_NEW / MULT_DEFAULT)
  B  mult 5,  phi 0.7  -- the backtest flat-refresh truth
  C  mult 2,  phi 0.7  -- a small refresh
  D  mult 10, phi 0.7  -- a big refresh
The saturated trophy (rate 100%) uses phi = 1 in every scenario: it is
the game's first trophy, everybody gets it, so a refresh does not
dilute it.  Truths are rounded to 2 decimals, as the page prints them.
The persistence baseline is scored on the same truths for comparison.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
KIT_IN = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
              "gta6-trophy-rarity-forecast-01-kit/input")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

SCENARIOS = {
    "A": (5.0, 0.5),
    "B": (5.0, 0.7),
    "C": (2.0, 0.7),
    "D": (10.0, 0.7),
}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(p):
    p = clamp_pct(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def pinball(y, qs):
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = logit_pct(y) - logit_pct(q)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
    return tot / len(LEVELS)


def score(truth, forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    vals = [pinball(truth[i], by_id[i + 1]) for i in range(len(truth))]
    return sum(vals) / len(vals)


def refresh_truth(last_rates, mult, phi):
    out = []
    for r in last_rates:
        p = 1.0 if r >= 99.99 else phi
        out.append(round(r * (1.0 + p * (mult - 1.0)) / mult, 2))
    return out


def persistence_forecast(case):
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    z = [-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722]
    return {"trophies": [
        {"id": tr["id"],
         "q": [round(inv_logit_pct(logit_pct(last[i]) + zz * spread), 4)
               for zz in z]}
        for i, tr in enumerate(case["trophies"])]}


def run_main(case):
    with tempfile.TemporaryDirectory() as td:
        ind = Path(td) / "in"
        outd = Path(td) / "out"
        ind.mkdir()
        (ind / (case["id"] + ".in")).write_text(json.dumps(case))
        env = dict(__import__("os").environ,
                   MAKEFUN_INPUT_DIR=str(ind), MAKEFUN_OUTPUT_DIR=str(outd))
        r = subprocess.run([sys.executable, str(MAIN)], env=env,
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr[-1000:]
        return json.loads((outd / (case["id"] + ".out")).read_text())


def main():
    print("== practice proxy (real p00/p01, synthetic refresh truths) ==")
    print("%-6s %-4s %10s %10s %8s" % ("case", "scen", "ours", "persist",
                                      "ratio"))
    rows = []
    for name in ("p00", "p01"):
        case = json.loads((KIT_IN / (name + ".in")).read_text())
        ours = run_main(case)
        last = case["history"][-1]["rates"]
        pers = persistence_forecast(case)
        for scen, (mult, phi) in SCENARIOS.items():
            truth = refresh_truth(last, mult, phi)
            lo = score(truth, ours)
            lp = score(truth, pers)
            rows.append((scen, lo, lp))
            print("%-6s %-4s %10.4f %10.4f %8.3f" % (name, scen, lo, lp,
                                                    lo / lp))
    print("-- means over both cases --")
    for scen in SCENARIOS:
        vals = [(lo, lp) for s, lo, lp in rows if s == scen]
        mo = sum(v[0] for v in vals) / len(vals)
        mp = sum(v[1] for v in vals) / len(vals)
        print("scenario %s: ours %.4f  persistence %.4f  ratio %.3f"
              % (scen, mo, mp, mo / mp))
    all_o = [lo for _, lo, _ in rows]
    all_p = [lp for _, _, lp in rows]
    print("OVERALL mean: ours %.4f  persistence %.4f  ratio %.3f"
          % (sum(all_o) / len(all_o), sum(all_p) / len(all_p),
             (sum(all_o) / len(all_o)) / (sum(all_p) / len(all_p))))


if __name__ == "__main__":
    main()
