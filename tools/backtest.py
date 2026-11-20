"""Synthetic harness for the v3-rebuild forecaster.

Builds Exophase-like worlds with a known truth, runs main.py on them
as a subprocess (exactly as the grader would), and scores the result
against a persistence baseline with the contest's own pinball loss on
the logit scale.  Also checks the expanding-window one-step CV, the
practice cases for format / determinism / runtime, and degenerate
edge cases.

Worlds (all share the same observation mechanics: the page is frozen
between refreshes and jumps at a refresh, because a refresh brings in
a batch of newly tracked players):
  weibull      -- true rate p(t) = 1 - exp(-((t+2)/lam)^k)
  loglogistic  -- true rate p(t) = 1 / (1 + (t0/t)^b)
  delayed      -- true rate 0 until a per-trophy delay, then logistic
  flat-refresh -- history perfectly flat, one refresh lands at the
                  target (the practice shape); new players earn at
                  0.7x the incumbent rate in the truth
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
KIT_IN = Path(os.environ.get(
    "KIT_INPUT",
    "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
TIERS = (["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46)


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


def score_forecast(truth, forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    vals = [pinball(truth[i], by_id[i + 1]) for i in range(len(truth))]
    return sum(vals) / len(vals)


def persistence_forecast(case):
    """The house baseline, re-implemented here for comparison only."""
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last[i] if last[i] is not None else 0.0
        lg = logit_pct(r)
        rows.append({"id": tr["id"],
                     "q": [round(inv_logit_pct(lg + z * spread), 4)
                           for z in Z]})
    return {"trophies": rows}


# ------------------------------------------------------------- world gen

def _fmt(hours):
    base = 1763510400  # 2026-11-19T00:00:00Z as a unix time
    import datetime
    dt = datetime.datetime.fromtimestamp(base + hours * 3600,
                                         tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def true_rate(world, prm, t):
    if t <= 0:
        return 0.0
    if world == "weibull":
        lam, k = prm
        return 100.0 * (1.0 - math.exp(-((t + 2.0) / lam) ** k))
    if world == "loglogistic":
        t0, b = prm
        return 100.0 / (1.0 + (t0 / t) ** b)
    if world == "delayed":
        delay, s, cap = prm
        if t < delay:
            return 0.0
        return cap / (1.0 + math.exp(-(t - delay) / s))
    if world == "flat-refresh":
        base, = prm
        return base
    raise ValueError(world)


def make_params(world, rng):
    prm = []
    for tier in TIERS:
        scale = {"platinum": 3.0, "gold": 1.6, "silver": 1.0,
                 "bronze": 0.55}[tier]
        if world == "weibull":
            prm.append((rng.uniform(250, 900) * scale,
                        rng.uniform(0.7, 1.4)))
        elif world == "loglogistic":
            prm.append((rng.uniform(150, 700) * scale,
                        rng.uniform(0.8, 1.6)))
        elif world == "delayed":
            prm.append((rng.uniform(20, 300) * scale,
                        rng.uniform(8, 40),
                        rng.uniform(0.5, 60)))
        elif world == "flat-refresh":
            prm.append((rng.uniform(0.5, 3.0),))
    return prm


def build_case(world, seed, horizon_hours=108.0, flat=False):
    rng = random.Random(seed)
    prm = make_params(world, rng)
    # player-count path: stepwise jumps at refresh times
    n = 638.0
    refreshes = []
    t = rng.uniform(2.0, 5.0)
    while t < 48.0 + horizon_hours + 1.0:
        refreshes.append(t)
        t += rng.uniform(3.5, 6.5)
    if flat:
        refreshes = [48.0 + horizon_hours - 0.5]
    n_at = {0.0: 638.0}
    cur = 638.0
    for rt in refreshes:
        cur *= math.exp(rng.gauss(math.log(1.6), 0.25))
        n_at[rt] = cur

    def observed(t_read):
        # page state as of the last refresh at or before t_read
        past = [rt for rt in refreshes if rt <= t_read]
        rt = past[-1] if past else 0.0
        nn = n_at.get(rt, 638.0) if rt else 638.0
        rates = []
        for i in range(55):
            if world == "flat-refresh":
                # earners frozen from launch; new cohort earns at 0.7x
                e_old = prm[i][0] / 100.0 * 638.0
                e_new = 0.7 * prm[i][0] / 100.0 * (nn - 638.0) if rt else 0.0
                rates.append(round(100.0 * (e_old + e_new) / nn, 2))
            else:
                p = true_rate(world, prm[i], rt if rt else 0.1)
                rates.append(round(p, 2))
        return int(round(nn)), rates

    hist = []
    t = 0.1
    while t < 48.0:
        nn, rates = observed(t)
        hist.append({"slot": _fmt(t), "at": _fmt(t),
                     "players_tracked": nn,
                     "trophies_earned": int(sum(r * nn / 100.0
                                                for r in rates)),
                     "rates": rates})
        t += 0.5
    # truth at the target
    t_tgt = 48.0 + horizon_hours
    nn_t, truth = observed(t_tgt)
    if world == "flat-refresh":
        pass  # observed() already applies the 0.7x new-cohort truth
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "syn-%s-%d" % (world, seed),
        "at": _fmt(48.0),
        "horizon_minutes": int(horizon_hours * 60),
        "target": _fmt(t_tgt),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19", "source": "synthetic"},
        "trophies": [{"id": i + 1, "name": "Trophy %d" % (i + 1),
                      "description": "", "tier": TIERS[i],
                      "points": {"platinum": 300, "gold": 90,
                                 "silver": 30, "bronze": 15}[TIERS[i]],
                      "secret": False} for i in range(55)],
        "history": hist,
        "notes": "synthetic world %s seed %d" % (world, seed),
    }
    return case, truth


def run_main(case):
    with tempfile.TemporaryDirectory() as td:
        ind = Path(td) / "in"
        outd = Path(td) / "out"
        ind.mkdir()
        (ind / (case["id"] + ".in")).write_text(json.dumps(case))
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(ind),
                   MAKEFUN_OUTPUT_DIR=str(outd))
        t0 = time.time()
        r = subprocess.run([sys.executable, str(MAIN)], env=env,
                           capture_output=True, text=True, timeout=30)
        dt = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError("main.py failed: " + r.stderr[-2000:])
        return json.loads((outd / (case["id"] + ".out")).read_text()), dt


def check_valid(forecast, n=55):
    assert set(forecast.keys()) == {"trophies"}, forecast.keys()
    assert len(forecast["trophies"]) == n
    ids = sorted(r["id"] for r in forecast["trophies"])
    assert ids == list(range(1, n + 1)), ids[:5]
    for r in forecast["trophies"]:
        assert set(r.keys()) == {"id", "q"}
        q = r["q"]
        assert len(q) == 5
        assert all(0.0 <= v <= 100.0 for v in q), q
        assert all(q[i] <= q[i + 1] for i in range(4)), q


# ------------------------------------------------------------- the checks

def synthetic_worlds():
    print("== synthetic worlds (final-like horizon 108 h) ==")
    print("%-14s %-6s %10s %10s %8s" % ("world", "seed", "ours", "persist",
                                       "ratio"))
    totals = {}
    for world in ("weibull", "loglogistic", "delayed", "flat-refresh"):
        for seed in (11, 23):
            case, truth = build_case(world, seed,
                                     flat=(world == "flat-refresh"))
            ours, dt = run_main(case)
            check_valid(ours)
            lo = score_forecast(truth, ours)
            lp = score_forecast(truth, persistence_forecast(case))
            totals.setdefault(world, []).append((lo, lp))
            print("%-14s %-6d %10.4f %10.4f %8.3f   (%.1fs)"
                  % (world, seed, lo, lp, lo / lp, dt))
    print("-- means --")
    for w, vals in totals.items():
        mo = sum(v[0] for v in vals) / len(vals)
        mp = sum(v[1] for v in vals) / len(vals)
        print("%-14s ours %.4f  persistence %.4f  ratio %.3f"
              % (w, mo, mp, mo / mp))


def one_step_cv():
    print("== expanding-window one-step CV (weibull world) ==")
    case, _ = build_case("weibull", 7)
    hist = case["history"]
    for cut in (len(hist) - 3, len(hist) - 2, len(hist) - 1):
        sub = dict(case)
        sub["history"] = hist[:cut]
        sub["at"] = hist[cut - 1]["at"]
        sub["horizon_minutes"] = 30
        sub["target"] = hist[cut]["at"]
        sub["id"] = "cv-%d" % cut
        ours, _ = run_main(sub)
        check_valid(ours)
        truth = hist[cut]["rates"]
        lo = score_forecast(truth, ours)
        lp = score_forecast(truth, persistence_forecast(sub))
        print("cut %d/%d: ours %.4f  persistence %.4f  ratio %.3f"
              % (cut, len(hist), lo, lp, lo / lp))


def practice_cases():
    print("== practice cases: format / determinism / runtime ==")
    for name in ("p00", "p01"):
        src = KIT_IN / (name + ".in")
        with tempfile.TemporaryDirectory() as td:
            ind = Path(td) / "in"
            out1 = Path(td) / "o1"
            out2 = Path(td) / "o2"
            ind.mkdir()
            (ind / (name + ".in")).write_text(src.read_text())
            env = dict(os.environ)
            for outd in (out1, out2):
                env2 = dict(env, MAKEFUN_INPUT_DIR=str(ind),
                            MAKEFUN_OUTPUT_DIR=str(outd))
                t0 = time.time()
                r = subprocess.run([sys.executable, str(MAIN)], env=env2,
                                   capture_output=True, text=True,
                                   timeout=30)
                dt = time.time() - t0
                assert r.returncode == 0, r.stderr[-1000:]
                print("%s run: %.2fs" % (name, dt))
            b1 = (out1 / (name + ".out")).read_bytes()
            b2 = (out2 / (name + ".out")).read_bytes()
            assert b1 == b2, "non-deterministic output!"
            assert len(b1) <= 65536, len(b1)
            check_valid(json.loads(b1))
            print("%s: valid, %d bytes, byte-identical across runs"
                  % (name, len(b1)))


def edge_cases():
    print("== edge cases ==")
    base, _ = build_case("weibull", 3)
    cases = {}
    one = dict(base)
    one["history"] = base["history"][:1]
    one["id"] = "edge-1read"
    cases["1 read"] = one
    empty = dict(base)
    empty["history"] = []
    empty["id"] = "edge-empty"
    cases["empty history"] = empty
    nulls = dict(base)
    nulls["history"] = [dict(h, rates=[None if (i % 3 == 0) else r
                                      for i, r in enumerate(h["rates"])])
                        for h in base["history"]]
    nulls["id"] = "edge-nulls"
    cases["null rates (1/3)"] = nulls
    floor = dict(base)
    floor["history"] = [dict(h, rates=[0.01] * 55) for h in base["history"]]
    floor["id"] = "edge-floor"
    cases["all rates at floor"] = floor
    for name, c in cases.items():
        out, _ = run_main(c)
        check_valid(out)
        print("%-20s OK" % name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if "all" in which or "worlds" in which:
        synthetic_worlds()
    if "all" in which or "cv" in which:
        one_step_cv()
    if "all" in which or "practice" in which:
        practice_cases()
    if "all" in which or "edge" in which:
        edge_cases()
    print("ALL CHECKS DONE")
