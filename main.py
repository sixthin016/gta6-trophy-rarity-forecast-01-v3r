"""GTA VI trophy rarity forecaster (v3 rebuild, written fresh in this session).

Forecasts every trophy's Exophase earn rate at a case's target as five
quantiles, scored by pinball loss on the logit scale.

Idea, in one paragraph.  Exophase does not refresh continuously: the page
can sit frozen for hours and then jump, because a refresh brings in a batch
of newly tracked players who hold fewer trophies.  A forecaster that just
repeats the last rate is therefore wrong exactly when a refresh lands
inside the horizon.  This program handles the two regimes it can see in a
case's history:

* Flat history (no rate or player-count movement at all): the next refresh
  is still ahead of us, so the median is a dilution forecast -- the
  earners seen so far diluted by the players a refresh is expected to
  bring in -- and the quantiles come from the count process itself
  (Poisson new earners over a lognormal refresh multiplier), not from a
  symmetric normal fan.
* Moving history: a per-trophy ensemble of three growth curves
  (log-linear, log-logistic, two-stage Weibull), weighted by a
  rolling-origin backtest on the case's own history, blended -- only when
  an expanding-window cross-validation honestly beats that ensemble by
  at least 10% -- with a joint hierarchical binomial model whose trophy
  intercepts shrink toward PAVA-ordered tier means.

Standard library only, deterministic (every random draw comes from a
seed fixed per trophy id), and it reads $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR
with /input and /output as fallbacks.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- priors / tuning constants (stated here, none fitted to any answer) ---
PHI_NEW = 0.5        # a newly tracked player earns a trophy at half the
                     # incumbent cohort's rate, on average (prior only)
MULT_DEFAULT = 5.0   # expected player-count jump of one refresh when the
                     # case's own history shows no jump at all
S_MULT = 0.35        # lognormal spread of the refresh multiplier
N_MC = 3000          # Monte-Carlo draws for the flat-history count fan
CURVE_SIGMA_SCALE = 0.6  # v3: curve-path spread is shrunk by this factor
CAP_PCT = 90.0       # upper cap for every quantile (see saturated branch)
RW_SIGMA = 0.5       # random-walk logit sigma per sqrt(day), house scale
SEED_BASE = 20261119
TIER_ORDER = ["platinum", "gold", "silver", "bronze"]
TIER_FALLBACK = {"platinum": 0.5, "gold": 1.0, "silver": 2.0, "bronze": 3.0}


# ---------------------------------------------------------------- utilities

def clamp_pct(x):
    """Clamp a percent into the range the logit can represent."""
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half away from zero on the scaled value."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    if value == int(value):
        return int(value)
    return value


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def quantile_sorted(s, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def ols_line(xs, ys):
    """Least-squares y = a + b*x; returns (a, b, resid_std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 1.0
    mx = mean(xs)
    my = mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12 or n < 2:
        resid = [y - my for y in ys]
        sd = math.sqrt(mean([r * r for r in resid])) if resid else 1.0
        return my, 0.0, sd
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(mean([r * r for r in resid]))
    return a, b, sd


def pava_increasing(vals):
    """Pool-adjacent-violators: smallest non-decreasing sequence >= in
    least-squares sense (isotonic regression, equal weights)."""
    blocks = [[v, 1] for v in vals]  # [sum, count]
    out = []
    for v, c in blocks:
        out.append([v, c])
        while len(out) >= 2 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            s = out[-2][0] + out[-1][0]
            k = out[-2][1] + out[-1][1]
            out[-2:] = [[s, k]]
    res = []
    for s, c in out:
        res.extend([s / c] * c)
    return res


def _norm_ppf(u):
    """Acklam's rational approximation of the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if u < plow:
        q = math.sqrt(-2.0 * math.log(u))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if u > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - u))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = u - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1.0)


# Stratified standard-normal quantiles, computed once: the count fans
# draw from this table (rotated per trophy) instead of calling the RNG
# thousands of times, which keeps a 55-trophy case fast and exactly
# reproducible without any per-run randomness.
_Z_TABLE = [_norm_ppf((i + 0.5) / N_MC) for i in range(N_MC)]


def poisson_draw(rng, lam):
    """One Poisson(lam) draw: Knuth's product method only for small
    lam (where it is cheap); a normal approximation with rounding for
    lam >= 5, which is far faster and indistinguishable at that size."""
    if lam <= 0:
        return 0
    if lam < 5.0:
        limit = math.exp(-lam)
        prod = 1.0
        k = 0
        while True:
            prod *= rng.random()
            if prod <= limit:
                return k
            k += 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


# ------------------------------------------------------------ curve models

def _weibull_base(tau, k, lam):
    """log(expm1(((tau + 1) / lam) ** k)) -- the logit of a Weibull CDF
    with a 1-hour offset so tau = 0 is finite."""
    u = ((tau + 1.0) / lam) ** k
    if u > 40.0:
        return u  # log(expm1(u)) -> u for large u
    return math.log(math.expm1(u))


def fit_weibull(taus, ys):
    """Two-stage Weibull fit on logit rates.

    Coarse grid over (k, lam), then two local refinement rounds around
    the best cell (lam steps of x1.15, k steps of +/-0.15, then halved).
    The intercept is the mean residual for each (k, lam), fitted in
    closed form.  Returns (predict_fn, resid_std).
    """
    if len(taus) < 3:
        m = mean(ys) if ys else 0.0
        return (lambda t: m), 1.0
    best = None  # (sse, k, lam, a)
    ks = [0.3 + 0.25 * i for i in range(11)]           # 0.3 .. 2.8
    lams = [math.exp(i * (math.log(3000.0) / 15.0)) for i in range(16)]
    for k in ks:
        for lam in lams:
            base = [_weibull_base(t, k, lam) for t in taus]
            a = mean([y - b for y, b in zip(ys, base)])
            sse = sum((y - (a + b)) ** 2 for y, b in zip(ys, base))
            if best is None or sse < best[0]:
                best = (sse, k, lam, a)
    for step_k, step_lam in ((0.15, 1.15), (0.075, 1.07)):
        _, bk, blam, _ = best
        for dk in (-step_k, 0.0, step_k):
            for f in (1.0 / step_lam, 1.0, step_lam):
                k = max(0.1, bk + dk)
                lam = max(0.2, blam * f)
                base = [_weibull_base(t, k, lam) for t in taus]
                a = mean([y - b for y, b in zip(ys, base)])
                sse = sum((y - (a + b)) ** 2 for y, b in zip(ys, base))
                if sse < best[0]:
                    best = (sse, k, lam, a)
    _, k, lam, a = best
    resid = [y - (a + _weibull_base(t, k, lam)) for t, y in zip(taus, ys)]
    sd = math.sqrt(mean([r * r for r in resid]))
    return (lambda t, k=k, lam=lam, a=a: a + _weibull_base(t, k, lam)), sd


def fit_curves(taus, ys):
    """Fit the three curve families on (tau_hours, logit rate).

    Returns a list of dicts {pred, sd} with pred(tau) -> logit rate:
    log-linear on ln(1+tau), log-logistic on ln(tau+0.25), Weibull.
    """
    models = []
    a1, b1, sd1 = ols_line([math.log1p(t) for t in taus], ys)
    models.append({"pred": (lambda t, a=a1, b=b1: a + b * math.log1p(t)),
                   "sd": sd1})
    a2, b2, sd2 = ols_line([math.log(t + 0.25) for t in taus], ys)
    models.append({"pred": (lambda t, a=a2, b=b2: a + b * math.log(t + 0.25)),
                   "sd": sd2})
    wpred, wsd = fit_weibull(taus, ys)
    models.append({"pred": wpred, "sd": wsd})
    return models


def rolling_origin_weights(taus, ys, max_origins=4):
    """Weight the curve families by rolling-origin backtest.

    Each origin fits on the reads up to that origin and scores the next
    read by squared logit error.  Weights are inverse-MSE, normalised;
    with fewer than two origins every family weighs the same.
    """
    n = len(taus)
    if n < 4:
        return [1.0 / 3.0] * 3
    # Origin fits run on a thinned series (<= 16 points): the Weibull
    # grid is the expensive part and the weights barely move with more.
    if n > 16:
        keep = _subsample(list(range(n)), cap=16)
        taus = [taus[j] for j in keep]
        ys = [ys[j] for j in keep]
        n = len(taus)
    origins = list(range(max(2, n - 1 - max_origins), n - 1))
    if not origins:
        return [1.0 / 3.0] * 3
    errs = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for j in origins:
        sub = fit_curves(taus[:j + 1], ys[:j + 1])
        for m in range(3):
            e = sub[m]["pred"](taus[j + 1]) - ys[j + 1]
            errs[m] += e * e
            counts[m] += 1
    mse = [errs[m] / max(1, counts[m]) for m in range(3)]
    inv = [1.0 / (v + 0.02) for v in mse]
    tot = sum(inv)
    return [v / tot for v in inv]


def ensemble_predict(taus, ys, tau_target, weights):
    """Weighted curve ensemble at the target.  Returns (logit, resid_sd)."""
    models = fit_curves(taus, ys)
    pred = sum(w * m["pred"](tau_target) for w, m in zip(weights, models))
    var = sum(w * m["sd"] ** 2 for w, m in zip(weights, models))
    return pred, math.sqrt(max(var, 1e-4))


# ------------------------------------------- joint hierarchical binomial

def _smooth_rw2(obs, lam_pen=2.0, sweeps=400):
    """Penalised least squares with a second-difference (RW2) penalty,
    solved by Gauss-Seidel sweeps:  min sum (f-obs)^2 + lam * sum d2^2."""
    n = len(obs)
    f = list(obs)
    if n < 3:
        return f
    for _ in range(sweeps):
        delta = 0.0
        for j in range(n):
            # row j of (I + lam * D2' D2) f = obs, solved coordinate-wise:
            # second differences touching j are rows i in {j-2..j}
            num = obs[j]
            den = 1.0
            for i in range(max(0, j - 2), min(n - 2, j + 1)):
                # d2_i = f_i - 2 f_{i+1} + f_{i+2}; coefficient of f_j
                c = 0.0
                if i == j:
                    c = 1.0
                elif i + 1 == j:
                    c = -2.0
                elif i + 2 == j:
                    c = 1.0
                if c == 0.0:
                    continue
                den += lam_pen * c * c
                others = 0.0
                for jj, cc in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
                    if jj != j:
                        others += cc * f[jj]
                num -= lam_pen * c * others
            new = num / den
            delta = max(delta, abs(new - f[j]))
            f[j] = new
        if delta < 1e-10:
            break
    return f


def fit_joint(taus, logity, counts_n, tiers, horizon_tau):
    """Hierarchical binomial-style joint model.

    logit p[i,j] = alpha[i] + f(t_j) + delta * log(N_j / N0).
    alpha[i] shrinks toward its tier mean, tier means are PAVA-ordered
    (platinum < gold < silver < bronze in logit), f is RW2-smoothed,
    delta is shrunk toward 0.  Returns (pred_logit[i], sigma, info)
    with predictions at horizon_tau, or None when the data are too thin.
    """
    n_tr = len(logity)
    n_rd = len(taus)
    if n_tr == 0 or n_rd < 4:
        return None
    n0 = counts_n[0] if counts_n and counts_n[0] else 1.0
    logn = [math.log(max(1.0, nn) / max(1.0, n0)) for nn in counts_n]
    # per-trophy mean logit and tier means
    tier_vals = {t: [] for t in TIER_ORDER}
    means = []
    for i in range(n_tr):
        row = [v for v in logity[i] if v is not None]
        m = mean(row) if row else 0.0
        means.append(m)
        tier_vals[tiers[i]].append(m)
    tier_mean = {}
    for t in TIER_ORDER:
        tier_mean[t] = mean(tier_vals[t]) if tier_vals[t] else mean(means)
    ordered = pava_increasing([tier_mean[t] for t in TIER_ORDER])
    tier_mean = {t: v for t, v in zip(TIER_ORDER, ordered)}
    k0 = 3.0
    alpha = []
    for i in range(n_tr):
        row = [v for v in logity[i] if v is not None]
        ni = len(row)
        alpha.append((ni * means[i] + k0 * tier_mean[tiers[i]]) / (ni + k0))
    # common time curve f(t): mean residual per read, RW2-smoothed
    obs_f = []
    for j in range(n_rd):
        vals = [logity[i][j] - alpha[i] for i in range(n_tr)
                if logity[i][j] is not None]
        obs_f.append(mean(vals) if vals else 0.0)
    f_sm = _smooth_rw2(obs_f)
    # delta: pooled regression of the remaining residual on log(N/N0)
    num = den = 0.0
    resid_all = []
    for i in range(n_tr):
        for j in range(n_rd):
            if logity[i][j] is None:
                continue
            r = logity[i][j] - alpha[i] - f_sm[j]
            resid_all.append(r)
            num += logn[j] * r
            den += logn[j] * logn[j]
    delta_hat = num / den if den > 1e-9 else 0.0
    delta = 0.5 * delta_hat  # shrunk halfway toward 0
    resid2 = []
    for i in range(n_tr):
        for j in range(n_rd):
            if logity[i][j] is None:
                continue
            resid2.append(logity[i][j] - alpha[i] - f_sm[j]
                          - delta * logn[j])
    sigma = math.sqrt(mean([r * r for r in resid2])) if resid2 else 1.0
    sigma = max(sigma, 0.15)
    # extrapolate f to the target: slope of the last two smoothed points
    if n_rd >= 2 and taus[-1] > taus[-2]:
        slope = (f_sm[-1] - f_sm[-2]) / (taus[-1] - taus[-2])
    else:
        slope = 0.0
    dt = max(0.0, horizon_tau - taus[-1])
    slope = max(-0.05, min(0.05, slope))
    f_tgt = f_sm[-1] + slope * dt
    f_tgt = max(f_sm[-1] - 1.0, min(f_sm[-1] + 1.0, f_tgt))
    # extrapolate N: log-linear on the last reads, capped at 10x
    if n_rd >= 2:
        a_n, b_n, _ = ols_line(taus, [math.log(max(1.0, nn))
                                      for nn in counts_n])
        logn_tgt = a_n + b_n * horizon_tau
    else:
        logn_tgt = math.log(max(1.0, counts_n[-1]))
    logn_tgt = min(logn_tgt, math.log(max(1.0, n0)) + math.log(10.0))
    logn_tgt = max(logn_tgt, math.log(max(1.0, counts_n[-1])))
    pred = [alpha[i] + f_tgt + delta * (logn_tgt - math.log(max(1.0, n0)))
            for i in range(n_tr)]
    return pred, sigma, {"delta": delta}


# ------------------------------------------------------- CV gate + paths

def _pinball_item(y_pct, q_pct, tau):
    yc = clamp_pct(y_pct)
    qc = clamp_pct(q_pct)
    d = logit_pct(yc) - logit_pct(qc)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def cv_gate_weight(taus, logity, counts_n, tiers, horizon_tau):
    """Expanding-window CV: does the joint model beat the curve ensemble
    on one-step-ahead forecasts by at least 10%?

    Returns 0.30 when it does, else exactly 0.0.  The ensemble side of
    the comparison uses the two closed-form curves only (the Weibull
    grid is too slow to refit at every origin); the joint side uses the
    full joint fit.  Origins are the last three cut points with at
    least four reads behind them.
    """
    n_rd = len(taus)
    if n_rd < 7:
        return 0.0
    origins = [j for j in range(n_rd - 4, n_rd - 1) if j >= 3]
    if len(origins) < 2:
        return 0.0
    loss_ens = loss_joint = cnt = 0.0
    n_tr = len(logity)
    for j in origins:
        sub_taus = taus[:j + 1]
        nxt = taus[j + 1]
        ens_q = []
        for i in range(n_tr):
            ys = [logity[i][t] for t in range(j + 1)
                  if logity[i][t] is not None]
            xs = [taus[t] for t in range(j + 1) if logity[i][t] is not None]
            if len(ys) < 3:
                last = ys[-1] if ys else 0.0
                ens_q.append([inv_logit_pct(last + z * 0.5) for z in Z])
                continue
            a1, b1, sd1 = ols_line([math.log1p(t) for t in xs], ys)
            a2, b2, sd2 = ols_line([math.log(t + 0.25) for t in xs], ys)
            lg = 0.5 * ((a1 + b1 * math.log1p(nxt))
                        + (a2 + b2 * math.log(nxt + 0.25)))
            sd = CURVE_SIGMA_SCALE * math.sqrt(0.5 * (sd1 ** 2 + sd2 ** 2)
                                               + 0.25)
            ens_q.append([inv_logit_pct(lg + z * sd) for z in Z])
        sub_logity = [[logity[i][t] for t in range(j + 1)]
                      for i in range(n_tr)]
        joint = fit_joint(sub_taus, sub_logity, counts_n[:j + 1], tiers,
                          nxt)
        if joint is None:
            return 0.0
        pred_j, sig_j, _ = joint
        for i in range(n_tr):
            y = logity[i][j + 1]
            if y is None:
                continue
            y_pct = inv_logit_pct(y)
            for k, tau in enumerate(LEVELS):
                loss_ens += _pinball_item(y_pct, ens_q[i][k], tau)
                qj = inv_logit_pct(pred_j[i] + Z[k] * sig_j)
                loss_joint += _pinball_item(y_pct, qj, tau)
            cnt += 1
    if cnt == 0:
        return 0.0
    loss_ens /= (cnt * 5.0)
    loss_joint /= (cnt * 5.0)
    if loss_joint < 0.90 * loss_ens:
        return 0.30
    return 0.0


def flat_fan_quantiles(trophy_id, rate_pct, n_last, mult):
    """Count-process fan for a flat history.

    Earners now E (rounded from the last rate and player count); a
    refresh multiplies the player count by M ~ LogNormal(ln mult, 0.35)
    and adds K ~ Poisson(lam) new earners, lam = (mult - 1) * E * PHI_NEW.
    Quantiles come from 3000 deterministic draws of (E + K) / (N * M),
    shifted in logit space so the fan's median is exactly the dilution
    median (E + lam) / (N * mult).
    """
    e = int(round(rate_pct * n_last / 100.0))
    e = max(e, 0)
    lam = (mult - 1.0) * max(e, 0.5) * PHI_NEW
    med_dil = (e + lam) / (n_last * mult) * 100.0
    off = (104729 * int(trophy_id)) % N_MC
    sd_lam = math.sqrt(lam) if lam > 0 else 0.0
    log_mult = math.log(mult)
    draws = []
    rng = random.Random(SEED_BASE + 104729 * int(trophy_id))
    for j in range(N_MC):
        z1 = _Z_TABLE[(j + off) % N_MC]
        z2 = _Z_TABLE[(7 * j + off) % N_MC]
        if lam < 5.0:
            k = poisson_draw(rng, lam)
        else:
            k = max(0, int(round(lam + sd_lam * z1)))
        m = math.exp(log_mult + S_MULT * z2)
        draws.append((e + k) / (n_last * m) * 100.0)
    draws.sort()
    mc_med = quantile_sorted(draws, 0.5)
    shift = 0.0
    if 0.01 < med_dil < 99.99 and 0.01 < mc_med < 99.99:
        shift = logit_pct(med_dil) - logit_pct(mc_med)
    out = []
    for tau in LEVELS:
        v = quantile_sorted(draws, tau)
        if shift:
            v = inv_logit_pct(logit_pct(v) + shift)
        out.append(v)
    return out


def saturated_quantiles(trophy_id, n_last, mult):
    """A trophy every tracked player already holds (rate 100%).

    New players earn it at the incumbent rate too (it is the game's
    first trophy: everybody gets it), so the fan is the same count
    process with PHI_NEW = 1 and the cap is 100, not 90.
    """
    e = n_last
    off = (104729 * int(trophy_id)) % N_MC
    log_mult = math.log(mult)
    draws = []
    for j in range(N_MC):
        # Every new player earns this trophy too, so the new-earner
        # count tracks the multiplier itself instead of varying
        # independently of it: rate = (1 + (M-1)*phi) / M.
        z1 = _Z_TABLE[(j + off) % N_MC]
        z2 = _Z_TABLE[(7 * j + off) % N_MC]
        m = math.exp(log_mult + S_MULT * z1)
        phi = min(1.0, max(0.99, 0.999 + 0.001 * z2))
        draws.append(min(100.0, (1.0 + (m - 1.0) * phi) / m * 100.0))
    draws.sort()
    return [quantile_sorted(draws, tau) for tau in LEVELS]


def quantiles_from_logit(med_logit, sigma):
    qs = [inv_logit_pct(med_logit + z * sigma) for z in Z]
    qs = [min(max(q, 0.01), CAP_PCT) for q in qs]
    qs.sort()
    return qs


# ------------------------------------------------------------- case logic

def _subsample(idx, cap=48):
    if len(idx) <= cap:
        return list(idx)
    step = (len(idx) - 1) / (cap - 1)
    picks = sorted({int(round(i * step)) for i in range(cap)})
    return picks


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case.get("history") or [],
                     key=lambda h: parse_time(h["at"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    n_tr = len(trophies)
    tiers = [t.get("tier", "bronze") for t in trophies]
    ids = [t["id"] for t in trophies]

    # last non-null rate per trophy (tier-median fallback, then tier prior)
    last_rates = []
    for i in range(n_tr):
        v = None
        for h in reversed(history):
            r = h["rates"][i] if i < len(h["rates"]) else None
            if r is not None:
                v = float(r)
                break
        last_rates.append(v)
    for i in range(n_tr):
        if last_rates[i] is None:
            same = [last_rates[j] for j in range(n_tr)
                    if tiers[j] == tiers[i] and last_rates[j] is not None]
            last_rates[i] = median(same) if same else TIER_FALLBACK[tiers[i]]

    if not history:
        # No reads at all: tier priors with a wide, honest spread.
        rows = []
        for i in range(n_tr):
            lg = logit_pct(TIER_FALLBACK[tiers[i]])
            rows.append({"id": ids[i],
                         "q": [round4(q) for q in
                               quantiles_from_logit(lg, 1.5)]})
        return {"trophies": rows}

    t0 = parse_time(history[0]["at"])
    taus_all = [hours_between(t0, parse_time(h["at"])) for h in history]
    n_all = [float(h["players_tracked"]) if h.get("players_tracked")
             else 0.0 for h in history]
    n_last = n_all[-1] if n_all[-1] > 0 else 1.0
    tau_target = hours_between(t0, parse_time(case["target"]))

    # ---- flat-history detection: nothing moved, ever
    moved = False
    for i in range(n_tr):
        seen = [float(h["rates"][i]) for h in history
                if i < len(h["rates"]) and h["rates"][i] is not None]
        if seen and (max(seen) - min(seen)) > 1e-9:
            moved = True
            break
    n_distinct = len({round(v, 6) for v in n_all})
    is_flat = (not moved) and n_distinct <= 1

    # refresh multiplier from the case's own history, else the prior
    jumps = []
    for a, b in zip(n_all, n_all[1:]):
        if a > 0 and b / a > 1.02:
            jumps.append(b / a)
    mult = median(jumps) if jumps else MULT_DEFAULT
    mult = max(1.5, min(20.0, mult))

    rows = []
    if is_flat or len(history) < 3:
        for i in range(n_tr):
            if last_rates[i] >= 99.99:
                qs = saturated_quantiles(ids[i], int(n_last), mult)
            else:
                qs = flat_fan_quantiles(ids[i], last_rates[i], int(n_last),
                                        mult)
            qs = [min(max(q, 0.01),
                      100.0 if last_rates[i] >= 99.99 else CAP_PCT)
                  for q in qs]
            qs.sort()
            rows.append({"id": ids[i], "q": [round4(q) for q in qs]})
        return {"trophies": rows}

    # ---- moving history: curve ensemble, joint model gated by CV
    picks = _subsample(list(range(len(history))))
    taus = [taus_all[j] for j in picks]
    counts_n = [n_all[j] for j in picks]
    logity = []
    for i in range(n_tr):
        row = []
        for j in picks:
            h = history[j]
            r = h["rates"][i] if i < len(h["rates"]) else None
            row.append(logit_pct(float(r)) if r is not None else None)
        logity.append(row)

    weights_cache = {}
    joint = fit_joint(taus, logity, counts_n, tiers, tau_target)
    w_joint = cv_gate_weight(taus, logity, counts_n, tiers, tau_target)
    rw_var = (RW_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2

    for i in range(n_tr):
        if last_rates[i] >= 99.99:
            qs = saturated_quantiles(ids[i], int(n_last), mult)
            rows.append({"id": ids[i], "q": [round4(q) for q in qs]})
            continue
        ys = [v for v in logity[i] if v is not None]
        xs = [taus[j] for j, v in enumerate(logity[i]) if v is not None]
        if len(ys) < 3:
            lg = logit_pct(last_rates[i])
            sigma = CURVE_SIGMA_SCALE * math.sqrt(rw_var + 0.25)
            rows.append({"id": ids[i], "q": [round4(q) for q in
                        quantiles_from_logit(lg, sigma)]})
            continue
        key = len(ys)
        if key not in weights_cache:
            weights_cache[key] = rolling_origin_weights(xs, ys)
        w = weights_cache[key]
        ens_logit, ens_sd = ensemble_predict(xs, ys, tau_target, w)
        sigma = CURVE_SIGMA_SCALE * math.sqrt(ens_sd ** 2 + rw_var)
        med_logit, sig = ens_logit, sigma
        if w_joint > 0.0 and joint is not None:
            pred_j, sig_j, _ = joint
            med_logit = (1.0 - w_joint) * ens_logit + w_joint * pred_j[i]
            sig = math.sqrt((1.0 - w_joint) ** 2 * sigma ** 2
                            + w_joint ** 2 * sig_j ** 2)
        rows.append({"id": ids[i], "q": [round4(q) for q in
                    quantiles_from_logit(med_logit, sig)]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
