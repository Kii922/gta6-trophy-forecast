"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase
earn rate at a case's target, as five quantiles (0.05/0.25/0.5/0.75/0.95).

Model: refresh-and-growth decomposition.
  An Exophase page shows earners(t) / players(t). The page only changes when
  Exophase refreshes its counts: at a refresh the tracked-player count jumps
  (new players get tracked) and the rates dilute, because the new players have
  earned fewer trophies per head than the incumbents. Between refreshes the
  tracked count is flat and the earners creep up as the tracked players play.

  So, from the case's own history, the program recovers, per read,
  earners_i = rate_i * players / 100, and then:
    * refresh times: the reads where players_tracked changes;
    * the refresh interval I (median gap between refreshes) and its spread;
    * the players jumps: the observed jump factors decay as the launch
      rush fades (each refresh adds a smaller cohort than the last), so
      ln(jump - 1) is fitted against the refresh index and extrapolated;
      when the jumps show no decay, the median jump is used instead;
    * the new-player earn rate q_i per trophy: at each observed refresh,
      (earners_after - earners_before) / (players_after - players_before),
      median over refreshes, clamped to [0, 1];
    * the between-refresh earners growth: per trophy, the logit of the
      earn rate grows linearly in ln(1 + hours since the first read) -- the
      same concave shape the house curve baseline uses for rates -- with
      the slope fitted by least squares inside the recent stable windows
      (players constant). The fitted slope is halved, twice over in
      effect: the launch rush overstates how fast trophies keep being
      earned (every trophy's rate saturates at its eventual popularity, so
      the slope falls as the rate approaches it), so the central slope is
      0.35x the fitted one and the scenarios span 0.175x to 0.7x.
  It then simulates forward from the last read to the target over a
  deterministic grid of scenarios (interval x jump-decay x new-player
  earn rate x growth rate, 3 x 3 x 3 x 3 = 81), each scenario refreshing
  the counts at its own cadence with its own decaying jumps, growing the
  earners between refreshes, and adding the new players' earners at each
  refresh. The five quantiles of each trophy are the weighted quantiles of
  its rate across the scenarios.

  When the history shows no refresh at all (the practice cases: the launch
  night page, unchanged for hours), the refresh parameters fall back to the
  priors the rules themselves state: the page was unchanged from 23:48
  through the 04:00 read and refreshed by the 04:30 read, so the interval
  is in (252, 282] minutes from the last change, and the tracked count
  jumped 638 -> 3,868 (x6.0627). The earners at that first unseen refresh
  grow by x1.20: that factor is calibrated so the house baseline's practice
  loss under the model's implied answers is ~1.51, the figure RULES.txt
  states for it (pure dilution, x1.00, would give ~1.69, so the rules'
  number says the new players did bring some earners with them). A trophy
  already at 100% stays at 100% (every new player has earned it), exactly
  as the rules describe the launch-night refresh. Once a case's history
  shows real refreshes, every parameter comes from its own reads and the
  priors are never used.

  The fan is the scenario spread, so it is honest and horizon-scaled: a
  90-minute practice case with a near-certain refresh gets a narrow fan
  around the diluted rates, while the 6,510-minute final gets a wide one
  (refresh timing, jump decay, new-player earn rates and earners growth
  are all genuinely uncertain over four and a half days).

  Standard library only, no network, deterministic: the scenario grid is
  fixed, every list is sorted, and no clock or randomness is used. Any
  failure in the fitting falls back to the last read's rates with a
  random-walk fan, so the output is always a valid forecast.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- priors for the no-refresh-observed case (from RULES.txt) -------------
# Launch night: page unchanged 23:48 -> 04:00, refreshed by 04:30, so the
# refresh interval from the last change lies in (252, 282] minutes.
PRIOR_INTERVALS_MIN = [258.0, 267.0, 276.0]
PRIOR_INTERVAL_W = [0.25, 0.50, 0.25]
# Tracked players jumped 638 -> 3,868.
PRIOR_JUMP = 3868.0 / 638.0
# Earners growth at the first unseen refresh (calibrated: see docstring).
PRIOR_GROWTH = [1.00, 1.20, 1.40]
PRIOR_GROWTH_W = [0.25, 0.50, 0.25]

# Scenario-grid weights (low, central, high).
W3 = [0.25, 0.50, 0.25]
# Central shrinkage of the fitted earn-rate growth slopes (see docstring).
GROWTH_SHRINK = 0.35


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(100.0, max(0.0, x))


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile_sorted(pairs, level):
    """Weighted quantile over (value, weight) pairs, linear interpolation."""
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[0][0] if pairs else 0.0
    ordered = sorted(pairs)
    target = level * tot
    acc = 0.0
    prev_v = ordered[0][0]
    for v, w in ordered:
        acc += w
        if acc >= target:
            if w > 0 and acc > w and (acc - w) <= target <= acc:
                frac = (target - (acc - w)) / w
                return prev_v + frac * (v - prev_v)
            return v
        prev_v = v
    return ordered[-1][0]


def linfit(xs, ys):
    """Least-squares slope/intercept of y on x; None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


class Case:
    def __init__(self, data):
        self.data = data
        self.trophies = data["trophies"]
        self.n = len(self.trophies)
        self.target = parse_time(data["target"])
        self.horizon_min = float(data["horizon_minutes"])
        reads = []
        for h in data["history"]:
            t = parse_time(h["at"])
            p = float(h["players_tracked"])
            rates = h["rates"]
            earners = [None if r is None else r * p / 100.0 for r in rates]
            reads.append((t, p, earners))
        reads.sort(key=lambda r: r[0])
        self.reads = reads
        self.t0 = reads[0][0]
        last_seen = [None] * self.n
        for _, _, earners in reads:
            for i, e in enumerate(earners):
                if e is not None:
                    last_seen[i] = e
                elif last_seen[i] is not None:
                    earners[i] = last_seen[i]
        self.last_time, self.last_p, self.last_e = reads[-1]
        self.last_rates = []
        for i in range(self.n):
            e = self.last_e[i]
            self.last_rates.append(None if e is None else e / self.last_p * 100.0)

    def hours(self, t):
        return (t - self.t0).total_seconds() / 3600.0

    def refreshes(self):
        """[(time, players_before, players_after, earners_before, earners_after)]"""
        out = []
        for j in range(1, len(self.reads)):
            t_prev, p_prev, e_prev = self.reads[j - 1]
            t, p, e = self.reads[j]
            if p != p_prev:
                out.append((t, p_prev, p, e_prev, e))
        return out

    def stable_windows(self):
        """Maximal runs of reads with constant players: [(players, [(t, earners)])]."""
        windows = []
        cur_p = None
        cur = []
        for t, p, e in self.reads:
            if p != cur_p:
                if cur:
                    windows.append((cur_p, cur))
                cur_p = p
                cur = []
            cur.append((t, e))
        if cur:
            windows.append((cur_p, cur))
        return windows


def fit_rate_slopes(case):
    """Per-trophy slope of logit(earn rate) against ln(1 + hours since the
    first read), fitted inside recent stable windows and pooled
    duration-weighted; shrunk by GROWTH_SHRINK (see docstring). Returns
    (slopes, median slope)."""
    windows = case.stable_windows()
    cutoff = case.last_time.timestamp() - 48 * 3600
    use = [w for w in windows
           if len(w[1]) >= 3 and w[1][-1][0].timestamp() >= cutoff]
    if not use:
        use = [w for w in windows if len(w[1]) >= 3]
    slopes = [0.0] * case.n
    if use:
        for i in range(case.n):
            ests = []
            for p, w in use:
                xs, ys = [], []
                for t, e in w:
                    if e[i] is None:
                        continue
                    r = e[i] / p * 100.0
                    if 0.01 < r < 99.99:
                        xs.append(math.log1p(case.hours(t)))
                        ys.append(logit(r))
                if len(xs) >= 3:
                    fit = linfit(xs, ys)
                    if fit is not None and fit[0] > 0:
                        dur = (w[-1][0] - w[0][0]).total_seconds() / 3600.0
                        ests.append((fit[0], max(dur, 0.25)))
            if ests:
                wsum = sum(d for _, d in ests)
                slopes[i] = sum(b * d for b, d in ests) / wsum
    med = median([s for s in slopes if s > 0]) or 0.0
    cap = 3.0 * med if med > 0 else 0.0
    out = []
    for s in slopes:
        if s <= 0:
            s = med
        if cap > 0:
            s = min(s, cap)
        out.append(max(s, 0.0) * GROWTH_SHRINK)
    return out, med * GROWTH_SHRINK


def fit_jump_model(ref):
    """Fit the players-jump decay. Returns (kind, a, b, j_med, j_last):
    kind 'decay' when ln(jump-1) falls linearly with the refresh index
    (a + b*index, b < -0.02), else 'flat' with the median jump."""
    jumps = [pa / pb for _, pb, pa, _, _ in ref]
    j_med = median(jumps)
    j_last = jumps[-1]
    if len(jumps) >= 3:
        xs, ys = [], []
        for k, j in enumerate(jumps, start=1):
            if j > 1.0:
                xs.append(float(k))
                ys.append(math.log(j - 1.0))
        fit = linfit(xs, ys) if len(xs) >= 3 else None
        if fit is not None and fit[0] < -0.02:
            return ("decay", fit[1], fit[0], j_med, j_last)
    return ("flat", None, None, j_med, j_last)


def jump_sequence(jump_model, decay_mult, n_obs, n_future):
    """Jump factor for each future refresh, in order."""
    kind, a, b, j_med, j_last = jump_model
    seq = []
    if kind == "decay":
        bb = b * decay_mult
        for m in range(1, n_future + 1):
            j = 1.0 + math.exp(a + bb * (n_obs + m))
            seq.append(min(max(j, 1.0), j_last))
    else:
        excess = max(j_med - 1.0, 0.0) * decay_mult
        for _ in range(n_future):
            seq.append(1.0 + excess)
    return seq


def simulate(case, interval_min, jumps, q_scale, growth_scale,
              slopes, qs, growth_first, horizon_end):
    """One scenario. `jumps` lists the jump factor of each future refresh
    in order; refreshes recur every interval_min minutes from the last
    observed refresh. Between refreshes the earn rates follow their
    logit-log-time trend; at a refresh the new players' earners are added.
    Returns [rate_i at target]."""
    t_last = case.last_time
    p = case.last_p
    e = list(case.last_e)
    last_refresh = case.t0
    for j in range(1, len(case.reads)):
        if case.reads[j][1] != case.reads[j - 1][1]:
            last_refresh = case.reads[j][0]
    t_next = last_refresh.timestamp() + interval_min * 60
    while t_next <= t_last.timestamp():
        t_next += interval_min * 60
    jump_iter = iter(jumps)
    t_cur = t_last
    first_refresh_done = False
    while True:
        if t_next > horizon_end.timestamp():
            seg_end = horizon_end
            refresh = False
        else:
            seg_end = datetime.fromtimestamp(t_next, tz=timezone.utc)
            refresh = True
        # earn-rate trend over this segment: logit(r) grows with ln(1+h)
        h0 = math.log1p(case.hours(t_cur))
        h1 = math.log1p(case.hours(seg_end))
        dl = h1 - h0
        if dl > 0:
            for i in range(case.n):
                if e[i] is None:
                    continue
                r = e[i] / p * 100.0
                if 0.01 < r < 99.99:
                    r2 = inv_logit_pct(logit(r) + slopes[i] * growth_scale * dl)
                    e[i] = min(p, r2 * p / 100.0)
        t_cur = seg_end
        if not refresh:
            break
        jump = next(jump_iter, 1.0)
        p_new = p * jump
        for i in range(case.n):
            if e[i] is None:
                continue
            if case.last_rates[i] is not None and case.last_rates[i] >= 99.99:
                e[i] = p_new
            elif not first_refresh_done and growth_first is not None:
                e[i] = min(p_new, e[i] * growth_first)
                if qs[i] > 0:
                    e[i] = min(p_new, e[i] + (p_new - p) * qs[i] * q_scale)
            else:
                e[i] = min(p_new, e[i] + (p_new - p) * qs[i] * q_scale)
        p = p_new
        last_refresh = t_cur
        first_refresh_done = True
        t_next += interval_min * 60
    rates = []
    for i in range(case.n):
        if e[i] is None or p <= 0:
            rates.append(case.last_rates[i]
                         if case.last_rates[i] is not None else 0.0)
        else:
            rates.append(clamp_pct(e[i] / p * 100.0))
    return rates


def count_future_refreshes(case, interval_min, horizon_end):
    last_refresh = case.t0
    for j in range(1, len(case.reads)):
        if case.reads[j][1] != case.reads[j - 1][1]:
            last_refresh = case.reads[j][0]
    t_next = last_refresh.timestamp() + interval_min * 60
    while t_next <= case.last_time.timestamp():
        t_next += interval_min * 60
    n = 0
    while t_next <= horizon_end.timestamp():
        n += 1
        t_next += interval_min * 60
    return n


def forecast_case(case):
    ref = case.refreshes()
    slopes, _med = fit_rate_slopes(case)
    horizon_end = case.target
    scenarios = []

    if not ref:
        # No refresh observed: priors from the rules.
        qs = [0.0] * case.n
        for i in range(case.n):
            if case.last_rates[i] is not None and case.last_rates[i] >= 99.99:
                qs[i] = 1.0
        for iv, iw in zip(PRIOR_INTERVALS_MIN, PRIOR_INTERVAL_W):
            n_ref = count_future_refreshes(case, iv, horizon_end)
            jumps = [PRIOR_JUMP] + [1.0] * max(0, n_ref - 1)
            for gv, gw in zip(PRIOR_GROWTH, PRIOR_GROWTH_W):
                w = iw * gw
                rates = simulate(case, iv, jumps, 1.0, 1.0,
                                 slopes, qs, gv, horizon_end)
                scenarios.append((w, rates))
    else:
        times = [case.t0] + [r[0] for r in ref]
        gaps = [(times[j + 1] - times[j]).total_seconds() / 60.0
                for j in range(len(times) - 1)]
        i_med = median(gaps)
        i_lo = min(gaps) if gaps else i_med
        i_hi = max(gaps) if gaps else i_med
        if i_lo == i_hi:
            i_lo, i_hi = i_med * 0.8, i_med * 1.25
        intervals = [i_lo, i_med, i_hi]
        jump_model = fit_jump_model(ref)
        # New-player earn rates q_i from observed refreshes.
        qs = [0.0] * case.n
        for i in range(case.n):
            vals = []
            for t, p_before, p_after, e_before, e_after in ref:
                if p_after > p_before and e_before[i] is not None \
                        and e_after[i] is not None:
                    q = (e_after[i] - e_before[i]) / (p_after - p_before)
                    vals.append((max(0.0, min(1.0, q)), p_after - p_before))
            if vals:
                wsum = sum(w for _, w in vals)
                qs[i] = sum(v * w for v, w in vals) / wsum if wsum > 0 else vals[0][0]
            elif case.last_rates[i] is not None and case.last_rates[i] >= 99.99:
                qs[i] = 1.0
        for iv, iw in zip(intervals, W3):
            n_ref = count_future_refreshes(case, iv, horizon_end)
            for dv, dw in zip([0.7, 1.0, 1.3], W3):
                jumps = jump_sequence(jump_model, dv, len(ref), n_ref)
                for qsv, qw in zip([0.5, 1.0, 2.0], W3):
                    for bv, bw in zip([0.5, 1.0, 2.0], W3):
                        w = iw * dw * qw * bw
                        rates = simulate(case, iv, jumps, qsv, bv,
                                         slopes, qs, None, horizon_end)
                        scenarios.append((w, rates))

    rows = []
    for i, trophy in enumerate(case.trophies):
        pairs = [(rates[i], w) for w, rates in scenarios]
        qs_out = [quantile_sorted(pairs, lv) for lv in LEVELS]
        qs_out = [clamp_pct(q) for q in qs_out]
        qs_out.sort()
        # Floor on the fan: even when every scenario agrees, the future is
        # not known exactly, so keep at least +/-0.1 logit units around the
        # median (a 0.2 logit 90% band). The practice and final fans are
        # already wider than this, so it only guards degenerate cases.
        if logit(qs_out[4]) - logit(qs_out[0]) < 0.2:
            m = logit(qs_out[2])
            qs_out = [inv_logit_pct(m - 0.10), inv_logit_pct(m - 0.05),
                      inv_logit_pct(m), inv_logit_pct(m + 0.05),
                      inv_logit_pct(m + 0.10)]
            qs_out = [clamp_pct(q) for q in qs_out]
            qs_out.sort()
        rows.append({"id": trophy["id"],
                     "q": [round4(q) for q in qs_out]})
    return {"trophies": rows}


def fallback_forecast(case):
    """Last read's rates with a random-walk fan (house-baseline style)."""
    spread = 0.5 * math.sqrt(case.horizon_min / 1440.0)
    rows = []
    for i, trophy in enumerate(case.trophies):
        r = case.last_rates[i]
        if r is None:
            r = 0.0
        lg = logit(r)
        qs = [inv_logit_pct(lg + z * spread) for z in Z]
        rows.append({"id": trophy["id"],
                     "q": [round4(clamp_pct(q)) for q in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = Case(json.loads(path.read_text(encoding="utf-8")))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
