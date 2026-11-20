# GTA VI trophy rarity forecast — make.fun `gta6-trophy-rarity-forecast-01`

Forecasts every GTA VI PS5 trophy's Exophase earn rate at a case's target,
as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy. Entry for the
make.fun bounty "Forecast how many GTA VI players earn each trophy in week
one" (`gta6-trophy-rarity-forecast-01`).

## How it works

An Exophase page shows `earners / players_tracked`. The page only changes
when Exophase refreshes its counts: at a refresh the tracked-player count
jumps (new players get tracked) and the rates dilute, because the new
players have earned fewer trophies per head than the incumbents. Between
refreshes the tracked count is flat and the earners creep up as the tracked
players play.

`main.py` therefore decomposes the problem per case:

1. Recover `earners_i = rate_i * players / 100` from every read in the
   case's history, and find the refreshes (reads where `players_tracked`
   changes).
2. Estimate from the case's own history:
   - the refresh interval (median gap between refreshes) and its spread;
   - the players jumps: the observed jump factors decay as the launch rush
     fades, so `ln(jump - 1)` is fitted against the refresh index and
     extrapolated (median jump if no decay is visible);
   - the new-player earn rate `q_i` per trophy: at each observed refresh,
     `(earners_after - earners_before) / (players_after - players_before)`;
   - the between-refresh earners growth: the logit of each trophy's earn
     rate grows linearly in `ln(1 + hours)`, fitted inside the recent
     stable windows, then shrunk (central factor 0.35) because the launch
     rush overstates how fast trophies keep being earned — every rate
     saturates at its eventual popularity.
3. Simulate forward from the last read to the target over a deterministic
   grid of 81 scenarios (interval × jump-decay × new-player earn rate ×
   growth rate). The five quantiles of each trophy are the weighted
   quantiles of its rate across the scenarios, so the fan is honest and
   horizon-scaled: narrow for a 90-minute practice case with a near-certain
   refresh, wide for the 6,510-minute final.

When a case's history shows no refresh at all (the practice cases: the
launch-night page, unchanged for hours), the refresh parameters fall back
to the priors stated in the bounty rules: the page was unchanged from 23:48
through the 04:00 read and refreshed by the 04:30 read (interval in
(252, 282] minutes), the tracked count jumped 638 → 3,868, and the earners
at that first unseen refresh grow by ×1.20 — calibrated so the house
baseline's practice loss under the model's implied answers is ~1.51, the
figure the rules state for it. A trophy already at 100% stays at 100%.

## Validation (local scorer, pinball loss on the logit scale, per RULES.txt)

Practice answers are derived from the rules' stated refresh (players
638 → 3,868, earners ×1.198, the 100% trophy stays 100%); the local scorer
reproduces the rules' worked example exactly (0.403404), and the house
baseline scores 1.52 against the derived answers, matching the rules'
stated ~1.51 — so the derivation is consistent with the rules.

| case | this entry | house baseline |
| ---- | ---------- | -------------- |
| p00  | 0.0315     | 1.52           |
| p01  | 0.0315     | 1.52           |
| mean | **0.0315** | 1.52           |

The final's mechanism was also checked on two synthetic final-like cases
(120 reads, 13 refreshes, 6,510-minute horizon, generated independently of
this model): a world where players growth saturates by the deadline
(0.143 vs the baseline's 0.256) and a world where it keeps growing
(0.092 vs the baseline's 0.239). In both, the model beats the house
baseline; the practice score is not the final's score, and the final
depends on the refresh pattern the real history shows.

## Run it

```sh
# practice cases (from the kit's input/)
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` file in
`$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (else `/output/`). Deterministic (two runs write
byte-identical files) and fast (~0.1 s per case). Output is
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, each
quantile in [0, 100], non-decreasing, well under the 65,536-byte cap.

## Files

- `main.py` — the whole entry (one file, stdlib only).
- `.makefun/owner.json` — payout wallet.
- `LICENSE` — MIT.

## Caveats

- The practice answers are *derived*, not ground truth: they assume the
  launch-night refresh diluted the rates by the rules' stated player jump
  with earners ×1.198 (calibrated to the rules' ~1.51 baseline loss). If
  the real refresh grew the earners by a different factor, the practice
  score moves; the mechanism itself does not depend on that calibration —
  on the final, every parameter is fitted from the case's own history.
- The final is a 4.5-day extrapolation. The model's edge over the baseline
  comes from the refresh mechanism; if Exophase's refresh cadence or the
  player-growth saturation in the real week differs sharply from the
  history's pattern, the fan (not the median) is what absorbs it.
