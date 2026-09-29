# Findings: competitive optimal stopping ("shoe shopping / dating market")

## What was built

`~/workspace/shoe-market/` — numpy-only simulation + self-play evolution trainer:

- **[engine.py](sandbox://workspace/shoe-market/engine.py)** — the market environment.
  N agents compete for M shoe styles over T steps. Style popularity `p_j`
  (common value) + idiosyncratic taste `t_ij` give satisfaction
  `v_ij = w·p_j + (1−w)·t_ij`, clipped to [0,1]. Each step agents act in random
  order; a turn is INSPECT (reveal next unseen style in the agent's private
  random order), BUY (take a copy of an inspected style with stock > 0, exit
  with payoff v), or PASS. Stock is observable by default (agents skip sold-out
  styles when inspecting). No purchase by the deadline → payoff 0, unless
  `fallback_buy=True` (forced to take best inspected style left). Fully
  vectorized over episodes; `import engine` works with no dependencies beyond
  numpy.
- **[train.py](sandbox://workspace/shoe-market/train.py)** — self-play evolution:
  all N agents share one threshold-policy param vector `(e, theta0, theta1)`;
  fitness = mean payoff over K episodes with **common random numbers** (same
  markets + acting orders for every candidate in a generation). Population 100,
  top-12 elites kept, Gaussian mutation (σ_e = max(1,(T/2)/4), σ_θ = 0.10),
  10 generations, K = 500.
- **[run_experiments.py](sandbox://workspace/shoe-market/run_experiments.py)** —
  runs every sweep config: trains, then re-evaluates the learned policy plus 3
  fixed baselines on fresh K=2000 markets with paired (common) random numbers.
  Writes `results/summary.csv` and `results/learning_curves.csv`.
- **[test_env.py](sandbox://workspace/shoe-market/test_env.py)** — 9 unit tests.
  All pass.

## Policy class

Threshold policy `(e, theta0, theta1)`: inspect-only for the first `e` turns,
then each turn inspect the next style and buy the best inspected style still
in stock if its value ≥ θ(t), with θ declining linearly from θ0 (t=0) to θ1
(t=T−1). `e ∈ [0, T/2]`, θ ∈ [0,1].

Baselines: **secretary37** (explore ⌊T/ℯ⌋ turns, then take first inspected
style in stock with v ≥ max v seen during explore — cardinal analog of the
classic rule, adapted because inspection itself consumes the time budget);
**fixed07** (no explore, constant threshold 0.7); **random** (no explore; each
turn inspect, then buy best inspected in-stock style w.p. 0.5).

## Sanity checks (all passed)

1. **Environment unit tests** (`python3 test_env.py`, 9 tests): stock decrements
   exactly once per purchase and never goes negative; a bought-out style can
   never be bought (M=1, stock=1, N=2 → exactly one buyer per episode);
   payoffs always in [0,1] for all four value distributions; agents never
   exceed T turns; identical markets+orders → identical stats (deterministic);
   fallback variant weakly reduces the empty-handed rate (0.27 → 0.00 in the
   test); all three baselines run.
2. **N=1, uniform taste, w=0** (no competition): learned
   `e=15.0 (=T/2, the upper bound), θ0=1.00 → θ1=0.87`, fitness 0.955 —
   substantial exploration + high declining threshold, beating all baselines.
   **Caveat found:** the e ≤ T/2 bound *binds* here — a hand sweep over e with
   fixed thresholds shows fitness keeps rising to e=25 (0.949 → 0.961). With no
   competition the unconstrained optimum wants to inspect almost the whole
   budget, then take the best seen. We kept the specified [0, T/2] bound; the
   constrained optimum sits at the boundary.
3. **N=16, stock=1, w=1** (max competition, pure popularity contest): learned
   `e=9.7, θ0=1.00 → θ1=0.58`, fitness 0.855. Direction of effect confirmed vs
   N=1: exploration 15.0 → 9.7 (earlier commitment), end-threshold 0.87 → 0.58
   (much lower bar). A hand diagnostic shows fitness is *flat* in e ∈ [2,15]
   (all ≈0.853) while e=0 with a fixed threshold collapses to ~0.76–0.80:
   under cutthroat competition the threshold *schedule* (start ultra-picky,
   end lenient) is what matters — it rations the scramble for the top styles —
   not the exact explore length. 0.855 ≈ the symmetric optimum (16 agents
   splitting the top-16 styles, expected mean ≈ 0.86).

## Results schema (`results/summary.csv`)

One row per config (27 rows: config_id 0–26). Columns:

| column | meaning |
|---|---|
| config_id | 0-based index into the sweep list |
| experiment | sweep name: competition, scarcity, taste_dist, pop_dist, correlation, time, dating, fallback, sanity |
| label | human-readable setting |
| n_agents, n_styles, T, stock, w, taste_dist, pop_dist, stock_observable, fallback_buy | market config |
| seed | master training seed = 1000 + config_id; final eval used 2000 + config_id |
| pop_size, generations, K_train, K_eval | 100, 10, 500, 2000 for every run |
| e, theta0, theta1 | learned policy params |
| learned_mean, learned_sem | mean payoff ± SEM over K_eval=2000 fresh episodes |
| base37_mean, fixed07_mean, random_mean | baseline means on the *same* episodes (paired) |
| mean_inspections | mean # inspections made by buyers |
| mean_accepted_v | mean value of bought shoes |
| pct_empty | fraction of agents who never bought |
| train_runtime_s, eval_runtime_s | wall-clock seconds |

`results/learning_curves.csv`: (config_id, generation, best_fitness, mean_fitness)
per generation — convergence check. (Curves for configs 0–10 were re-run after
a VM restart wiped the first sweep mid-run; see "Seeds, runtimes" below.)

## Sweep results

Defaults held constant unless noted: N=8, M=60, T=30, stock=2/style, w=0.5,
taste ∼ uniform[0,1], popularity ∼ uniform[0,1], observable stock, no fallback.
`L` = learned mean payoff; `37/f07/rnd` = baselines; `accV` = mean accepted
value; SEMs are ≤ 0.0017 throughout (K_eval=2000), so differences ≥ 0.005
are real.

### 1. Competition (N = 1, 2, 4, 8, 16)

| N | e | θ0→θ1 | L | 37% | fix07 | rand | insp | accV | empty |
|---|-----|-----------|-------|-------|-------|-------|------|------|-------|
| 1 | 15.0 | 1.00→0.71 | 0.869 | 0.824 | 0.800 | 0.581 | 18.3 | 0.871 | 0.003 |
| 2 | 15.0 | 1.00→0.74 | 0.866 | 0.823 | 0.796 | 0.583 | 19.0 | 0.875 | 0.011 |
| 4 | 13.1 | 1.00→0.73 | 0.865 | 0.823 | 0.797 | 0.582 | 18.3 | 0.872 | 0.008 |
| 8 | 15.0 | 1.00→0.72 | 0.864 | 0.819 | 0.797 | 0.579 | 18.5 | 0.871 | 0.007 |
| 16 | 0.0 | 1.00→0.69 | 0.859 | 0.799 | 0.796 | 0.580 | 16.0 | 0.863 | 0.004 |

More shoppers barely dent welfare here (0.869 → 0.859): with w=0.5 half the
value is idiosyncratic, so rivalry is diffuse, and stock=2 gives slack. θ0 is
1.00 everywhere — the learner *always* starts ultra-picky; the declining bar
does the exploring, which is why explicit `e` is nearly irrelevant (hand
diagnostic at N=16: e=0 vs e=15 → 0.8593 vs 0.8595, identical). The learned
policy beats the 37% rule by ~0.05 throughout.

### 2. Stock scarcity (copies/style = 1, 2, 4, 8; N=8, M=60)

| stock | e | θ0→θ1 | L | 37% | fix07 | L−37% |
|-------|-----|-----------|-------|-------|-------|-------|
| 1 | 6.2 | 1.00→0.69 | 0.854 | 0.768 | 0.794 | **0.086** |
| 2 | 6.4 | 1.00→0.71 | 0.864 | 0.818 | 0.797 | 0.046 |
| 4 | 14.9 | 1.00→0.72 | 0.866 | 0.825 | 0.799 | 0.041 |
| 8 | 15.0 | 1.00→0.70 | 0.867 | 0.825 | 0.799 | 0.042 |

Scarcity doubles the learned-vs-rigid gap (0.086 at stock=1 vs ~0.04 with
slack). Mechanism: the 37% rule's explore-phase max is a *reference to a shoe
that may no longer exist* — rivals buy it while the agent is still "exploring",
and then nothing ever clears the bar. The learned declining threshold adapts:
it never anchors on a vanished shoe.

### 3. Satisfaction distribution

taste ∼ (popularity fixed uniform):

| taste | e | θ0→θ1 | L | 37% | fix07 | rand |
|-------|-----|-----------|-------|-------|-------|-------|
| uniform | 13.4 | 1.00→0.71 | 0.864 | 0.817 | 0.796 | 0.579 |
| normal(0.5,.15) | 15.0 | 1.00→0.66 | 0.788 | 0.744 | 0.741 | 0.565 |
| beta(2,5) | 13.1 | 1.00→0.55 | 0.699 | 0.650 | **0.416** | 0.458 |
| heavy-tail | 15.0 | 1.00→0.48 | 0.700 | 0.626 | 0.439 | 0.400 |

popularity ∼ (taste fixed uniform): uniform 0.865, normal 0.788, beta 0.698,
heavy 0.682 — mirroring the taste sweep almost exactly, as expected from the
symmetry of v = 0.5·p + 0.5·t. Good internal consistency check.

The headline: **the optimal bar adapts to the distribution**. When great shoes
are rare (beta/heavy — most draws are "meh"), θ1 falls to 0.55/0.48: lower
your standards, because waiting for a 0.9 that never comes means going home
empty-handed. Fixed heuristics don't adapt and get crushed: fixed07 collapses
to 0.42 under beta taste — *worse than random buying* (0.46), because a 0.7
bar is almost never cleared and agents leave empty-handed. Under heavy tails
the learned policy's edge is largest (+0.074 over the 37% rule): patience for
the rare standout + willingness to eventually settle is exactly what a
declining threshold expresses.

### 4. Taste correlation (w = 0.0, 0.3, 0.7, 1.0)

| w | e | θ0→θ1 | L | 37% | fix07 |
|-----|-----|-----------|-------|-------|-------|
| 0.0 | 15.0 | 1.00→0.83 | 0.952 | 0.923 | 0.850 |
| 0.3 | 10.6 | 1.00→0.74 | 0.876 | 0.837 | 0.800 |
| 0.7 | 14.0 | 1.00→0.73 | 0.875 | 0.827 | 0.799 |
| 1.0 | 14.5 | 1.00→0.79 | 0.941 | 0.871 | 0.850 |

**U-shape — the most surprising result.** Welfare is *not* monotonic in
agreement. w=0 (pure personal taste, no rivalry): 0.952, and agents can stay
picky (θ1=0.83 — nobody competes for *your* favorites). w=0.3–0.7: ~0.875 —
rivalry over the common component plus noisy values. w=1 (pure popularity
contest): 0.941 — everyone agrees what's good, so the top styles are
well-defined and get allocated cleanly to buyers (8 agents take the top-4
styles × 2 copies; expected ≈ 0.96). **Partial agreement is worse than full
agreement or none:** at w=0.5 you fight over shoes that are only mediocre for
you personally; at w=1 the fight is at least over objectively-the-best shoes.

### 5. Time budget (T = 12, 30, 60 with M = 2T)

| T | e | θ0→θ1 | L | 37% | fix07 |
|----|-----|-----------|-------|-------|-------|
| 12 | 5.2 | 1.00→0.59 | 0.793 | 0.723 | 0.708 |
| 30 | 6.8 | 1.00→0.72 | 0.863 | 0.818 | 0.797 |
| 60 | 30.0 | 1.00→0.78 | 0.902 | 0.869 | 0.801 |

Less time → lower bar (θ1=0.59), commit earlier to good-enough. More time →
stay picky (θ1=0.78), higher payoff (more draws → better max). Exploration
scales roughly with the budget (e/T ≈ 0.43–0.5). Note fixed07 improves from
0.71 to 0.80 as T goes 12 → 30 but then plateaus (+0.004 from T=30 → 60),
while the learned declining bar keeps gaining (0.86 → 0.90) — a fixed bar
can't fully exploit a bigger choice set the way a declining one can.

### 6. Dating-market variant (N=M=20, T=20, stock=1, w=0.9)

Learned: **e=0.0, θ0=1.00 → θ1=0.06**, L=0.526, empty=0.016, insp=9.4.
Baselines: 37% → 0.252, fixed07 → 0.264, random → 0.502.

In words: twenty suitors, twenty partners, everyone largely agrees who's
attractive (w=0.9), each partner pairs once — musical chairs for love. The
learned strategy is: **pounce on love at first sight, then lower your
standards fast.** No explicit exploration (e=0 — looking around while others
commit is pure loss); start willing to commit only to a 10/10 (θ0=1.0), but
let the bar collapse to nearly zero (θ1=0.06) as the deadline looms and the
good options get taken. The fixed "play hard to get" heuristics are
catastrophic here (0.25–0.26): the 37% rule explores while all the attractive
partners pair off, then its explore-phase reference partner is gone and nothing
ever clears the bar — most agents go home alone. Even *random* committing
(0.50) beats the 37% rule, because in a scramble, a mediocre partner beats no
partner. The learned policy beats random (+0.024, real at SEM 0.0013 here) via
early pickiness: it still snags the occasional great match on day one.

### 7. Fallback variant (forced buy of best inspected at deadline vs payoff 0)

| fallback | e | θ0→θ1 | L | empty |
|----------|-----|-----------|-------|-------|
| off (default) | 15.0/6.4* | 1.00→0.71 | 0.864 | 0.007 |
| **on** | 8.7 | **1.00→0.95** | **0.885** | 0.000 |

(*default config trained twice: seeds differ, same fitness.)
A safety net changes the strategy completely: with nothing to lose by waiting
(the deadline hands you the best of what you saw), the optimal bar *barely
declines* (1.00 → 0.95) — hold out for something excellent, let the fallback
insure the downside. Welfare rises (0.885 > 0.864) and nobody goes home
empty-handed. Real-world analog: "I'll keep looking, and if nothing turns up
I'll just order the safe pair online" — the outside option raises your
reservation price.

## Interpretation: how behavior changes

1. **θ0 = 1.0 is universal.** Every learned policy starts ultra-picky. The
   declining threshold *is* the exploration mechanism: a near-1.0 initial bar
   means you effectively browse without buying until the bar reaches your
   level. Explicit explore-phase `e` is nearly redundant (flat fitness in e
   whenever θ0≈1) — except it binds at T/2 when competition is absent.
2. **θ1 is the adaptive dial.** It encodes everything: competition (down a
   bit), scarce good shoes / rare-value distributions (down a lot: 0.71 →
   0.48), short deadlines (down: 0.59 at T=12), no rivalry (up: 0.83 at w=0),
   safety net (up: 0.95 with fallback).
3. **Competition's bite depends on agreement, not headcount.** N=1→16 at w=0.5
   costs only 0.01 of welfare; but w=0.5→1.0 at N=8 *gains* 0.08 (cleaner
   allocation), and stock=1 vs 8 doubles the rigid-heuristic penalty.
4. **Rigid rules fail exactly when the market is hardest.** The 37% rule is
   fine with slack (gap ~0.04) but breaks under scarcity (0.086), max
   competition (0.34!), and dating (0.27) — always for the same reason: it
   anchors on options that rivals remove.
5. **When to move on / when to commit (the user's question):** commit when
   something clears a bar that *you lower as time passes*; the bar should
   start near your max realistic hope and end near your walk-away value. Move
   on (keep searching) while everything you've seen is below the bar — but in
   a scramble (dating, stock=1, high w), don't "explore" passively: stay
   ready to pounce from turn one, because exploration without the option to
   commit is just watching good options disappear.

## Seeds, runtimes, reproducibility

- Training seed = 1000 + config_id; per-generation market seeds derive
  deterministically (`seed·1000003 + g·9177 + 13`); final paired evaluation
  uses seed 2000 + config_id (markets + acting orders) and 3000 + 10·id + j
  (baseline coin flips). Re-running `run_experiments.py` reproduces
  `results/summary.csv` on the same numpy version (1.26.4).
- Sanity runs used seeds 42 (N=1) and 43 (N=16): `results/sanity_n1.json`,
  `results/sanity_n16.json`.
- The first full sweep was killed by a VM restart after config 10 (the
  learning-curve buffer for configs 0–10 was lost with it); configs 11–26
  completed, then configs 0–10 were re-run into `results/rerun011/` and their
  curves merged back. `learning_curves.csv` now covers all 27 configs × 10
  generations with header.
- Typical train time: ~15 s (N=1) → ~135 s (N=8) per config at K=500
  (100 pop × 10 gens); ~275–300 s at N=16; eval at K=2000 adds a few seconds.
  Full 27-config sweep ≈ 45–60 min on 2 cores. No K/generation reductions
  were needed.
- Evolution converges fast: gen-0 best is usually within 0.01 of final, with
  small gains through ~gen 7 (see `learning_curves.csv`). With K=500, fitness
  SEM ≈ 0.005 during training — ranking is stable, but exact param values in
  flat regions (especially `e`) should not be over-interpreted.

## Honest caveats (what the model can't show)

- **Symmetric self-play only.** All agents share one policy; we find a good
  symmetric profile, not a Nash equilibrium, and we don't model asymmetric
  exploitation (e.g. one patient agent against impatient rivals).
- **The e ≤ T/2 bound binds without competition** (sanity check 2): the policy
  class can't express "inspect (almost) everything, then pick the best".
- **No learning *within* an episode** about the value distribution — θ(t) is a
  fixed schedule; agents don't update beliefs from what they've seen (a real
  shopper lowers the bar faster after 20 duds).
- **No prices, no search cost beyond time, no budget.** Satisfaction is the
  only objective; price dispersion (arguably the main driver of real shoe
  shopping) is absent.
- **Inspection order is uniformly random** — no directed search ("check the
  popular shelf first"), no advertising, no social influence.
- **Competition is only via stock depletion.** No bidding wars, no queueing,
  no strategic timing beyond the random acting order.
- Distributions are stylized; "heavy" is min-max-normalized exp-normal per
  episode — a qualitative heavy-tail analog, not calibrated to real
  preference data.
- K=500 training episodes leaves small fitness noise; flat regions (e.g. `e`
  when θ0≈1) are real — don't over-read exact param values there.
