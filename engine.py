"""
engine.py -- Competitive optimal-stopping market ("shoe shopping / dating").

N agents compete for M shoe styles over T time steps.  Style j has a
popularity p_j (common-value component) and a stock s_j (copies available).
Agent i's true satisfaction for style j is

    v_ij = w * p_j + (1 - w) * t_ij,   clipped to [0, 1],

where t_ij is idiosyncratic taste.  w in [0, 1] controls taste correlation
across agents:
    w = 1  -> every agent agrees on every style (pure popularity contest),
    w = 0  -> purely personal taste.

Episode protocol
----------------
For t = 0 .. T-1, agents act one at a time in a fresh random order each step.
On its turn an agent may

  * INSPECT: reveal its own v for its next unseen style.  Each agent inspects
    styles in its own private random order, i.e. sampling without
    replacement.  Costs the turn (1 time step).
  * BUY: take one copy of any previously inspected style that still has
    stock > 0.  The agent then exits with payoff = v of the bought shoe.
  * PASS: do nothing, keep searching.

A purchase decrements that style's stock; at 0 the style is gone for everyone.
Remaining stock per style is observable to all agents by default
(cfg.stock_observable=True); in that case agents never waste an inspection on
a style that is already sold out (they skip to the next style in their order).

If time runs out with no purchase the payoff is 0 ("go home empty-handed"),
unless cfg.fallback_buy=True, in which case at the deadline each remaining
agent is forced to take its best inspected style that still has stock
(in agent order, sequentially, so the last copy goes to whoever claims it
first).

Policies
--------
* Threshold policy, params (e, theta0, theta1): inspect-only for the first
  e turns ("explore"), then on each turn inspect the next style and buy the
  best inspected style still in stock if its v >= theta(t), where theta(t)
  declines linearly from theta0 at t=0 to theta1 at t=T-1.
  e in [0, T/2], theta0/theta1 in [0, 1].
* Baselines: "secretary37" (37%-rule analog), "fixed07" (threshold 0.7,
  no explore), "random" (coin-flip buy).  See make_baseline() for exact
  definitions.

Everything is vectorized over episodes with numpy (no torch), so a
Streamlit app can `import engine` directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Value distributions
# ---------------------------------------------------------------------------
# Taste and popularity draws support four distributions:
#   "uniform" : U[0, 1]
#   "normal"  : N(0.5, 0.15), clipped to [0, 1]
#   "beta"    : Beta(2, 5), skewed low (most shoes are "meh", few are great)
#   "heavy"   : exp(N(0, 1)) then min-max normalized to [0, 1] *per episode*
#               (over styles).  Heavy right tail: almost everything is near 0
#               with a rare standout near 1.
DISTRIBUTIONS = ("uniform", "normal", "beta", "heavy")


def draw_dist(dist: str, shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw values in [0, 1] from the named distribution.

    The last axis is the "styles" axis; "heavy" is min-max normalized per
    slice along that axis (i.e. per episode, resp. per episode x agent).
    """
    if dist == "uniform":
        return rng.random(shape)
    if dist == "normal":
        return np.clip(rng.normal(0.5, 0.15, size=shape), 0.0, 1.0)
    if dist == "beta":
        return rng.beta(2.0, 5.0, size=shape)
    if dist == "heavy":
        x = np.exp(rng.normal(0.0, 1.0, size=shape))
        lo = x.min(axis=-1, keepdims=True)
        hi = x.max(axis=-1, keepdims=True)
        span = np.maximum(hi - lo, 1e-12)
        return (x - lo) / span
    raise ValueError(f"unknown distribution {dist!r}; choose from {DISTRIBUTIONS}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MarketConfig:
    n_agents: int = 8
    n_styles: int = 60
    T: int = 30
    stock: object = 2            # int copies per style, or array-like of len M
    w: float = 0.5               # taste correlation in [0, 1]
    taste_dist: str = "uniform"
    pop_dist: str = "uniform"
    stock_observable: bool = True
    fallback_buy: bool = False   # force a buy of best inspected at deadline

    def __post_init__(self):
        assert self.n_agents >= 1 and self.n_styles >= 1 and self.T >= 1
        assert 0.0 <= self.w <= 1.0
        assert self.taste_dist in DISTRIBUTIONS and self.pop_dist in DISTRIBUTIONS


def sample_markets(cfg: MarketConfig, K: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw K independent market episodes.

    Returns dict with:
      v    (K, N, M) float in [0,1]  -- agent x style satisfaction
      stock(K, M)     int            -- copies available per style
      perm (K, N, M)  int            -- each agent's private inspection order
    """
    N, M = cfg.n_agents, cfg.n_styles
    p = draw_dist(cfg.pop_dist, (K, M), rng)          # popularity
    t = draw_dist(cfg.taste_dist, (K, N, M), rng)     # idiosyncratic taste
    v = np.clip(cfg.w * p[:, None, :] + (1.0 - cfg.w) * t, 0.0, 1.0)
    s = np.asarray(cfg.stock)
    if s.ndim == 0:
        stock = np.full((K, M), int(s), dtype=np.int64)
    else:
        assert s.shape == (M,), f"stock array must have shape ({M},)"
        stock = np.broadcast_to(s.astype(np.int64), (K, M)).copy()
    perm = np.argsort(rng.random((K, N, M)), axis=2)
    return {"v": v, "stock": stock, "perm": perm}


def sample_orders(cfg: MarketConfig, K: int, rng: np.random.Generator):
    """Pre-draw the per-step acting orders: list of T arrays of shape (K, N).

    Passing fixed orders to run_episodes() implements common random numbers
    across candidate policies (less noise when comparing them).
    """
    return [np.argsort(rng.random((K, cfg.n_agents)), axis=1) for _ in range(cfg.T)]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def threshold_policy(e: float, theta0: float, theta1: float) -> Dict:
    return {"kind": "threshold", "e": float(e),
            "theta0": float(theta0), "theta1": float(theta1)}


def make_baseline(kind: str, cfg: Optional[MarketConfig] = None) -> Dict:
    """Fixed heuristic baselines.

    * "secretary37": explore floor(T/ℯ) ~= 37% of the time budget
      inspecting only, then buy the first inspected style still in stock with
      v >= max v seen during the explore phase.  This is the cardinal-value
      analog of the classic secretary problem (which is ordinal and assumes
      all candidates are seen; here inspection itself consumes the budget).
    * "fixed07": no explore, buy the best inspected style in stock with
      v >= 0.7 (constant threshold).
    * "random": no explore; each turn inspect, then with probability 0.5 buy
      the best inspected style still in stock.
    """
    if kind == "secretary37":
        if cfg is None:
            raise ValueError("secretary37 needs cfg (to compute floor(T/e))")
        return {"kind": "secretary", "e": float(max(1, int(math.floor(cfg.T / math.e))))}
    if kind == "fixed07":
        return threshold_policy(0.0, 0.7, 0.7)
    if kind == "random":
        return {"kind": "random", "p_buy": 0.5}
    raise ValueError(f"unknown baseline {kind!r}")


def random_params(cfg: MarketConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n random threshold-policy param vectors: columns (e, theta0, theta1)."""
    e_max = cfg.T / 2.0
    return np.column_stack([rng.uniform(0, e_max, n), rng.random(n), rng.random(n)])


# ---------------------------------------------------------------------------
# Episode runner (vectorized over episodes)
# ---------------------------------------------------------------------------
def run_episodes(cfg: MarketConfig, markets: Dict[str, np.ndarray], policy: Dict,
                 rng: Optional[np.random.Generator] = None,
                 orders=None) -> Dict[str, float]:
    """Run K episodes with all N agents sharing `policy` (symmetric play).

    `orders`: optional pre-drawn acting orders from sample_orders() (common
    random numbers).  If None, drawn from `rng` (which is then required).

    Returns summary stats (means over all K*N agent-episodes):
      mean_payoff, std_payoff, sem_payoff, pct_bought, pct_empty,
      mean_inspections (inspections made by buyers), mean_accepted_v,
      mean_turns_used.
    """
    K = markets["v"].shape[0]
    N, M, T = cfg.n_agents, cfg.n_styles, cfg.T
    v = markets["v"]
    stock = markets["stock"].astype(np.int64).copy()
    perm = markets["perm"]
    Kidx = np.arange(K)

    kind = policy["kind"]
    if kind == "threshold":
        e = float(policy["e"]); th0 = float(policy["theta0"]); th1 = float(policy["theta1"])
    elif kind == "secretary":
        e = float(policy["e"]); th0 = th1 = float("nan")
    elif kind == "random":
        e = 0.0; p_buy = float(policy.get("p_buy", 0.5))
    else:
        raise ValueError(f"unknown policy kind {kind!r}")

    inspected = np.zeros((K, N, M), dtype=bool)
    done = np.zeros((K, N), dtype=bool)
    bought = np.zeros((K, N), dtype=bool)
    payoff = np.zeros((K, N))
    turns = np.zeros((K, N), dtype=np.int64)
    n_inspect = np.zeros((K, N), dtype=np.int64)
    explore_max = np.zeros((K, N))  # max v seen during explore (secretary rule)
    t_denom = max(T - 1, 1)
    obs = cfg.stock_observable

    for t in range(T):
        order = orders[t] if orders is not None else np.argsort(rng.random((K, N)), axis=1)
        theta = th0 + (th1 - th0) * (t / t_denom) if kind == "threshold" else float("nan")
        for s in range(N):
            a = order[:, s]
            ka = (Kidx, a)
            active = ~done[ka]
            if not active.any():
                continue
            exploring = turns[ka] < e

            # ---- INSPECT: next unseen style in this agent's order ----
            pk = perm[ka]                                   # (K, M) styles in order
            ins = inspected[ka]                             # (K, M) bool, style-index space
            style_avail = (stock > 0) if obs else True      # (K, M) or broadcast True
            cand = (~ins) & style_avail
            cand_order = cand[Kidx[:, None], pk]            # (K, M) in inspection order
            has = cand_order.any(axis=1)
            do_inspect = active & has
            first_pos = cand_order.argmax(axis=1)
            j = pk[Kidx, first_pos]
            ii, aa, jj = Kidx[do_inspect], a[do_inspect], j[do_inspect]
            inspected[ii, aa, jj] = True
            n_inspect[ka] = n_inspect[ka] + do_inspect
            if do_inspect.any():
                new_v = v[ii, aa, jj]
                upd = exploring[do_inspect]
                iu, au = ii[upd], aa[upd]
                explore_max[iu, au] = np.maximum(explore_max[iu, au], new_v[upd])

            # ---- BUY? best inspected style still in stock ----
            ins2 = inspected[ka]
            vv = v[ka]                                      # (K, M)
            buyable = ins2 & (stock > 0)
            masked = np.where(buyable, vv, -np.inf)
            best_av = masked.max(axis=1)
            jbuy = masked.argmax(axis=1)
            if kind == "threshold":
                want = active & (~exploring) & (best_av >= theta)
            elif kind == "secretary":
                want = active & (~exploring) & (best_av >= explore_max[ka])
            else:  # random
                want = active & (rng.random(K) < p_buy) & (best_av > -np.inf)
            bi, ba, bj = Kidx[want], a[want], jbuy[want]
            if want.any():
                stock[bi, bj] -= 1
                bought[bi, ba] = True
                done[bi, ba] = True
                payoff[bi, ba] = v[bi, ba, bj]
            turns[ka] = turns[ka] + active

    # ---- deadline: optional forced fallback buy ----
    if cfg.fallback_buy:
        for a_ in range(N):
            nd = ~done[:, a_]
            if not nd.any():
                continue
            buyable = inspected[:, a_, :] & (stock > 0)
            masked = np.where(buyable, v[:, a_, :], -np.inf)
            best_av = masked.max(axis=1)
            jbuy = masked.argmax(axis=1)
            want = nd & (best_av > -np.inf)
            wi = np.flatnonzero(want)
            if wi.size:
                stock[wi, jbuy[wi]] -= 1
                bought[wi, a_] = True
                done[wi, a_] = True
                payoff[wi, a_] = v[wi, a_, jbuy[wi]]

    ep_mean = payoff.mean(axis=1)
    stats = {
        "mean_payoff": float(payoff.mean()),
        "std_payoff": float(payoff.std()),
        "sem_payoff": float(ep_mean.std(ddof=1) / math.sqrt(K)) if K > 1 else float("nan"),
        "pct_bought": float(bought.mean()),
        "pct_empty": float((~bought).mean()),
        "mean_inspections": float(n_inspect[bought].mean()) if bought.any() else float("nan"),
        "mean_accepted_v": float(payoff[bought].mean()) if bought.any() else float("nan"),
        "mean_turns_used": float(turns.mean()),
    }
    return stats
