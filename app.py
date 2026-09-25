"""Shoe-market: when should you stop searching and commit?

Interactive lab for competitive optimal stopping. N agents shop for M shoe
styles over T time steps; each style has limited stock and each agent's
satisfaction v = w*popularity + (1-w)*personal_taste. Agents learn a threshold
strategy (how long to browse, how picky to be) by self-play evolution.

Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import engine

st.set_page_config(page_title="Shoe Market: when to commit?", layout="wide")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False})

DISTS = ["uniform", "normal", "beta", "heavy"]
DIST_LABELS = {"uniform": "Uniform (even mix)",
               "normal": "Normal (most are average)",
               "beta": "Beta(2,5) (most are meh)",
               "heavy": "Heavy-tail (rare gems)"}

# --------------------------------------------------------------------------
# evolution trainer (compact self-play, mirrors train.py)
# --------------------------------------------------------------------------
def evolve(cfg, pop_size=36, gens=6, K=150, seed=7, progress=None):
    rng = np.random.default_rng(seed)
    e_max = cfg.T / 2.0
    pop = engine.random_params(cfg, rng, pop_size)
    hist_best, hist_mean = [], []
    best, best_fit = None, -1.0
    n_elite = max(2, pop_size // 8)
    for g in range(gens):
        mrng = np.random.default_rng(seed * 1000003 + g * 9177 + 13)
        markets = engine.sample_markets(cfg, K, mrng)
        orders = engine.sample_orders(cfg, K, mrng)
        fits = np.array([
            engine.run_episodes(cfg, markets, engine.threshold_policy(*p),
                                orders=orders)["mean_payoff"] for p in pop])
        order = np.argsort(fits)[::-1]
        pop, fits = pop[order], fits[order]
        hist_best.append(float(fits[0])); hist_mean.append(float(fits.mean()))
        if fits[0] > best_fit:
            best_fit, best = float(fits[0]), pop[0].copy()
        kids = []
        while len(kids) < pop_size - n_elite:
            parent = pop[rng.integers(0, n_elite)]
            child = parent + np.array([rng.normal(0, max(1.0, e_max / 4)),
                                       rng.normal(0, 0.10), rng.normal(0, 0.10)])
            child[0] = np.clip(child[0], 0, e_max)
            child[1:] = np.clip(child[1:], 0, 1)
            kids.append(child)
        pop = np.vstack([pop[:n_elite], np.array(kids)])
        if progress is not None:
            progress.progress((g + 1) / gens)
    return best, best_fit, hist_best, hist_mean


def evaluate_all(cfg, policy, K=2000, seed=99):
    """Paired evaluation of a policy + the three baselines on fresh markets."""
    rng = np.random.default_rng(seed)
    markets = engine.sample_markets(cfg, K, rng)
    orders = engine.sample_orders(cfg, K, rng)
    out = {}
    cands = {"learned": policy,
             "37% rule": engine.make_baseline("secretary37", cfg),
             "fixed 0.7": engine.make_baseline("fixed07"),
             "random": engine.make_baseline("random")}
    for name, pol in cands.items():
        out[name] = engine.run_episodes(cfg, markets, pol, rng=rng,
                                        orders=orders)
    return out


def fig_threshold(e, th0, th1, T):
    t = np.arange(T)
    theta = th0 + (th1 - th0) * t / max(T - 1, 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(t, theta, lw=2.5, color="#7c3aed")
    ax.axvspan(0, e, color="#7c3aed", alpha=0.08)
    ax.text(e / 2 if e > 0 else 1, 0.97, "browse only", ha="center", va="top",
            fontsize=9, color="#7c3aed")
    ax.set(xlim=(0, T - 1), ylim=(0, 1.02), xlabel="time step",
           ylabel="buy threshold θ(t)",
           title="Learned strategy: commit when satisfaction ≥ θ(t)")
    ax.grid(alpha=0.25)
    return fig


def fig_bars(results):
    names = list(results.keys())
    vals = [results[n]["mean_payoff"] for n in names]
    colors = ["#7c3aed", "#94a3b8", "#94a3b8", "#94a3b8"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(names, vals, color=colors)
    bars[0].set_edgecolor("black")
    ax.set(ylim=(0, 1), ylabel="mean satisfaction",
           title="Learned policy vs fixed heuristics (same 2000 markets)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    return fig


def fig_curve(hb, hm):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    g = np.arange(1, len(hb) + 1)
    ax.plot(g, hb, "o-", lw=2, label="best", color="#7c3aed")
    ax.plot(g, hm, "s--", lw=1.5, label="population mean", color="#94a3b8")
    ax.set(xlabel="generation", ylabel="fitness (mean payoff)",
           title="Evolution of the strategy")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    return fig


# --------------------------------------------------------------------------
# sidebar: market setup
# --------------------------------------------------------------------------
st.sidebar.header("Market setup")
preset = st.sidebar.selectbox("Preset", ["Custom", "Calm mall (no rush)",
                                        "Black Friday (scramble)",
                                        "Dating market"])
if preset == "Calm mall (no rush)":
    d_n, d_m, d_t, d_s, d_w, d_fb = 2, 60, 30, 8, 0.2, False
elif preset == "Black Friday (scramble)":
    d_n, d_m, d_t, d_s, d_w, d_fb = 16, 60, 30, 1, 0.9, False
elif preset == "Dating market":
    d_n, d_m, d_t, d_s, d_w, d_fb = 20, 20, 20, 1, 0.9, False
else:
    d_n, d_m, d_t, d_s, d_w, d_fb = 8, 60, 30, 2, 0.5, False

n = st.sidebar.slider("Shoppers (agents)", 1, 24, d_n)
m = st.sidebar.slider("Shoe styles", 10, 120, d_m)
T = st.sidebar.slider("Time steps", 8, 80, d_t)
stock = st.sidebar.slider("Pairs per style (stock)", 1, 12, d_s)
w = st.sidebar.slider("Taste agreement w", 0.0, 1.0, d_w, 0.05,
                      help="1 = everyone agrees what's good (popularity contest); "
                           "0 = purely personal taste")
taste = st.sidebar.selectbox("Satisfaction distribution (taste)",
                             DISTS, format_func=DIST_LABELS.get, index=0)
popd = st.sidebar.selectbox("Satisfaction distribution (popularity)",
                            DISTS, format_func=DIST_LABELS.get, index=0)
fallback = st.sidebar.checkbox("Safety net: forced best-buy at closing time",
                               d_fb)
st.sidebar.caption("Payoff 0 if you leave empty-handed (unless safety net).")

cfg = engine.MarketConfig(n_agents=n, n_styles=m, T=T, stock=stock, w=w,
                          taste_dist=taste, pop_dist=popd,
                          fallback_buy=fallback)

tab_lab, tab_sweeps, tab_how = st.tabs(
    ["Live lab: train agents", "Sweep findings", "How it works"])

# --------------------------------------------------------------------------
# TAB 1: live lab
# --------------------------------------------------------------------------
with tab_lab:
    st.title("When should you stop searching and commit?")
    st.markdown("Agents learn a **threshold strategy**: browse for a while, then "
                "buy the first shoe whose satisfaction clears a bar θ that "
                "_falls_ as closing time approaches. Press train and watch what "
                "they learn under your market settings.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        pop_size = st.slider("Evolution population", 16, 80, 36, step=4)
        gens = st.slider("Generations", 3, 12, 6)
    with col_b:
        K_train = st.slider("Training episodes per candidate", 60, 400, 150,
                            step=10)
        seed = st.number_input("Seed", 1, 9999, 7)

    if st.button("Train agents", type="primary"):
        bar = st.progress(0.0, "Evolving strategies…")
        best, fit, hb, hm = evolve(cfg, pop_size, gens, K_train, seed,
                                   progress=bar)
        bar.progress(1.0, "Evaluating on fresh markets…")
        res = evaluate_all(cfg, engine.threshold_policy(*best))
        bar.empty()
        st.session_state["lab"] = dict(best=best, res=res, hb=hb, hm=hm,
                                       cfg_desc=f"N={n} M={m} T={T} "
                                       f"stock={stock} w={w:.2f} "
                                       f"{taste}/{popd}"
                                       f"{' +safety net' if fallback else ''}")

    lab = st.session_state.get("lab")
    if lab is None:
        st.info("Set up the market in the sidebar, then press **Train agents**.")
    else:
        e, th0, th1 = lab["best"]
        st.caption(f"Market: {lab['cfg_desc']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Browse-only steps", f"{e:.1f}")
        c2.metric("Starting pickiness θ₀", f"{th0:.2f}")
        c3.metric("Closing-time bar θ₁", f"{th1:.2f}")
        c4.metric("Mean satisfaction", f"{lab['res']['learned']['mean_payoff']:.3f}")
        st.pyplot(fig_threshold(e, th0, th1, T))

        st.subheader("Learned vs fixed heuristics")
        st.pyplot(fig_bars(lab["res"]))
        r = lab["res"]["learned"]
        st.markdown(f"Buyers inspect **{r['mean_inspections']:.1f}** shoes on "
                    f"average, accept mean satisfaction **{r['mean_accepted_v']:.3f}**, "
                    f"and **{100*r['pct_empty']:.1f}%** go home empty-handed.")
        gap = (r["mean_payoff"] - lab["res"]["37% rule"]["mean_payoff"])
        st.markdown(f"The learned strategy beats the classic 37% rule by "
                    f"**{gap:+.3f}** here — the rigid rule anchors on shoes "
                    f"rivals may snatch mid-search.")
        with st.expander("Evolution curve"):
            st.pyplot(fig_curve(lab["hb"], lab["hm"]))

# --------------------------------------------------------------------------
# TAB 2: precomputed sweep findings
# --------------------------------------------------------------------------
with tab_sweeps:
    st.title("What the full study found")
    st.markdown("27 market configurations, each trained by self-play evolution "
                "and re-evaluated on 2,000 fresh markets. Hover-free static "
                "charts; every difference quoted is far above noise "
                "(SEMs ≤ 0.0013).")
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "sweep_results.csv"))

    st.subheader("1 · Competition barely hurts — until stock runs out")
    d = df[df.experiment == "competition"].sort_values("n_agents")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(d.n_agents, d.learned_mean, "o-", lw=2.5, color="#7c3aed",
            label="learned")
    ax.plot(d.n_agents, d.base37_mean, "s--", color="#94a3b8",
            label="37% rule")
    ax.set(xlabel="shoppers", ylabel="mean satisfaction",
           title="More shoppers, same slack (stock=2, w=0.5)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    st.pyplot(fig)
    st.markdown("1 → 16 shoppers costs only **0.01** of welfare here: half the "
                "value is personal taste, so rivalry is diffuse. The learned "
                "policy beats the 37% rule by ~0.05 throughout.")

    st.subheader("2 · Scarcity punishes rigid rules")
    d = df[df.experiment == "scarcity"].sort_values("stock")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(d)); wd = 0.35
    ax.bar(x - wd/2, d.learned_mean, wd, label="learned", color="#7c3aed")
    ax.bar(x + wd/2, d.base37_mean, wd, label="37% rule", color="#94a3b8")
    ax.set_xticks(x, [f"{s}" for s in d.stock])
    ax.set(xlabel="pairs per style", ylabel="mean satisfaction",
           title="Learned vs 37% rule as stock thins (N=8)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    st.markdown("At 1 pair per style the learned edge **doubles** (0.086): the "
                "37% rule's explore-phase reference shoe gets bought by a "
                "rival, and then nothing ever clears its bar.")

    st.subheader("3 · The bar adapts to the satisfaction distribution")
    d = df[df.experiment == "taste_dist"]
    order = ["uniform", "normal", "beta", "heavy"]
    d = d.set_index("taste_dist").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(d)); wd = 0.2
    for i, (col, lab_) in enumerate([("learned_mean", "learned"),
                                     ("base37_mean", "37% rule"),
                                     ("fixed07_mean", "fixed 0.7"),
                                     ("random_mean", "random")]):
        ax.bar(x + (i - 1.5) * wd, d[col], wd, label=lab_,
               color="#7c3aed" if i == 0 else "#94a3b8",
               alpha=1.0 if i == 0 else 0.55)
    ax.set_xticks(x, [DIST_LABELS[o].split(" (")[0] for o in order], rotation=12)
    ax.set(ylabel="mean satisfaction", title="Rare gems → lower your bar")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    st.markdown("When great shoes are rare (beta/heavy-tail), the learned "
                "closing bar falls to **0.55 / 0.48** — waiting for a 0.9 that "
                "never comes means leaving empty-handed. Fixed 0.7 collapses "
                "to 0.42 under beta taste: **worse than buying at random**.")

    st.subheader("4 · Taste agreement is U-shaped (the surprise)")
    d = df[df.experiment == "correlation"].sort_values("w")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(d.w, d.learned_mean, "o-", lw=2.5, color="#7c3aed")
    ax.set(xlabel="taste agreement w", ylabel="mean satisfaction",
           title="Partial agreement is worse than none or total")
    for _, r_ in d.iterrows():
        ax.annotate(f"{r_.learned_mean:.3f}", (r_.w, r_.learned_mean),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9)
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    st.markdown("**w=0 → 0.952** (nobody fights over *your* favorites, stay "
                "picky: θ₁=0.83). **w=0.3–0.7 → ~0.875** (you brawl over shoes "
                "that are only mediocre for you). **w=1 → 0.941** (everyone "
                "agrees what's best, so the top styles get allocated cleanly). "
                "Partial agreement is the worst of both worlds.")

    st.subheader("5 · Less time → commit earlier to good-enough")
    d = df[df.experiment == "time"].sort_values("T")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.4))
    a1.plot(d["T"], d.theta1, "o-", lw=2.5, color="#7c3aed")
    a1.set(xlabel="time budget T", ylabel="closing bar θ₁",
           title="Shorter deadline → lower bar")
    a2.plot(d["T"], d.learned_mean, "o-", lw=2.5, color="#7c3aed",
            label="learned")
    a2.plot(d["T"], d.fixed07_mean, "s--", color="#94a3b8", label="fixed 0.7")
    a2.set(xlabel="time budget T", ylabel="mean satisfaction",
           title="A fixed bar can't exploit more time")
    for a in (a1, a2):
        a.grid(alpha=0.25)
    a2.legend(frameon=False)
    st.pyplot(fig)

    st.subheader("6 · Dating market: pounce, then lower standards fast")
    d = df[df.experiment == "dating"].iloc[0]
    st.markdown(f"20 suitors, 20 partners, everyone mostly agrees who's "
                f"attractive (w=0.9), each pairs once. Learned: **no browsing "
                f"(e=0), θ: 1.00 → 0.06**, payoff **{d.learned_mean:.3f}** vs "
                f"37% rule {d.base37_mean:.3f}, fixed {d.fixed07_mean:.3f}, "
                f"random {d.random_mean:.3f}.")
    st.markdown("*Pounce on love at first sight, then lower your standards "
                "fast.* Exploring while others commit is pure loss; even "
                "random committing beats the 37% rule here — a mediocre "
                "partner beats no partner.")

    st.subheader("7 · A safety net makes you pickier")
    r1 = df[df.experiment == "fallback"].iloc[0]
    r0 = df[(df.experiment == "competition") & (df.n_agents == 8)].iloc[0]
    st.markdown(f"With a forced best-buy at closing time, the bar barely "
                f"declines (**1.00 → 0.95** vs 1.00 → 0.71), welfare rises "
                f"({r1.learned_mean:.3f} vs {r0.learned_mean:.3f}), nobody "
                f"leaves empty-handed. Real-world analog: *keep looking — if "
                f"nothing turns up, order the safe pair online.* The outside "
                "option raises your reservation price.")

# --------------------------------------------------------------------------
# TAB 3: how it works
# --------------------------------------------------------------------------
with tab_how:
    st.title("How it works")
    st.markdown("""
**The market.** N agents shop M shoe styles over T steps. Style *j* has a
popularity *pⱼ* (common value) and a stock of identical pairs. Agent *i*'s
satisfaction is

> vᵢⱼ = w · pⱼ + (1 − w) · tᵢⱼ, clipped to [0, 1]

*tᵢⱼ* is personal taste. **w** = taste agreement: w=1 is a pure popularity
contest, w=0 is purely personal taste.

**A turn.** Agents act in random order. On your turn: **inspect** the next
unseen style (costs the turn), **buy** an inspected style that still has
stock (you exit with that satisfaction), or **pass**. Stock is visible —
agents skip sold-out styles. No purchase by closing → 0.

**What agents learn.** A threshold strategy *(e, θ₀, θ₁)*: browse-only for
*e* steps, then buy the first shoe with satisfaction ≥ θ(t), where θ falls
linearly from θ₀ to θ₁. Learned by **self-play evolution**: a population of
strategies competes, the fittest reproduce with mutations, for a few
generations — all agents share the winning strategy.

**The headline answer to "when to move on, when to commit":** commit when a
shoe clears a bar that *you lower as time passes*; keep moving on while
everything is below the bar. Start the bar near your max realistic hope,
end it near your walk-away value — and in a scramble, never "explore"
passively: stay ready to pounce from step one.

**Caveats.** Symmetric self-play (no asymmetric trickery); no prices or
budgets; no learning *within* a trip from what you've seen; competition only
via stock depletion. Stylized distributions, not calibrated to real shoppers.
""")
    st.caption("Engine: numpy-only `engine.py` · full study: 27 configs, "
               "seeds recorded, see the paper for details.")
