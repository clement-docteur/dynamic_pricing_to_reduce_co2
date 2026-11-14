"""Step 3 building blocks: candidate tariff menu, exact optimal policy tree (shallow prescriptive tree on model-based
action costs, Lecture 11), the average-price feasibility rule and Lagrange-multiplier helpers.

Prices are always (n_days, n_actions) arrays (a per-day row; the time-average price of a schedule is the same every day).
"""
from __future__ import annotations

import numpy as np

from .common import H, PRICE, schedule_price

NORMAL = 0          # menu index of the all-Normal day (always first)
MAX_LOW = 12        # carbon rule: at most 6 h of Low per day
POLICIES = ["Historical TOU (2013 trial)", "Simple carbon-based tariff", "Optimal policy tree", "Direct optimisation",
            "Direct optimisation, perfect carbon forecast", "Oracle (perfect hindsight)", "Budget-aware policy tree",
            "Budget-aware direct optimisation"]
SHORT = dict(zip(POLICIES, ["Historical TOU", "Carbon rule (no demand model)", "Optimal policy tree",
                            "Direct optimisation", "Direct opt., perfect carbon forecast", "Oracle (hindsight)",
                            "Budget-aware tree", "Budget-aware direct opt."]))


def slot_time(s):
    """Start clock time (GMT) of slot s."""
    return f"{((s - 1) * 30) // 60:02d}:{((s - 1) * 30) % 60:02d}"


# ------------------------------------------------------------------------------------------------ menu
def make_schedule(high_block=None, low_block=None):
    """(low, high) 0/1 arrays for one day; blocks are inclusive (first_slot, last_slot)."""
    lo, hi = np.zeros(H), np.zeros(H)
    if high_block:
        hi[high_block[0]: high_block[1] + 1] = 1
    if low_block:
        lo[low_block[0]: low_block[1] + 1] = 1
    return lo, hi


def describe(high_block=None, low_block=None):
    parts = [f"{kind} {slot_time(b[0])}-{slot_time(b[1] + 1)}" for kind, b in (("High", high_block), ("Low", low_block)) if b]
    return " + ".join(parts) or "Normal all day"


def candidate_pool():
    """Plausible single-block schedules (block shapes and times like the 2013 trial's), kept inside slots 3-44 so
    that the 2 h edge effects stay within the day. Returns a list of dicts(name, low, high, price); entry 0 is the
    all-Normal day."""
    blocks = [(None, None)]
    blocks += [((s, s + L - 1), None) for L in (4, 6, 8) for s in range(3, 42, 2) if s + L - 1 <= 44]   # High 2-4 h, 01-20 h
    blocks += [(None, (s, s + L - 1)) for L in (6, 12) for s in range(3, 34, 2) if s + L - 1 <= 44]     # Low 3/6 h, 01-16 h
    blocks += [((hs, hs + 5), (ls, ls + 11)) for hs in (33, 35, 37, 39) for ls in (3, 7)]              # evening High + night Low
    pool = []
    for hb, lb in blocks:
        lo, hi = make_schedule(hb, lb)
        pool.append(dict(name=describe(hb, lb), low=lo, high=hi, price=float(schedule_price(lo, hi).mean())))
    return pool


# ------------------------------------------------------------------------------------------------ policy tree
class PolicyTree:
    """Exact optimal prescriptive tree of a given depth minimising the summed cost of the prescribed actions.
    Split thresholds: all midpoints at the last level, `n_thresh` quantiles above it (n_thresh=None: every threshold at
    every level, i.e. the global optimum). Each leaf keeps the ranking of the actions by their summed training cost
    (the fallback order used by the feasibility rule)."""

    def __init__(self, depth, min_leaf=10, n_thresh=12, feature_names=None):
        self.depth, self.min_leaf, self.n_thresh, self.names = depth, min_leaf, n_thresh, feature_names

    def fit(self, X, C):
        self.X, self.C = np.asarray(X, float), np.asarray(C, float)
        self.root = self._build(np.arange(len(X)), self.depth)
        del self.X, self.C
        return self

    def _leaf(self, idx):
        tot = self.C[idx].sum(0)
        return {"leaf": True, "action": int(tot.argmin()), "n": int(len(idx)), "ranking": np.argsort(tot, kind="stable").tolist()}

    def _stump(self, idx, leaf):
        """Best depth-1 split of idx, vectorised over features: (cost, feature, threshold)."""
        Xs, Cs = self.X[idx], self.C[idx]
        O = np.argsort(Xs, axis=0, kind="stable")
        xs = np.take_along_axis(Xs, O, 0)
        cum = np.cumsum(Cs[O], axis=0)                       # (n, p, A)
        k = np.arange(self.min_leaf - 1, len(idx) - self.min_leaf)
        s = cum[k].min(-1) + (cum[-1][None] - cum[k]).min(-1)  # (len k, p)
        s[~(xs[k] < xs[k + 1])] = np.inf
        i, j = np.unravel_index(int(np.argmin(s)), s.shape)
        if not s[i, j] < leaf - 1e-9:
            return leaf, None, None
        return float(s[i, j]), int(j), float(0.5 * (xs[k[i], j] + xs[k[i] + 1, j]))

    def _search(self, idx, depth):
        """Optimal (cost, feature, threshold) of a depth-`depth` tree on idx (feature None = leaf)."""
        rowmin = self.C[idx].min(1)
        leaf, lower_bound = self.C[idx].sum(0).min(), rowmin.sum()
        if depth == 0 or len(idx) < 2 * self.min_leaf or leaf - lower_bound < 1e-9:
            return leaf, None, None
        if depth == 1:
            return self._stump(idx, leaf)
        best = (leaf, None, None)
        Xs = self.X[idx]
        if self.n_thresh:
            Q = np.quantile(Xs, np.linspace(0, 1, self.n_thresh + 2)[1:-1], axis=0, method="lower")
            thresholds = [np.unique(Q[:, j]) for j in range(Xs.shape[1])]
        else:
            thresholds = [np.unique(Xs[:, j])[:-1] for j in range(Xs.shape[1])]
        for j in range(Xs.shape[1]):
            for t in thresholds[j]:
                m = Xs[:, j] <= t
                nl = int(m.sum())
                if nl < self.min_leaf or len(idx) - nl < self.min_leaf or lower_bound >= best[0] - 1e-9:
                    continue
                cl = self._search(idx[m], depth - 1)[0]
                if cl + rowmin[~m].sum() >= best[0] - 1e-9:
                    continue
                cr = self._search(idx[~m], depth - 1)[0]
                if cl + cr < best[0] - 1e-9:
                    best = (cl + cr, j, float(t))
        return best

    def _build(self, idx, depth):
        _, j, t = self._search(idx, depth)
        if j is None:
            return self._leaf(idx)
        m = self.X[idx, j] <= t
        return {"leaf": False, "feature": j, "threshold": t, "n": int(len(idx)),
                "left": self._build(idx[m], depth - 1), "right": self._build(idx[~m], depth - 1)}

    def leaf_of(self, x):
        node = self.root
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node

    def rankings(self, X):
        return [self.leaf_of(x)["ranking"] for x in np.asarray(X, float)]

    def n_leaves(self, node=None):
        node = self.root if node is None else node
        return 1 if node["leaf"] else self.n_leaves(node["left"]) + self.n_leaves(node["right"])

    def text(self, action_names, node=None, indent=""):
        node = self.root if node is None else node
        if node["leaf"]:
            return f"{indent}-> {action_names[node['action']]}  (n={node['n']})\n"
        f = self.names[node["feature"]] if self.names else f"x{node['feature']}"
        return (f"{indent}if {f} <= {node['threshold']:.3g}:\n" + self.text(action_names, node["left"], indent + "    ")
                + f"{indent}else:\n" + self.text(action_names, node["right"], indent + "    "))


# ------------------------------------------------------------------------------------------------ constraint handling
def budget_slack(spent, k, D, target, p_normal):
    """Remaining budget per remaining day, as extra price above an all-Normal day (p/kWh), before day k."""
    return (D * target - spent) / (D - k) - p_normal


def feasibility_rule(rankings, prices, target):
    """Online average-price rule over an evaluation period of D days (prices: (D, A), known at decision time).
    Day k takes the first action in its ranking such that the target stays attainable if every remaining day were
    all-Normal ('cancel the event'; day k's Normal price stands in for future days'). `rankings` is a list of
    per-day rankings, or a function (k, budget_slack) -> ranking for budget-aware policies.
    Returns (actions, n_adjustments)."""
    P = np.asarray(prices, float)
    D = len(P)
    assert (P[:, NORMAL] <= target + 1e-9).all(), "an all-Normal day must always fit the target"
    S, out, adj = 0.0, [], 0
    for k in range(D):
        rank = rankings(k, budget_slack(S, k, D, target, P[k, NORMAL])) if callable(rankings) else rankings[k]
        for i, a in enumerate(rank):
            if S + P[k, a] + (D - k - 1) * P[k, NORMAL] <= D * target + 1e-9:
                out.append(a)
                adj += i > 0
                S += P[k, a]
                break
    return np.array(out), adj


def tune_lambda(choose, prices, target, hi=0.05):
    """Smallest lambda (bisection) such that the mean price of choose(lambda) -> actions is <= target."""
    P = np.asarray(prices, float)
    mp = lambda lam: P[np.arange(len(P)), choose(lam)].mean()
    lo = 0.0
    while mp(hi) > target and hi < 1e6:
        hi *= 2
    if mp(lo) <= target:
        return lo
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if mp(mid) <= target else (mid, hi)
    return hi


def pick_near_best(grid, tiebreak, key="val_emissions_kg", rel=1e-4):
    """Among grid entries within `rel` of the best `key`, the one that minimises `tiebreak`."""
    best = min(g[key] for g in grid)
    return min((g for g in grid if g[key] <= best * (1 + rel)), key=tiebreak)


# ------------------------------------------------------------------------------------------------ carbon-only baseline
def carbon_rule_schedule(ci_forecast_day, k_high, k_low):
    """High on the forecast-dirtiest contiguous k_high slots, Low on the forecast-cleanest contiguous k_low slots
    not overlapping it."""
    lo, hi = np.zeros(H), np.zeros(H)
    if k_high:
        s = int(np.convolve(ci_forecast_day, np.ones(k_high), "valid").argmax())
        hi[s: s + k_high] = 1
    if k_low:
        w = np.convolve(ci_forecast_day, np.ones(k_low), "valid").astype(float)
        w[[s for s in range(len(w)) if hi[s: s + k_low].any()]] = np.inf
        s = int(w.argmin())
        lo[s: s + k_low] = 1
    return lo, hi


def carbon_rule_counts(target):
    """(k_high, k_low): as many High slots as possible (Low <= MAX_LOW) with a daily average price <= target."""
    best = (0, 0)
    for kh in range(13):
        for kl in range(MAX_LOW + 1):
            if (kh * PRICE["H"] + kl * PRICE["L"] + (H - kh - kl) * PRICE["N"]) / H <= target + 1e-9:
                best = (kh, kl) if kh > best[0] else best
                break
    return best
