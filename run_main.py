"""Frozen J-CAP main experiment. All paths are relative to this release."""
from pathlib import Path
from functools import lru_cache
import argparse
import itertools
import json
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WARDS = ["W1", "W2", "W3", "W4"]


@lru_cache(maxsize=4)
def candidates(k):
    if k not in (2, 4, 6, 8):
        raise ValueError("K must be 2, 4, 6 or 8")
    n = k // 2
    rows = [x for x in itertools.product(range(n + 1), repeat=5)
            if x[0] + x[1] == n and x[2] + x[3] + x[4] == n]
    rows.sort(key=lambda x: (x[1] + x[3] + x[4], x))
    return np.asarray(rows, dtype=int)


def ward_totals(x):
    return np.stack([x[..., 0], x[..., 2], x[..., 1] + x[..., 3], x[..., 4]], axis=-1)


def load_data():
    cfg = json.loads((ROOT / "config/frozen.json").read_text(encoding="utf-8"))
    cal = pd.read_csv(ROOT / "data/calendar.csv")
    raw = pd.read_csv(ROOT / "data/ward_day.csv", float_precision="round_trip")
    days = cal.day_index.to_numpy(int)
    np.testing.assert_array_equal(days, np.arange(len(cal)))
    cn = np.full((len(cal), 4, 2), np.nan, np.float32)
    beds = np.full((len(cal), 4), np.nan, np.float32)
    for w, ward in enumerate(WARDS):
        g = raw[raw.ward.eq(ward)].set_index("day_index").reindex(days)
        c = g.child_count + g.adult_count
        ok = g.active.fillna(False).astype(bool) & c.ge(0) & g.nurse_count.ge(0) & g.analysis_beds.gt(0)
        cn[:, w, 0] = c.where(ok)
        cn[:, w, 1] = g.nurse_count.where(ok)
        beds[:, w] = g.analysis_beds
    tau = np.asarray(cfg["tau"], np.float32)
    slack = np.stack([tau * cn[:, :, 1] - cn[:, :, 0], beds - cn[:, :, 0]], -1)
    return cfg, cal, cn, beds, tau, slack


def joint_scenarios(cn, beds, tau, day, cfg):
    valid = np.isfinite(cn).all((1, 2))
    anchor = np.flatnonzero(valid & (np.arange(len(cn)) <= day - 2))[-1]
    horizon = day - anchor
    ends = np.arange(max(horizon, day - cfg["history_window_days"]), day - 1)
    ends = ends[(ends >= cfg["history_change_first_day"]) & valid[ends] & valid[ends - horizon]]
    if len(ends) < cfg["minimum_history"]:
        raise ValueError("Insufficient observed history; missing values are never zero-filled")
    future = np.maximum(0, cn[anchor] + cn[ends] - cn[ends - horizon])
    # Frozen implementation repeats the empirical sample. Retain numeric tie behavior.
    future = np.tile(future, (cfg["scenario_repeats"], 1, 1))
    scenario = np.stack([tau * future[:, :, 1] - future[:, :, 0], beds[day] - future[:, :, 0]], -1)
    assert anchor <= day - 2 and ends.max() <= day - 2
    return scenario, int(anchor), ends


def costs(slack, k):
    a = ward_totals(candidates(k))
    return (np.maximum(a[None, :, :, None] - slack[:, None, :, :], 0)
            - np.maximum(-slack[:, None, :, :], 0)).sum(2)


def reference_cost(demand, beds, cfg):
    """Minimal full-pooling reference needed by J-CAP screening.

    Adapted from Wang Algorithm A1's pooling mechanism, with the frozen daily
    forecast interface. This is not a full paper replication or a benchmark run.
    Attribution and daily-data assumptions are documented in METHOD.md.
    """
    r = cfg["internal_reference"]
    alpha, c, delta = r["alpha"], r["transfer_cost"], r["stress_discount"]
    reserve = (1 - alpha) * beds
    pool = alpha * beds.sum()
    stress = demand.sum(1) >= r["stress_threshold"]
    lam = np.maximum(r["minimum_efficiency"], 1 - alpha * delta * (1 + r["rho"]) * stress)
    need = np.maximum(demand - reserve, 0) / lam[:, None]
    y = np.zeros_like(demand)
    remaining = np.full(len(demand), pool, float)
    scores = lam.mean() * np.full(4, 10.) - c / r["qscale"]
    for w in np.argsort(-scores, kind="stable"):
        if scores[w] > 0:
            y[:, w] = np.minimum(need[:, w], remaining)
            remaining -= y[:, w]
    effective = reserve + lam[:, None] * y
    physical = reserve + y
    short = np.maximum(demand - effective, 0)
    idle = np.maximum(effective - demand, 0)
    transfer = np.abs(physical - beds).sum(1) / 2
    return (10 * short.sum(1) + idle.sum(1) + remaining
            + c * transfer / r["qscale"] + .05 * short.sum(1) ** 2)


def reference_action(scenario, beds, k, cfg):
    scenario = scenario[:len(scenario) // cfg["scenario_repeats"]]
    census = beds[None] - scenario[:, :, 1]
    a = ward_totals(candidates(k))
    demand = (census[:, None] + a[None]).reshape(-1, 4)
    scores = reference_cost(demand, beds, cfg).reshape(len(census), len(a)).mean(0)
    return int(np.lexsort((np.arange(len(a)), scores))[0])


def select(scenario, k, reference, cfg):
    hp = costs(scenario, k)
    p = (hp <= cfg["success_tolerance"]).all(2).mean(0)
    mean = hp.mean(0)
    j = (cfg["weight_H"] * mean[:, 0] + cfg["weight_P"] * mean[:, 1]) / k
    eligible = ((p >= p[reference]) & (mean[:, 0] <= mean[reference, 0])
                & (mean[:, 1] <= mean[reference, 1]))
    ix = np.flatnonzero(eligible)
    best = float(j[ix].min())
    chosen = reference if j[reference] == best else int(ix[np.lexsort((ix, j[ix]))[0]])
    assert eligible[reference] and eligible[chosen]
    return chosen, p, mean, j


def verify(output, cfg):
    actual = pd.read_csv(output / "daily.csv", float_precision="round_trip")
    expected = pd.read_csv(ROOT / "results/reference_daily.csv", float_precision="round_trip")
    keys = ["day_index", "K"]
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)
    for field in keys + ["path", "action", "success"]:
        np.testing.assert_array_equal(a[field], e[field])
    for field in ["H", "P", "J", "pred_p", "pred_H", "pred_P", "pred_J"]:
        np.testing.assert_allclose(a[field], e[field], rtol=0, atol=1e-8)
    ward = pd.read_csv(output / "ward.csv")
    expected_ward = pd.read_csv(ROOT / "results/reference_ward.csv")
    ward_keys = keys + ["ward"]
    wa = ward.sort_values(ward_keys).reset_index(drop=True)
    we = expected_ward.sort_values(ward_keys).reset_index(drop=True)
    for field in ward_keys + ["admissions"]:
        np.testing.assert_array_equal(wa[field], we[field])
    for field in ["H", "P", "background_H", "background_P"]:
        np.testing.assert_allclose(wa[field], we[field], rtol=0, atol=1e-8)
    np.testing.assert_allclose(ward.groupby(keys)[["H", "P"]].sum().sort_index(),
                               actual.set_index(keys)[["H", "P"]].sort_index(), rtol=0, atol=1e-8)
    assert len(a) == cfg["expected_cases"] and a.day_index.nunique() == cfg["expected_days"]
    assert int(a.success.sum()) == cfg["expected_successes"]
    return {"passed": True, "all_case_actions_match": True, "all_case_metrics_match": True,
            "all_ward_metrics_match": True,
            "ward_costs_sum_to_system_costs": True, "equal_K_source_quotas": True,
            "cases": len(a), "dates": int(a.day_index.nunique()), "successes": int(a.success.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg, cal, cn, beds, tau, slack = load_data()
    rows, wards, provenance = [], [], []
    for day in cal.loc[cal.evaluate, "day_index"].to_numpy(int):
        scenario, anchor, ends = joint_scenarios(cn, beds, tau, day, cfg)
        # Selection sees no unavailable occupancy/staffing. Test this at every target.
        poisoned = cn.copy()
        poisoned[day - 1:] = 123456
        check, _, _ = joint_scenarios(poisoned, beds, tau, day, cfg)
        np.testing.assert_array_equal(check, scenario)
        provenance.append(dict(day_index=day, latest_observed_day=anchor,
                               last_change_day=int(ends[-1]), distinct_scenarios=len(ends),
                               repeated_scenarios=len(scenario)))
        for k in cfg["K"]:
            reference = reference_action(scenario, beds[day], k, cfg)
            idx, p, mean, j = select(scenario, k, reference, cfg)
            x = candidates(k)[idx]
            assert x[:2].sum() == k // 2 and x[2:].sum() == k // 2 and x.sum() == k
            a = ward_totals(x)
            ward_hp = np.maximum(a[:, None] - slack[day], 0) - np.maximum(-slack[day], 0)
            h, b = ward_hp.sum(0)
            rows.append(dict(day_index=day, K=k, model="J_CAP", action=idx,
                             path=json.dumps(x.tolist()), H=h, P=b,
                             J=(cfg["weight_H"] * h + cfg["weight_P"] * b) / k,
                             success=bool(h <= cfg["success_tolerance"] and b <= cfg["success_tolerance"]),
                             pred_p=p[idx], pred_H=mean[idx, 0], pred_P=mean[idx, 1], pred_J=j[idx],
                             internal_reference_action=reference))
            for w, name in enumerate(WARDS):
                wards.append(dict(day_index=day, K=k, model="J_CAP", ward=name,
                                  admissions=int(a[w]), H=ward_hp[w, 0], P=ward_hp[w, 1],
                                  background_H=max(0, -slack[day, w, 0]),
                                  background_P=max(0, -slack[day, w, 1])))
    daily = pd.DataFrame(rows)
    daily.to_csv(out / "daily.csv", index=False)
    pd.DataFrame(wards).to_csv(out / "ward.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "scenario_provenance.csv", index=False)
    summary = dict(model="J_CAP", dates=int(daily.day_index.nunique()), cases=len(daily),
                   successes=int(daily.success.sum()), success_rate=float(daily.success.mean()),
                   **{m: float(daily[m].mean()) for m in ["H", "P", "J"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checks = verify(out, cfg)
    checks.update(unavailable_state_poison_invariant=True, python=sys.version.split()[0],
                  numpy=np.__version__, pandas=pd.__version__, platform=platform.system())
    (out / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("PASS: all frozen actions and costs reproduced; source quotas, ward totals and information timing checked.")


if __name__ == "__main__":
    main()
