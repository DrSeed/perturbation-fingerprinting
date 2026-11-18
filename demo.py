# Self-contained demo: a reference perturbation dictionary and a query experiment with a different baseline.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fingerprint import learn_fingerprints, map_cells

G, K = 200, 12


def make_world(seed=11):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, G)
    Phi = np.linalg.qr(rng.normal(size=(G, 3)))[0] * np.array([6.0, 4.0, 3.0])  # cell cycle, differentiation, stress
    fps = np.zeros((G, K))
    for k in range(K):
        genes = rng.choice(G, 25, replace=False)
        fps[genes, k] = rng.choice([-1, 1], 25) * rng.uniform(0.5, 1.2, 25)
    for a, b in [(0, 1), (2, 3)]:  # two protein complexes: knocking down either member looks alike
        fps[:, b] = 0.85 * fps[:, a] + 0.4 * np.where(fps[:, b] != 0, fps[:, b], 0)
    fps[:, 6] += 0.35 * Phi[:, 0] / 6.0 * np.linalg.norm(fps[:, 6])  # perturbation 6 also slows the cell cycle
    return rng, mu, Phi, fps


def cells(rng, mu, Phi, fps, n, pert=None, strength=(0.6, 1.6), cycle_shift=0.0):
    f = rng.normal(0, 1, (n, 3))
    f[:, 0] += cycle_shift
    Y = mu + f @ Phi.T + rng.normal(0, 1, (n, G))
    if pert is not None:
        for k in np.atleast_1d(pert):
            Y += np.outer(rng.uniform(*strength, n), fps[:, k])
    return Y


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rng, mu, Phi, fps = make_world()

    ref_ctrl = cells(rng, mu, Phi, fps, 600)
    ref = np.vstack([cells(rng, mu, Phi, fps, 60, k) for k in range(K)])
    ref_lab = np.repeat(np.arange(K), 60)
    learned, names = learn_fingerprints(ref_ctrl, ref, ref_lab)

    # query: a new lab where treated cells cycle more than the query's own controls
    q_ctrl = cells(rng, mu, Phi, fps, 400)
    truth, Q = [], []
    for k in range(K):
        Q.append(cells(rng, mu, Phi, fps, 15, k, cycle_shift=0.6)); truth += [(k,)] * 15
    for a, b in [(4, 9), (5, 10), (7, 11)]:
        Q.append(cells(rng, mu, Phi, fps, 15, [a, b], cycle_shift=0.6)); truth += [(a, b)] * 15
    Q.append(cells(rng, mu, Phi, fps, 30, cycle_shift=0.6)); truth += [()] * 30
    Q = np.vstack(Q)

    results = map_cells(q_ctrl, Q, learned)

    # naive baseline: correlate (cell - query control mean) with reference mean-difference signatures
    sig = np.array([ref[ref_lab == k].mean(0) - ref_ctrl.mean(0) for k in range(K)])
    D = Q - q_ctrl.mean(0)
    corr = np.array([[np.corrcoef(d, s)[0, 1] for s in sig] for d in D])

    rows = []
    for i, t in enumerate(truth):
        sets = results[i][0]
        kind = "unperturbed" if len(t) == 0 else ("pair" if len(t) == 2 else
               ("complex member" if t[0] in (0, 1, 2, 3) else "single"))
        if not sets:
            fp = "correct" if len(t) == 0 else "abstained"
        else:
            hits = [any(k in s for k in t) for s in sets]   # each reported set contains a true perturbation
            found = [any(k in s for s in sets) for k in t]  # each true perturbation is in some set
            fp = ("correct" if all(hits) and all(found) and len(sets) == len(t)
                  else "partly (one of two)" if all(hits) and any(found) else "wrong")
        # the naive matcher always names its top hit; it is even told how many perturbations to name
        naive = set(np.argsort(corr[i])[::-1][: max(1, len(t))])
        rows.append({"kind": kind, "truth": "+".join(map(str, t)) or "none",
                     "fingerprint_call": fp, "credible_set_sizes": " ".join(str(len(s)) for s in sets),
                     "naive_call": "correct" if len(t) and naive == set(t) else "wrong"})
    per_cell = pd.DataFrame(rows)
    per_cell.to_csv("results/per_cell_calls.csv", index=False)
    order = ["single", "complex member", "pair", "unperturbed"]
    table = pd.DataFrame({
        "n": per_cell.groupby("kind").size(),
        "naive_correct_pct": per_cell.groupby("kind")["naive_call"].apply(lambda s: (s == "correct").mean()),
        "fingerprint_correct_pct": per_cell.groupby("kind")["fingerprint_call"].apply(lambda s: (s == "correct").mean()),
        "fingerprint_partly_pct": per_cell.groupby("kind")["fingerprint_call"].apply(lambda s: (s == "partly (one of two)").mean()),
        "fingerprint_abstained_pct": per_cell.groupby("kind")["fingerprint_call"].apply(lambda s: (s == "abstained").mean()),
        "fingerprint_wrong_pct": per_cell.groupby("kind")["fingerprint_call"].apply(lambda s: (s == "wrong").mean()),
    }).reindex(order)
    table.iloc[:, 1:] = (100 * table.iloc[:, 1:]).round(1)

    # group level: average the residuals of each 15-cell block before regressing (the paper's group mode)
    blocks = np.cumsum([0] + [15] * (K + 3) + [30])
    group_rows = []
    for s_, e_ in zip(blocks[:-1], blocks[1:]):
        t = truth[s_]
        res = map_cells(q_ctrl, Q[s_:e_].mean(axis=0, keepdims=True), learned)[0][0]
        group_rows.append(len(res) == len(t) and all(any(k in c for c in res) for k in t))
    table["group_level_correct_pct"] = np.nan
    table.loc["all groups", "group_level_correct_pct"] = round(100 * np.mean(group_rows), 1)
    table.to_csv("results/summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    cos = learned.T @ learned
    im = axes[0].imshow(np.abs(cos), cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(K)); axes[0].set_yticks(range(K))
    axes[0].set_title("Learned fingerprints: |cosine|\n(0-1 and 2-3 are complex partners)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    examples = {"single (5)": next(i for i, t in enumerate(truth) if t == (5,)),
                "complex member": max((i for i, t in enumerate(truth) if t and t[0] in (0, 1, 2, 3) and len(t) == 1),
                                      key=lambda i: max([len(s) for s in results[i][0]] or [0])),
                "pair (4+9)": next(i for i, t in enumerate(truth) if t == (4, 9)),
                "unperturbed": next(i for i, t in enumerate(truth) if t == ())}
    examples = {(f"complex member ({truth[i][0]})" if k == "complex member" else k): i for k, i in examples.items()}
    pip = np.array([1 - np.prod(1 - results[i][1][results[i][2] >= 3], axis=0) for i in examples.values()])
    axes[1].imshow(pip, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[1].set_yticks(range(len(examples)), list(examples))
    axes[1].set_xticks(range(K))
    axes[1].set_xlabel("perturbation")
    axes[1].set_title("Posterior inclusion probability for four query cells\n(complex partners share probability; pairs get two effects)")
    for r_ in range(pip.shape[0]):
        for c_ in range(K):
            if pip[r_, c_] > 0.05:
                axes[1].text(c_, r_, f"{pip[r_, c_]:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if pip[r_, c_] < 0.6 else "black")

    x = np.arange(len(order))
    axes[2].bar(x - 0.2, table.loc[order, "naive_correct_pct"], 0.4, color="#c44e52", label="naive: correct")
    axes[2].bar(x - 0.2, 100 - table.loc[order, "naive_correct_pct"], 0.4, bottom=table.loc[order, "naive_correct_pct"],
                color="#c44e52", alpha=0.3, label="naive: wrong")
    bottom = np.zeros(len(order))
    for col, c, lab in [("fingerprint_correct_pct", "#4c72b0", "fingerprinting: correct"),
                        ("fingerprint_partly_pct", "#9db7de", "fingerprinting: one of two found"),
                        ("fingerprint_abstained_pct", "#bbbbbb", "fingerprinting: abstained"),
                        ("fingerprint_wrong_pct", "#4c72b0", "fingerprinting: wrong")]:
        v = table.loc[order, col].values
        axes[2].bar(x + 0.2, v, 0.4, bottom=bottom, color=c, alpha=0.3 if "wrong" in col else 1, label=lab,
                    hatch="//" if "wrong" in col else None)
        bottom += v
    axes[2].set_xticks(x, order, rotation=15)
    axes[2].set_ylabel("% of query cells")
    axes[2].set_title(f"Single cells: abstain rather than guess\n(group level, 15 cells averaged: {table.loc['all groups', 'group_level_correct_pct']:.0f}% correct)")
    axes[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print(table.to_string())


if __name__ == "__main__":
    main()
