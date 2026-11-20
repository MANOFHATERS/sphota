"""Step 5: image features + file-size predictor (Linear / Ridge / Lasso / Elastic Net).

    python scripts/train_size_predictor.py                    # features (resumable) then models
    python scripts/train_size_predictor.py --stage features
    python scripts/train_size_predictor.py --stage model

Target: log(JPEG bpp at quality 10). Models: StandardScaler + {Linear, Ridge, Lasso, ElasticNet}.
Reported R^2 is NESTED 5-fold CV (alpha chosen by an inner 5-fold CV inside each outer training fold),
on train+val images, plus a held-out R^2 on each test set (fit on train+val only).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.analysis.features import FEATURES, extract_all  # noqa: E402

OUT = ROOT / "experiments" / "step5_size_predictor"
FIGS = ROOT / "report" / "figures"
SKEWED = ["grad_mean", "lap_var", "edge_density"]
ALPHAS = np.logspace(-5, 2, 36)


def make(kind, alpha=1.0, l1=0.5):
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    m = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=alpha),
         "Lasso": Lasso(alpha=alpha, max_iter=50000), "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=50000)}[kind]
    return Pipeline([("scale", StandardScaler()), ("model", m)])


def design(df, cols, log_skewed):
    X = df[cols].copy()
    if log_skewed:
        for c in SKEWED:
            if c in X:
                X[c] = np.log1p(X[c])
    return X.to_numpy(dtype=float)


def tuned(kind):
    from sklearn.model_selection import GridSearchCV, KFold
    if kind == "Linear":
        return make("Linear")
    grid = {"model__alpha": ALPHAS if kind != "Ridge" else np.logspace(-3, 4, 36)}
    if kind == "ElasticNet":
        grid["model__l1_ratio"] = [0.2, 0.5, 0.8]
    return GridSearchCV(make(kind), grid, cv=KFold(5, shuffle=True, random_state=1), scoring="r2", n_jobs=1)


def nested_cv(X, y, kind):
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    scores = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = tuned(kind).fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def regularisation_plot(X, y, cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import validation_curve, KFold
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    cv = KFold(5, shuffle=True, random_state=0)
    best = {}
    for a, (kind, alphas) in zip(ax, [("Ridge", np.logspace(-3, 5, 30)), ("Lasso", np.logspace(-5, 0.5, 30)), ("ElasticNet", np.logspace(-5, 0.5, 30))]):
        est = make(kind, 1.0, 0.5)
        tr, va = validation_curve(est, X, y, param_name="model__alpha", param_range=alphas, cv=cv, scoring="neg_mean_squared_error")
        tr, va = -tr.mean(1), -va.mean(1)
        i = int(np.argmin(va)); best[kind] = float(alphas[i])
        a.semilogx(alphas, tr, "o-", ms=3, label="train MSE")
        a.semilogx(alphas, va, "s-", ms=3, label="5-fold validation MSE")
        a.axvline(alphas[i], c="k", ls=":"); a.annotate(f"best alpha = {alphas[i]:.2g}", (alphas[i], va[i]), xytext=(10, 25), textcoords="offset points", fontsize=8)
        a.set(xlabel="regularisation strength alpha", ylabel="MSE of log(JPEG bpp q10)", title=f"{kind}: bias-variance trade-off"); a.grid(alpha=.3); a.legend()
    # 4th panel: the same Ridge on only 40 images with degree-2 features (35 inputs) -> real overfitting at small alpha
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    rng, alphas = np.random.default_rng(0), np.logspace(-3, 3, 25)
    tr_m, va_m = np.zeros(len(alphas)), np.zeros(len(alphas))
    for _ in range(30):
        idx = rng.permutation(len(X)); a_, b_ = idx[:40], idx[40:]
        for k, al in enumerate(alphas):
            m = Pipeline([("s", StandardScaler()), ("p", PolynomialFeatures(2, include_bias=False)), ("s2", StandardScaler()), ("r", Ridge(alpha=al))]).fit(X[a_], y[a_])
            tr_m[k] += np.mean((m.predict(X[a_]) - y[a_]) ** 2) / 30; va_m[k] += np.mean((m.predict(X[b_]) - y[b_]) ** 2) / 30
    a = ax[3]
    a.semilogx(alphas, tr_m, "o-", ms=3, label="train MSE (40 images)"); a.semilogx(alphas, va_m, "s-", ms=3, label="validation MSE (rest)")
    i = int(np.argmin(va_m)); a.axvline(alphas[i], c="k", ls=":"); a.annotate(f"best alpha = {alphas[i]:.2g}", (alphas[i], va_m[i]), xytext=(10, 25), textcoords="offset points", fontsize=8)
    a.set(xlabel="regularisation strength alpha", ylabel="MSE", yscale="log", title="Ridge, tiny-sample demo: overfitting at small alpha"); a.grid(alpha=.3); a.legend()
    best["Ridge_small_sample_demo"] = float(alphas[i])
    fig.tight_layout(); fig.savefig(FIGS / "size_predictor_regularisation.png", dpi=120); plt.close(fig)

    # Lasso coefficient path
    al = np.logspace(-4, 0, 40)
    coefs = np.array([Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=a, max_iter=50000))]).fit(X, y).named_steps["m"].coef_ for a in al])
    fig, a = plt.subplots(figsize=(7.5, 4.6))
    for j, c in enumerate(cols):
        a.semilogx(al, coefs[:, j], label=c)
    a.set(xlabel="Lasso alpha", ylabel="standardised coefficient", title="Lasso path: which features survive"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "size_predictor_lasso_path.png", dpi=120); plt.close(fig)
    return best


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "model", "all"], default="all")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.stage in ("features", "all"):
        extract_all(workers=a.workers)
    if a.stage == "features":
        return
    from sklearn.metrics import r2_score
    OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    df = df[df.ok.astype(bool)]
    df["y"] = np.log(df["jpeg10_bpp"])
    df["log_pixels"] = np.log(df["w"] * df["h"])       # resolution: JPEG headers weigh more on small images
    fit = df[df.role.isin(["train", "val"])]
    tests = {r: g for r, g in df.groupby("role") if r.startswith("test_")}
    print(f"fit images (train+val): {len(fit)}; test sets: { {k: len(v) for k, v in tests.items()} }", flush=True)
    print(f"target log(JPEG q10 bpp): mean {df.y.mean():.2f}, std {df.y.std():.2f}", flush=True)

    res = {"n_fit": len(fit), "target": "log(jpeg10_bpp)", "variants": {}}
    variants = {"guide (7 raw features)": (FEATURES, False), "7 features, log1p on skewed": (FEATURES, True),
                "without jpeg50_bpp (6 features, log1p)": ([c for c in FEATURES if c != "jpeg50_bpp"], True),
                "7 features + log_pixels, log1p": (FEATURES + ["log_pixels"], True)}
    for vname, (cols, lg) in variants.items():
        X, y = design(fit, cols, lg), fit.y.to_numpy()
        v = {}
        for kind in ("Linear", "Ridge", "Lasso", "ElasticNet"):
            mu, sd = nested_cv(X, y, kind)
            final = tuned(kind).fit(X, y)
            held = {r: float(r2_score(g.y, final.predict(design(g, cols, lg)))) for r, g in tests.items()}
            rmse = {r: float(np.sqrt(np.mean((g.y - final.predict(design(g, cols, lg))) ** 2))) for r, g in tests.items()}
            v[kind] = {"cv_r2_mean": round(mu, 4), "cv_r2_std": round(sd, 4), "heldout_test_r2": {k: round(x, 4) for k, x in held.items()},
                       "heldout_rmse_log": {k: round(x, 4) for k, x in rmse.items()}}
            if kind != "Linear":
                est = final.best_estimator_
                v[kind]["best_params"] = {k.replace("model__", ""): float(x) for k, x in final.best_params_.items()}
            else:
                est = final
            v[kind]["coefficients"] = dict(zip(cols, np.round(est.named_steps["model"].coef_, 4).tolist()))
            print(f"[{vname}] {kind:<10} nested-CV R2 = {mu:.4f} +- {sd:.4f} | held-out R2: {v[kind]['heldout_test_r2']} RMSE: {v[kind]['heldout_rmse_log']}", flush=True)
        res["variants"][vname] = v
    cols, lg = FEATURES, False
    best_alpha = regularisation_plot(design(fit, cols, lg), fit.y.to_numpy(), cols)
    res["regularisation_plot_best_alpha"] = best_alpha
    best_name = max(res["variants"]["guide (7 raw features)"], key=lambda k: res["variants"]["guide (7 raw features)"][k]["cv_r2_mean"])
    res["best_guide_model"] = {"name": best_name, **{k: res["variants"]["guide (7 raw features)"][best_name][k] for k in ("cv_r2_mean", "cv_r2_std")}}
    res["target_r2_ge_0.6"] = res["best_guide_model"]["cv_r2_mean"] >= 0.6
    lasso = res["variants"]["guide (7 raw features)"]["Lasso"]["coefficients"]
    res["lasso_surviving_features"] = {k: v for k, v in lasso.items() if abs(v) > 1e-6}
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print("best (guide variant):", res["best_guide_model"], "| target >= 0.6:", res["target_r2_ge_0.6"])
    print("Lasso coefficients:", lasso)


if __name__ == "__main__":
    main()
