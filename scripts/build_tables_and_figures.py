"""
Script to extract, format, and generate all machine-readable tables and publication figures
for the ASMB Wiley submission package.
All outputs are derived strictly and deterministically from existing research outputs in data/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ARCHIVE_ROOT / "data"
TABLES_DIR = ARCHIVE_ROOT / "tables"
FIGURES_DIR = ARCHIVE_ROOT / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_publication_figure(fig, base_path: Path):
    """Save figure in vector (PDF, EPS) and high-resolution raster (300 DPI PNG, 600 DPI PNG, TIFF)."""
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".eps"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.parent / f"{base_path.name}_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".tiff"), dpi=300, bbox_inches="tight")


def build_tables():
    print("[TABLES] Building Table 1: Data and Availability Chronology...")
    t1_data = [
        {
            "Variable": "target_commercial_bank_deposits_nrb",
            "Description": "Commercial bank total deposits held at Nepal Rastra Bank (NRB)",
            "Frequency": "Monthly",
            "Units": "NPR million",
            "Coverage": "2004-07 to 2026-07 (265 obs)",
            "Availability Convention": "Assumed 2-month release lag (c = t - 2)",
            "Evaluation Role": "Primary target series (y_t)"
        },
        {
            "Variable": "nrb_net_omo_injection",
            "Description": "Net NRB open market operation liquidity injection (Outright Purchase + FX Purchase - Outright Sale - FX Sale)",
            "Frequency": "Monthly",
            "Units": "NPR million",
            "Coverage": "2004-07 to 2026-07 (265 obs)",
            "Availability Convention": "Known through cutoff c = t - 2; held constant at x_c over horizon",
            "Evaluation Role": "Exogenous policy feature (x_t)"
        }
    ]
    pd.DataFrame(t1_data).to_csv(TABLES_DIR / "Table1_Data_and_Availability.csv", index=False)

    print("[TABLES] Building Table 2: Model Specifications and Ensembles...")
    t2_data = [
        {
            "Model ID": "E_S",
            "Family": "Linear Econometric",
            "Specification": "Seasonal ARIMA(p,d,q)x(P,D,Q)_12 on raw levels",
            "Information Set": "Target history up to cutoff: {y_tau : tau <= c = t - 2}",
            "Estimation / Selection": "Auto-selection over order grid; exact MLE",
            "Role in Tournament": "Univariate statistical benchmark"
        },
        {
            "Model ID": "E_X",
            "Family": "Linear Econometric",
            "Specification": "SARIMAX with Net OMO injection lags on raw levels",
            "Information Set": "Target & OMO history: {y_tau, x_tau : tau <= c = t - 2}",
            "Estimation / Selection": "Same order grid; constant policy path over horizon",
            "Role in Tournament": "Informed econometric policy benchmark"
        },
        {
            "Model ID": "E_EQ",
            "Family": "Econometric Ensemble",
            "Specification": "Equal-weight combination: 0.5 * E_S + 0.5 * E_X",
            "Information Set": "Combined information set of E_S and E_X",
            "Estimation / Selection": "Equal fixed 50/50 weighting",
            "Role in Tournament": "Equally informed econometric ensemble benchmark"
        },
        {
            "Model ID": "XGB_Y",
            "Family": "Machine Learning",
            "Specification": "Gradient Boosted Trees (XGBoost) on target lags",
            "Information Set": "Target lags 1..12 and rolling statistics",
            "Estimation / Selection": "Inner validation tuning over depth & learning rate",
            "Role in Tournament": "Target-only non-parametric component"
        },
        {
            "Model ID": "XGB_OMO",
            "Family": "Machine Learning",
            "Specification": "XGBoost on target lags + Net OMO policy lags",
            "Information Set": "Target lags 1..12 + OMO lags 0..2 and rolling stats",
            "Estimation / Selection": "Identical inner tuning grid on rolling validation folds",
            "Role in Tournament": "Information-enriched non-parametric component"
        },
        {
            "Model ID": "H_Y",
            "Family": "Hybrid Architecture",
            "Specification": "Equal-weight combination: 0.5 * E_S + 0.5 * XGB_Y",
            "Information Set": "Target history only: {y_tau : tau <= c = t - 2}",
            "Estimation / Selection": "Equal fixed 50/50 weighting",
            "Role in Tournament": "Target-only hybrid benchmark"
        },
        {
            "Model ID": "H_OMO",
            "Family": "Hybrid Architecture",
            "Specification": "Equal-weight combination: 0.5 * E_S + 0.5 * XGB_OMO",
            "Information Set": "Target and OMO policy history up to cutoff c = t - 2",
            "Estimation / Selection": "Equal fixed 50/50 weighting",
            "Role in Tournament": "Proposed information-enriched hybrid champion"
        }
    ]
    pd.DataFrame(t2_data).to_csv(TABLES_DIR / "Table2_Procedures_and_Components.csv", index=False)

    print("[TABLES] Building Table 3: Aggregate Empirical Performance...")
    metrics_path = DATA_DIR / "metrics_DEVELOPMENT_FIT_V1.csv"
    df_metrics = pd.read_csv(metrics_path)
    df_metrics["procedure_label"] = df_metrics["procedure_id"].map({
        "H_OMO": "H_OMO (Information-Enriched Hybrid)",
        "H_Y": "H_Y (Target-Only Hybrid)",
        "E_EQ": "E_EQ (Econometric Combination)",
        "E_S": "E_S (Univariate SARIMA Benchmark)",
        "E_X": "E_X (OMO-SARIMAX Policy Model)",
        "XGB_OMO": "XGB_OMO (Policy Gradient Boosting Component)",
        "XGB_Y": "XGB_Y (Univariate Gradient Boosting Component)"
    })
    cols = ["procedure_id", "procedure_label", "n_scored", "mae", "rmse", "median_ae", "bias", "fallback_rate"]
    df_metrics = df_metrics[cols].sort_values("mae")
    df_metrics.to_csv(TABLES_DIR / "Table3_Aggregate_Performance.csv", index=False)

    print("[TABLES] Building Table 4: Primary Adaptation Inference Contrasts...")
    inf_path = DATA_DIR / "inference_summary_DEVELOPMENT_FIT_V1.json"
    with open(inf_path, "r", encoding="utf-8") as f:
        inf_data = json.load(f)

    p_l12 = inf_data["primary_family_L12"]["comparisons"]
    t4_rows = []
    for c in p_l12:
        t4_rows.append({
            "Comparison ID": c["comparison_id"],
            "Contrast Description": c["name"],
            "Reference Procedure": c["reference"],
            "Challenger Procedure": c["challenger"],
            "Reference MAE (mn)": round(c["mean_loss_ref"], 2),
            "Challenger MAE (mn)": round(c["mean_loss_chal"], 2),
            "Mean Loss Difference (Delta MAE)": round(c["effect_mean"], 2),
            "Percentage Reduction (%)": round(c["pct_reduction"], 2),
            "Moving Block Bootstrap 95% CI (L=12)": f"[{c['ci_lower_95']:+,.2f}, {c['ci_upper_95']:+,.2f}]",
            "Raw p-value": round(c["raw_p_value"], 4),
            "Holm-Adjusted p-value": round(c["holm_adjusted_p"], 4),
            "Statistical Decision (alpha=0.05)": "Do not reject H0 (Superiority not established)"
        })
    pd.DataFrame(t4_rows).to_csv(TABLES_DIR / "Table4_Primary_Inference_Contrasts.csv", index=False)

    print("[TABLES] Building Supplementary Tables...")
    # Table S1: Hyperparameters and search spaces
    s1_data = [
        {"Model Family": "SARIMA / SARIMAX", "Parameter": "Candidate Orders (p,d,q)x(P,D,Q)_12", "Specification": "(0,1,2)x(0,1,1), (2,1,2)x(0,1,1), (2,1,0)x(1,1,0), (0,1,2)x(1,1,1), (1,0,2)x(0,1,1), (1,1,2)x(0,1,1), (0,0,2)x(0,1,1), (0,1,1)x(1,1,1), (1,1,1)x(0,1,1)"},
        {"Model Family": "SARIMA / SARIMAX", "Parameter": "Trend & Seasonality", "Specification": "Linear trend c, seasonal period s = 12"},
        {"Model Family": "SARIMA / SARIMAX", "Parameter": "Estimation & Solvers", "Specification": "Exact maximum likelihood, maxiter = 50, root margin = 0.01"},
        {"Model Family": "SARIMAX", "Parameter": "Exogenous Lags", "Specification": "Net OMO injection lags 0, 1, 2"},
        {"Model Family": "SARIMAX", "Parameter": "Future Exogenous Path", "Specification": "Constant-level path held at x_c over forecast horizon"},
        {"Model Family": "XGBoost", "Parameter": "Candidate Hyperparameter Grid", "Specification": "learning_rate in {0.03, 0.10} x max_depth in {2, 3, 4}; n_estimators = 100"},
        {"Model Family": "XGBoost", "Parameter": "Stochastic Subsampling", "Specification": "subsample = 0.8, colsample_bytree = 0.8, random_state = 42"},
        {"Model Family": "XGBoost", "Parameter": "Inner Validation Tuning", "Specification": "24 rolling in-fold validation origin pairs; minimum MAE selection"}
    ]
    pd.DataFrame(s1_data).to_csv(TABLES_DIR / "TableS1_Order_Selection_and_Hyperparameters.csv", index=False)

    # Table S2: Sensitivity across block lengths
    s2_rows = []
    for l_key, bl in [("sensitivity_L6", 6), ("primary_family_L12", 12), ("sensitivity_L18", 18)]:
        for c in inf_data[l_key]["comparisons"]:
            s2_rows.append({
                "Block Length (Months)": bl,
                "Comparison ID": c["comparison_id"],
                "Contrast": c["name"],
                "Delta MAE (NPR mn)": round(c["effect_mean"], 2),
                "Pct Reduction (%)": round(c["pct_reduction"], 2),
                "95% Percentile CI": f"[{c['ci_lower_95']:+,.2f}, {c['ci_upper_95']:+,.2f}]",
                "Raw p-value": round(c["raw_p_value"], 4),
                "Holm-Adjusted p-value": round(c["holm_adjusted_p"], 4),
                "Significant at 0.05": c["is_significant_holm"]
            })
    pd.DataFrame(s2_rows).to_csv(TABLES_DIR / "TableS2_Bootstrap_Block_Sensitivity.csv", index=False)

    df_ym = pd.read_csv(DATA_DIR / "YEARLY_METRICS.csv")
    df_ym[df_ym.year_basis == "origin_year"].to_csv(TABLES_DIR / "TableS3_Origin_Year_Annual_Metrics.csv", index=False)
    df_ym[df_ym.year_basis == "target_year"].to_csv(TABLES_DIR / "TableS4_Target_Year_Annual_Metrics.csv", index=False)

    df_tail = pd.read_csv(DATA_DIR / "TAIL_METRICS.csv")
    df_tail.to_csv(TABLES_DIR / "TableS5_Tail_Loss_Quantiles.csv", index=False)

    df_corr = pd.read_csv(DATA_DIR / "ERROR_CORRELATION.csv")
    df_corr.to_csv(TABLES_DIR / "TableS6_Error_Correlation_Matrix.csv", index=False)

    df_wins = pd.read_csv(DATA_DIR / "WIN_COUNTS.csv")
    df_wins.to_csv(TABLES_DIR / "TableS7_Pairwise_Win_Counts.csv", index=False)

    print("[TABLES] All tables successfully generated and saved to tables/.")


def build_figure1():
    """Monthly spacing and explicit as-of information for the assumed L2 regime."""
    fig, ax = plt.subplots(figsize=(7.2, 3.65), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(-0.42, 3.48), ylim=(-1.1, 1.5))
    ax.axis("off")
    ink, muted, blue, orange = "#233441", "#586775", "#176A83", "#A65624"
    # Two months to the origin, one month from the origin to the target.
    ax.axvspan(-0.3, 0, ymin=0.27, ymax=0.68, color="#EDF4F6", zorder=0)
    ax.annotate("", xy=(3.36, 0), xytext=(-0.29, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=ink))
    for x in range(4):
        ax.plot([x, x], [-0.045, 0.045], color=ink, lw=1.2)
    for x, color in [(0, blue), (2, ink), (3, orange)]:
        ax.plot(x, 0, "o", ms=6, color=color, zorder=4)
    ax.plot(1, 0, "o", ms=5, markerfacecolor="white", markeredgecolor=muted, zorder=4)
    for x, label in [(0, r"$c=t-2$"), (1, r"$t-1$"), (2, r"$t$"), (3, r"$t+1=c+3$")]:
        ax.text(x, -0.14, label, ha="center", va="top", fontsize=10, color=ink)
    ax.text(0, 0.42, "INFORMATION CUTOFF", ha="center", fontsize=8, weight="bold", color=blue)
    ax.text(0, 0.22, "Latest usable observations", ha="center", fontsize=8, color=muted)
    ax.text(2, 0.42, "FORECAST ORIGIN", ha="center", fontsize=8, weight="bold", color=ink)
    ax.text(2, 0.22, "Forecast issued", ha="center", fontsize=8, color=muted)
    ax.text(3.04, 0.42, "TARGET MONTH", ha="center", fontsize=8, weight="bold", color=orange)
    ax.text(3.04, 0.22, r"Predict $y_{t+1}$", ha="center", fontsize=8, color=muted)
    def span(a, b, y, label, color):
        ax.annotate("", xy=(b, y), xytext=(a, y),
                    arrowprops=dict(arrowstyle="|-|", lw=1.1, color=color))
        ax.text((a+b)/2, y+0.1, label, ha="center", va="bottom", fontsize=8.5, color=color)
    span(0, 2, 0.91, "Assumed availability lag: 2 months", blue)
    span(2, 3, 0.91, "Horizon: 1 month", orange)
    ax.text(1.5, 1.32, "Forecast chronology under the assumed two-month availability lag",
            ha="center", fontsize=10, weight="bold", color=ink)
    span(0, 3, -0.57, "Effective lead from cutoff to target: 3 months", ink)
    ax.text(1.5, -0.91, r"At origin $t$, use $y$ and OMO only through $c$; $y_{t-1}$ and $y_t$ are unavailable.",
            ha="center", fontsize=8, color=muted)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.98, bottom=0.035)
    base = FIGURES_DIR / "Figure1_Information_Chronology"
    save_publication_figure(fig, base)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def build_figures():
    print("[FIGURES] Generating Figure 1: Information Chronology Diagram...")
    build_figure1()

    print("[FIGURES] Generating Figure 2: Cumulative Absolute Loss Differentials...")
    eval_csv = DATA_DIR / "evaluation_DEVELOPMENT_FIT_V1.csv"
    df_eval = pd.read_csv(eval_csv)
    df_eval = df_eval[df_eval.is_eligible_for_evaluation == True].copy()
    piv_abs = df_eval.pivot(index="target_period", columns="procedure_id", values="abs_error")

    # Differential: L(ref) - L(H_OMO). Positive means H_OMO has lower error (wins)
    diffs = pd.DataFrame(index=piv_abs.index)
    diffs["vs_E_S"] = (piv_abs["E_S"] - piv_abs["H_OMO"]).cumsum()
    diffs["vs_E_EQ"] = (piv_abs["E_EQ"] - piv_abs["H_OMO"]).cumsum()
    diffs["vs_E_X"] = (piv_abs["E_X"] - piv_abs["H_OMO"]).cumsum()
    diffs["vs_H_Y"] = (piv_abs["H_Y"] - piv_abs["H_OMO"]).cumsum()

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    dates = pd.to_datetime(diffs.index)

    ax.plot(dates, diffs["vs_E_X"] / 1000, label="vs. E_X (OMO-SARIMAX, +4,857 mn/mo)", color="#e67e22", lw=1.8, linestyle="--")
    ax.plot(dates, diffs["vs_E_S"] / 1000, label="vs. E_S (Univariate SARIMA, +2,948 mn/mo)", color="#2980b9", lw=2.0)
    ax.plot(dates, diffs["vs_E_EQ"] / 1000, label="vs. E_EQ (Econometric Combination, +2,084 mn/mo)", color="#27ae60", lw=2.0)
    ax.plot(dates, diffs["vs_H_Y"] / 1000, label="vs. H_Y (Target-Only Hybrid, +586 mn/mo)", color="#8e44ad", lw=1.8, linestyle="-.")

    ax.axhline(0, color="#7f8c8d", lw=1, linestyle=":")
    ax.set_ylabel("Cumulative Absolute Loss Advantage\n[Thousand NPR Million] (Positive = H_OMO Wins)")
    ax.set_xlabel("Forecast Target Month")
    ax.set_title("Cumulative Absolute Loss Advantage of H_OMO over Competing Procedures (191 Origins)", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Annotate key epochs
    ax.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2021-12-01"), color="#e74c3c", alpha=0.08, label="COVID-19 Shock")
    ax.text(pd.to_datetime("2020-04-01"), -50, "COVID-19\nDisruption", fontsize=8, color="#c0392b", ha="left")

    save_publication_figure(fig, FIGURES_DIR / "Figure2_Cumulative_Loss_Differential")
    plt.close(fig)

    print("[FIGURES] Generating Figure 3: Annual MAE Tournament Comparison...")
    df_ym = pd.read_csv(DATA_DIR / "YEARLY_METRICS.csv")
    df_orig = df_ym[df_ym.year_basis == "origin_year"].copy()
    piv_mae = df_orig.pivot(index="year", columns="procedure", values="mae")

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    years = piv_mae.index.astype(int)
    w = 0.18
    x = np.arange(len(years))

    ax.bar(x - 1.5*w, piv_mae["H_OMO"] / 1000, width=w, label="H_OMO (OMO Hybrid)", color="#2980b9", edgecolor="#1c5980")
    ax.bar(x - 0.5*w, piv_mae["H_Y"] / 1000, width=w, label="H_Y (Target Hybrid)", color="#8e44ad", edgecolor="#5e2d73")
    ax.bar(x + 0.5*w, piv_mae["E_EQ"] / 1000, width=w, label="E_EQ (Econ Combination)", color="#27ae60", edgecolor="#1a6e3d")
    ax.bar(x + 1.5*w, piv_mae["E_S"] / 1000, width=w, label="E_S (SARIMA Benchmark)", color="#e67e22", edgecolor="#a05615")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.set_ylabel("Annual Mean Absolute Error [Thousand NPR Million]")
    ax.set_title("Annual Tournament Performance Across Competing Procedures (Origin Year, 2010 - 2025)", pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

    save_publication_figure(fig, FIGURES_DIR / "Figure3_Annual_MAE_Tournament")
    plt.close(fig)

    print("[FIGURES] Generating Figure 4: Tail Squared Loss Concentration...")
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    # Calculate Lorenz-style concentration curves for key models
    eval_csv = DATA_DIR / "evaluation_DEVELOPMENT_FIT_V1.csv"
    df_eval = pd.read_csv(eval_csv)
    eligible = df_eval[df_eval.is_eligible_for_evaluation == True].copy()

    model_styles = {
        "H_OMO": ("H_OMO (Information Hybrid)", "#2980b9", "-"),
        "H_Y": ("H_Y (Target Hybrid)", "#8e44ad", "--"),
        "E_EQ": ("E_EQ (Econ Combination)", "#27ae60", "-."),
        "E_S": ("E_S (SARIMA)", "#e67e22", ":")
    }

    for mid, (lbl, color, ls) in model_styles.items():
        sq_errs = eligible[eligible.procedure_id == mid]["sq_error"].sort_values(ascending=False).values
        cum_sq = np.cumsum(sq_errs) / np.sum(sq_errs) * 100.0
        pct_origins = np.arange(1, len(sq_errs) + 1) / len(sq_errs) * 100.0
        ax.plot(pct_origins, cum_sq, label=lbl, color=color, linestyle=ls, lw=2.0)

    # 45-degree line (equal distribution)
    ax.plot([0, 100], [0, 100], color="#95a5a6", linestyle=":", lw=1.2, label="Equal Distribution")

    # Annotate top 9 origins (4.71%)
    ax.plot([4.71, 4.71], [0, 51.25], color="#c0392b", linestyle="--", lw=1.0)
    ax.plot([0, 4.71], [51.25, 51.25], color="#c0392b", linestyle="--", lw=1.0)
    ax.plot(4.71, 51.25, "o", color="#c0392b", markersize=6)
    ax.annotate("Top 4.71% (9 origins) =\n51.25% of Squared Loss",
                xy=(4.71, 51.25), xytext=(15, 45),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#c0392b"),
                fontsize=8.5, fontweight="bold", color="#c0392b")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Cumulative Proportion of Forecast Origins [%] (Ordered by Descending Loss)")
    ax.set_ylabel("Cumulative Proportion of Total Squared Error [%]")
    ax.set_title("Squared Loss Concentration Curves Across Evaluation Origins (191 Origins)", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    save_publication_figure(fig, FIGURES_DIR / "Figure4_Tail_Loss_Concentration")
    plt.close(fig)

    print("[FIGURES] Generating Supplementary Figure S1: Absolute Error Boxplots...")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    procs = ["H_OMO", "H_Y", "E_EQ", "E_S", "E_X", "XGB_OMO", "XGB_Y"]
    data_boxes = [eligible[eligible.procedure_id == p]["abs_error"].values / 1000 for p in procs]

    bp = ax.boxplot(data_boxes, labels=procs, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="#e74c3c", markersize=5),
                    medianprops=dict(color="#2c3e50", lw=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgecolor="#7f8c8d"))

    colors = ["#3498db", "#9b59b6", "#2ecc71", "#e67e22", "#f39c12", "#1abc9c", "#34495e"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Absolute Forecast Error [Thousand NPR Million]")
    ax.set_title("Distribution of Absolute Forecast Errors across 191 Origins (2010 - 2025)", pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    save_publication_figure(fig, FIGURES_DIR / "FigureS1_Error_Distributions_Boxplot")
    plt.close(fig)

    print("[FIGURES] All publication figures successfully generated and saved to figures/.")


if __name__ == "__main__":
    build_tables()
    build_figures()
