"""
Statistical analysis, tables and figures for the larval behaviour results.

Reads the files written by larval_tracker.py:
    results/larva_summary.csv
    results/larva_per_minute.csv

Writes to analysis/:
    Table1_descriptive.csv          mean, SD, SE and n per treatment for every endpoint
    Table2_mixed_model_tests.csv    treatment effect per endpoint (likelihood-ratio test)
    Table3_planned_contrasts.csv    planned contrasts against each vehicle, Holm-adjusted
                                    (Table3_pairwise_tukey.csv if COMPARISONS = "tukey")
    Tables.xlsx                     the three tables in one workbook (if openpyxl is installed)
    Fig_box_<endpoint>              box plot (median, IQR, 1.5 x IQR whiskers) + every larva
    Fig_bar_<endpoint>              mean +/- SE bar chart + every larva
    Fig_panel_box / Fig_panel_bar   six main endpoints, panels A-F, full-page width
    Fig_time_course                 per-minute distance, activity and thigmotaxis
Each figure is saved as PNG and LZW-compressed TIFF (600 dpi) and as vector PDF.

Model, for each endpoint:
    endpoint ~ treatment + (1 | replicate tank)
The two larvae from the same replicate tank share a random intercept, so the
replicate tank is the experimental unit. With COMPARISONS = "planned" (default), each
exposure is compared with its vehicle and the figures show brackets with asterisks for
significant contrasts (* p < 0.05, ** p < 0.01, *** p < 0.001, Holm-adjusted).
With COMPARISONS = "tukey", all pairs are tested and the figures show compact letters.

Requirements:  pip install pandas numpy scipy statsmodels matplotlib openpyxl
Run:           python larval_analysis.py
"""

import itertools
import os
import warnings

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
SUMMARY_FILE = os.path.join("results", "larva_summary.csv")
PER_MINUTE_FILE = os.path.join("results", "larva_per_minute.csv")
OUTPUT_FOLDER = "analysis"

# Treatments are drawn in this order; names must match the video file names.
TREATMENT_ORDER = ["Control", "DMSO", "Tween80", "MP", "TCS", "MP+TCS"]
TREATMENT_ALIASES = {"TCS+MP": "MP+TCS"}   # other spellings mapped to the names above

# Comparisons. "planned" tests only the contrasts listed below; "tukey" tests every pair
# and labels the figures with compact letters.
COMPARISONS = "planned"
# (treatment, reference). Each exposure is compared with the vehicle it was delivered in:
# TCS in DMSO, PE-MP suspended with Tween80, MP+TCS containing both vehicles.
PLANNED_CONTRASTS = [
    ("DMSO", "Control"),      # solvent check
    ("Tween80", "Control"),   # solvent (MP suspension agent) check
    ("TCS", "DMSO"),          # TCS against its vehicle
    ("MP", "Tween80"),        # PE-MP against its vehicle
    ("MP+TCS", "TCS"),        # does adding MP change the TCS effect?
    ("MP+TCS", "MP"),         # does adding TCS change the MP effect?
    ("MP+TCS", "Control"),    # combined exposure against untreated larvae
]
P_ADJUST = "holm"             # multiplicity correction across the planned contrasts of each endpoint
SHOW_NS_BRACKETS = False      # True = also draw brackets for non-significant contrasts

# Box/bar fill colours: greys for the control and solvent groups, colour for exposures.
# The palette stays distinguishable in greyscale print and for colour-blind readers.
TREATMENT_COLOURS = {"Control": "#d9d9d9", "DMSO": "#bdbdbd", "Tween80": "#969696",
                     "MP": "#92c5de", "TCS": "#f4a582", "MP+TCS": "#b2abd2", "TCS+MP": "#b2abd2"}
# Darker versions for lines and markers in the time-course figure.
TREATMENT_LINE_COLOURS = {"Control": "#252525", "DMSO": "#636363", "Tween80": "#969696",
                          "MP": "#2166ac", "TCS": "#b2182b", "MP+TCS": "#542788", "TCS+MP": "#542788"}
TREATMENT_MARKERS = {"Control": "o", "DMSO": "s", "Tween80": "D", "MP": "^",
                     "TCS": "v", "MP+TCS": "P", "TCS+MP": "P"}

# endpoint column -> axis label. The first six go into the panel figures.
ENDPOINTS = {
    "total_distance_mm": "Total distance (mm)",
    "mean_speed_mm_s": "Mean speed (mm s$^{-1}$)",
    "pct_time_active": "Time active (%)",
    "n_bouts": "Movement bouts (n)",
    "pct_time_outer_zone": "Time in outer zone (%)",
    "mean_speed_when_active_mm_s": "Speed when active (mm s$^{-1}$)",
    "max_speed_mm_s": "Maximum speed (mm s$^{-1}$)",
}
PANEL_ENDPOINTS = list(ENDPOINTS)[:6]
ALPHA = 0.05

# Figure appearance
SINGLE_COLUMN_MM = 85        # typical single-column width (Elsevier, Springer, Wiley)
DOUBLE_COLUMN_MM = 174       # typical full-page width
FIG_DPI = 600                # raster resolution for PNG and TIFF
SAVE_FORMATS = ("png", "tif", "pdf")
POINT_COLOUR = "#252525"
POINT_ALPHA = 0.45           # transparency of the individual-larva points
POINT_SIZE = 9
LETTER_POSITION = "top"     # "top" = one aligned row at the top of each panel; "bar" = above each bar
LETTER_GAP_PT = 1.5          # gap above each box/bar when LETTER_POSITION = "bar", in points
HIDE_LETTERS_IF_ALL_SAME = False  # True = no letters on a panel where no pair differs (all "a")
X_LABEL_ROTATION = 35
# --------------------------------------------------------------------------

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": FIG_DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load_summary():
    df = pd.read_csv(SUMMARY_FILE)
    df["treatment"] = df["treatment"].replace(TREATMENT_ALIASES)
    present = [t for t in TREATMENT_ORDER if t in set(df["treatment"])]
    extra = sorted(set(df["treatment"]) - set(TREATMENT_ORDER))
    if extra:
        print(f"Note: treatments not in TREATMENT_ORDER are added at the end: {extra}")
    order = present + extra
    df["treatment"] = pd.Categorical(df["treatment"], categories=order, ordered=True)
    # R1 in Control and R1 in TCS are different tanks, so each tank gets its own ID
    df["tank"] = df["treatment"].astype(str) + "_" + df["replicate"].astype(str)
    return df, order


# ------------------------------------------------------------------ tables
def descriptive_table(df, order):
    rows = []
    for ep, label in ENDPOINTS.items():
        for t in order:
            v = df.loc[df["treatment"] == t, ep].dropna()
            tank_means = df.loc[df["treatment"] == t].groupby("tank")[ep].mean()
            rows.append({
                "endpoint": label.replace("$^{-1}$", "-1"), "treatment": t,
                "n_larvae": len(v), "n_replicates": tank_means.notna().sum(),
                "mean": v.mean(), "SD": v.std(ddof=1), "SE": v.std(ddof=1) / np.sqrt(len(v)),
                "median": v.median(), "min": v.min(), "max": v.max(),
                "mean_SE": f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}",
            })
    return pd.DataFrame(rows)


class Fit:
    """Common view of a mixed-model fit, or of the OLS fallback used when it fails."""

    def __init__(self, fe, cov, llf, resid, tank_var, resid_var, kind):
        self.fe_params, self.cov, self.llf, self.resid = fe, cov, llf, resid
        self.tank_var, self.resid_var, self.kind = tank_var, resid_var, kind


def fit_mixed(df, ep, reml=True, formula="y ~ C(treatment)"):
    """Fit y ~ treatment + (1 | tank). If the tank variance is estimated at zero the
    mixed model can fail to converge; it then reduces to an ordinary linear model,
    which is fitted instead and flagged in the output (model = 'OLS (tank var = 0)')."""
    data = df[[ep, "treatment", "tank"]].dropna().copy()
    data["treatment"] = data["treatment"].cat.remove_unused_categories()
    data = data.rename(columns={ep: "y"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "powell", "nm", "bfgs"):
            try:
                res = smf.mixedlm(formula, data, groups=data["tank"]).fit(reml=reml, method=method)
                fe = res.fe_params
                cov = res.cov_params().loc[fe.index, fe.index]
                if np.all(np.isfinite(cov.values)) and np.all(np.diag(cov.values) > 0):
                    return Fit(fe, cov, res.llf, res.resid, float(res.cov_re.iloc[0, 0]),
                               float(res.scale), "Mixed model"), data
            except Exception:
                continue
        ols = smf.ols(formula, data).fit()
    resid_var = ols.mse_resid if reml else ols.ssr / ols.nobs
    return Fit(ols.params, ols.cov_params(), ols.llf, ols.resid, 0.0, float(resid_var),
               "OLS (tank var = 0)"), data


def treatment_test(df, ep):
    """Likelihood-ratio test of treatment (ML fits), plus model diagnostics."""
    full, data = fit_mixed(df, ep, reml=False)
    null, _ = fit_mixed(df, ep, reml=False, formula="y ~ 1")
    lr = max(2 * (full.llf - null.llf), 0.0)
    df_lr = len(full.fe_params) - 1
    p = stats.chi2.sf(lr, df_lr)
    reml_fit, _ = fit_mixed(df, ep, reml=True)
    tv, rv = reml_fit.tank_var, reml_fit.resid_var
    shapiro_p = stats.shapiro(reml_fit.resid)[1] if len(reml_fit.resid) >= 3 else np.nan
    return {
        "endpoint": ENDPOINTS[ep].replace("$^{-1}$", "-1"),
        "model": reml_fit.kind,
        "LR_chi2": lr, "df": df_lr, "p_value": p,
        "significance": stars(p),
        "tank_variance": tv, "residual_variance": rv,
        "ICC_tank": tv / (tv + rv) if (tv + rv) > 0 else np.nan,
        "residual_Shapiro_p": shapiro_p,
        "n_larvae": len(data), "n_tanks": data["tank"].nunique(),
    }


def contrast_tools(df, ep):
    """REML fit plus a helper giving the model mean vector of any treatment."""
    res, data = fit_mixed(df, ep, reml=True)
    levels = list(data["treatment"].cat.categories)
    fe = res.fe_params
    cov = res.cov.loc[fe.index, fe.index]
    names = list(fe.index)

    def vec(level):  # contrast vector giving the model mean of one treatment
        v = np.zeros(len(names))
        v[names.index("Intercept")] = 1
        term = f"C(treatment)[T.{level}]"
        if term in names:
            v[names.index(term)] = 1
        return v

    return levels, fe, cov, vec


def planned_comparisons(df, ep):
    """Planned contrasts from the REML mixed model, Holm-adjusted within each endpoint."""
    levels, fe, cov, vec = contrast_tools(df, ep)
    rows = []
    for g, ref in PLANNED_CONTRASTS:
        if g not in levels or ref not in levels:
            print(f"Note: contrast {g} vs {ref} skipped ({ep}): treatment not in the data.")
            continue
        c = vec(g) - vec(ref)
        est = float(c @ fe.values)
        se = float(np.sqrt(c @ cov.values @ c))
        z = est / se if se > 0 else np.nan
        ref_mean = float(vec(ref) @ fe.values)
        rows.append({
            "endpoint": ENDPOINTS[ep].replace("$^{-1}$", "-1"),
            "contrast": f"{g} vs {ref}", "group": g, "reference": ref,
            "difference": est, "SE": se,
            "CI95_low": est - 1.96 * se, "CI95_high": est + 1.96 * se,
            "pct_change": 100 * est / ref_mean if ref_mean else np.nan,
            "z": z, "p_raw": 2 * stats.norm.sf(abs(z)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj"] = multipletests(out["p_raw"], method=P_ADJUST)[1]
        out["significance"] = out["p_adj"].apply(stars)
    return out


def pairwise_tukey(df, ep, order):
    """All pairwise treatment differences from the REML mixed model, Tukey-adjusted."""
    levels, fe, cov, vec = contrast_tools(df, ep)
    k = len(levels)
    rows = []
    for a, b in itertools.combinations(levels, 2):
        c = vec(b) - vec(a)
        est = float(c @ fe.values)
        se = float(np.sqrt(c @ cov.values @ c))
        z = est / se if se > 0 else np.nan
        # Tukey adjustment: studentized range with the asymptotic (large df) reference
        p_adj = stats.studentized_range.sf(abs(z) * np.sqrt(2), k, 1e4) if se > 0 else np.nan
        rows.append({
            "endpoint": ENDPOINTS[ep].replace("$^{-1}$", "-1"),
            "comparison": f"{b} - {a}", "group_1": b, "group_2": a,
            "difference": est, "SE": se,
            "CI95_low": est - 1.96 * se, "CI95_high": est + 1.96 * se,
            "z": z, "p_tukey": min(p_adj, 1.0), "significance": stars(p_adj),
        })
    out = pd.DataFrame(rows)
    model_means = {lv: float(vec(lv) @ fe.values) for lv in levels}
    letters = compact_letters(levels, out, model_means)
    return out, letters


def compact_letters(levels, pairs, means):
    """Compact letter display: treatments that share a letter are not significantly different."""
    sig = {frozenset((r.group_1, r.group_2)) for r in pairs.itertuples() if r.p_tukey < ALPHA}
    groups = [set(levels)]
    for pair in sig:
        a, b = tuple(pair)
        new = []
        for g in groups:
            if a in g and b in g:
                new += [g - {a}, g - {b}]
            else:
                new.append(g)
        # drop groups contained in another group
        new = [g for g in new if g and not any(g < h for h in new)]
        groups = list({frozenset(g) for g in new})
        groups = [set(g) for g in groups]
    # order letters by the first treatment each group contains
    # conventional ordering: "a" goes to the group with the highest mean
    groups.sort(key=lambda g: -max(means[t] for t in g))
    letters = {t: "" for t in levels}
    for i, g in enumerate(groups):
        for t in g:
            letters[t] += "abcdefghijklmnopqrstuvwxyz"[i]
    return letters


def stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ----------------------------------------------------------------- figures
def jitter_points(ax, df, ep, order):
    rng = np.random.default_rng(1)
    for i, t in enumerate(order):
        v = df.loc[df["treatment"] == t, ep].dropna().values
        if len(v):
            x = i + rng.uniform(-0.13, 0.13, len(v))
            ax.scatter(x, v, s=POINT_SIZE, color=POINT_COLOUR, alpha=POINT_ALPHA,
                       linewidths=0, zorder=3)


def add_brackets(ax, order, contrasts, tops):
    """Draw a bracket with asterisks for each planned contrast (significant ones only,
    unless SHOW_NS_BRACKETS). Brackets are stacked without overlapping, and the spacing
    is set in points so the asterisks never touch the bracket above, whatever the panel size."""
    rows = [r for r in contrasts.itertuples()
            if (r.p_adj < ALPHA or SHOW_NS_BRACKETS) and r.group in order and r.reference in order]
    if not rows:
        return
    spans = sorted(((sorted((order.index(r.group), order.index(r.reference))), r.significance)
                    for r in rows), key=lambda s: (s[0][1] - s[0][0], s[0][0]))
    levels_used, placed = [], []
    for (x1, x2), label in spans:
        level = 0
        while level < len(levels_used) and any(not (x2 < a or x1 > b) for a, b in levels_used[level]):
            level += 1
        if level == len(levels_used):
            levels_used.append([])
        levels_used[level].append((x1, x2))
        placed.append((x1, x2, label, level))

    # vertical layout in points, converted to axis fractions
    height_pt = ax.get_window_extent().height / ax.figure.dpi * 72
    level_f, pad_f, tick_f, top_f = 12 / height_pt, 4 / height_pt, 2.5 / height_pt, 9 / height_pt
    n_levels = len(levels_used)
    data_f = max(0.45, 1 - pad_f - (n_levels - 1) * level_f - top_f)
    lo = ax.get_ylim()[0]
    total = (max(tops) - lo) / data_f
    ax.set_ylim(lo, lo + total)
    for x1, x2, label, level in placed:
        y = lo + total * (data_f + pad_f + level * level_f)
        t = total * tick_f
        ax.plot([x1, x1, x2, x2], [y - t, y, y, y - t], color="black", linewidth=0.6, clip_on=False)
        ax.annotate(label, xy=((x1 + x2) / 2, y), xytext=(0, -1.5 if label != "ns" else 0.5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8 if label != "ns" else 6.5)


def add_letters(ax, order, letters, tops):
    """Write the Tukey letters, either in one aligned row at the top of the panel
    (LETTER_POSITION = "top") or just above each box or bar ("bar").
    A DataFrame of planned contrasts is drawn as brackets instead."""
    if isinstance(letters, pd.DataFrame):
        add_brackets(ax, order, letters, tops)
        return
    shown = [letters.get(t, "") for t in order]
    if HIDE_LETTERS_IF_ALL_SAME and len(set(shown)) <= 1:
        return
    if LETTER_POSITION == "top":
        lo, hi = ax.get_ylim()
        data_top = max(max(tops), hi if hi < max(tops) else max(tops))
        # leave a band above the highest point for the letter row
        ax.set_ylim(lo, data_top + 0.14 * (data_top - lo))
        row = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for i, lab in enumerate(shown):
            ax.text(i, 0.985, lab, transform=row, ha="center", va="top", fontsize=7.5)
        return
    for i, (lab, top) in enumerate(zip(shown, tops)):
        ax.annotate(lab, xy=(i, top), xytext=(0, LETTER_GAP_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, max(tops) + 0.09 * (max(tops) - lo)))


def draw_box(ax, df, ep, order, letters):
    data = [df.loc[df["treatment"] == t, ep].dropna().values for t in order]
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.6, patch_artist=True,
                    showfliers=False, whis=1.5,
                    boxprops=dict(linewidth=0.6, edgecolor="black"),
                    whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
                    medianprops=dict(color="black", linewidth=1.0))
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(TREATMENT_COLOURS.get(t, "#d9d9d9"))
    jitter_points(ax, df, ep, order)
    finish_axis(ax, ep, order)
    tops = [np.max(d) if len(d) else 0 for d in data]
    add_letters(ax, order, letters, tops)


def draw_bar(ax, df, ep, order, letters):
    means = [df.loc[df["treatment"] == t, ep].mean() for t in order]
    ses = [df.loc[df["treatment"] == t, ep].sem() for t in order]
    ax.bar(range(len(order)), means, width=0.62,
           color=[TREATMENT_COLOURS.get(t, "#d9d9d9") for t in order],
           edgecolor="black", linewidth=0.6, zorder=1)
    ax.errorbar(range(len(order)), means, yerr=ses, fmt="none", ecolor="black",
                elinewidth=0.6, capsize=2.5, capthick=0.6, zorder=2)
    jitter_points(ax, df, ep, order)
    finish_axis(ax, ep, order)
    tops = [np.nanmax([m + s, df.loc[df["treatment"] == t, ep].max()])
            for m, s, t in zip(means, ses, order)]
    add_letters(ax, order, letters, tops)


def finish_axis(ax, ep, order):
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=X_LABEL_ROTATION, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(ENDPOINTS[ep])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


def save(fig, name):
    for ext in SAVE_FORMATS:
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(OUTPUT_FOLDER, f"{name}.{ext}"), **kw)
    plt.close(fig)


def time_course(order):
    if not os.path.exists(PER_MINUTE_FILE):
        return
    pm = pd.read_csv(PER_MINUTE_FILE)
    pm["treatment"] = pm["treatment"].replace(TREATMENT_ALIASES)
    eps = ["total_distance_mm", "pct_time_active", "pct_time_outer_zone"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_MM * MM, 52 * MM))
    offsets = np.linspace(-0.12, 0.12, len(order))   # small dodge so error bars don't overlap
    for k, (ax, ep) in enumerate(zip(axes, eps)):
        for t, dx in zip(order, offsets):
            sub = pm[pm["treatment"] == t].groupby("bin")[ep]
            m, se = sub.mean(), sub.sem()
            ax.errorbar(m.index + dx, m.values, yerr=se.values,
                        marker=TREATMENT_MARKERS.get(t, "o"), markersize=3.2,
                        markerfacecolor="white", markeredgewidth=0.7, capsize=1.5,
                        elinewidth=0.6, linewidth=0.8,
                        color=TREATMENT_LINE_COLOURS.get(t, "black"), label=t)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ENDPOINTS[ep].replace("Total distance", "Distance per minute"))
        ax.set_xticks(sorted(pm["bin"].unique()))
        ax.set_ylim(bottom=0)
        ax.text(-0.3, 1.04, "ABC"[k], transform=ax.transAxes, fontsize=9, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "Fig_time_course")


def write_excel(sheets):
    """Write the tables to one workbook. A failure here never stops the analysis,
    because the same tables are already saved as CSV files."""
    path = os.path.join(OUTPUT_FOLDER, "Tables.xlsx")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("openpyxl not installed: Tables.xlsx skipped (CSV tables were written).")
        return
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, table in sheets.items():
                table.to_excel(xw, sheet_name=name, index=False)
    except PermissionError:
        print("Could not write Tables.xlsx: close it in Excel and run again (CSV tables were written).")
    except Exception as e:
        print(f"Could not write Tables.xlsx ({e}). The CSV tables were written.")


# -------------------------------------------------------------------- main
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df, order = load_summary()
    print(f"Loaded {len(df)} larvae from {df['tank'].nunique()} replicate tanks, "
          f"treatments: {', '.join(order)}")

    table1 = descriptive_table(df, order)
    table2, pair_tables, all_letters = [], [], {}
    for ep in ENDPOINTS:
        if ep not in df.columns:
            continue
        table2.append(treatment_test(df, ep))
        if COMPARISONS == "planned":
            pairs = planned_comparisons(df, ep)
            all_letters[ep] = pairs            # drawn as brackets
        else:
            pairs, letters = pairwise_tukey(df, ep, order)
            all_letters[ep] = letters
        pair_tables.append(pairs)
    table2 = pd.DataFrame(table2)
    table3 = pd.concat(pair_tables, ignore_index=True)

    if COMPARISONS != "planned":
        table1["letters"] = [all_letters[ep].get(t, "") for ep in ENDPOINTS for t in order]

    table1.round(3).to_csv(os.path.join(OUTPUT_FOLDER, "Table1_descriptive.csv"), index=False)
    table2.round(4).to_csv(os.path.join(OUTPUT_FOLDER, "Table2_mixed_model_tests.csv"), index=False)
    t3_name = "Table3_planned_contrasts.csv" if COMPARISONS == "planned" else "Table3_pairwise_tukey.csv"
    table3.round(4).to_csv(os.path.join(OUTPUT_FOLDER, t3_name), index=False)
    write_excel({"Descriptive": table1.round(3),
                 "Treatment effect": table2.round(4),
                 ("Planned contrasts" if COMPARISONS == "planned" else "Pairwise Tukey"): table3.round(4)})

    # single-endpoint figures (single-column width)
    for ep in ENDPOINTS:
        for style, draw in (("box", draw_box), ("bar", draw_bar)):
            fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_MM * MM * 0.62, 62 * MM))
            draw(ax, df, ep, order, all_letters[ep])
            fig.tight_layout()
            save(fig, f"Fig_{style}_{ep}")

    # multi-panel figures (full-page width)
    for style, draw in (("box", draw_box), ("bar", draw_bar)):
        fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_MM * MM, 112 * MM))
        for i, (ax, ep) in enumerate(zip(axes.flat, PANEL_ENDPOINTS)):
            draw(ax, df, ep, order, all_letters[ep])
            ax.text(-0.3, 1.04, "ABCDEF"[i], transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.2, h_pad=1.0)
        fig.align_ylabels(axes[:, 0])
        save(fig, f"Fig_panel_{style}")

    time_course(order)

    print("\nTreatment effect (mixed model, likelihood-ratio test):")
    print(table2[["endpoint", "model", "LR_chi2", "df", "p_value", "significance"]].round(4).to_string(index=False))
    print(f"\nDone. Tables and figures are in '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    main()
