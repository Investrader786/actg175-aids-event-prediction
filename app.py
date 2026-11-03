"""ACTG-175 Outcome Evaluator: Streamlit front-end for ML Assignment 2.

Scores five pre-trained classifiers (saved in model/) on an uploaded test CSV from
the AIDS Clinical Trials Group Study 175. The app never trains or re-fits a model:
it only loads the saved pipelines and calls predict / predict_proba.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

import actg_eval as ev  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"
README_METRICS = APP_DIR / "results" / "metrics_comparison.csv"
SHORT_LABELS = ["No event", "Event"]
HIGHLIGHT = "background-color: #CDEBE5; font-weight: 600"

st.set_page_config(page_title="ACTG-175 Outcome Evaluator", page_icon="🧬", layout="wide")


# ----------------------------------------------------------------------------
# Cached loaders: models are loaded once per server process, never re-fitted
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the five saved classifiers...")
def load_classifiers() -> dict:
    return {name: joblib.load(MODEL_DIR / file) for name, file in ev.MODEL_REGISTRY.items()}


@st.cache_data
def load_model_card() -> dict:
    return json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))


@st.cache_data
def read_table(raw_bytes: bytes) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(io.BytesIO(raw_bytes)), None
    except Exception as exc:  # noqa: BLE001 - any parse failure gets the same friendly message
        return None, (f"The file could not be read as a CSV ({type(exc).__name__}). "
                      "Please upload a comma-separated .csv file with a header row.")


@st.cache_data(show_spinner="Scoring the five models...")
def score_all(X: pd.DataFrame, y: pd.Series) -> dict:
    scored = {}
    for name, model in load_classifiers().items():
        metrics, y_pred, y_score = ev.evaluate_model(model, X, y)
        scored[name] = {"metrics": metrics, "y_pred": y_pred, "y_score": y_score}
    return scored


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def fmt(value) -> str:
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def confusion_figure(y_true, y_pred, title: str):
    cm = ev.confusion_frame(y_true, y_pred).values
    row_pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    annot = np.array([[f"{cm[i, j]}\n({row_pct[i, j]:.1f}% of row)" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(cm, annot=annot, fmt="", cmap="crest", cbar=False, linewidths=1.5, linecolor="white",
                xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, annot_kws={"fontsize": 10}, ax=ax)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def comparison_figure(table: pd.DataFrame):
    long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="crest", ax=ax)
    ax.set_ylim(min(0.0, float(np.nanmin(long["Score"])) - 0.05), 1.0)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("")
    ax.legend(title="", fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    return fig


def roc_figure(y_true, scored: dict):
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    palette = sns.color_palette("crest", n_colors=len(scored))
    for color, (name, result) in zip(palette, scored.items()):
        fpr, tpr, _ = roc_curve(y_true, result["y_score"], pos_label=ev.POSITIVE_CLASS)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} ({result['metrics']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (AUC in brackets)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def feature_dictionary() -> pd.DataFrame:
    group = {c: "continuous" for c in ev.CONTINUOUS_FEATURES}
    group.update({c: "binary" for c in ev.BINARY_FEATURES})
    group.update({c: "nominal" for c in ev.NOMINAL_FEATURES})
    rows = [{"column": c, "type": group[c], "meaning": ev.FEATURE_DESCRIPTIONS[c]} for c in ev.FEATURE_COLUMNS]
    rows.append({"column": ev.TARGET_COL, "type": "target", "meaning": ev.TARGET_DESCRIPTION})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Header and sidebar
# ----------------------------------------------------------------------------
st.title("🧬 ACTG-175 Outcome Evaluator")
st.markdown(
    "Will a patient in the **AIDS Clinical Trials Group Study 175** have an AIDS-defining event or die "
    "during follow-up (`cid = 1`)? Upload held-out patients, pick one of five pre-trained classifiers, "
    "and see how well it separates the two outcomes."
)
st.caption("M.Tech (AIML) · Machine Learning · Assignment 2 · Mani Rathinam Subbaiah (2026AJ05440)")

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV", type=["csv"],
        help="It needs the 21 feature columns and the target column 'cid'. Extra columns are ignored.")
    st.caption("Please upload **test data only**: the free Streamlit tier has limited memory.")
    st.download_button("Download sample test_data.csv", data=SAMPLE_CSV.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    use_sample = st.toggle("Use the bundled test_data.csv", value=False,
                           help="Scores the 428 held-out patients that ship with the app, without uploading.")

    st.header("2 · Model")
    model_name = st.selectbox("Classifier", list(ev.MODEL_REGISTRY), index=0)

    st.divider()
    st.caption(ev.METRIC_CONVENTION)
    st.caption("Models are loaded from `model/` and never retrained here.")

# ----------------------------------------------------------------------------
# Get the data to score
# ----------------------------------------------------------------------------
if uploaded is not None:
    df, read_error = read_table(uploaded.getvalue())
    source = uploaded.name
elif use_sample:
    df, read_error = read_table(SAMPLE_CSV.read_bytes())
    source = "bundled test_data.csv"
else:
    st.info("**Start here:** upload a test CSV in the sidebar, or switch on *Use the bundled test_data.csv*. "
            "You can also download the sample file first and then upload it.")
    with st.expander("Columns the file must contain", expanded=True):
        st.dataframe(feature_dictionary(), hide_index=True)
    st.stop()

if read_error:
    st.error(read_error)
    st.stop()

X, y, problems, notes = ev.check_schema(df)
if problems:
    st.error("This file can't be scored yet:\n\n" + "\n".join(f"- {p}" for p in problems))
    with st.expander("Columns the file must contain"):
        st.dataframe(feature_dictionary(), hide_index=True)
    st.stop()
for note in notes:
    st.info(note)

two_classes = y.nunique() == 2
summary = st.columns(3)
summary[0].metric("Patients scored", f"{len(y):,}")
summary[1].metric("Actual events (cid = 1)", f"{int(y.sum()):,}")
summary[2].metric("Event share", f"{y.mean():.1%}")
st.caption(f"Data source: **{source}** · Selected model: **{model_name}**")
if not two_classes:
    st.warning("The file contains only one outcome class, so AUC and the ROC curve can't be computed "
               "(shown as N/A). The other metrics are still valid.")

scored = score_all(X, y)
tab_eval, tab_compare, tab_about = st.tabs(
    ["📊 Evaluate selected model", "🏁 Compare all five models", "📘 About the data & models"])

# ----------------------------------------------------------------------------
# Tab 1: one model in detail
# ----------------------------------------------------------------------------
with tab_eval:
    result = scored[model_name]
    st.subheader(f"{model_name} on {len(y):,} patients")
    cards = st.columns(len(ev.METRIC_NAMES))
    for card, metric in zip(cards, ev.METRIC_NAMES):
        with card.container(border=True):
            st.metric(metric, fmt(result["metrics"][metric]))

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        fig = confusion_figure(y, result["y_pred"], model_name)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = ev.report_frame(y, result["y_pred"])
        for col in ("precision", "recall", "f1-score"):  # pre-format: st.dataframe shows NaN as "None"
            report[col] = report[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        st.dataframe(report, column_config={c: st.column_config.TextColumn(c) for c in ("precision", "recall", "f1-score")})
        st.caption("Rows 'No event' and 'AIDS event / death' are the two classes. The assignment's Precision, "
                   "Recall and F1 are the values for the event class.")

    with st.expander("Row-level predictions"):
        predictions = X.assign(**{"actual cid": y.values, "predicted cid": result["y_pred"],
                                  "P(event)": np.round(result["y_score"], 4)})
        st.dataframe(predictions, height=280)
        st.download_button("Download predictions as CSV", predictions.to_csv(index=False).encode("utf-8"),
                           file_name=f"predictions_{ev.MODEL_REGISTRY[model_name].split('.')[0]}.csv",
                           mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 2: all five models side by side on the same file
# ----------------------------------------------------------------------------
with tab_compare:
    table = pd.DataFrame({name: r["metrics"] for name, r in scored.items()}).T[ev.METRIC_NAMES]
    table.index.name = "ML Model Name"
    st.subheader("Comparison on the scored file")
    st.dataframe(table.style.format(fmt).highlight_max(axis=0, props=HIGHLIGHT))

    if README_METRICS.exists():
        reported = pd.read_csv(README_METRICS, index_col=0).reindex(table.index)[ev.METRIC_NAMES]
        if np.allclose(table.to_numpy(float), reported.to_numpy(float), atol=1e-9, equal_nan=True):
            st.caption("✅ These numbers match the comparison table in README.md exactly: this file is "
                       "the project's held-out test set.")

    leader = table["MCC"].idxmax()
    st.success(f"Highest MCC on this file: **{leader}** ({table.loc[leader, 'MCC']:.4f}). "
               "MCC is the headline metric because only about 1 in 4 patients has an event, so "
               "accuracy alone rewards models that rarely predict one.")

    chart_col, roc_col = st.columns([1.6, 1], gap="large")
    with chart_col:
        fig = comparison_figure(table)
        st.pyplot(fig)
        plt.close(fig)
    with roc_col:
        if two_classes:
            fig = roc_figure(y, scored)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ROC curves need both outcome classes in the file.")

# ----------------------------------------------------------------------------
# Tab 3: context
# ----------------------------------------------------------------------------
with tab_about:
    card = load_model_card()
    split = card["split"]
    st.markdown(
        f"""
**Dataset:** [AIDS Clinical Trials Group Study 175]({card['dataset']['source']}) (UCI ML Repository, id 890).
It has {card['dataset']['rows_used']:,} HIV-infected adults, randomised to four antiretroviral regimens.
**Target:** `cid`, {ev.TARGET_DESCRIPTION}.

**Training setup:** a stratified {int((1 - split['test_size']) * 100)}/{int(split['test_size'] * 100)} split
(train {split['train_rows']:,} / test {split['test_rows']:,}), seed {split['seed']}. Hyper-parameters were
tuned by 5-fold stratified cross-validation on the training rows only, choosing the best MCC. Every saved
model is a full pipeline: median/most-frequent imputation, standard scaling of continuous columns and
one-hot encoding of `trt` and `strat`, followed by the classifier.
"""
    )
    st.markdown("**Saved models**")
    st.dataframe(pd.DataFrame([
        {"ML Model Name": name, "estimator": m["estimator"],
         "tuned settings": ", ".join(f"{k}={v}" for k, v in m["best_params"].items()),
         "CV MCC (train folds)": m["cv_mcc_mean"]}
        for name, m in card["models"].items()]), hide_index=True,
        column_config={"CV MCC (train folds)": st.column_config.NumberColumn(format="%.4f")})
    st.markdown("**Input columns**")
    dictionary = feature_dictionary()
    st.dataframe(dictionary, hide_index=True, height=(len(dictionary) + 1) * 35 + 3)
    st.markdown("**Columns removed before training**")
    st.dataframe(pd.DataFrame([{"column": c, "reason": r} for c, r in ev.DROPPED_COLUMNS.items()]),
                 hide_index=True)
    versions = card["versions"]
    st.caption("Trained with " + ", ".join(f"{k} {v}" for k, v in versions.items()) + ".")
