"""Starter app for Streamlit Community Cloud: three plain sections that read results/ and show correct numbers.

Run locally:  bash run_app.sh  (your agent writes it; it runs streamlit run streamlit_app.py)
Build each section up to its required minimum (see the activity website). Keep every number
coming from m11.py, and label every table and plot "Training" or "Test".
"""

import pandas as pd
import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def load(name: str):
    return m11.load(name)


def starter_note(text: str) -> None:
    st.info(f"Starter display: {text} Build this section up to its required minimum.")


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis`, then commit and push.")
        for k in m11.DATASETS:
            if any(recs.values()):
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


# ---- 1. Beyond Accuracy ------------------------------------------------------
def confusion_table(counts: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Predicted positive (flagged)": [counts["tp"], counts["fp"]],
            "Predicted negative (not flagged)": [counts["fn"], counts["tn"]],
        },
        index=pd.Index(["Actual positive", "Actual negative"], name="Actual outcome"),
    )


def beyond_accuracy(res: dict, model: str) -> None:
    if res["dataset"] == "readmission":
        positive = "a patient returned to the hospital within 30 days of discharge"
    else:
        positive = "an adult reported a doctor's diabetes diagnosis or had an HbA1c of at least 6.5%"

    scores, actual = m11.preds(res, model)
    counts = m11.confusion(scores, actual, 0.50)
    measures = m11.metrics(counts)

    st.write(f"**Positive means** {positive}.")
    st.write(f"**Test prevalence (share who were positive)** {m11.fmt_pct(measures['prevalence'])}")
    st.write(f"**Test accuracy if we predict 'no' for everyone** {m11.fmt_pct(m11.majority_accuracy(actual))}")
    st.caption("Most people in these datasets are negative, so predicting 'no' can look accurate while missing every positive.")

    st.subheader(f"Test | {m11.MODEL_LABEL[model]} at 0.50")
    st.table(confusion_table(counts))
    st.caption(
        "Top left: positives caught. Top right: positives missed. "
        "Bottom left: false alarms. Bottom right: negatives correctly left unflagged."
    )

    st.write(f"**Sensitivity** {m11.fmt_pct(measures['sensitivity'])}. Of those who were actually positive, how many did we flag?")
    st.write(f"**Specificity** {m11.fmt_pct(measures['specificity'])}. Of those who were actually negative, how many did we leave unflagged?")
    st.write(f"**Precision** {m11.fmt_pct(measures['precision'])}. Of those we flagged, how many were actually positive?")

    metric_columns = {
        "Accuracy": "accuracy",
        "Sensitivity (recall / TPR)": "sensitivity",
        "Specificity": "specificity",
        "False positive rate": "fpr",
        "Precision": "precision",
        "F1": "f1",
    }
    comparison = []
    st.subheader("Test | Compare models at 0.50")
    for name in m11.MODELS:
        if name not in res["models"]:
            continue
        scores, actual = m11.preds(res, name)
        model_counts = m11.confusion(scores, actual, 0.50)
        model_metrics = m11.metrics(model_counts)
        label = m11.MODEL_LABEL[name]
        comparison.append({
            "Model": label,
            **{column: m11.fmt_pct(model_metrics[key]) for column, key in metric_columns.items()},
        })
        with st.expander(f"Test confusion matrix | {label}"):
            st.table(confusion_table(model_counts))

    st.caption("Test metrics at 0.50 for all available models")
    st.table(pd.DataFrame(comparison).set_index("Model"))
    st.caption(
        "False positive rate is the share of actual negatives that were flagged. "
        "F1 combines precision and sensitivity. Precision is undefined when nobody is flagged."
    )


# ---- 2. A Threshold for the Job ------------------------------------------------
def threshold(res: dict, model: str) -> None:
    scores, actual = m11.preds(res, model)
    st.markdown(
        f"**Current dataset** {m11.DATASET_LABEL[res['dataset']]}  \n"
        f"**Current model** {m11.MODEL_LABEL[model]}"
    )
    st.caption("Use Dataset and Model in the sidebar to change what you are comparing.")
    if model == "baseline":
        st.info(
            "The prevalence baseline gives everyone the same score. "
            "The counts change only when the cutoff crosses that score."
        )
    selected = st.slider("Threshold for flagging someone", 0.0, 1.0, 0.50, 0.01, format="%.2f")
    st.caption(
        "People are flagged when their score is at or above the threshold. "
        "Moving the slider changes who is flagged. The model's scores and AUROC stay the same."
    )

    default_counts = m11.confusion(scores, actual, 0.50)
    selected_counts = m11.confusion(scores, actual, selected)
    default_metrics = m11.metrics(default_counts)
    selected_metrics = m11.metrics(selected_counts)

    def comparison_values(counts, measures):
        false_alarms = 100 * counts["fp"] / measures["n"] if measures["n"] else None
        return {
            "Flagged per 100 people": m11.fmt_num(measures["flagged_per_100"], 1),
            "False alarms per 100 people": m11.fmt_num(false_alarms, 1),
            "Sensitivity (share of actual positives caught)": m11.fmt_pct(measures["sensitivity"]),
            "Specificity (share of actual negatives left unflagged)": m11.fmt_pct(measures["specificity"]),
            "Precision (share of flagged people who were positive)": m11.fmt_pct(measures["precision"]),
            "Accuracy": m11.fmt_pct(measures["accuracy"]),
        }

    st.subheader(f"Test | {m11.MODEL_LABEL[model]} at two cutoffs")
    default_values = comparison_values(default_counts, default_metrics)
    selected_values = comparison_values(selected_counts, selected_metrics)
    rows = [
        f"| Test metric | Default 0.50 | Selected {selected:.2f} |",
        "| --- | ---: | ---: |",
    ]
    for label, value in default_values.items():
        rows.append(f"| {label} | {value} | {selected_values[label]} |")
    st.markdown("\n".join(rows))
    st.caption("False alarms per 100 uses everyone screened as the denominator.")

    if res["dataset"] == "diabetes":
        sensitivity = selected_metrics["sensitivity"]
        if sensitivity is not None and sensitivity >= 0.90:
            st.success("This cutoff meets the 90% sensitivity target on these test adults.")
        else:
            st.info("The diabetes screening goal is to catch at least 90% of actual positives.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Test patient counts | Default 0.50**")
        st.table(confusion_table(default_counts))
    with right:
        st.markdown(f"**Test patient counts | Selected {selected:.2f}**")
        st.table(confusion_table(selected_counts))
    st.caption("Top left: positives caught. Top right: positives missed. Bottom left: false alarms.")


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, model: str) -> None:
    summary = m11.auroc_summary(res)
    rows = []
    for name in m11.MODELS:
        if name not in summary:
            continue
        scores = summary[name]
        rows.append({
            "Model": m11.MODEL_LABEL[name],
            "Training AUROC": m11.fmt_num(scores["train"]),
            "Test AUROC": m11.fmt_num(scores["test"]),
            "Gap (Training minus Test)": m11.fmt_num(scores["gap"]),
        })

    st.subheader("Training and Test | How well each model ranks patients")
    st.write(f"**Training patients** {res['train']['n']:,} | **Test patients** {res['test']['n']:,}")
    st.table(pd.DataFrame(rows).set_index("Model"))
    st.caption(
        "AUROC measures how often a model ranks a person who had the outcome "
        "above a person who did not. A score of 0.5 is chance-level ranking."
    )
    st.caption(
        "Use the Test column to compare how the models ranked patients they had never seen. "
        "A large drop from Training to Test can be a sign of overfitting."
    )


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
if model in res["models"]:
    RENDER[section](res, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
