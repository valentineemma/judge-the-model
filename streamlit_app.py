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
    starter_note("metrics at 0.50 only.")
    p, y = m11.preds(res, model)
    m = m11.metrics(m11.confusion(p, y, 0.5))
    st.caption(f"{TEST} Â· {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{"Sensitivity": m11.fmt_pct(m["sensitivity"]), "Specificity": m11.fmt_pct(m["specificity"]),
                            "Precision": m11.fmt_pct(m["precision"]), "Flagged per 100": f"{m['flagged_per_100']:.1f}"}]))


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, model: str) -> None:
    starter_note("training and test AUROC of every model, in a table.")
    s = m11.auroc_summary(res)
    rows = [{"Model": m11.MODEL_LABEL[m], f"{TRAIN} AUROC": m11.fmt_num(v["train"]), f"{TEST} AUROC": m11.fmt_num(v["test"])}
            for m, v in s.items()]
    st.caption(f"{TRAIN} and {TEST} Â· AUROC on the rows each model learned from, and on rows it never saw")
    st.table(pd.DataFrame(rows).set_index("Model"))


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
