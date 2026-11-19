"""Results: how good is each model, how robust, and where does it fail."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH)]
from plots import (MODE_NAMES, per_class_recall, plot_accuracy_bars, plot_confusion,  # noqa: E402
                   plot_curves, plot_missing, plot_per_class)
from utils import CLASS_COLORS, MODE_LABEL, SPLIT_LABEL, load_runs, summary_table  # noqa: E402

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("Results")
st.caption("Every number here is read from the saved result files, measured on test recordings "
           "the models never saw.")

tab = summary_table()
runs = load_runs()
if tab.empty:
    st.error("No results found in `results/`.")
    st.stop()

c1, c2, c3 = st.columns(3)
split = c1.radio("Test", list(SPLIT_LABEL), format_func=SPLIT_LABEL.get, horizontal=True,
                 help="Standard: random recordings held out. Unseen-severity: entire severity "
                      "levels held out, the harder and more honest test.")
metric = c2.radio("Score", ["accuracy", "f1", "file_accuracy"], horizontal=True,
                  format_func={"accuracy": "Accuracy", "f1": "Macro-F1",
                               "file_accuracy": "Per recording"}.get,
                  help="Macro-F1 gives every class equal weight, so the tiny 'normal' class counts. "
                       "Per recording averages the nine windows of a file.")
seeds = sorted({r["args"]["seed"] for r in runs})
seed_choice = c3.selectbox("Seed (for matrix and curves)", ["all seeds combined"] + seeds)

st.header("1. Which model is best?")
a, b = st.columns(2)
with a:
    st.pyplot(plot_accuracy_bars(tab, split, metric))
    st.caption(f"{SPLIT_LABEL[split]}. Bars: mean over seeds; whiskers: ± one standard deviation.")
with b:
    if (tab[(tab["split"] == split) & (tab["mode"] == "fusion")]["input"] != "full").any():
        st.pyplot(plot_missing(tab, split, metric))
        st.caption("What if a sensor fails? The fusion model keeps working when sound is lost, "
                   "but loses a lot when vibration is lost.")
with st.expander("What am I looking at?"):
    st.markdown(
        "- On the **standard split** vibration and fusion are both near 99%, so fusion adds "
        "little.\n"
        "- On the **unseen-severity split** every model drops (about 76 to 83%) because it must "
        "recognise fault sizes it never trained on. This is the honest number.\n"
        "- Fusion is slightly ahead there, but the seeds share one split, so the whiskers "
        "understate the real uncertainty. Do not claim more than 'slightly ahead'.")

st.header("2. Where does a model go wrong?")
mode = st.radio("Model", list(MODE_LABEL), format_func=MODE_LABEL.get, horizontal=True, key="cm_mode")
sel = [r for r in runs if r["args"]["mode"] == mode and r["args"]["split"] == split
       and (seed_choice == "all seeds combined" or r["args"]["seed"] == seed_choice)]
if not sel:
    st.info("No runs for this selection.")
else:
    classes = sel[0]["classes"]
    cm = np.sum([r["test"]["confusion_matrix"] for r in sel], axis=0)
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plot_confusion(cm, classes))
        st.caption("Each row is a true class; the dark cell shows where its windows were sent. "
                   "A perfect model is a dark diagonal.")
    with right:
        st.pyplot(plot_per_class(cm, classes, CLASS_COLORS))
        weakest = classes[int(np.argmin(per_class_recall(cm)))]
        st.caption(f"Weakest class: **{weakest}**.")
    st.header("3. Training curves")
    st.pyplot(plot_curves(sel, mode))
    st.caption("Validation macro-F1 after each epoch. We keep the best epoch (checkpoint).")

st.header("4. Numbers table")
show = tab[tab["split"] == split].copy()
show["model"] = show["mode"].map(MODE_NAMES)
cols = ["model", "input", "n_seeds"] + [f"{m}_{s}" for m in ("accuracy", "f1", "file_accuracy")
                                        for s in ("mean", "std")]
st.dataframe(show[cols].round(4), use_container_width=True, hide_index=True)
st.caption("Copy of docs/EXPERIMENT_LOG.md's summary. Three seeds share one data split.")
