"""PredictIQ dashboard - home page (Overview).  Run:  python -m streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODE_LABEL, SPLIT_LABEL, load_rul_runs, summary_table  # noqa: E402

st.set_page_config(page_title="PredictIQ", page_icon="⚙️", layout="wide")

st.title("PredictIQ")
st.caption("Fault detection and remaining-life prediction for industrial machines")

st.markdown(
    "**The problem.** A machine that fails unexpectedly is expensive. We try to spot *what is wrong* "
    "(fault type) and *how long it will last* (remaining useful life) from what the machine gives off: "
    "vibration and sound."
)

tab = summary_table()
rul = load_rul_runs()

# ---- headline numbers: all read from the real result files ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fault classes", 10, help="1 healthy state + 9 fault types (MAFAULDA dataset)")
c2.metric("Recordings", "1,951", help="5-second recordings of a real test rig")
if not tab.empty:
    f = tab[(tab["input"] == "full") & (tab["split"] == "file") & (tab["mode"] == "fusion")]
    h = tab[(tab["input"] == "full") & (tab["split"] == "severity") & (tab["mode"] == "fusion")]
    if len(f):
        c3.metric("Accuracy, standard split", f"{f.iloc[0].accuracy_mean:.1%}",
                  help="Fusion model; test recordings never seen in training")
    if len(h):
        c4.metric("Accuracy, unseen severity", f"{h.iloc[0].accuracy_mean:.1%}",
                  help="Harder, honest test: whole severity levels held out")
else:
    st.warning("No fault-classification results found in `results/` yet.")

# ---- how it works ----
st.subheader("How it works")
st.markdown(
    "```\n"
    "Vibration (6 channels) ──► 1D-CNN ─┐\n"
    "                                   ├─► Fusion ─► Fault class (10) + Severity (3)\n"
    "Sound (Mel spectrogram) ─► 2D-CNN ─┘\n"
    "\n"
    "Engine sensors (C-MAPSS) ──► LSTM ──────────────► Remaining useful life (cycles)\n"
    "```"
)

# ---- result in one sentence ----
st.subheader("Result in one sentence")
if not tab.empty:
    rows = tab[tab["input"] == "full"].copy()
    rows["Model"] = rows["mode"].map(MODE_LABEL)
    rows["Split"] = rows["split"].map(SPLIT_LABEL)
    pv = rows.pivot(index="Model", columns="Split", values="accuracy_mean")
    st.dataframe(pv.style.format("{:.1%}"), use_container_width=True)
    st.caption("Mean test accuracy over 3 random seeds. The unseen-severity column is the honest one.")
if rul:
    import numpy as np
    rm = np.mean([r["test"]["rmse"] for r in rul])
    st.markdown(f"**Remaining life (engines, NASA C-MAPSS FD001):** typical error of about "
                f"**{rm:.0f} cycles** (RMSE, mean of {len(rul)} seeds).")

with st.expander("What am I looking at?"):
    st.write("The cards are measured on recordings the model never saw during training. "
             "The standard split mixes severities of each fault across train and test; the "
             "unseen-severity split holds out whole severity levels, so it is the fairer test.")
