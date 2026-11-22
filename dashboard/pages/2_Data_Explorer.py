"""Data Explorer: what the MAFAULDA data contains and how it is split."""
import sys
from pathlib import Path

import streamlit as st

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src")]
from explore_utils import (assign_splits, class_counts, plot_class_counts, plot_speed,  # noqa: E402
                           plot_split_stack, severity_table)
from utils import CLASS_COLORS, SPLIT_LABEL, load_index  # noqa: E402

st.set_page_config(page_title="Data Explorer", page_icon="🗂️", layout="wide")
st.title("Data Explorer")
st.caption("MAFAULDA: real vibration and sound recordings from a laboratory machine-fault simulator (UFRJ).")

try:
    index = load_index()
except FileNotFoundError:
    st.error("Processed data not found. Run the MAFAULDA preprocessing first.")
    st.stop()

normal_share = (index["fault"] == "normal").mean()
m = st.columns(4)
m[0].metric("Recordings", f"{len(index):,}", help="Each is about 5 seconds of data")
m[1].metric("1-second windows", f"{int(index['n_windows'].sum()):,}", help="What the models actually classify")
m[2].metric("Fault classes", index["fault"].nunique())
m[3].metric("Share that is 'normal'", f"{normal_share:.1%}")

st.header("1. What is in the dataset?")
unit = st.radio("Count", ["recordings", "windows"], horizontal=True)
a, b = st.columns([3, 2])
with a:
    st.pyplot(plot_class_counts(class_counts(index, unit), CLASS_COLORS, unit))
with b:
    st.markdown(
        f"**Why `normal` matters.** Only {normal_share:.1%} of the recordings are healthy. A model that "
        "never says 'normal' would still be right about "
        f"{1 - normal_share:.1%} of the time. That is why we report **macro-F1**, which gives every "
        "class the same weight, next to accuracy.\n\n"
        "**Each recording** has 8 channels: a tachometer (shaft speed), 6 accelerometer "
        "channels (vibration) and a microphone. We use the vibration and the microphone.")
with st.expander("What am I looking at?"):
    st.write("Bars show how many examples exist per fault class. The dataset is not balanced: some "
             "faults have 300+ recordings, the healthy state has 49.")

st.header("2. Severity levels")
st.markdown("Most faults come in several **severities** (for example the imbalance weight 6 g to 35 g). "
            "This is what lets us test whether a model can handle a fault size it has never seen.")
split = st.radio("Split", list(SPLIT_LABEL), format_func=SPLIT_LABEL.get, horizontal=True,
                 help="The exact split the models were trained and tested with (seed 42).")
sp = assign_splits(index, split)
c, d = st.columns([2, 3])
with c:
    st.pyplot(plot_split_stack(sp))
    st.caption("Recordings per class in train / validation / test.")
with d:
    tab = severity_table(sp)
    st.dataframe(tab, use_container_width=True, hide_index=True, height=380)
    st.caption("Look at one class, e.g. imbalance: in the standard split every severity level has "
               "recordings in train AND test; in the unseen-severity split whole levels move to "
               "test, so the model faces fault sizes it never trained on.")
with st.expander("Why two splits?"):
    st.markdown(
        "- **Standard:** random recordings of each class go to train/val/test. Neighbouring "
        "recordings (same fault, same severity, similar speed) look almost identical, so scores are optimistic.\n"
        "- **Unseen severity:** whole severity levels of a class are held out. Harder, and closer "
        "to what happens in practice. This is the honest number.\n"
        "- In both cases the split is by **recording**, never by window; otherwise windows of one "
        "recording would sit in both train and test (leakage).")

st.header("3. Operating speed")
st.pyplot(plot_speed(index, CLASS_COLORS))
st.caption("Shaft speed varies from recording to recording. A fault changes the vibration, but so does "
           "speed, which is one reason the models must generalise across speeds.")
