"""Live Prediction: pick a recording, let a trained model diagnose it, hide inputs to stress it."""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src")]
from inference import SEVERITY_NAMES, load_bundle, score_windows  # noqa: E402
from plots import plot_mel, plot_vibration  # noqa: E402
from utils import CHECKPOINTS, CLASS_COLORS, MODE_LABEL, SPLIT_LABEL, load_index, load_recording  # noqa: E402

st.set_page_config(page_title="Live Prediction", page_icon="🩺", layout="wide")
st.title("Live Prediction")
st.caption("Choose a recording, let the model diagnose it, then hide an input and watch what changes.")

try:
    index = load_index()
except FileNotFoundError:
    st.error("Processed data not found. Run the MAFAULDA preprocessing first.")
    st.stop()


@st.cache_data(show_spinner=False)
def recording(key):
    return load_recording(key)


@st.cache_resource(show_spinner="Loading model...")
def bundle(tag):
    return load_bundle(tag)


@st.cache_data(show_spinner=False)
def split_ids(split):
    from predictiq.data.mafaulda import relabel_for_task, split_files
    ids = split_files(relabel_for_task(index, "fault")[0], split, 42)  # same call as training
    return {name: set(map(int, v)) for name, v in ids.items()}


# ------------------------------------------------------------------ controls
top = st.columns([2, 2, 1])
mode = top[0].radio("Model", list(MODE_LABEL), format_func=MODE_LABEL.get, horizontal=True,
                    help="Which inputs the model is allowed to use.")
split = top[1].radio("Trained with", list(SPLIT_LABEL), format_func=SPLIT_LABEL.get, horizontal=True,
                     help="Standard: random recordings held out. Unseen-severity: whole severity "
                          "levels held out (harder).")
seed = top[2].selectbox("Seed", [1, 2, 3])
tag = f"{mode}_fault_{split}_seed{seed}"
if not (CHECKPOINTS / f"{tag}.pt").exists():
    st.error(f"Checkpoint `checkpoints/{tag}.pt` not found. Train it first (see scripts/run_experiments.ps1).")
    st.stop()
b = bundle(tag)
classes = b["classes"]

st.divider()
c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
only_test = c1.checkbox("Only recordings the model never saw (test set)", value=True,
                        help="Untick to also see training recordings; those are easy and prove nothing.")
sets = split_ids(split)
pool = index[index["id"].isin(sets["test"])] if only_test else index
faults = sorted(pool["fault"].unique())
fault = c2.selectbox("Fault class", faults, index=faults.index("imbalance") if "imbalance" in faults else 0)
sub = pool[pool.fault == fault].sort_values("id")
keys = list(sub["key"])
if c4.button("🎲 Random"):
    st.session_state["live_pos"] = int(np.random.randint(len(keys)))
pos = min(st.session_state.get("live_pos", 0), len(keys) - 1)
key = c3.selectbox("Recording", keys, index=pos, key=f"live_rec_{fault}_{only_test}_{split}")
row = sub[sub.key == key].iloc[0]
which = next((n for n in ("train", "val", "test") if int(row["id"]) in sets[n]), "?")

hc = st.columns(4)
can_hide = b["modalities"] == ("sensor", "audio")
hide_s = hc[0].checkbox("Hide vibration", disabled=not can_hide,
                        help="Only for the fusion model.")
hide_a = hc[1].checkbox("Hide sound", disabled=not can_hide, help="Only for the fusion model.")
reveal = hc[2].toggle("Reveal true label", value=False)
hc[3].markdown(f"Recording belongs to the **{which}** set")

vib, aud = recording(key)
fprob, sprob = score_windows(b, vib, aud, hide_s, hide_a)
if fprob is None:
    st.warning("Both inputs are hidden, so there is nothing to diagnose.")
    st.stop()

win = st.slider("Window (1 second each)", 1, len(vib), 5) - 1
replay = st.button("▶ Replay windows one by one")
slot = st.empty()


def show(w):
    p, sp = fprob[w], sprob[w]
    top_i = int(p.argmax())
    pred = classes[top_i]
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plot_vibration(vib[w], color=CLASS_COLORS.get(pred, "#2f7fd6")))
    with right:
        import torch
        with torch.no_grad():
            spec = b["mel"](torch.from_numpy(aud[w]))[0, 0].numpy()
        st.pyplot(plot_mel(spec, "Mel spectrogram"))
    m = st.columns(3)
    m[0].metric("Predicted fault", pred, f"{p[top_i]:.0%} confident", delta_color="off")
    m[1].metric("Severity bucket", SEVERITY_NAMES[int(sp.argmax())], f"{sp.max():.0%}", delta_color="off")
    if reveal:
        ok = pred == row["fault"]
        m[2].metric("True label", row["fault"], "✓ correct" if ok else "✗ wrong",
                    delta_color="normal" if ok else "inverse")
    else:
        m[2].metric("True label", "hidden", "tick 'Reveal true label'", delta_color="off")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    order = np.argsort(p)
    ax.barh([classes[i] for i in order], p[order],
            color=[CLASS_COLORS.get(classes[i], "#888") for i in order])
    ax.set(xlim=(0, 1), xlabel="probability")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close("all")


if replay:
    for w in range(len(vib)):
        with slot.container():
            st.markdown(f"**Window {w + 1} of {len(vib)}**")
            show(w)
        time.sleep(0.7)
else:
    with slot.container():
        show(win)

rec = fprob.mean(0)
st.success(f"**Whole-recording verdict** (average of all {len(vib)} windows): "
           f"**{classes[int(rec.argmax())]}** ({rec.max():.0%})"
           + (f" · true label: {row['fault']}" if reveal else ""))

with st.expander("What am I looking at?"):
    st.markdown(
        "- The model reads **one 1-second window** and returns a probability for each of the 10 "
        "classes. The tallest bar is its answer.\n"
        "- The **whole-recording verdict** averages the nine windows, which is steadier.\n"
        "- **Hide vibration / Hide sound** switches one input off (the fusion model was trained to "
        "cope). Compare how the confidence changes: the model relies mostly on vibration.\n"
        "- By default only **test recordings** are offered, i.e. ones the model never trained on.")
