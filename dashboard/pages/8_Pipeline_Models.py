"""Pipeline & Models: how raw recordings become model inputs, and what the networks look like."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src")]
from inference import layer_table, load_bundle  # noqa: E402
from pipeline_utils import (params_by_block, plot_before_after, plot_params,  # noqa: E402
                            plot_timeline, stitch)
from plots import plot_mel  # noqa: E402
from utils import CHECKPOINTS, CLASS_COLORS, MODE_LABEL, load_index, load_recording  # noqa: E402

st.set_page_config(page_title="Pipeline & Models", page_icon="🧩", layout="wide")
st.title("Pipeline & Models")
st.caption("From a raw recording to a prediction: every step, and every layer of the networks.")

cfg = yaml.safe_load(open(DASH.parent / "configs" / "default.yaml"))
M, A = cfg["mafaulda"], cfg["audio"]

# ============================================================== 1. data pipeline
st.header("1. From recording to model input")
st.markdown(
    "```\n"
    f"Raw CSV: 8 channels x ~5 s at {M['fs']:,} Hz   (tachometer, 6 vibration, 1 microphone)\n"
    f"   |-- vibration: low-pass filter + downsample to {M['vib_rate']:,} Hz  -> (6 x {int(M['vib_rate'] * M['win_seconds']):,}) per window\n"
    f"   |-- microphone: resample to {M['audio_rate']:,} Hz               -> ({int(M['audio_rate'] * M['win_seconds']):,}) per window\n"
    f"   |-- cut into {M['win_seconds']:g}-second windows every {M['stride_seconds']:g} s   -> 9 windows per recording\n"
    "   |-- vibration: standardise each channel (stats from TRAIN files only)\n"
    f"   '-- sound: log-Mel spectrogram, {A['n_mels']} bands -> picture of size {A['n_mels']} x 32\n"
    "```")

try:
    index = load_index()
except FileNotFoundError:
    st.warning("Processed data not found; the interactive part is skipped.")
    index = None

if index is not None:
    faults = sorted(index["fault"].unique())
    c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
    fault = c1.selectbox("Fault class", faults, index=faults.index("imbalance") if "imbalance" in faults else 0)
    sub = index[index.fault == fault]
    sevs = sorted(sub["severity_label"].fillna("none").unique())
    sev = c2.selectbox("Severity", sevs)
    sub = sub[sub["severity_label"].fillna("none") == sev].sort_values("id")
    keys = list(sub["key"])
    if c4.button("🎲 Random"):
        st.session_state["pipe_pos"] = int(np.random.randint(len(keys)))
    key = c3.selectbox("Recording", keys, index=min(st.session_state.get("pipe_pos", 0), len(keys) - 1),
                       key=f"pipe_rec_{fault}_{sev}")

    @st.cache_data(show_spinner=False)
    def recording(k):
        return load_recording(k)

    @st.cache_resource(show_spinner=False)
    def bundle(tag):
        return load_bundle(tag)

    vib, aud = recording(key)
    n_win = len(vib)
    d1, d2 = st.columns([3, 1])
    win = d1.slider("Move the 1-second window along the recording", 1, n_win, 1,
                    help="Windows overlap by half a second.") - 1
    ch = d2.selectbox("Vibration channel", list(range(1, 7)), index=0)
    color = CLASS_COLORS.get(fault, "#2f7fd6")

    full_v, full_a = stitch(vib), stitch(aud)
    t1, t2 = st.columns(2)
    t1.pyplot(plot_timeline(full_v[ch - 1], M["vib_rate"], win, M["win_seconds"], M["stride_seconds"],
                            color, f"Whole recording: vibration channel {ch}", "amplitude"))
    t2.pyplot(plot_timeline(full_a, M["audio_rate"], win, M["win_seconds"], M["stride_seconds"],
                            "#555", "Whole recording: microphone", "amplitude"))

    st.subheader(f"What the models receive for window {win + 1} of {n_win}")
    s1, s2 = st.columns(2)
    stats = None
    tag = "fusion_fault_file_seed1"
    if (CHECKPOINTS / f"{tag}.pt").exists():
        b = bundle(tag)
        stats = b["stats"]
    with s1:
        st.markdown("**Vibration branch input**")
        if stats is not None:
            mean, std = stats
            norm = (vib[win][ch - 1] - mean[ch - 1]) / std[ch - 1]
            st.pyplot(plot_before_after(vib[win][ch - 1], norm, M["vib_rate"], color=color))
            st.caption("The network receives all 6 channels as a 6 x 10,000 table, standardised like this.")
        else:
            st.info("Needs `checkpoints/fusion_fault_file_seed1.pt` to show the standardisation step.")
    with s2:
        st.markdown("**Sound branch input**")
        if stats is not None:
            import torch
            with torch.no_grad():
                spec = b["mel"](torch.from_numpy(aud[win]))[0, 0].numpy()
            st.pyplot(plot_mel(spec, "log-Mel spectrogram (64 x 32)"))
            st.audio((aud[win] / (np.abs(aud[win]).max() + 1e-9)).astype("float32"), sample_rate=M["audio_rate"])
            st.caption("The sound is turned into a picture; a small image CNN reads it.")

# ============================================================== 2. architecture
st.header("2. The networks, layer by layer")
mode = st.radio("Model", list(MODE_LABEL), format_func=MODE_LABEL.get, horizontal=True, index=2)


@st.cache_data(show_spinner="Building model...")
def table(mode):
    return layer_table(mode)


layers, total = table(mode)
k1, k2 = st.columns([1, 2])
with k1:
    st.metric("Total learned parameters", f"{total:,}")
    st.pyplot(plot_params(params_by_block(layers)))
with k2:
    st.dataframe(layers, use_container_width=True, hide_index=True, height=420)
    st.caption("One row per layer, in the order data flows through. 'Output shape' is for a single "
               "1-second example (channels x length, or channels x height x width).")

with st.expander("What does each part do?"):
    st.markdown(
        "- **Vibration encoder (1D-CNN):** 4 blocks of convolution + batch-norm + ReLU + pooling. Early "
        "blocks use wide filters to catch slow patterns, later ones narrow filters. Global average "
        "pooling collapses time into one 128-number summary.\n"
        "- **Sound encoder (2D-CNN):** 3 blocks of 3x3 convolutions that treat the spectrogram as a "
        "small image, then a 64 -> 128 summary.\n"
        "- **Fusion layer:** concatenates the two 128-number summaries (256), then a 256-unit layer with "
        "ReLU and dropout.\n"
        "- **Two heads:** one scores the 10 fault classes, one scores the 3 severity buckets.\n"
        "- **Missing input:** if a sensor is switched off, its 128 numbers are replaced by zeros. "
        "Training randomly does this (modality dropout) so the model copes.")

# ============================================================== 3. training recipe
st.header("3. How they were trained")
st.table({
    "Setting": ["Loss", "Optimiser", "Learning rate", "Weight decay", "Epochs", "Batch size",
                "Modality dropout", "Model selection", "Seeds"],
    "Value": ["cross-entropy(fault) + 0.5 x cross-entropy(severity)", "Adam", str(M["lr"]),
              str(cfg["train"]["weight_decay"]), "20 (all MAFAULDA runs)", "32 (config default 64)",
              str(M["modality_dropout"]), "best validation macro-F1", "1, 2, 3"],
})
st.caption("Epochs and batch size are the values used for the reported runs. The RUL LSTM and the "
           "vision ResNet18 have their own recipes (see their pages).")
