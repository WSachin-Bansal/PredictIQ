"""Signal Lab: look at the raw inputs (vibration, spectrum, Mel spectrogram, sound)."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src")]
from plots import plot_mel, plot_spectrum, plot_vibration  # noqa: E402
from utils import CLASS_COLORS, load_index, load_recording  # noqa: E402

st.set_page_config(page_title="Signal Lab", page_icon="📈", layout="wide")
st.title("Signal Lab")
st.caption("What the machine gives off, and what the models actually receive.")

try:
    index = load_index()
except FileNotFoundError:
    st.error("Processed data not found. Run the MAFAULDA preprocessing first.")
    st.stop()


@st.cache_data(show_spinner=False)
def recording(key):
    return load_recording(key)


@st.cache_resource(show_spinner=False)
def mel_extractor():
    from predictiq.data.audio import MelExtractor
    return MelExtractor(16000, 1.0, 1024, 512, 64)


def mel_of(wav):
    import torch
    with torch.no_grad():
        return mel_extractor()(torch.tensor(wav))[0, 0].numpy()


def picker(side: str, default_fault: str):
    """Three dropdowns (class -> severity -> recording) plus a Random button; returns a row."""
    faults = sorted(index["fault"].unique())
    fault = st.selectbox("Fault class", faults, index=faults.index(default_fault)
                         if default_fault in faults else 0, key=f"{side}_fault")
    sub = index[index.fault == fault]
    sevs = sorted(sub["severity_label"].fillna("none").unique())
    sev = st.selectbox("Severity level", sevs, key=f"{side}_sev")
    sub = sub[sub["severity_label"].fillna("none") == sev].sort_values("id")
    keys = list(sub["key"])
    pos_key = f"{side}_pos"
    if st.button("🎲 Random recording", key=f"{side}_rand"):
        st.session_state[pos_key] = int(np.random.randint(len(keys)))
    pos = min(st.session_state.get(pos_key, 0), len(keys) - 1)
    key = st.selectbox("Recording", keys, index=pos, key=f"{side}_rec_{fault}_{sev}")
    return sub[sub.key == key].iloc[0]


def panel(row, win: int):
    vib, aud = recording(row["key"])
    color = CLASS_COLORS.get(row["fault"], "#2f7fd6")
    st.markdown(f"### :{'green' if row['fault'] == 'normal' else 'orange'}[{row['fault']}]"
                f" · severity {row['severity_label'] if isinstance(row['severity_label'], str) else 'n/a'}"
                f" · speed {row['speed_hz']:.0f} Hz")
    st.pyplot(plot_vibration(vib[win], color=color))
    st.pyplot(plot_spectrum(vib[win], color=color))
    st.pyplot(plot_mel(mel_of(aud[win])))
    wav = aud[win]
    st.audio((wav / (np.abs(wav).max() + 1e-9)).astype("float32"), sample_rate=16000)
    st.caption("Sound of this 1-second window (normalised volume).")


compare = st.toggle("Compare two recordings side by side", value=True)
n_win = 9
win = st.slider("Window (each recording is cut into 9 one-second windows)", 1, n_win, 5,
                help="The models classify one 1-second window at a time.") - 1

if compare:
    left, right = st.columns(2)
    with left:
        a = picker("A", "normal")
        panel(a, win)
    with right:
        b = picker("B", "imbalance")
        panel(b, win)
else:
    a = picker("A", "normal")
    panel(a, win)

with st.expander("What am I looking at?"):
    st.markdown(
        "- **Vibration traces:** six sensors on the machine. A healthy machine gives a smooth, regular "
        "wave; faults add larger or more irregular movement.\n"
        "- **Frequency spectrum:** the same signal broken into frequencies. Faults show up as peaks "
        "at multiples of the shaft speed, and these peaks are what the 1D-CNN learns.\n"
        "- **Mel spectrogram:** the sound turned into a picture (time across, pitch up, brightness = "
        "loudness). This picture is the actual input of the audio model.\n"
        "- Try: normal vs imbalance, then vs a bearing fault, and listen to the difference.")
    st.info("Sensor channel names follow the dataset documentation and are not verified by us.")
