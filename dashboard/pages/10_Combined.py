"""Combined model: one network for vibration, sound, images and engine sensors, vs separate models."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src"), str(DASH.parent / "scripts")]
from compare_combined import rows_for  # noqa: E402
from utils import RESULTS, SPLIT_LABEL  # noqa: E402

st.set_page_config(page_title="Combined model", page_icon="🧬", layout="wide")
st.title("Combined model")
st.caption("One network that takes all four inputs and answers all four questions.")

with st.expander("What am I looking at?", expanded=True):
    st.markdown(
        "The earlier pages use **separate** models: one for vibration/sound faults, one for images, one for "
        "engine life. This page tests the **combined** model: four encoders feed one shared layer, and four "
        "heads answer *fault type, severity, good/defect, remaining life*.\n\n"
        "The point is to check one thing honestly: **does sharing one network cost accuracy?** "
        "Tables show mean ± standard deviation over 3 seeds, same test sets for both."
    )

st.header("1. Architecture")
st.markdown(
    "```\n"
    "vibration (6 x 10000) --> 1D CNN  --\\\n"
    "sound  (Mel 64 x 32)  --> 2D CNN  ---\\  concat (4 x 128 = 512)\n"
    "image  (3 x 224 x 224)--> ResNet18 ---/--> Linear 256 + ReLU + Dropout --> fault (10)\n"
    "engine (50 cycles x 17)--> LSTM   --/                                  --> severity (3)\n"
    "                                                                       --> defect (2)\n"
    "                                                                       --> RUL (1)\n"
    "```\n"
    "Each training batch comes from **one** dataset and trains only the head it has labels for. "
    "Inputs that a batch does not have are replaced by zeros."
)

st.header("2. Combined vs separate models")
init = st.radio("Encoder start", ["warm", "scratch"], horizontal=True,
                help="warm = encoders start from the separately trained models; scratch = random start")
tabs = st.tabs([SPLIT_LABEL[s] for s in ("file", "severity")])
for tab, split in zip(tabs, ("file", "severity")):
    with tab:
        try:
            rows, n = rows_for(RESULTS, split, init)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not read results: {e}")
            continue
        if n == 0:
            st.info(f"No combined runs found for init={init}, split={split}. Train with "
                    f"`python -m predictiq.training.train_combined --init {init} --split {split} --seed 1`.")
            continue
        df = pd.DataFrame(rows, columns=["Task", "Metric", "Separate models", "Combined"])
        st.caption(f"{n} combined runs")
        st.dataframe(df, hide_index=True, use_container_width=True)

        def num(s):
            try:
                return float(s.split(" ±")[0])
            except ValueError:
                return None

        pick = df[df.Metric.isin(["accuracy", "macro-F1", "defects caught"])].copy()
        pick["Label"] = pick.Task + " " + pick.Metric
        pick["Separate"] = pick["Separate models"].map(num)
        pick["Combined "] = pick["Combined"].map(num)
        st.bar_chart(pick.set_index("Label")[["Separate", "Combined "]])

st.header("3. What this does and does not show")
st.markdown(
    "- **Shows:** one shared network matches the separate models on fault, defect and RUL, within the "
    "spread between seeds. One model does the work of three.\n"
    "- **Does not show:** that vibration and images from the *same machine* help each other. The datasets "
    "never contain the same machine, so that cannot be trained or claimed.\n"
    "- **Noise:** only 3 seeds sharing one MAFAULDA split; gaps smaller than the ± are not real differences.\n"
    "- **Missing-input robustness** (vibration or sound hidden) is mixed and noisy."
)
