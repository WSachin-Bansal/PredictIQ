"""RUL: how many flight cycles does an engine have left? (NASA C-MAPSS FD001, simulated)"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src")]
from inference import load_rul, predict_rul  # noqa: E402
from rul_utils import (cycle_windows, load_rul_runs, load_test, plot_pred_vs_true,  # noqa: E402
                       plot_rul_curve, plot_sensors, status, true_rul_curve)
from utils import CHECKPOINTS  # noqa: E402

st.set_page_config(page_title="Remaining life", page_icon="⏳", layout="wide")
st.title("Remaining Useful Life")
st.caption("Simulated jet engines (NASA C-MAPSS FD001): how many flight cycles are left before failure?")

runs = load_rul_runs()
try:
    test_df, final_rul, engines = load_test()
except FileNotFoundError:
    st.error("C-MAPSS files not found in `data/raw/cmapss/`.")
    st.stop()
if not runs:
    st.error("No RUL results found. Train with `python -m predictiq.training.train_cmapss`.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def model_for(seed):
    return load_rul(f"rul_FD001_seed{seed}")


seeds = sorted(r["args"]["seed"] for r in runs if (CHECKPOINTS / f"rul_FD001_seed{r['args']['seed']}.pt").exists())
if not seeds:
    st.error("No RUL checkpoints found in `checkpoints/`.")
    st.stop()

c1, c2 = st.columns([1, 3])
seed = c1.selectbox("Seed", seeds)
b = model_for(seed)
engine = c2.selectbox("Engine (test set; the data stops before failure)", engines, index=0)
default_sensors = [s for s in ("s2", "s3", "s4", "s7", "s11", "s12") if s in b["features"]][:4]
chosen = st.multiselect("Sensors to plot", [f for f in b["features"] if f.startswith("s")],
                        default=default_sensors)

g = test_df[test_df["unit"] == engine].sort_values("cycle")
n = len(g)
final = float(final_rul[engines.index(engine)])
feats = ((g[b["features"]].to_numpy(dtype="float32") - b["mean"]) / b["std"])
pred = np.clip(predict_rul(b, cycle_windows(feats, b["window"])), 0, 125)
true = true_rul_curve(n, final, b["max_rul"])

cycle = st.slider("Flight cycle (drag to move through the engine's life)", 1, n, n)
p, t = float(pred[cycle - 1]), float(true[cycle - 1])
label, colour = status(p)

m = st.columns(4)
m[0].metric("Cycles observed", n, help="Test engines are cut off before they fail.")
m[1].metric("Predicted remaining life", f"{p:.0f} cycles")
m[2].metric("True remaining life", f"{t:.0f} cycles", f"error {p - t:+.0f}", delta_color="off")
m[3].markdown(f"<div style='background:{colour};color:white;padding:14px;border-radius:8px;"
              f"text-align:center;font-weight:700'>{label}</div>", unsafe_allow_html=True)
st.progress(min(p, 125) / 125, text="Estimated remaining life (125 = 'healthy', the label cap)")

left, right = st.columns(2)
with left:
    if chosen:
        st.pyplot(plot_sensors(g, chosen, cycle, b["mean"], b["std"], b["features"]))
    else:
        st.info("Pick at least one sensor.")
with right:
    st.pyplot(plot_rul_curve(g["cycle"].to_numpy(), pred, true, cycle))

with st.expander("What am I looking at?"):
    st.markdown(
        "- Each engine reports ~17 values per flight cycle. Slowly, as the engine wears out, some "
        "sensors drift (left chart).\n"
        "- The LSTM reads the **last 50 cycles** and outputs the remaining life. The right chart "
        "shows its prediction at every cycle (blue) against the truth (black).\n"
        "- Early on nobody can tell 200 from 300 cycles left, so the labels are capped at **125**; "
        "the curve is flat at the start and falls linearly near the end.\n"
        "- The coloured bands and the status box use **illustrative thresholds** (above 75 healthy, "
        "30 to 75 plan maintenance, below 30 urgent), not a validated maintenance policy.")

st.header("Over all 100 test engines")
k = st.columns(5)
for col, (name, key, fmt) in zip(k, [("RMSE", "rmse", "{:.1f}"), ("MAE", "mae", "{:.1f}"),
                                     ("R²", "r2", "{:.3f}"), ("NASA score", "nasa_score", "{:.0f}"),
                                     ("Predicted too late", "late_fraction", "{:.0%}")]):
    v = np.array([r["test"][key] for r in runs])
    col.metric(name, fmt.format(v.mean()), f"± {fmt.format(v.std(ddof=1) if len(v) > 1 else 0)}",
               delta_color="off")
st.caption(f"Mean ± std over {len(runs)} seeds, official test protocol: the last window of each "
           "engine compared with NASA's true RUL. RMSE and MAE are in cycles; the NASA score "
           "punishes late predictions harder (lower is better).")
a, c = st.columns([2, 3])
sel = next(r for r in runs if r["args"]["seed"] == seed)
with a:
    st.pyplot(plot_pred_vs_true(sel["test_true"], sel["test_pred"]))
with c:
    st.markdown(
        "**Honest reading**\n\n"
        "- A typical error of about 11 cycles (MAE), 15 (RMSE), in line with published LSTM baselines.\n"
        "- About half the predictions are too late. A late prediction is the dangerous direction.\n"
        "- Simulated data, one operating condition (FD001), 100 test engines.\n"
        "- This model is separate from the vibration, sound and image models: no dataset has all of them.")
