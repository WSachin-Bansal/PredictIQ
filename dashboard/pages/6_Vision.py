"""Vision: good-vs-defect verdict on MVTec images, with a Grad-CAM heatmap."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

DASH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DASH), str(DASH.parent / "src")]
from inference import diagnose_image, load_vision  # noqa: E402
from utils import CHECKPOINTS  # noqa: E402
from vision_utils import (defect_detection_table, load_vision_runs, mask_path,  # noqa: E402
                          overlay, plot_defect_detection, resolve_image)

st.set_page_config(page_title="Vision", page_icon="🔍", layout="wide")
st.title("Vision")
st.caption("Is this part good or defective, and where did the model look?")

runs = load_vision_runs()
ckpts = sorted(p.stem for p in CHECKPOINTS.glob("vision_*_seed*.pt"))
if not runs or not ckpts:
    st.error("No vision results/checkpoints found. Train with `python -m predictiq.training.train_vision`.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def model_for(tag):
    return load_vision(tag)


top = st.columns([2, 1, 2, 2])
tag = top[0].selectbox("Model (categories, seed)", ckpts, format_func=lambda t: t.replace("vision_", ""))
run = next((r for r in runs if r["file"] == f"{tag}.json"), None)
if run is None:
    st.error(f"results/{tag}.json is missing.")
    st.stop()
imgs = run["test_images"]
cats = sorted({i["category"] for i in imgs})
cat = top[1].selectbox("Category", cats)
pool = [i for i in imgs if i["category"] == cat]
types = ["any"] + sorted({i["defect_type"] for i in pool})
dtype = top[2].selectbox("Defect type", types)
if dtype != "any":
    pool = [i for i in pool if i["defect_type"] == dtype]
names = [f"{i['defect_type']}/{Path(i['path'].replace(chr(92), '/')).name}" for i in pool]
if top[3].button("🎲 Random image"):
    st.session_state["vis_pos"] = int(np.random.randint(len(pool)))
pos = min(st.session_state.get("vis_pos", 0), len(pool) - 1)
pick = st.selectbox("Image (only images the model never saw)", range(len(pool)), index=pos,
                    format_func=lambda k: names[k], key=f"vis_img_{tag}_{cat}_{dtype}")
item = pool[pick]

c1, c2 = st.columns(2)
reveal = c1.toggle("Reveal true label", value=False)
target = c2.radio("Heatmap shows", ["What the model decided", "Evidence for 'defect'"], horizontal=True,
                  help="Red = regions that pushed the decision most.")

try:
    path = resolve_image(item["path"], item["category"])
    pil = Image.open(path)
except (FileNotFoundError, ValueError):
    st.error("Could not find this image under `data/raw/mvtec`. Is the dataset in that folder?")
    st.stop()

p_def, cam, img = diagnose_image(model_for(tag), pil, target=1 if target.startswith("Evidence") else None)
verdict = "DEFECT" if p_def >= 0.5 else "GOOD"

m = st.columns(3)
m[0].metric("Model says", verdict, f"P(defect) = {p_def:.0%}", delta_color="off")
if reveal:
    truth = "DEFECT" if item["label"] == 1 else "GOOD"
    ok = truth == verdict
    m[1].metric("True label", f"{truth} ({item['defect_type']})", "✓ correct" if ok else "✗ wrong",
                delta_color="normal" if ok else "inverse")
else:
    m[1].metric("True label", "hidden", "tick 'Reveal true label'", delta_color="off")
m[2].metric("Decision threshold", "50%", help="Flagged as defect when P(defect) is at least 50%.")

a, b, c = st.columns(3)
a.image(img, caption="Image given to the model", use_container_width=True)
b.image(overlay(img, cam), caption="Grad-CAM heatmap (red = important)", use_container_width=True)
mp = mask_path(path) if reveal else None
if mp is not None:
    c.image(Image.open(mp).resize(img.shape[:2][::-1]), caption="True defect location (from MVTec)",
            use_container_width=True)
else:
    c.info("Reveal the true label to see the real defect location (when the image has one).")

with st.expander("What am I looking at?"):
    st.markdown(
        "- The model (a ResNet18 pretrained on everyday photos, with its last block retrained) outputs "
        "the probability that the part is defective.\n"
        "- **Grad-CAM** highlights the image regions that most influenced the decision. If the red "
        "area sits on the real defect, the model looks at the right thing; if it sits on the "
        "background, be suspicious.\n"
        "- The third picture is the true defect mask from the dataset. Compare it with the heatmap.")

st.header("How well does it work?")
accs = [r["test"] for r in runs if r["categories"] == run["categories"]]
k = st.columns(5)
for col, (name, key) in zip(k, [("Accuracy", "accuracy"), ("Macro-F1", "f1"), ("AUC", "roc_auc"),
                                ("Defects caught", "defect_recall"), ("Good passed", "good_recall")]):
    v = np.array([a[key] for a in accs])
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    col.metric(name, f"{v.mean():.3f}" if key == "roc_auc" else f"{v.mean():.1%}",
               f"± {sd:.3f}" if key == "roc_auc" else f"± {sd * 100:.1f} pts", delta_color="off")
st.caption(f"Mean over {len(accs)} seeds on about {run['split_sizes']['test']} test images per seed "
           "(the seeds also change the split).")
left, right = st.columns([3, 2])
with left:
    st.pyplot(plot_defect_detection(defect_detection_table([r for r in runs if r["categories"] == run["categories"]])))
    st.caption("Red: defect types (higher is better). Green: good parts wrongly flagged (lower is better).")
with right:
    st.markdown(
        "**Honest reading**\n\n"
        "- Large defects are found almost every time.\n"
        "- Subtle screw defects (thread, front) are often missed.\n"
        "- About 1 in 4 defects is missed: a baseline, not an inspection system.\n"
        "- Only 2 of 15 categories; every defect type was seen in training.")
