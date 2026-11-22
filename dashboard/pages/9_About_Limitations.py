"""About & Limitations: what is real, what is simulated, what we do NOT claim."""
import streamlit as st

st.set_page_config(page_title="About & Limitations", page_icon="⚠️", layout="wide")
st.title("About & Limitations")
st.caption("Read this before trusting (or presenting) any number in the dashboard.")

st.header("What is real and what is simulated")
st.table({
    "Data": ["MAFAULDA (vibration + sound)", "NASA C-MAPSS (engine sensors)", "MVTec AD (images)"],
    "Nature": ["Real recordings from a laboratory test rig (UFRJ)",
               "Simulated jet-engine run-to-failure data",
               "Real photos of industrial objects with real and artificial defects"],
    "Used for": ["Fault type + severity bucket", "Remaining useful life (RUL)",
                 "Good / defect detection + heatmap"],
    "Has the other data's labels?": ["No RUL, no images", "No sound, no vibration, no images",
                                      "No vibration, no sound, no RUL"],
})
st.info("No single machine in our data has vibration, sound, images and a run-to-failure curve at the "
        "same time. Such a dataset does not exist publicly. So the three tracks are trained on "
        "different datasets and joined only at the model level.")

st.header("What we claim")
st.markdown(
    "- A 1D-CNN on vibration, a CNN on sound and a fusion of both can classify the 10 MAFAULDA "
    "conditions on a laboratory rig.\n"
    "- On random test recordings the scores are very high; on **unseen fault severities** they are "
    "clearly lower. We report both, and the second is the honest one.\n"
    "- An LSTM predicts engine RUL on C-MAPSS FD001 with an error in line with published baselines.")

st.header("What we do NOT claim")
st.markdown(
    "- **No real industrial results.** Everything is on one laboratory rig and on simulated engines.\n"
    "- **No RUL for MAFAULDA.** It has no run-to-failure data, and we do not invent it.\n"
    "- **No proven benefit of fusion.** On the standard split fusion equals vibration alone; on "
    "unseen severities it is slightly ahead, but the three seeds share one split, so this is not "
    "statistically conclusive.\n"
    "- **No claim that images, sound and vibration come from the same machine.** Any pairing of "
    "MVTec images with MAFAULDA windows is a labelled simulation for demonstration, not evidence.")

st.header("Known limitations")
st.markdown(
    "- **Near-duplicate recordings.** Neighbouring recordings of the same fault, severity and speed "
    "look alike, so the standard split is optimistic.\n"
    "- **Severity is coarse.** We use three buckets (low/medium/high) built from rank order within "
    "a class, not a physical fault size.\n"
    "- **One rig, one sensor placement.** We did not test other machines, other speeds held out, or noise.\n"
    "- **Small class `normal`** (49 recordings), which is why we use macro-F1 next to accuracy.\n"
    "- **C-MAPSS: FD001 only** (one operating condition and one fault mode), with a 100-engine test set.\n"
    "- **Unverified details:** the meaning of the six vibration channels and of the bearing-fault "
    "severity levels comes from the dataset documentation and has not been checked by us.\n"
    "- **Validation noise.** Validation scores jump from epoch to epoch, so small differences "
    "(about one point) are within noise.")

with st.expander("Questions you may be asked, and honest answers"):
    st.markdown(
        "**Why is the standard-split accuracy so high?** Because the test recordings come from the same "
        "fault settings as the training ones. That is why we also test on unseen severities.\n\n"
        "**Does fusion actually help?** Not on the standard split. On unseen severities it is slightly "
        "better and much more robust when sound is lost, but the evidence is limited.\n\n"
        "**Would this work on a real factory machine?** We do not know. It is evidence on a lab rig, not a deployment.\n\n"
        "**Where is the remaining-life prediction for the vibration data?** It does not exist; the "
        "data has no run-to-failure labels. RUL comes from the engine dataset only.\n\n"
        "**Is the image part connected to the vibration part?** Only in the combined model's design, "
        "and any joint demo is simulated and labelled as such.")
