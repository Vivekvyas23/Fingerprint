import streamlit as st
import os, zipfile
from roi_detector import detect_and_crop
from fingerprint_enhance import enhance_fingerprint

MODEL_PATH = "best_float16.tflite"
ROI_DIR = "output/rois"
FP_DIR = "output/fingerprints"

st.set_page_config("Fingerprint Processor", layout="centered")
st.title("🖐️ Fingerprint ROI & Binarization")

uploaded = st.file_uploader("Upload 4-finger image", type=["jpg","jpeg","png"])

if uploaded:
    os.makedirs("temp", exist_ok=True)
    img_path = f"temp/{uploaded.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded.read())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Process"):
        st.info("Detecting finger ROIs...")
        crops = detect_and_crop(img_path, MODEL_PATH, ROI_DIR)

        os.makedirs(FP_DIR, exist_ok=True)
        outputs = []

        for i, c in enumerate(crops):
            out = f"{FP_DIR}/finger_{i+1}_fp.png"
            enhance_fingerprint(c, out)
            outputs.append(out)

        st.success("Processing complete")

        for o in outputs:
            st.image(o, caption=os.path.basename(o), width=200)

        zip_path = "fingerprints.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for f in outputs:
                z.write(f, arcname=os.path.basename(f))

        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download ZIP", f, file_name="fingerprints.zip")
