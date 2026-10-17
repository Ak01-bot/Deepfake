"""
app.py
Streamlit dashboard for the Deepfake Detection Tool.
Forensics + confidence fusion visualization. Model verdict/Grad-CAM
sections activate once Charlie's model is integrated (Week 6).

Run with: streamlit run dashboard/app.py


import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "forensics"))
sys.path.insert(0, str(PROJECT_ROOT / "fusion"))

from forensics_pipeline import run_forensics  # noqa: E402
from confidence_fusion import fuse_confidence  # noqa: E402

st.set_page_config(page_title="Deepfake Forensics Dashboard", layout="wide")

st.title("🔍 Deepfake & Fake Media Detection — Forensics Module")
st.caption("Forensic analysis (hashing, metadata, ELA, malware scan) + confidence fusion preview. Model predictions arrive in a later phase.")

uploaded_files = st.file_uploader(
    "Upload one or more images/videos to analyze",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    accept_multiple_files=True,
)


def render_result(uploaded_file, results: dict):
    st.subheader(f"Results for `{uploaded_file.name}`")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔑 File Hash (SHA-256)")
        if results.get("sha256"):
            st.code(results["sha256"], language="text")
        else:
            st.warning(f"Hash failed: {results.get('sha256_error', 'unknown error')}")

    with col2:
        st.markdown("### 🛡️ Malware Scan")
        scan = results.get("malware_scan")
        if scan:
            if scan["status"] == "clean":
                st.success(f"Clean — {scan['details']}")
            elif scan["status"] == "infected":
                st.error(f"Infected! — {scan['details']}")
            else:
                st.warning(f"Scan error — {scan['details']}")
        else:
            st.warning(f"Malware scan failed: {results.get('malware_scan_error', 'unknown error')}")

    st.markdown("### 📋 Metadata")
    if results.get("metadata"):
        with st.expander("View full metadata", expanded=False):
            st.json(results["metadata"])
    else:
        st.warning(f"Metadata extraction failed: {results.get('metadata_error', 'unknown error')}")

    st.markdown("### 🧪 Error Level Analysis")
    if results.get("ela"):
        ela = results["ela"]
        ela_col1, ela_col2 = st.columns([1, 1])
        with ela_col1:
            st.image(ela["ela_image_path"], caption="ELA difference map")
        with ela_col2:
            st.metric("Anomaly score", ela["anomaly_score"])
            st.caption("Higher = more inconsistent compression history = more suspicious.")
    elif results.get("ela_note"):
        st.info(results["ela_note"])
    else:
        st.warning(f"ELA failed: {results.get('ela_error', 'unknown error')}")

    # Forensics-only fusion preview — model_confidence=None until Week 6
    st.markdown("### ⚖️ Combined Confidence (preview)")
    fusion = fuse_confidence(results, model_confidence=None, model_verdict=None)
    st.info(
        f"Fused verdict: **{fusion['fused_verdict']}** (score: {fusion['fused_score']}) — "
        "forensics-only signal for now; model confidence will be added once integrated."
    )

    with st.expander("Raw forensics output (JSON)"):
        st.json(results)
    with st.expander("Raw fusion output (JSON)"):
        st.json(fusion)


if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            with st.spinner(f"Running forensic analysis on {uploaded_file.name}..."):
                results = run_forensics(tmp_path)
            render_result(uploaded_file, results)
        except Exception as e:
            # forensics_pipeline already catches per-check errors internally;
            # this guards against something totally unreadable (e.g. a
            # corrupted upload) from killing the rest of the batch.
            st.error(f"Unexpected error analyzing `{uploaded_file.name}`: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

        st.divider()
else:
    st.info("Upload one or more files above to begin.")



    """



"""
app.py
Streamlit dashboard for the Deepfake Detection Tool.
Forensics + model classification + confidence fusion, all in one view.
Model classification runs on video files only (predict_video() is
frame-based); images get forensics + fusion-preview only.

Run with: streamlit run dashboard/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "forensics"))
sys.path.insert(0, str(PROJECT_ROOT / "fusion"))
sys.path.insert(0, str(PROJECT_ROOT / "data_pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "reports"))

from forensics_pipeline import run_forensics    # noqa: E402
from confidence_fusion import fuse_confidence    # noqa: E402
from predict import predict_video                # noqa: E402
from pdf_report import generate_pdf_report        # noqa: E402

XCEPTION_CKPT = str(PROJECT_ROOT / "checkpoints" / "xception_best.pth")
TEMPORAL_CKPT = str(PROJECT_ROOT / "checkpoints" / "temporal_attention_cdf.pth")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

st.set_page_config(page_title="Deepfake Forensics Dashboard", layout="wide")

st.title("🔍 Deepfake & Fake Media Detection — Full Pipeline")
st.caption("Forensic analysis + CNN/LSTM/RNN model classification + combined confidence score.")

uploaded_files = st.file_uploader(
    "Upload one or more images/videos to analyze",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    accept_multiple_files=True,
)


def render_result(uploaded_file, forensics_result, model_result, model_error):
    st.subheader(f"Results for `{uploaded_file.name}`")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔑 File Hash (SHA-256)")
        if forensics_result.get("sha256"):
            st.code(forensics_result["sha256"], language="text")
        else:
            st.warning(f"Hash failed: {forensics_result.get('sha256_error', 'unknown error')}")

    with col2:
        st.markdown("### 🛡️ Malware Scan")
        scan = forensics_result.get("malware_scan")
        if scan:
            if scan["status"] == "clean":
                st.success(f"Clean — {scan['details']}")
            elif scan["status"] == "infected":
                st.error(f"Infected! — {scan['details']}")
            else:
                st.warning(f"Scan error — {scan['details']}")
        else:
            st.warning(f"Malware scan failed: {forensics_result.get('malware_scan_error', 'unknown error')}")

    st.markdown("### 📋 Metadata")
    if forensics_result.get("metadata"):
        with st.expander("View full metadata", expanded=False):
            st.json(forensics_result["metadata"])
    else:
        st.warning(f"Metadata extraction failed: {forensics_result.get('metadata_error', 'unknown error')}")

    st.markdown("### 🧪 Error Level Analysis")
    if forensics_result.get("ela"):
        ela = forensics_result["ela"]
        ela_col1, ela_col2 = st.columns([1, 1])
        with ela_col1:
            st.image(ela["ela_image_path"], caption="ELA difference map")
        with ela_col2:
            st.metric("Anomaly score", ela["anomaly_score"])
            st.caption("Higher = more inconsistent compression history = more suspicious.")
    elif forensics_result.get("ela_note"):
        st.info(forensics_result["ela_note"])
    else:
        st.warning(f"ELA failed: {forensics_result.get('ela_error', 'unknown error')}")

    st.markdown("### 🧠 Model Classification")
    if model_result:
        verdict = model_result["verdict"]
        confidence = model_result["confidence"]
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            if verdict == "fake":
                st.error(f"Verdict: **{verdict.upper()}**")
            else:
                st.success(f"Verdict: **{verdict.upper()}**")
        with m_col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with m_col3:
            st.metric("Frames used", model_result.get("num_frames_used", "N/A"))
        st.caption(
            f"Real probability: {model_result.get('real_probability', 0):.2%} · "
            f"Fake probability: {model_result.get('fake_probability', 0):.2%} · "
            f"Frames discarded: {model_result.get('num_frames_discarded', 0)}"
        )
    elif model_error:
        st.warning(f"Model classification unavailable: {model_error}")
    else:
        st.info("Model classification runs on video files only — this is a static image.")

    st.markdown("### ⚖️ Combined Confidence")
    fusion = fuse_confidence(
        forensics_result,
        model_confidence=model_result["confidence"] if model_result else None,
        model_verdict=model_result["verdict"] if model_result else None,
    )
    if model_result:
        st.info(f"Fused verdict: **{fusion['fused_verdict']}** (score: {fusion['fused_score']})")
    else:
        st.info(
            f"Fused verdict: **{fusion['fused_verdict']}** (score: {fusion['fused_score']}) — "
            "forensics-only signal; no model classification available for this file."
        )

    with st.expander("Raw forensics output (JSON)"):
        st.json(forensics_result)
    with st.expander("Raw fusion output (JSON)"):
        st.json(fusion)

    if st.button(f"📄 Generate PDF report for `{uploaded_file.name}`", key=f"pdf_{uploaded_file.name}"):
        pdf_model_result = None
        if model_result:
            pdf_model_result = {
                "verdict": model_result["verdict"],
                "confidence": f"{model_result['confidence']:.2%}",
            }
        output_path = f"/tmp/{Path(uploaded_file.name).stem}_evidence_report.pdf"
        generate_pdf_report(forensics_result, pdf_model_result, fusion, output_path=output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF report",
                data=f.read(),
                file_name=f"{Path(uploaded_file.name).stem}_evidence_report.pdf",
                mime="application/pdf",
                key=f"dl_{uploaded_file.name}",
            )


if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        model_result = None
        model_error = None

        try:
            with st.spinner(f"Running forensic analysis on {uploaded_file.name}..."):
                forensics_result = run_forensics(tmp_path)

            if suffix in VIDEO_EXTENSIONS:
                try:
                    with st.spinner(f"Running model classification on {uploaded_file.name}..."):
                        model_result = predict_video(
                            video_path=str(tmp_path),
                            xception_checkpoint=XCEPTION_CKPT,
                            temporal_checkpoint=TEMPORAL_CKPT,
                        )
                except ValueError as e:
                    model_error = str(e)  # predict_video raises this on no-face-detected
                except Exception as e:
                    model_error = f"Unexpected error: {e}"

            render_result(uploaded_file, forensics_result, model_result, model_error)

        except Exception as e:
            st.error(f"Unexpected error analyzing `{uploaded_file.name}`: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

        st.divider()
else:
    st.info("Upload one or more files above to begin.")