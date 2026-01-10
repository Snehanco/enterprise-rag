import streamlit as st
import requests
import json
from typing import Dict, Any

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

# --- Page Config ---
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling ---
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
    }
    .chunk-box {
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        background-color: #fafafa;
    }
    .score-tag {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .file-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Helper Functions ---
def make_api_request(endpoint: str, payload: Dict[str, Any]):
    """Generic wrapper for API calls with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection Error: Is the FastAPI backend running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e}")
        st.json(response.json())  # Show detailed error from backend
        return None


def display_evidence(relevant_chunks):
    """Renders the evidence chunks in a clean, expandable format."""
    if not relevant_chunks:
        st.info("No detailed evidence chunks returned.")
        return

    st.markdown("### 🔍 Source Evidence")

    for i, chunk in enumerate(relevant_chunks):
        # Determine icon based on score
        score = chunk.get("score", 0)
        icon = "🟢" if score > 5.0 else "🟡" if score > 3.0 else "🔴"

        source_name = chunk.get("source", "Unknown")
        page_num = chunk.get("page_number", "N/A")
        file_name = chunk.get("file_name", "Unknown File")

        with st.expander(f"{icon} Evidence #{i+1}: {file_name} (Page {page_num})"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Entity/Protocol:** `{source_name}`")
            with col2:
                st.markdown(f"**Relevance:** `{score}`")

            st.markdown("**Extract:**")
            st.info(chunk.get("content", ""))

            # Show file path/name as a footer tag
            st.markdown(f"📄 **Source File:** `{file_name}`")


# --- Sidebar Navigation ---
st.sidebar.title("🩺 Medical RAG Chatbot")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Select Functionality:",
    ["Generate Treatment Plan", "Search Patient Records", "Search Protocols"],
)

st.sidebar.markdown("---")
# st.sidebar.info(
#     "**System Status:**\n" "✅ Hybrid Search Active\n" "✅ ColBERT Reranking Active"
# )

# --- Main Interface ---

if mode == "Generate Treatment Plan":
    st.title("📋 Treatment Plan Generator")
    st.markdown("_Synthesizes Patient Data with Medical Protocols_")

    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input(
            "Patient ID (Document Name)",
            value="",
            placeholder="e.g., patient_name",
        )
    with col2:
        disease_name = st.text_input(
            "Disease/Protocol Name",
            value="",
            placeholder="e.g., Diabetes",
        )

    query = st.text_area(
        "Clinical Question / Instruction",
        height=100,
        placeholder="Describe the patient's condition and ask for a plan...",
    )

    if st.button("Generate Plan", type="primary"):
        if not query or not patient_id or not disease_name:
            st.warning("Please fill in all fields.")
        else:
            with st.spinner(
                "🔄 Retrieving records, cross-referencing protocols, and generating plan..."
            ):
                payload = {
                    "query": query,
                    "patient_id": patient_id,
                    "disease_name": disease_name,
                }
                data = make_api_request("/generate/treatment-plan", payload)

                if data:
                    st.success("Analysis Complete")
                    st.markdown("### 💡 Recommended Treatment Plan")
                    st.markdown(data.get("answer"))

                    st.divider()
                    display_evidence(data.get("relevant_chunks"))

elif mode == "Search Patient Records":
    st.title("👤 Patient Record Search")
    st.markdown("_Deep search within specific patient history_")

    patient_id = st.text_input("Patient ID", placeholder="e.g., patient_001")
    query = st.text_input(
        "Search Query", placeholder="e.g., historical blood pressure trends"
    )

    if st.button("Search Records"):
        with st.spinner("Searching patient archives..."):
            payload = {"query": query, "patient_id": patient_id}
            data = make_api_request("/search/patient", payload)

            if data:
                st.markdown("### 📝 Analysis")
                st.write(data.get("answer"))
                st.divider()
                display_evidence(data.get("relevant_chunks"))

elif mode == "Search Protocols":
    st.title("📚 Medical Protocol Search")
    st.markdown("_Consult organizational guidelines and SOPs_")

    disease_name = st.text_input(
        "Disease / Protocol Name", placeholder="e.g., cancer_protocol"
    )
    query = st.text_input(
        "Search Query", placeholder="e.g., contraindications for chemotherapy"
    )

    if st.button("Consult Protocols"):
        with st.spinner("Consulting medical database..."):
            payload = {"query": query, "disease_name": disease_name}
            data = make_api_request("/search/protocol", payload)

            if data:
                st.markdown("### 📝 Protocol Summary")
                st.write(data.get("answer"))
                st.divider()
                display_evidence(data.get("relevant_chunks"))
