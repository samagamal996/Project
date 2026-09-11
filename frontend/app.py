import requests
import streamlit as st
from PIL import Image

# API Endpoint Configuration
import os
from dotenv import load_dotenv

load_dotenv()

# Read from env variable with a local fallback
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL}/api/v1/analyze"


st.set_page_config(
    page_title="Tell-Tale Dashboard Assistant",
    page_icon="🚗",
    layout="wide",
)

st.title("Car Dashboard Warning Light Assistant")
st.markdown(
    "Upload an image of your car's dashboard warning light and ask any question. "
    "Our AI detects the symbol and retrieves actionable steps directly from your owner's manual."
)

st.divider()

# Sidebar options
with st.sidebar:
    st.header("⚙️ Settings")
    server_status = "Disconnected"
    try:
        health_check = requests.get("http://127.0.0.1:8000/", timeout=2)
        if health_check.status_code == 200:
            server_status = "Connected"
    except Exception:
        pass

    st.write(f"**Backend API Status:** {server_status}")
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("- **Vision:** YOLOv8 Fine-Tuned")
    st.markdown("- **Vector Store:** ChromaDB")
    st.markdown("- **LLM:** Ollama (Llama 3)")

# UI Layout: Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Step 1: Upload Warning Light Image")
    uploaded_file = st.file_uploader(
        "Choose a dashboard image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image, caption="Uploaded Dashboard Image", use_container_width=True
        )

    st.subheader("Step 2: Ask Your Question")
    user_query = st.text_input(
        "Question about the warning light:",
        value="What action should I take right now for this warning light?",
        help="Tip: If the symbol isn't detected automatically, mention the color or icon shape (e.g., 'red battery light').",
    )

    submit_button = st.button("Analyze Dashboard", type="primary")

with col2:
    st.subheader("Analysis & Manual Instructions")

    if submit_button:
        if server_status == "Disconnected ❌":
            st.error(
                "Cannot connect to FastAPI backend! Ensure `uvicorn backend.main:app --reload` is running."
            )
        else:
            with st.spinner("Analyzing image and searching owner manual..."):
                try:
                    # Prepare payload for FastAPI Form/File request
                    files = (
                        {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }
                        if uploaded_file
                        else None
                    )
                    data = {"user_query": user_query}

                    # Call FastAPI backend
                    response = requests.post(
                        API_URL, data=data, files=files, timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # Display detected symbols
                        symbols = result.get("detected_symbols", [])
                        if symbols:
                            st.success(
                                f"**Detected Symbol(s):** {', '.join(symbols)}"
                            )
                        else:
                            st.warning(
                                "**No specific symbol detected.** (Falling back to query context)"
                            )

                        # Display generated answer
                        st.markdown("### **Recommended Action:**")
                        st.info(result.get("answer", "No answer generated."))

                        # Display sources
                        sources = result.get("sources", [])
                        if sources:
                            st.markdown("**Manual Sources:**")
                            for src in sources:
                                st.caption(f"• {src}")
                    else:
                        st.error(
                            f"API Error {response.status_code}: {response.text}"
                        )

                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    else:
        st.info("Upload an image or enter a question, then click **Analyze Dashboard**.")