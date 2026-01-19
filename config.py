import os
from dotenv import load_dotenv

# Version number - increment by 0.1 for each code change
APP_VERSION = "0.5"

# Load environment variables (for local development)
load_dotenv()

# OpenAI Configuration
# Try Streamlit secrets first (for Streamlit Cloud), then fall back to environment variables (for local)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except (AttributeError, KeyError, ImportError):
    # If st.secrets is not available or key doesn't exist, use environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file or Streamlit Cloud secrets.")

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "AI Vibe Agents",
    "page_icon": "🤖",
    "layout": "wide",  # Keep wide for desktop, but we'll add mobile-specific CSS
    "initial_sidebar_state": "expanded"
}
