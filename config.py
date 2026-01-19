import os
from dotenv import load_dotenv

# Version number - increment by 0.001 for each code change
APP_VERSION = "0.069"

# Load environment variables (for local development)
load_dotenv()

# OpenAI Configuration
# Try Streamlit secrets first (for Streamlit Cloud), then fall back to environment variables (for local)
OPENAI_API_KEY = None
try:
    import streamlit as st
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
    try:
        # Try to access secrets (works on Streamlit Cloud)
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    except (StreamlitSecretNotFoundError, AttributeError, KeyError, TypeError):
        # Secrets file not found (local development) or not accessible
        pass
except (ImportError, AttributeError):
    # Streamlit not available or StreamlitSecretNotFoundError not found
    pass

# Fall back to environment variables if secrets not found
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file or Streamlit Cloud secrets.")

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Denken Labs",
    "page_icon": "🤖",
    "layout": "wide",  # Keep wide for desktop, but we'll add mobile-specific CSS
    "initial_sidebar_state": "expanded"
}
