import os
from dotenv import load_dotenv

# Version number - increment by 0.001 for each code change
APP_VERSION = "0.147"

# Load environment variables (for local development)
load_dotenv()

# OpenAI Configuration
# Lazy function to get API key (secrets only available after Streamlit initializes)
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY")
                if api_key:
                    return api_key
            except (AttributeError, KeyError, TypeError):
                pass
    except (ImportError, AttributeError):
        pass
    
    # Fall back to environment variables (for local development)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file or Streamlit Cloud secrets.")
    return api_key

# For backward compatibility, try to get it at import time if possible
# But don't raise error if Streamlit isn't initialized yet
OPENAI_API_KEY = None
try:
    # Try environment variable first (works before Streamlit initializes)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
except:
    pass

# If not found, it will be None and will need to be retrieved via get_openai_api_key() after Streamlit initializes

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Denken Labs",
    "page_icon": "🤖",
    "layout": "wide",  # Keep wide for desktop, but we'll add mobile-specific CSS
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get Help": None,
        "Report a bug": None,
        "About": "Denken Labs - Begin your space adventure"
    }
    # Note: Dark mode is allowed. CSS ensures text is light/readable in dark mode
}
