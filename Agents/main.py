import streamlit as st
import os
from config import STREAMLIT_CONFIG, APP_VERSION

def main():
    # Set page config with logo as favicon
    config = STREAMLIT_CONFIG.copy()
    # Use relative path for logo (works locally and on Streamlit Cloud)
    logo_path = os.path.join(os.path.dirname(__file__), "agents", "Logo-DenkenLabs.png")
    if os.path.exists(logo_path):
        config['page_icon'] = logo_path
    else:
        config['page_icon'] = "🤖"  # Fallback to emoji if image not found
    st.set_page_config(**config)
    
    # Mobile-responsive CSS with minimal spacing and logo-matching colors
    st.markdown("""
    <style>
    :root {
        --primary-color: #2563eb;
        --primary-dark: #1e40af;
        --primary-light: #3b82f6;
        --accent-color: #0ea5e9;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --background: #ffffff;
        --border-color: #e2e8f0;
    }
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        background-color: var(--background);
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2px !important;
        margin-top: 0 !important;
    }
    .logo-container {
        flex-shrink: 0;
    }
    .title-version-container {
        flex: 1;
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        border-color: var(--border-color) !important;
        background-color: var(--border-color) !important;
    }
    .main-content {
        padding: 8px 0 !important;
        margin-top: 5px !important;
        color: var(--text-primary);
    }
    h1 {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        color: var(--text-primary) !important;
    }
    h2 {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        color: var(--text-primary) !important;
    }
    .stMarkdown h2 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.5rem !important;
        color: var(--text-primary) !important;
    }
    .stMarkdown p {
        margin-bottom: 0.3rem !important;
        margin-top: 0.2rem !important;
        color: var(--text-secondary) !important;
    }
    .stMarkdown strong {
        color: var(--text-primary) !important;
    }
    .stButton > button {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: var(--primary-dark) !important;
    }
    .stInfo {
        background-color: #dbeafe !important;
        border-left: 4px solid var(--primary-color) !important;
    }
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column !important;
            text-align: center !important;
            gap: 8px !important;
        }
        .title-version-container {
            flex-direction: column !important;
            gap: 3px !important;
        }
        .stTitle {
            font-size: 1.5rem !important;
        }
        .logo-image {
            width: 200px !important;
            max-width: 100% !important;
        }
        .main-content {
            padding: 10px 0 !important;
        }
        .stButton > button {
            width: 100% !important;
            font-size: 1rem !important;
            padding: 0.75rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Compact header with logo, title, and version in one row
    # Try multiple possible paths for the logo
    logo_paths = [
        os.path.join(os.path.dirname(__file__), "agents", "Logo-DenkenLabs.png"),
        os.path.join(os.path.dirname(__file__), "Agents", "Logo-DenkenLabs.png"),
        os.path.join(os.path.dirname(__file__), "Agents", "agents", "Logo-DenkenLabs.png"),
    ]
    logo_path = None
    for path in logo_paths:
        if os.path.exists(path):
            logo_path = path
            break
    
    # Create compact header layout with logo on the left
    st.markdown("""
    <div class="header-container">
        <div class="logo-container">
    """, unsafe_allow_html=True)
    
    try:
        if logo_path:
            st.image(logo_path, width=280)
        else:
            # Debug: show which paths were checked
            st.info(f"Logo not found. Checked: {logo_paths}")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
    
    st.markdown(f"""
        </div>
        <div class="title-version-container">
            <div></div>
            <h2 style="color: var(--primary-color); margin: 0; font-size: 1.1rem; font-weight: 400;">v{APP_VERSION}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content - minimal spacing
    st.markdown("""
    <div class="main-content" style='text-align: center;'>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'show_agent_builder' not in st.session_state:
        st.session_state.show_agent_builder = False
    
    if not st.session_state.show_agent_builder:
        st.markdown("## Welcome to Denken Labs")
        st.markdown("**Build your own AI agent eco-system with ease**")
        
        # Build your own agent button - compact
        if st.button("🚀 Build Your Own Agent", type="primary", use_container_width=True):
            st.session_state.show_agent_builder = True
            st.rerun()
    else:
        # Show bot logo and agent description input
        # Find bot.png file
        bot_paths = [
            os.path.join(os.path.dirname(__file__), "bot.png"),
            os.path.join(os.path.dirname(__file__), "Agents", "bot.png"),
            os.path.join(os.path.dirname(__file__), "..", "bot.png"),
        ]
        bot_path = None
        for path in bot_paths:
            if os.path.exists(path):
                bot_path = path
                break
        
        # Display bot logo
        if bot_path:
            st.image(bot_path, width=100)
        else:
            st.info("🤖 Bot logo")
        
        # Agent description text input
        st.markdown("### Describe Your Agent")
        agent_description = st.text_area(
            "Enter a detailed description of the AI agent you want to build:",
            placeholder="Example: An AI agent that helps users create marketing content for social media posts...",
            height=150,
            key="agent_description"
        )
        
        # Back button
        if st.button("← Back", use_container_width=False):
            st.session_state.show_agent_builder = False
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
