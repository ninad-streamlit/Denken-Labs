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
    
    # Mobile-responsive CSS
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 10px;
    }
    .logo-container {
        flex-shrink: 0;
    }
    .title-version-container {
        flex: 1;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column !important;
            text-align: center !important;
            gap: 10px !important;
        }
        .title-version-container {
            flex-direction: column !important;
            gap: 5px !important;
        }
        .stTitle {
            font-size: 1.5rem !important;
        }
        .logo-image {
            width: 150px !important;
            max-width: 100% !important;
        }
        .main-content {
            padding: 20px 0 !important;
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
    
    # Create compact header layout
    st.markdown("""
    <div class="header-container">
        <div class="logo-container">
    """, unsafe_allow_html=True)
    
    try:
        if logo_path:
            st.image(logo_path, width=120)
        else:
            # Debug: show which paths were checked
            st.info(f"Logo not found. Checked: {logo_paths}")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
    
    st.markdown(f"""
        </div>
        <div class="title-version-container">
            <h1 style="margin: 0; font-size: 2rem;">Denken Labs</h1>
            <h2 style="color: #1f77b4; margin: 0; font-size: 1.2rem; font-weight: 400;">v{APP_VERSION}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content - more compact
    st.markdown("""
    <div class="main-content" style='text-align: center; padding: 30px 0;'>
    """, unsafe_allow_html=True)
    
    st.markdown("## Welcome to Denken Labs")
    st.markdown("**Build your own AI agent eco-system with ease**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Build your own agent button
    if st.button("🚀 Build Your Own Agent", type="primary", use_container_width=True):
        st.info("🚀 Agent builder coming soon!")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
