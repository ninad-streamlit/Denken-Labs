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
    @media (max-width: 768px) {
        .main-header {
            flex-direction: column !important;
            text-align: center !important;
        }
        .stTitle {
            font-size: 1.5rem !important;
        }
        .logo-image {
            width: 200px !important;
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
    
    # Header - responsive
    col1, col2 = st.columns([4, 1])
    with col1:
        # Display logo above the title
        logo_path = os.path.join(os.path.dirname(__file__), "agents", "Logo-DenkenLabs.png")
        try:
            if os.path.exists(logo_path):
                st.image(logo_path, width=250)
        except Exception as e:
            pass
        
        # Create a container with title and version on the same line
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0;">Denken Labs</h1>
            </div>
            <div style="text-align: right;">
                <h2 style="color: #1f77b4; margin: 0; font-size: 1.5rem; font-weight: 400;">v{APP_VERSION}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Empty column for layout consistency
        pass
    
    st.markdown("---")
    
    # Main content - centered and mobile-friendly
    st.markdown("""
    <div class="main-content" style='text-align: center; padding: 50px 0;'>
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
