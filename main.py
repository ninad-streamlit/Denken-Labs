import streamlit as st
import os
from config import STREAMLIT_CONFIG
from utils.auth import check_authentication, get_user_display_info, GoogleAuth

# Version number - increment by 0.1 for each code change
APP_VERSION = "0.4"

def main():
    # Set page config with bot icon as favicon
    config = STREAMLIT_CONFIG.copy()
    # Use relative path for bot icon (works locally and on Streamlit Cloud)
    bot_icon_path = os.path.join(os.path.dirname(__file__), "bot.png")
    if os.path.exists(bot_icon_path):
        config['page_icon'] = bot_icon_path
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
        .bot-icon {
            width: 80px !important;
            height: 80px !important;
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
    
    # Check authentication
    if not check_authentication():
        return
    
    # Get user info for display
    user_info = get_user_display_info()
    
    # Header - responsive
    col1, col2 = st.columns([4, 1])
    with col1:
        # Create a container with title and version on the same line
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0;">🤖 AI Vibe Agents</h1>
            </div>
            <div style="text-align: right;">
                <h2 style="color: #1f77b4; margin: 0; font-size: 1.5rem; font-weight: 400;">v{APP_VERSION}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Add custom bot icon
        bot_icon_path = os.path.join(os.path.dirname(__file__), "bot.png")
        try:
            if os.path.exists(bot_icon_path):
                st.image(bot_icon_path, width=100)
        except Exception as e:
            pass
    
    with col2:
        st.markdown(f"👤 **{user_info['name']}**")
        if st.button("🚪 Logout", help="Sign out of the application", use_container_width=True):
            auth = GoogleAuth()
            auth.logout()
            return
    
    st.markdown("---")
    
    # Main content - centered and mobile-friendly
    st.markdown("""
    <div class="main-content" style='text-align: center; padding: 50px 0;'>
    """, unsafe_allow_html=True)
    
    st.markdown("## Welcome to AI Vibe Agents")
    st.markdown("**Build your own AI agent with ease**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Build your own agent button
    if st.button("🚀 Build Your Own Agent", type="primary", use_container_width=True):
        st.info("🚀 Agent builder coming soon!")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
