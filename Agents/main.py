import streamlit as st
import os
from config import STREAMLIT_CONFIG
from utils.auth import check_authentication, get_user_display_info, GoogleAuth

def main():
    # Set page config with bot icon as favicon
    config = STREAMLIT_CONFIG.copy()
    config['page_icon'] = "/Users/ninadjoshi/Documents/GitHub/Denken-Labs/Projects/bot.png"
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
        st.title("🤖 AI Vibe Agents")
        # Add custom bot icon
        try:
            st.image("/Users/ninadjoshi/Documents/GitHub/Denken-Labs/Projects/bot.png", width=100)
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
