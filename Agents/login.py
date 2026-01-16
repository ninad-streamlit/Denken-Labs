import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.auth import GoogleAuth
from config import STREAMLIT_CONFIG

def main():
    st.set_page_config(**STREAMLIT_CONFIG)
    
    # Initialize authentication
    auth = GoogleAuth()
    
    # Check if this is a callback from Google OAuth
    query_params = st.query_params
    
    if 'code' in query_params and 'state' in query_params:
        # Handle OAuth callback
        code = query_params['code']
        state = query_params['state']
        
        st.info("🔄 Processing authentication...")
        
        if auth.handle_callback(code, state):
            st.success("✅ Successfully authenticated!")
            st.balloons()
            
            # Clear query parameters to prevent re-processing
            st.query_params.clear()
            
            # Redirect to main app
            st.markdown("""
            <script>
                setTimeout(function() {
                    window.location.href = "/";
                }, 2000);
            </script>
            """, unsafe_allow_html=True)
            
            st.info("🔄 Redirecting to main application...")
        else:
            st.error("❌ Authentication failed. Please try again.")
            # Clear query parameters to prevent re-processing
            st.query_params.clear()
            auth.create_login_ui()
    else:
        # Show login interface
        auth.create_login_ui()

if __name__ == "__main__":
    main()
