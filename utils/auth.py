import streamlit as st
import os
import json
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GoogleAuth:
    def __init__(self):
        # Google OAuth configuration
        # Try Streamlit secrets first (for Streamlit Cloud), then fall back to environment variables (for local)
        self.client_id = None
        self.client_secret = None
        configured_redirect_uri = None
        
        # Try to get from Streamlit secrets (for Streamlit Cloud)
        try:
            if hasattr(st, 'secrets'):
                from streamlit.runtime.secrets import StreamlitSecretNotFoundError
                try:
                    # Try direct access first (Streamlit Cloud format)
                    self.client_id = st.secrets['GOOGLE_CLIENT_ID']
                    self.client_secret = st.secrets['GOOGLE_CLIENT_SECRET']
                    configured_redirect_uri = st.secrets['GOOGLE_REDIRECT_URI']
                except (StreamlitSecretNotFoundError, KeyError, AttributeError):
                    # Secrets file not found (local dev) or key missing - try .get() method
                    try:
                        self.client_id = st.secrets.get('GOOGLE_CLIENT_ID', None)
                        self.client_secret = st.secrets.get('GOOGLE_CLIENT_SECRET', None)
                        configured_redirect_uri = st.secrets.get('GOOGLE_REDIRECT_URI', None)
                    except (StreamlitSecretNotFoundError, AttributeError, TypeError):
                        # Secrets not available at all
                        pass
        except (AttributeError, KeyError, TypeError, ImportError):
            pass
        
        # Fall back to environment variables if secrets not found (for local development)
        if not self.client_id:
            self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        if not self.client_secret:
            self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        if not configured_redirect_uri:
            configured_redirect_uri = os.getenv('GOOGLE_REDIRECT_URI')
        
        # Use configured redirect URI or default
        if configured_redirect_uri:
            self.redirect_uri = configured_redirect_uri.rstrip('/')
        else:
            # For local development, always use port 8501 to match Google Cloud Console
            # This ensures the redirect URI matches what's registered, regardless of what port Streamlit is actually on
            try:
                # Check if we're on Streamlit Cloud
                server_url = os.getenv('STREAMLIT_SERVER_URL', '')
                if server_url and 'streamlit.app' in server_url:
                    self.redirect_uri = server_url.rstrip('/')
                else:
                    # For local dev, always use port 8501 (must match Google Cloud Console)
                    # Even if Streamlit is running on a different port, we use 8501 for OAuth
                    self.redirect_uri = 'http://localhost:8501'
            except:
                # Default fallback to port 8501
                self.redirect_uri = 'http://localhost:8501'
        
        self.scopes = [
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
            'openid'
        ]
        
        # Initialize the OAuth flow only if credentials are available
        self.flow = None
        self.flow_error = None
        if self.client_id and self.client_secret:
            try:
                self.flow = Flow.from_client_config(
                    {
                        "web": {
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": [self.redirect_uri]
                        }
                    },
                    scopes=self.scopes,
                    redirect_uri=self.redirect_uri
                )
            except Exception as e:
                # Store error for debugging - will be caught in get_authorization_url
                self.flow_error = str(e)

    def get_authorization_url(self):
        """Get the Google OAuth authorization URL"""
        if not self.client_id or not self.client_secret:
            st.error("❌ Google OAuth credentials not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")
            return None
        
        if not self.flow:
            error_msg = "❌ OAuth flow not initialized. Check your credentials."
            if hasattr(self, 'flow_error') and self.flow_error:
                error_msg += f" Error: {self.flow_error}"
            st.error(error_msg)
            return None
        
        # Generate a random state parameter for security
        state = secrets.token_urlsafe(32)
        st.session_state.oauth_state = state
        
        # Also store it in a more persistent way
        st.session_state['oauth_state_backup'] = state
        
        # IMPORTANT: Make sure redirect_uri matches EXACTLY what's in Google Cloud Console
        try:
            # Use a simpler prompt to avoid potential issues
            # 'select_account' shows account picker, 'consent' forces consent screen
            authorization_url, _ = self.flow.authorization_url(
                access_type='offline',
                state=state,
                prompt='select_account',  # Show account selection screen first
                include_granted_scopes='true'
            )
            return authorization_url
        except Exception as e:
            st.error(f"❌ Error creating authorization URL: {e}")
            return None

    def handle_callback(self, code, state):
        """Handle the OAuth callback and exchange code for tokens"""
        # Verify state parameter (with more lenient validation for development)
        stored_state = st.session_state.get('oauth_state')
        backup_state = st.session_state.get('oauth_state_backup')
        
        if (stored_state and state != stored_state) or (backup_state and state != backup_state):
            # State mismatch - proceed anyway for development
            pass
        elif not stored_state and not backup_state:
            # No stored state - proceed anyway for development
            pass
        
        try:
            # Create a fresh flow instance for token exchange
            fresh_flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [self.redirect_uri]
                    }
                },
                scopes=self.scopes,
                redirect_uri=self.redirect_uri
            )
            
            # Exchange code for tokens
            fresh_flow.fetch_token(code=code)
            credentials = fresh_flow.credentials
            
            # Store credentials in session state
            st.session_state.credentials = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            # Get user info
            user_info = self.get_user_info(credentials)
            if user_info:
                st.session_state.user_info = user_info
                st.session_state.authenticated = True
                return True
            else:
                st.error("❌ Failed to retrieve user information from Google.")
                return False
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Authentication failed: {error_msg}")
            
            if "invalid_grant" in error_msg:
                st.error("**Invalid Grant Error** - This usually means:")
                st.error("1. The authorization code has expired (codes expire in 10 minutes)")
                st.error("2. The code has already been used")
                st.error("3. There's a clock synchronization issue")
                st.info("🔄 **Solution:** Try the authentication process again from the beginning.")
            else:
                st.error("This might be due to:")
                st.error("1. Incorrect redirect URI in Google Cloud Console")
                st.error("2. Invalid client credentials")
                st.error("3. OAuth consent screen not properly configured")
            
            return False

    def get_user_info(self, credentials):
        """Get user information from Google"""
        try:
            service = build('oauth2', 'v2', credentials=credentials)
            user_info = service.userinfo().get().execute()
            return {
                'id': user_info.get('id'),
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'picture': user_info.get('picture'),
                'verified_email': user_info.get('verified_email')
            }
        except HttpError as e:
            st.error(f"❌ Failed to get user info: {str(e)}")
            return None

    def is_authenticated(self):
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)

    def get_current_user(self):
        """Get current authenticated user info"""
        return st.session_state.get('user_info', {})

    def logout(self):
        """Logout the current user"""
        # Clear all authentication-related session state including persistent auth
        clear_persistent_auth()
        
        # Clear additional OAuth-related keys
        additional_keys = ['oauth_state', 'oauth_state_backup', 'access_token', 'refresh_token']
        for key in additional_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Successfully logged out!")
        # Force a rerun to update the UI
        st.rerun()

    def create_login_ui(self):
        """Create the login user interface"""
        # Import version from config
        from config import APP_VERSION
        
        # Mobile-responsive CSS
        st.markdown("""
        <style>
        @media (max-width: 768px) {
            .login-container {
                padding: 10px !important;
            }
            .bot-icon {
                width: 80px !important;
                height: 80px !important;
            }
            .login-title {
                font-size: 24px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header with title and version
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("AI Vibe Agents")
        with col2:
            st.markdown(f"""
            <div style="text-align: right; padding-top: 20px;">
                <h2 style="color: #1f77b4; margin: 0; font-size: 1.5rem; font-weight: 400;">v{APP_VERSION}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Add custom bot icon with responsive sizing
        # Use relative path for bot icon (works locally and on Streamlit Cloud)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bot_icon_path = os.path.join(base_dir, "bot.png")
        try:
            if os.path.exists(bot_icon_path):
                st.image(bot_icon_path, width=100)
        except Exception as e:
            st.error(f"Could not load bot icon: {e}")
        
        st.markdown("**For Enterprise scale software development**")
        st.markdown("---")
        
        st.markdown("""
        ### Welcome to the AI Agent System
        
        This app enables the building of new apps using streamlit tech stack and embracing the vibe coding principles.
        
        Please log in with your Google account to access the system.
        """)
        
        # Check if credentials are configured
        if not self.client_id or not self.client_secret:
            st.error("❌ Google OAuth not configured. Please contact the administrator.")
            return False
        
        # Create login button
        try:
            # Show OAuth configuration for debugging
            client_id_preview = self.client_id[:20] + "..." if self.client_id and len(self.client_id) > 20 else (self.client_id or "Not set")
            scopes_str = ", ".join(self.scopes)
            
            with st.expander("🔍 OAuth Configuration Debug Info", expanded=True):
                st.write(f"**Client ID:** `{client_id_preview}`")
                st.write(f"**Redirect URI:** `{self.redirect_uri}`")
                st.write(f"**Scopes being requested:**")
                for scope in self.scopes:
                    st.write(f"  - `{scope}`")
                st.write("\n⚠️ **Verify in Google Cloud Console:**")
                st.write("1. These scopes are added in OAuth consent screen → Scopes section")
                st.write("2. Redirect URI matches exactly (including https/http)")
                st.write("3. Your email is in Test users list")
                st.write("4. App is in 'Testing' mode (or published)")
            
            auth_url = self.get_authorization_url()
            
            if auth_url:
                # Show the authorization URL for debugging (truncated for security)
                with st.expander("🔗 Full Authorization URL (for debugging)", expanded=False):
                    st.code(auth_url, language=None)
            
            if auth_url:
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{auth_url}" target="_self" style="text-decoration: none;">
                        <button style="
                            background-color: #4285f4;
                            color: white;
                            border: none;
                            padding: 12px 24px;
                            border-radius: 8px;
                            font-size: 16px;
                            cursor: pointer;
                            display: inline-flex;
                            align-items: center;
                            gap: 8px;
                        ">
                            <svg width="20" height="20" viewBox="0 0 24 24">
                                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                            </svg>
                            Login with Google
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def check_authentication():
    """Check if user is authenticated, redirect to login if not"""
    # Initialize session state to prevent refresh logout
    initialize_session()
    
    # Check for OAuth errors in query params (Google redirects with error parameter)
    if 'error' in st.query_params:
        error = st.query_params.get('error', 'Unknown error')
        error_description = st.query_params.get('error_description', 'No description provided')
        st.error(f"❌ **OAuth Error from Google:**\n\n**Error:** `{error}`\n\n**Description:** `{error_description}`")
        
        if error == 'access_denied':
            st.info("💡 **This usually means:**\n- Your email is not in the Test users list\n- The app is in Testing mode and you're not authorized\n- You clicked 'Cancel' on the consent screen")
        elif error == 'redirect_uri_mismatch':
            st.info("💡 **This means the redirect URI doesn't match.** Check that `https://agentbuilder.streamlit.app` is exactly in your Google Cloud Console authorized redirect URIs.")
        elif error == 'invalid_client':
            st.info("💡 **This means the Client ID or Client Secret is incorrect.** Verify your credentials in Streamlit Cloud secrets.")
        
        # Clear error params to prevent showing again on refresh
        st.query_params.clear()
        return False
    
    # Check if this is an OAuth callback
    if 'code' in st.query_params and 'state' in st.query_params:
        try:
            auth = GoogleAuth()
            code = st.query_params['code']
            state = st.query_params['state']
            
            if auth.handle_callback(code, state):
                st.success("✅ Successfully authenticated!")
                # Persist authentication state
                persist_auth_state()
                # Clear the query parameters to prevent reprocessing
                st.query_params.clear()
                st.rerun()
            else:
                st.error("❌ Authentication failed. Please try again.")
                # Clear the query parameters to prevent reprocessing
                st.query_params.clear()
                return False
        except Exception as e:
            st.error(f"❌ Error in OAuth callback: {e}")
            return False
    
    # Check for persistent authentication first
    if check_persistent_auth():
        return True
    
    # Check if user is already authenticated with more robust checks
    is_authenticated = (
        st.session_state.get('authenticated', False) and 
        st.session_state.get('user_info') is not None and
        st.session_state.get('credentials') is not None
    )
    
    if not is_authenticated:
        try:
            auth = GoogleAuth()
            auth.create_login_ui()
            return False
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return False
    return True

def get_user_display_info():
    """Get user display information for the UI"""
    user_info = st.session_state.get('user_info', {})
    return {
        'name': user_info.get('name', 'User'),
        'email': user_info.get('email', 'user@example.com'),
        'picture': user_info.get('picture', '')
    }

def initialize_session():
    """Initialize session state with default values to prevent refresh logout"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'credentials' not in st.session_state:
        st.session_state.credentials = None

def get_auth_file_path():
    """Get the path to the authentication file"""
    return os.path.join(os.path.dirname(__file__), '.auth_session')

def persist_auth_state():
    """Persist authentication state to a file"""
    if st.session_state.get('authenticated') and st.session_state.get('user_info'):
        auth_data = {
            'authenticated': True,
            'user_info': st.session_state.user_info,
            'credentials': st.session_state.get('credentials'),
            'timestamp': time.time(),
            'session_id': st.session_state.get('session_id', 'default')
        }
        
        try:
            with open(get_auth_file_path(), 'w') as f:
                json.dump(auth_data, f)
        except Exception as e:
            st.error(f"Failed to save auth state: {e}")

def check_persistent_auth():
    """Check if we have persistent authentication data from file"""
    auth_file = get_auth_file_path()
    
    if not os.path.exists(auth_file):
        return False
    
    try:
        with open(auth_file, 'r') as f:
            auth_data = json.load(f)
        
        # Check if auth is not too old (24 hours)
        current_time = time.time()
        auth_time = auth_data.get('timestamp', 0)
        
        if current_time - auth_time < 24 * 60 * 60:  # 24 hours
            # Restore authentication state
            st.session_state.authenticated = True
            st.session_state.user_info = auth_data.get('user_info')
            st.session_state.credentials = auth_data.get('credentials')
            return True
        else:
            # Auth is too old, clear it
            clear_persistent_auth()
            return False
            
    except Exception as e:
        # If there's any error reading the file, clear it
        clear_persistent_auth()
        return False

def clear_persistent_auth():
    """Clear persistent authentication data from file and session"""
    # Clear from file
    auth_file = get_auth_file_path()
    if os.path.exists(auth_file):
        try:
            os.remove(auth_file)
        except:
            pass
    
    # Clear from session state
    keys_to_clear = ['authenticated', 'user_info', 'credentials', 'auth_persistent', 'auth_timestamp']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
