# Google OAuth Setup Guide

## Prerequisites

1. A Google account
2. Access to Google Cloud Console

## Step-by-Step Setup

### 1. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter a project name (e.g., "AI Agent System")
4. Click "Create"

### 2. Enable Google+ API

1. In the Google Cloud Console, go to "APIs & Services" → "Library"
2. Search for "Google+ API"
3. Click on it and press "Enable"

### 3. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in the required fields:
     - App name: "AI Agent System"
     - User support email: your email
     - Developer contact: your email
   - Add your email to test users
   - Save and continue through all steps

4. For Application type, choose "Web application"
5. Add authorized redirect URIs:
   - `http://localhost:8501` (for local development)
   - `https://yourdomain.com` (for production, if applicable)

### 4. Get Your Credentials

1. After creating the OAuth client, you'll see:
   - Client ID
   - Client Secret
2. Copy these values

### 5. Configure Environment Variables

Add the following to your `.env` file:

```env
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_client_id_here
GOOGLE_CLIENT_SECRET=your_client_secret_here
GOOGLE_REDIRECT_URI=http://localhost:8501
```

### 6. Install Dependencies

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

### 7. Test the Setup

1. Start the application:
   ```bash
   streamlit run main.py
   ```

2. You should see a login page with a "Login with Google" button
3. Click the button and complete the OAuth flow
4. You should be redirected back to the main application

## Troubleshooting

### Common Issues

1. **"Site cannot be reached" or "This site can't be reached" error**
   - **Cause:** The redirect URI in your `.env` file doesn't match the port your app is running on
   - **Solution:**
     1. Check the URL in your browser (e.g., `http://localhost:8503`)
     2. Update `GOOGLE_REDIRECT_URI` in your `.env` file to match your current port:
        ```env
        GOOGLE_REDIRECT_URI=http://localhost:8503
        ```
     3. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
     4. Click on your OAuth 2.0 Client ID
     5. Under "Authorized redirect URIs", add the redirect URI matching your current port
     6. Save and restart your app
   - **Tip:** Add multiple redirect URIs for different ports (8501, 8502, 8503, etc.) to avoid this issue

2. **"redirect_uri_mismatch" error**
   - Make sure the redirect URI in your Google Cloud Console matches exactly with `GOOGLE_REDIRECT_URI` in your `.env` file
   - The redirect URI must match character-for-character (including `http://` vs `https://`)

3. **"access_denied" error**
   - Make sure your email is added to the test users in the OAuth consent screen

4. **"invalid_client" error**
   - Double-check your `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in the `.env` file
   - Make sure there are no extra spaces or quotes in the `.env` file

5. **"This app isn't verified" warning**
   - This is normal for development. Click "Advanced" → "Go to AI Agent System (unsafe)"

### Security Notes

- Never commit your `.env` file to version control
- Use environment variables in production
- Regularly rotate your OAuth credentials
- Monitor usage in Google Cloud Console

## Production Deployment

For production deployment:

1. Update the OAuth consent screen to "Published" status
2. Add your production domain to authorized redirect URIs
3. Update `GOOGLE_REDIRECT_URI` to your production URL
4. Consider using a more secure redirect URI pattern

