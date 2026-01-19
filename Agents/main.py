import streamlit as st
import os
import openai
from config import STREAMLIT_CONFIG, APP_VERSION, OPENAI_API_KEY

def main():
    # Set page config with logo as favicon
    config = STREAMLIT_CONFIG.copy()
    # Use relative path for logo (works locally and on Streamlit Cloud)
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
    
    if logo_path:
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
        padding-top: 0.2rem !important;
        padding-bottom: 0.5rem !important;
        background-color: var(--background);
    }
    .header-container {
        display: flex;
        align-items: flex-start;
        justify-content: flex-end;
        gap: 12px;
        margin-bottom: -0.3rem !important;
        margin-top: 0 !important;
    }
    .logo-container {
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.2rem;
    }
    .title-version-container {
        display: none;
    }
    hr {
        margin-top: -0.5rem !important;
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
    
    # Use the logo_path found earlier for the header display
    
    # Create compact header layout with logo above version on the right
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
            <h2 style="color: var(--primary-color); margin: 0; font-size: 1.1rem; font-weight: 400; text-align: right;">v{APP_VERSION}</h2>
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
    if 'created_bots' not in st.session_state:
        st.session_state.created_bots = []
    if 'delete_confirm' not in st.session_state:
        st.session_state.delete_confirm = {}
    if 'editing_bot' not in st.session_state:
        st.session_state.editing_bot = None
    if 'used_numbers' not in st.session_state:
        st.session_state.used_numbers = set()
    
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
        
        # Use form to handle submission and clear input
        with st.form("agent_creation_form", clear_on_submit=True):
            agent_description = st.text_area(
                "Enter a detailed description of the AI agent you want to build:",
                placeholder="Example: An AI agent that helps users create marketing content for social media posts...",
                height=150,
                key="agent_description_input"
            )
            
            # Create button
            submitted = st.form_submit_button("Create", type="primary", use_container_width=True)
            
            if submitted:
                if agent_description and agent_description.strip():
                    try:
                        # Generate bot name and short description using OpenAI
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that creates concise bot names and descriptions. Respond in JSON format with 'name' and 'description' fields. Name should be 2-4 words, description should be 1-2 sentences."},
                                {"role": "user", "content": f"Based on this agent description, create a catchy name and a short description:\n\n{agent_description}"}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.7
                        )
                        
                        import json
                        bot_data = json.loads(response.choices[0].message.content)
                        bot_name = bot_data.get("name", "AI Agent")
                        bot_desc = bot_data.get("description", agent_description[:100])
                        
                        # Generate unique 3-digit number
                        import random
                        while True:
                            bot_number = random.randint(100, 999)
                            if bot_number not in st.session_state.used_numbers:
                                st.session_state.used_numbers.add(bot_number)
                                break
                        
                        # Add bot to session state
                        bot_id = len(st.session_state.created_bots)
                        st.session_state.created_bots.append({
                            "id": bot_id,
                            "number": bot_number,
                            "name": bot_name,
                            "description": bot_desc,
                            "full_description": agent_description
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating bot: {str(e)}")
                else:
                    st.warning("Please enter an agent description first.")
        
        # Display all created bots
        if st.session_state.created_bots:
            st.markdown("---")
            st.markdown("### Created Agents")
            
            for bot in st.session_state.created_bots:
                with st.container():
                    # Check if this bot is being edited
                    is_editing = st.session_state.editing_bot == bot['id']
                    
                    if is_editing:
                        # Edit mode
                        bot_number = bot.get('number', 'N/A')
                        st.markdown(f"**Editing Agent #{bot_number}**")
                        
                        with st.form(f"edit_form_{bot['id']}", clear_on_submit=False):
                            edited_description = st.text_area(
                                "Edit agent description:",
                                value=bot['full_description'],
                                height=150,
                                key=f"edit_desc_{bot['id']}"
                            )
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                save_clicked = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                            with col_cancel:
                                cancel_clicked = st.form_submit_button("✗ Cancel", use_container_width=True)
                            
                            if save_clicked and edited_description and edited_description.strip():
                                try:
                                    # Regenerate name based on new description
                                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                                    response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant that creates concise bot names and descriptions. Respond in JSON format with 'name' and 'description' fields. Name should be 2-4 words, description should be 1-2 sentences."},
                                            {"role": "user", "content": f"Based on this agent description, create a catchy name and a short description:\n\n{edited_description}"}
                                        ],
                                        response_format={"type": "json_object"},
                                        temperature=0.7
                                    )
                                    
                                    import json
                                    bot_data = json.loads(response.choices[0].message.content)
                                    new_name = bot_data.get("name", "AI Agent")
                                    new_desc = bot_data.get("description", edited_description[:100])
                                    
                                    # Update bot (keep same number and id)
                                    for i, b in enumerate(st.session_state.created_bots):
                                        if b['id'] == bot['id']:
                                            st.session_state.created_bots[i]['name'] = new_name
                                            st.session_state.created_bots[i]['description'] = new_desc
                                            st.session_state.created_bots[i]['full_description'] = edited_description
                                            break
                                    
                                    st.session_state.editing_bot = None
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating bot: {str(e)}")
                            
                            if cancel_clicked:
                                st.session_state.editing_bot = None
                                st.rerun()
                    else:
                        # Display mode
                        col1, col2, col3 = st.columns([1, 8, 1])
                        
                        with col1:
                            if bot_path:
                                st.image(bot_path, width=80)
                            else:
                                st.info("🤖")
                            
                            # Handle bots that might not have a number (created before this feature)
                            bot_number = bot.get('number', 'N/A')
                            if bot_number == 'N/A' and 'number' not in bot:
                                # Generate a number for old bots without one
                                import random
                                while True:
                                    bot_number = random.randint(100, 999)
                                    if bot_number not in st.session_state.used_numbers:
                                        st.session_state.used_numbers.add(bot_number)
                                        bot['number'] = bot_number
                                        break
                            
                            # Display number below bot, left-aligned
                            st.markdown(f"<div style='text-align: left; margin-top: 5px;'><strong>#{bot.get('number', bot_number)}</strong></div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**{bot['name']}**")
                            st.markdown(f"{bot['description']}")
                        
                        with col3:
                            delete_key = f"delete_{bot['id']}"
                            edit_key = f"edit_{bot['id']}"
                            confirm_key = f"confirm_{bot['id']}"
                            
                            if st.session_state.delete_confirm.get(confirm_key, False):
                                st.warning(f"Delete {bot['name']}?")
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button("✓ Yes", key=f"yes_{bot['id']}", type="primary", use_container_width=True):
                                        # Remove bot from list and free up the number
                                        if 'number' in bot:
                                            st.session_state.used_numbers.discard(bot['number'])
                                        st.session_state.created_bots = [b for b in st.session_state.created_bots if b['id'] != bot['id']]
                                        st.session_state.delete_confirm[confirm_key] = False
                                        st.rerun()
                                with col_no:
                                    if st.button("✗ No", key=f"no_{bot['id']}", use_container_width=True):
                                        st.session_state.delete_confirm[confirm_key] = False
                                        st.rerun()
                            else:
                                col_edit, col_delete = st.columns(2)
                                with col_edit:
                                    if st.button("✏️", key=edit_key, help="Edit this agent", use_container_width=True):
                                        st.session_state.editing_bot = bot['id']
                                        st.rerun()
                                with col_delete:
                                    if st.button("🗑️", key=delete_key, help="Delete this agent", use_container_width=True):
                                        st.session_state.delete_confirm[confirm_key] = True
                                        st.rerun()
                    
                    st.markdown("---")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
