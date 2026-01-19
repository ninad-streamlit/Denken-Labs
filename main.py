import streamlit as st
import os
import openai
from config import STREAMLIT_CONFIG, APP_VERSION, OPENAI_API_KEY
import random # Added for unique bot numbers
import json # Added for OpenAI response parsing
from io import BytesIO
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def generate_agent_example():
    """Generate a short random agent description example (around 10 words) inspired by Star Trek characters"""
    star_trek_roles = [
        "Captain", "Science Officer", "Chief Engineer", "Medical Officer", 
        "Security Officer", "Communications Officer", "Navigator", "Counselor",
        "Tactical Officer", "Operations Officer", "Helmsman", "First Officer"
    ]
    star_trek_traits = [
        "bold and brave", "logical and analytical", "curious and scientific", 
        "caring and compassionate", "protective and vigilant", "diplomatic and friendly",
        "precise and focused", "wise and understanding", "strategic and tactical",
        "efficient and organized", "skilled and experienced", "loyal and trustworthy"
    ]
    star_trek_qualities = [
        "exploring new worlds", "solving complex problems", "helping others in need",
        "protecting the crew", "discovering new knowledge", "maintaining peace",
        "navigating through space", "healing and caring", "communicating with aliens",
        "analyzing data", "engineering solutions", "leading missions"
    ]
    
    role = random.choice(star_trek_roles)
    trait = random.choice(star_trek_traits)
    quality = random.choice(star_trek_qualities)
    
    # Generate short Star Trek-style examples (around 10 words)
    examples = [
        f"A {role} who is {trait} and loves {quality}.",
        f"An agent who is a {role}, {trait}, specializing in {quality}.",
        f"A {trait} {role} that enjoys {quality} and teamwork.",
        f"An {role} agent who is {trait} and excels at {quality}.",
        f"A {role} who is {trait} and dedicated to {quality}."
    ]
    
    return random.choice(examples)

def generate_story_question_example(story_title, story_content, existing_questions=None):
    """Generate a relevant example question about the story with variety"""
    # Extract key elements from the story for context
    story_preview = story_content[:300] if story_content else ""
    
    # Get list of already asked questions to avoid repetition
    asked_questions = existing_questions or []
    asked_questions_text = "\n".join([f"- {q}" for q in asked_questions]) if asked_questions else "None yet"
    
    # Different question types for variety
    question_types = [
        "about a specific character or agent",
        "about what happened during the mission",
        "about how the agents solved a problem",
        "about the outcome or ending",
        "about a specific event or moment",
        "about the teamwork or collaboration",
        "about a challenge they faced",
        "about what they discovered or learned"
    ]
    import random
    selected_type = random.choice(question_types)
    
    # Generate a question example using OpenAI with emphasis on variety
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse, relevant questions about children's stories. Generate ONE simple question that a child (5-10 years old) might ask about the story. IMPORTANT: Make each question DIFFERENT from previous questions. Vary the question type, focus, and perspective. The question should reference specific elements from the story. Keep it simple and engaging. Return ONLY the question, nothing else."},
                {"role": "user", "content": f"Story Title: {story_title}\n\nStory Preview: {story_preview}\n\nPreviously asked questions:\n{asked_questions_text}\n\nGenerate a NEW, DIFFERENT question that a child might ask about this story. Focus on: {selected_type}. Make sure it's completely different from the previously asked questions. Reference something specific from the story."}
            ],
            temperature=1.0,  # Higher temperature for more variety
            max_tokens=60
        )
        question = response.choices[0].message.content.strip()
        # Remove any quotes if present
        question = question.strip('"').strip("'")
        # Remove "Question:" prefix if present
        if question.lower().startswith("question:"):
            question = question[9:].strip()
        return question
    except Exception as e:
        # Fallback examples with more variety
        fallback_questions = [
            "What was the most exciting part of the mission?",
            "How did the agents work together?",
            "What problem did the team solve?",
            "What was each agent's special skill?",
            "How did the mission end?",
            "What challenge did they face?",
            "What did the agents discover?",
            "How did they help each other?",
            "What was the biggest surprise?",
            "What made the mission successful?"
        ]
        # Filter out already asked questions
        available_questions = [q for q in fallback_questions if q not in asked_questions]
        if available_questions:
            return random.choice(available_questions)
        else:
            return random.choice(fallback_questions)

def generate_mission_example():
    """Generate a random Star Trek-style mission example (around 20 words)"""
    star_trek_missions = [
        "explore an unknown planet and make contact with friendly alien civilizations",
        "investigate strange energy readings from a distant nebula and discover its source",
        "rescue a stranded spaceship crew from a dangerous asteroid field",
        "establish diplomatic relations with a newly discovered planet's inhabitants",
        "study an unusual space anomaly that could reveal secrets about the universe",
        "help a peaceful alien species protect their homeworld from a natural disaster",
        "investigate reports of mysterious signals coming from an abandoned space station",
        "explore a new star system and map planets that might support life",
        "rescue scientists trapped on a research outpost during a solar storm",
        "discover the origin of ancient artifacts found floating in deep space",
        "mediate a peace agreement between two warring alien civilizations",
        "investigate a time distortion that threatens the fabric of space"
    ]
    mission_objectives = [
        "ensuring the safety of all crew members and new friends",
        "working together to solve complex problems and challenges",
        "using science, teamwork, and friendship to complete the mission",
        "gathering valuable information while respecting new cultures",
        "bringing peace and understanding to the galaxy"
    ]
    
    mission = random.choice(star_trek_missions)
    objective = random.choice(mission_objectives)
    
    # Generate Star Trek-style mission examples (~20 words)
    examples = [
        f"Mission: {mission}, {objective}.",
        f"Your mission is to {mission}, {objective}.",
        f"Explore and discover: {mission}, {objective}."
    ]
    
    return random.choice(examples)

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
        --purple-button: #6b46c1;
        --purple-button-dark: #553c9a;
        --green-button: #059669;
        --green-button-dark: #047857;
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
    /* Purple for agent creation form submit button */
    form[data-testid*="agent_creation_form"] button[kind="primary"],
    form:has(textarea[placeholder*="agent"]) button[kind="primary"],
    form:has(textarea[placeholder*="Agent"]) button[kind="primary"],
    div[data-testid*="stForm"]:has(textarea[placeholder*="agent"]) button[kind="primary"],
    div[data-testid*="stForm"]:has(textarea[placeholder*="Agent"]) button[kind="primary"] {
        background-color: #6b46c1 !important;
        border-color: #6b46c1 !important;
        color: white !important;
    }
    form[data-testid*="agent_creation_form"] button[kind="primary"]:hover,
    form:has(textarea[placeholder*="agent"]) button[kind="primary"]:hover,
    form:has(textarea[placeholder*="Agent"]) button[kind="primary"]:hover {
        background-color: #553c9a !important;
        border-color: #553c9a !important;
    }
    /* Dark green for mission form submit button */
    form[data-testid*="mission_form"] button[kind="primary"],
    form:has(textarea[placeholder*="mission"]) button[kind="primary"],
    form:has(textarea[placeholder*="Mission"]) button[kind="primary"],
    div[data-testid*="stForm"]:has(textarea[placeholder*="mission"]) button[kind="primary"],
    div[data-testid*="stForm"]:has(textarea[placeholder*="Mission"]) button[kind="primary"] {
        background-color: #059669 !important;
        border-color: #059669 !important;
        color: white !important;
    }
    form[data-testid*="mission_form"] button[kind="primary"]:hover,
    form:has(textarea[placeholder*="mission"]) button[kind="primary"]:hover,
    form:has(textarea[placeholder*="Mission"]) button[kind="primary"]:hover {
        background-color: #047857 !important;
        border-color: #047857 !important;
    }
    .stInfo {
        background-color: #dbeafe !important;
        border-left: 4px solid var(--primary-color) !important;
    }
    </style>
    <script>
    // ULTRA AGGRESSIVE button styling - intercept ALL style changes
    function forcePurpleButton(btn) {
        // Remove all existing styles and set our own
        btn.style.removeProperty('background-color');
        btn.style.removeProperty('background');
        btn.style.removeProperty('border-color');
        btn.style.removeProperty('border');
        btn.style.cssText = 'background-color: #6b46c1 !important; border-color: #6b46c1 !important; color: white !important;';
        btn.setAttribute('style', 'background-color: #6b46c1 !important; border-color: #6b46c1 !important; color: white !important;');
        btn.setAttribute('data-button-color', 'purple');
        
        // Override any Streamlit classes
        btn.classList.add('custom-purple-button');
        
        // Set hover handlers
        btn.onmouseenter = function() {
            this.style.cssText = 'background-color: #553c9a !important; border-color: #553c9a !important; color: white !important;';
        };
        btn.onmouseleave = function() {
            this.style.cssText = 'background-color: #6b46c1 !important; border-color: #6b46c1 !important; color: white !important;';
        };
    }
    
    function forceGreenButton(btn) {
        btn.style.removeProperty('background-color');
        btn.style.removeProperty('background');
        btn.style.removeProperty('border-color');
        btn.style.removeProperty('border');
        btn.style.cssText = 'background-color: #059669 !important; border-color: #059669 !important; color: white !important;';
        btn.setAttribute('style', 'background-color: #059669 !important; border-color: #059669 !important; color: white !important;');
        btn.setAttribute('data-button-color', 'green');
        btn.classList.add('custom-green-button');
        
        btn.onmouseenter = function() {
            this.style.cssText = 'background-color: #047857 !important; border-color: #047857 !important; color: white !important;';
        };
        btn.onmouseleave = function() {
            this.style.cssText = 'background-color: #059669 !important; border-color: #059669 !important; color: white !important;';
        };
    }
    
    function styleButtons() {
        // Get ALL buttons on the page
        var allButtons = document.querySelectorAll('button');
        allButtons.forEach(function(btn) {
            var text = (btn.textContent || btn.innerText || '').trim();
            
            // Find the form parent to check context
            var form = btn.closest('form');
            var formParent = form || btn.closest('[data-testid*="stForm"]') || btn.closest('[data-testid*="form"]');
            
            // Check for textarea to identify form type
            var textarea = formParent ? formParent.querySelector('textarea') : null;
            var placeholder = textarea ? (textarea.getAttribute('placeholder') || '') : '';
            
            // Purple for Build Your Own Agent button
            if (text.includes('Build Your Own Agent') || text.includes('🚀 Build Your Own Agent')) {
                forcePurpleButton(btn);
            }
            
            // Purple for Create button - target ALL Create buttons unless in mission form
            if (text === 'Create' || (text.includes('Create') && !text.includes('Mission') && !text.includes('Save'))) {
                var isMissionForm = placeholder && (placeholder.toLowerCase().includes('mission') || placeholder.toLowerCase().includes('Mission'));
                
                // If it's NOT a mission form, make it purple
                if (!isMissionForm && formParent) {
                    forcePurpleButton(btn);
                }
            }
            
            // Dark green for Activate Mission button
            if (text.includes('Activate Mission') || text.includes('🚀 Activate Mission')) {
                forceGreenButton(btn);
            }
        });
    }
    
    // Add global CSS that overrides everything
    var globalStyle = document.createElement('style');
    globalStyle.id = 'global-button-overrides';
    globalStyle.textContent = `
        button.custom-purple-button,
        button[data-button-color="purple"],
        form[data-testid*="agent_creation_form"] button[kind="primary"],
        form:has(textarea[placeholder*="agent"]) button[kind="primary"],
        form:has(textarea[placeholder*="Agent"]) button[kind="primary"],
        form:has(textarea[placeholder*="description"]) button[kind="primary"],
        form:has(textarea[placeholder*="Example"]) button[kind="primary"] {
            background-color: #6b46c1 !important;
            border-color: #6b46c1 !important;
            color: white !important;
        }
        button.custom-purple-button:hover,
        button[data-button-color="purple"]:hover {
            background-color: #553c9a !important;
            border-color: #553c9a !important;
        }
        button.custom-green-button,
        button[data-button-color="green"],
        form[data-testid*="mission_form"] button[kind="primary"],
        form:has(textarea[placeholder*="mission"]) button[kind="primary"],
        form:has(textarea[placeholder*="Mission"]) button[kind="primary"] {
            background-color: #059669 !important;
            border-color: #059669 !important;
            color: white !important;
        }
        button.custom-green-button:hover,
        button[data-button-color="green"]:hover {
            background-color: #047857 !important;
            border-color: #047857 !important;
        }
    `;
    if (!document.getElementById('global-button-overrides')) {
        document.head.appendChild(globalStyle);
    }
    
    // Intercept style attribute changes using MutationObserver
    var styleObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                var btn = mutation.target;
                var text = (btn.textContent || btn.innerText || '').trim();
                if (text === 'Create' || text.includes('Create')) {
                    var form = btn.closest('form');
                    var textarea = form ? form.querySelector('textarea') : null;
                    var placeholder = textarea ? (textarea.getAttribute('placeholder') || '') : '';
                    var isMissionForm = placeholder && (placeholder.toLowerCase().includes('mission'));
                    if (!isMissionForm) {
                        forcePurpleButton(btn);
                    }
                }
            }
        });
    });
    
    // Observe all buttons for style changes
    document.querySelectorAll('button').forEach(function(btn) {
        styleObserver.observe(btn, { attributes: true, attributeFilter: ['style', 'class'] });
    });
    
    // Run immediately and continuously
    styleButtons();
    setInterval(styleButtons, 100); // Run every 100ms
    
    // Also on specific events
    setTimeout(styleButtons, 50);
    setTimeout(styleButtons, 200);
    setTimeout(styleButtons, 500);
    setTimeout(styleButtons, 1000);
    
    window.addEventListener('load', function() {
        styleButtons();
        setInterval(styleButtons, 100);
    });
    
    document.addEventListener('DOMContentLoaded', function() {
        styleButtons();
        setInterval(styleButtons, 100);
    });
    
    // Use MutationObserver to catch dynamically added buttons
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    if (node.tagName === 'BUTTON') {
                        styleButtons();
                        styleObserver.observe(node, { attributes: true, attributeFilter: ['style', 'class'] });
                    } else if (node.querySelectorAll) {
                        var buttons = node.querySelectorAll('button');
                        buttons.forEach(function(btn) {
                            styleButtons();
                            styleObserver.observe(btn, { attributes: true, attributeFilter: ['style', 'class'] });
                        });
                    }
                }
            });
        });
        styleButtons();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Listen for Streamlit's custom events
    window.addEventListener('streamlit:rerun', function() {
        setTimeout(function() {
            styleButtons();
            document.querySelectorAll('button').forEach(function(btn) {
                styleObserver.observe(btn, { attributes: true, attributeFilter: ['style', 'class'] });
            });
        }, 100);
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Additional CSS for mobile responsiveness
    st.markdown("""
    <style>
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
    if 'mission_story' not in st.session_state:
        st.session_state.mission_story = ""
    if 'mission_story_title' not in st.session_state:
        st.session_state.mission_story_title = ""
    if 'story_qa_history' not in st.session_state:
        st.session_state.story_qa_history = []
    if 'story_question_example' not in st.session_state:
        st.session_state.story_question_example = ""
    if 'agent_example' not in st.session_state:
        st.session_state.agent_example = generate_agent_example()
    if 'mission_example' not in st.session_state:
        st.session_state.mission_example = generate_mission_example()
    
    if not st.session_state.show_agent_builder:
        st.markdown("## Welcome to Denken Labs")
        st.markdown("**Build your own AI agent eco-system with ease**")
        
        # Build your own agent button - compact (purple)
        st.markdown("""
        <style>
        button[kind="primary"]:contains("Build Your Own Agent"),
        div.stButton:has(> button:contains("Build Your Own Agent")) button {
            background-color: #6b46c1 !important;
            border-color: #6b46c1 !important;
        }
        button[kind="primary"]:contains("Build Your Own Agent"):hover,
        div.stButton:has(> button:contains("Build Your Own Agent")) button:hover {
            background-color: #553c9a !important;
            border-color: #553c9a !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🚀 Build Your Own Agent", type="primary", use_container_width=True, key="build_agent_btn"):
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
        st.markdown("**Create individual AI agents with specific capabilities and personalities. When you have 2 or more agents, they form a team and can be assigned missions to work together.**")
        
        # Use form to handle submission and clear input
        # Generate example if not exists (will be regenerated after agent creation)
        if 'agent_example' not in st.session_state:
            st.session_state.agent_example = generate_agent_example()
        
        with st.form("agent_creation_form", clear_on_submit=True):
            agent_description = st.text_area(
                "Enter a detailed description of the AI agent you want to build:",
                placeholder=f"Example: {st.session_state.agent_example}",
                height=150,
                key="agent_description_input"
            )
            
            # Add inline CSS right before the Create button to ensure it's purple
            st.markdown("""
            <style>
            /* Force purple on Create button in agent creation form */
            form[data-testid*="agent_creation_form"] button[kind="primary"],
            form[data-testid*="agent_creation_form"] > div:last-child button[kind="primary"] {
                background-color: #6b46c1 !important;
                border-color: #6b46c1 !important;
                color: white !important;
            }
            form[data-testid*="agent_creation_form"] button[kind="primary"]:hover,
            form[data-testid*="agent_creation_form"] > div:last-child button[kind="primary"]:hover {
                background-color: #553c9a !important;
                border-color: #553c9a !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create button (should be purple)
            submitted = st.form_submit_button("Create", type="primary", use_container_width=True)
            
            if submitted:
                # Use example if description is blank
                if not agent_description or not agent_description.strip():
                    agent_description = st.session_state.agent_example
                
                if agent_description and agent_description.strip():
                    try:
                        # Generate bot name, description, and elaborate character using OpenAI
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                        # Get existing agent names to avoid duplicates
                        existing_names = [bot.get('name', '').lower() for bot in st.session_state.created_bots]
                        
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a creative assistant that creates AI agent profiles. Respond in JSON format with 'name', 'description', and 'character' fields. The name should be inspired by cartoon characters, superheroes, or famous personalities - it should feel like an actual character name (e.g., 'Flash Writer', 'Captain Code', 'Sparky Bot') rather than an adjective. Make it unique and original, catchy, adventurous, and playful. Description should be 1-2 sentences, and character should be an elaborate personality profile (3-5 sentences) describing the agent's traits, working style, expertise, and approach."},
                                {"role": "user", "content": f"Based on this agent description, create a unique character name (inspired by cartoon characters/superheroes/famous personalities but original), short description, and elaborate character profile. Avoid these existing names: {', '.join(existing_names) if existing_names else 'none'}\n\nAgent description:\n{agent_description}"}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.9
                        )
                        
                        bot_data = json.loads(response.choices[0].message.content)
                        bot_name = bot_data.get("name", "AI Agent")
                        bot_desc = bot_data.get("description", agent_description[:100])
                        bot_character = bot_data.get("character", "A versatile AI agent ready to assist.")
                        
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
                            "character": bot_character,
                            "full_description": agent_description
                        })
                        
                        # Regenerate example for next agent creation
                        st.session_state.agent_example = generate_agent_example()
                        
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
                                    # Regenerate name, description, and character based on new description
                                    # Get existing agent names to avoid duplicates (excluding current bot)
                                    existing_names = [b.get('name', '').lower() for b in st.session_state.created_bots if b['id'] != bot['id']]
                                    
                                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                                    response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a creative assistant that creates AI agent profiles. Respond in JSON format with 'name', 'description', and 'character' fields. The name should be inspired by cartoon characters, superheroes, or famous personalities - it should feel like an actual character name (e.g., 'Flash Writer', 'Captain Code', 'Sparky Bot') rather than an adjective. Make it unique and original, catchy, adventurous, and playful. Description should be 1-2 sentences in SIMPLE English. Character should be a personality profile (3-5 sentences) in VERY SIMPLE English words that a 5-10 year old can understand. Use short sentences (6-10 words each). Describe the agent's traits, working style, expertise, and approach using simple words like 'help', 'work', 'team', 'friend', 'smart', 'kind', 'brave'."},
                                            {"role": "user", "content": f"Based on this agent description, create a unique character name (inspired by cartoon characters/superheroes/famous personalities but original), short description, and elaborate character profile. Avoid these existing names: {', '.join(existing_names) if existing_names else 'none'}\n\nAgent description:\n{edited_description}"}
                                        ],
                                        response_format={"type": "json_object"},
                                        temperature=0.9
                                    )
                                    
                                    bot_data = json.loads(response.choices[0].message.content)
                                    new_name = bot_data.get("name", "AI Agent")
                                    new_desc = bot_data.get("description", edited_description[:100])
                                    new_character = bot_data.get("character", "A versatile AI agent ready to assist.")
                                    
                                    # Update bot (keep same number and id)
                                    for i, b in enumerate(st.session_state.created_bots):
                                        if b['id'] == bot['id']:
                                            st.session_state.created_bots[i]['name'] = new_name
                                            st.session_state.created_bots[i]['description'] = new_desc
                                            st.session_state.created_bots[i]['character'] = new_character
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
                            # Display character profile if available
                            if bot.get('character'):
                                with st.expander("👤 Character Profile", expanded=False):
                                    st.markdown(f"*{bot['character']}*")
                        
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
        
        # Team mission assignment (when 2+ agents exist)
        if len(st.session_state.created_bots) >= 2:
            st.markdown("---")
            st.markdown("### 🎯 Assign Mission to Team")
            st.markdown(f"**Team Size:** {len(st.session_state.created_bots)} agents")
            
            # Initialize mission state
            if 'team_mission' not in st.session_state:
                st.session_state.team_mission = ""
            
            # Regenerate mission example on each view to show variety
            if 'mission_example' not in st.session_state or st.session_state.get('refresh_mission_example', False):
                st.session_state.mission_example = generate_mission_example()
                st.session_state.refresh_mission_example = False
            
            with st.form("mission_form", clear_on_submit=False):
                mission_description = st.text_area(
                    "Describe the mission for your team:",
                    placeholder=f"Example: {st.session_state.mission_example}",
                    height=150,
                    value=st.session_state.team_mission,
                    key="mission_description_input"
                )
                
                activate_clicked = st.form_submit_button("🚀 Activate Mission", type="primary", use_container_width=True)
                
                if activate_clicked:
                    # Use example if mission description is blank
                    if not mission_description or not mission_description.strip():
                        mission_description = st.session_state.mission_example
                    
                    if mission_description and mission_description.strip():
                        st.session_state.team_mission = mission_description
                        
                        # Generate collaboration story
                        try:
                            # Get agent names and descriptions
                            agent_list = []
                            for bot in st.session_state.created_bots:
                                agent_list.append(f"- {bot.get('name', 'Agent')} (#{bot.get('number', 'N/A')}): {bot.get('description', 'A helpful agent')}")
                            
                            agent_info = "\n".join(agent_list)
                            
                            client = openai.OpenAI(api_key=OPENAI_API_KEY)
                            story_response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a creative children's storyteller inspired by Star Trek adventures. Write engaging stories for children aged 5-10. Use VERY SIMPLE English words. Write short sentences (6-10 words each). CRITICAL: The story MUST have exactly 4 paragraphs separated by blank lines. Each paragraph MUST have exactly 5-6 sentences. Use double line breaks (\\n\\n) to separate paragraphs. Stories should be about teamwork, friendship, and space adventures. Respond in JSON format with 'title' and 'story' fields. Title should be catchy and fun (5-10 words). Story must have 4 paragraphs with 5-6 short sentences each, separated by \\n\\n."},
                                    {"role": "user", "content": f"Write a Star Trek-style adventure story with a catchy title about how these AI agents worked together to complete a mission:\n\nAgents:\n{agent_info}\n\nMission: {mission_description}\n\nCRITICAL REQUIREMENTS:\n1. Title: A fun, catchy Star Trek-style title (5-10 words)\n2. Story MUST have exactly 4 paragraphs\n3. Use double line breaks (\\n\\n) to separate each paragraph\n4. Each paragraph MUST have exactly 5-6 sentences\n5. Each sentence MUST be short (6-10 words only)\n6. Use VERY SIMPLE English words that a 5-10 year old can understand\n7. Paragraph 1: Agents gather, receive mission briefing, and plan together (5-6 sentences)\n8. Paragraph 2: They begin the mission, face first challenges (5-6 sentences)\n9. Paragraph 3: They overcome obstacles and work together creatively (5-6 sentences)\n10. Paragraph 4: Mission success, agents compliment and thank each other (5-6 sentences)\n\nIMPORTANT: Separate paragraphs with \\n\\n. Use simple words like 'help', 'work', 'team', 'friend', 'space', 'ship'. Avoid complex words. Respond in JSON with 'title' and 'story' fields."}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.8
                            )
                            
                            story_data = json.loads(story_response.choices[0].message.content)
                            story_title = story_data.get("title", "The Amazing Team Adventure")
                            story_content = story_data.get("story", "")
                            
                            # Validate story has paragraphs
                            if story_content and len(story_content.strip()) > 50:
                                # Ensure story has proper paragraph breaks
                                # Replace single \n with \n\n if not already present
                                if '\n\n' not in story_content and '\n' in story_content:
                                    # Split by \n and rejoin with \n\n
                                    paragraphs = [p.strip() for p in story_content.split('\n') if p.strip()]
                                    story_content = '\n\n'.join(paragraphs)
                                
                                st.session_state.mission_story_title = story_title
                                st.session_state.mission_story = story_content
                                
                                # Clear Q&A history for the new mission
                                st.session_state.story_qa_history = []
                                st.session_state.story_question_example = ""
                            else:
                                # If story is too short or empty, try again with a simpler prompt
                                raise ValueError("Generated story is too short or empty")
                        except Exception as e:
                            # Show error and try again with a simpler prompt
                            st.error(f"Error generating story: {str(e)}. Retrying with simpler prompt...")
                            try:
                                # Retry with a simpler, more direct prompt
                                retry_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Write simple stories for children aged 5-10. Use very simple words. Write exactly 4 paragraphs. Each paragraph has 5-6 short sentences. Separate paragraphs with two line breaks. Respond in JSON with 'title' and 'story' fields."},
                                        {"role": "user", "content": f"Write a simple story in 4 paragraphs. Each paragraph has 5-6 short sentences (6-10 words each). Use simple words.\n\nStory about: {mission_description}\n\nAgents: {agent_info}\n\nParagraph 1: Agents meet and plan (5-6 sentences).\nParagraph 2: They start the mission (5-6 sentences).\nParagraph 3: They solve problems together (5-6 sentences).\nParagraph 4: Mission done! They celebrate (5-6 sentences).\n\nUse \\n\\n to separate paragraphs. JSON format with 'title' and 'story'."}
                                    ],
                                    response_format={"type": "json_object"},
                                    temperature=0.9
                                )
                                retry_data = json.loads(retry_response.choices[0].message.content)
                                retry_title = retry_data.get("title", "The Amazing Team Adventure")
                                retry_story = retry_data.get("story", "")
                                
                                if retry_story and len(retry_story.strip()) > 50:
                                    # Fix paragraph breaks if needed
                                    if '\n\n' not in retry_story and '\n' in retry_story:
                                        paragraphs = [p.strip() for p in retry_story.split('\n') if p.strip()]
                                        retry_story = '\n\n'.join(paragraphs)
                                    
                                    st.session_state.mission_story_title = retry_title
                                    st.session_state.mission_story = retry_story
                                    st.success("Story generated successfully!")
                                else:
                                    raise ValueError("Retry story generation also failed")
                            except Exception as retry_error:
                                st.session_state.mission_story_title = "The Amazing Team Adventure"
                                st.session_state.mission_story = f"Once upon a time, the agents worked together to complete the mission! They planned, executed, and thanked each other for their wonderful teamwork! Error: {str(retry_error)}"
                                st.error(f"Story generation failed. Please try again. Error: {str(retry_error)}")
                        
                        # Regenerate mission example for next mission
                        st.session_state.mission_example = generate_mission_example()
                        
                        st.rerun()
                    else:
                        st.warning("Please enter a mission description first.")
            
            # Display collaboration story if mission is activated
            if st.session_state.team_mission and st.session_state.mission_story:
                st.markdown("---")
                st.markdown("### 📖 The Story of Teamwork")
                
                # PDF download button
                col_title, col_pdf = st.columns([3, 1])
                with col_title:
                    # Display story title
                    if st.session_state.mission_story_title:
                        st.markdown(f"""
                        <div style='text-align: center; font-size: 1.5rem; font-weight: bold; color: var(--primary-color); margin-bottom: 15px;'>
                            {st.session_state.mission_story_title}
                        </div>
                        """, unsafe_allow_html=True)
                with col_pdf:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    if REPORTLAB_AVAILABLE:
                        try:
                            # Generate PDF using reportlab
                            buffer = BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=letter)
                            story = []
                            
                            # Title style
                            title_style = ParagraphStyle(
                                'CustomTitle',
                                parent=getSampleStyleSheet()['Heading1'],
                                fontSize=18,
                                textColor='#2563eb',
                                spaceAfter=30,
                                alignment=TA_CENTER
                            )
                            
                            # Body style
                            body_style = ParagraphStyle(
                                'CustomBody',
                                parent=getSampleStyleSheet()['Normal'],
                                fontSize=12,
                                leading=18,
                                spaceAfter=12,
                                leftIndent=0,
                                rightIndent=0
                            )
                            
                            # Add title
                            if st.session_state.mission_story_title:
                                story.append(Paragraph(st.session_state.mission_story_title, title_style))
                                story.append(Spacer(1, 0.3*inch))
                            
                            # Add story content (split by paragraphs)
                            story_paragraphs = st.session_state.mission_story.split('\n\n')
                            for para in story_paragraphs:
                                if para.strip():
                                    # Replace newlines within paragraphs with <br/>
                                    para_html = para.replace('\n', '<br/>')
                                    story.append(Paragraph(para_html, body_style))
                                    story.append(Spacer(1, 0.2*inch))
                            
                            # Add Q&A section if there are questions and answers
                            if st.session_state.story_qa_history:
                                story.append(Spacer(1, 0.4*inch))
                                
                                # Q&A Section Title
                                qa_title_style = ParagraphStyle(
                                    'QATitle',
                                    parent=getSampleStyleSheet()['Heading2'],
                                    fontSize=14,
                                    textColor='#6b46c1',
                                    spaceAfter=15,
                                    spaceBefore=20
                                )
                                story.append(Paragraph("Questions & Answers", qa_title_style))
                                story.append(Spacer(1, 0.2*inch))
                                
                                # Q&A Question style
                                qa_question_style = ParagraphStyle(
                                    'QAQuestion',
                                    parent=getSampleStyleSheet()['Normal'],
                                    fontSize=11,
                                    textColor='#553c9a',
                                    spaceAfter=8,
                                    leftIndent=0,
                                    fontName='Helvetica-Bold'
                                )
                                
                                # Q&A Answer style
                                qa_answer_style = ParagraphStyle(
                                    'QAAnswer',
                                    parent=getSampleStyleSheet()['Normal'],
                                    fontSize=10,
                                    textColor='#1e293b',
                                    spaceAfter=15,
                                    leftIndent=20
                                )
                                
                                # Add each Q&A pair
                                for idx, qa in enumerate(st.session_state.story_qa_history, 1):
                                    question_text = f"Q{idx}: {qa['question']}"
                                    answer_text = f"A{idx}: {qa['answer']}"
                                    
                                    story.append(Paragraph(question_text.replace('\n', '<br/>'), qa_question_style))
                                    story.append(Spacer(1, 0.1*inch))
                                    story.append(Paragraph(answer_text.replace('\n', '<br/>'), qa_answer_style))
                                    story.append(Spacer(1, 0.15*inch))
                            
                            # Build PDF
                            doc.build(story)
                            buffer.seek(0)
                            
                            # Download button
                            st.download_button(
                                label="📄 Download PDF",
                                data=buffer.getvalue(),
                                file_name=f"{st.session_state.mission_story_title.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"PDF generation error: {str(e)}")
                    else:
                        # Fallback if reportlab not available - use fpdf2
                        try:
                            from fpdf import FPDF
                            
                            pdf = FPDF()
                            pdf.set_auto_page_break(auto=True, margin=15)
                            pdf.add_page()
                            
                            # Title
                            pdf.set_font("Arial", "B", 16)
                            if st.session_state.mission_story_title:
                                title = st.session_state.mission_story_title[:50]  # Limit title length
                                pdf.cell(0, 10, title, ln=True, align='C')
                                pdf.ln(10)
                            
                            # Story content
                            pdf.set_font("Arial", size=12)
                            story_text = st.session_state.mission_story.replace('\n\n', '\n')
                            for line in story_text.split('\n'):
                                if line.strip():
                                    # Handle special characters
                                    line_clean = line.strip().encode('latin-1', 'replace').decode('latin-1')
                                    pdf.multi_cell(0, 8, line_clean, align='L')
                                    pdf.ln(3)
                            
                            # Add Q&A section if there are questions and answers
                            if st.session_state.story_qa_history:
                                pdf.ln(10)
                                
                                # Q&A Section Title
                                pdf.set_font("Arial", "B", 14)
                                pdf.set_text_color(107, 70, 193)  # Purple color #6b46c1
                                pdf.cell(0, 10, "Questions & Answers", ln=True, align='L')
                                pdf.ln(5)
                                
                                # Add each Q&A pair
                                pdf.set_text_color(0, 0, 0)  # Black for content
                                for idx, qa in enumerate(st.session_state.story_qa_history, 1):
                                    # Question
                                    pdf.set_font("Arial", "B", 11)
                                    pdf.set_text_color(85, 60, 154)  # Dark purple #553c9a
                                    question_text = f"Q{idx}: {qa['question']}"
                                    question_clean = question_text.encode('latin-1', 'replace').decode('latin-1')
                                    pdf.multi_cell(0, 8, question_clean, align='L')
                                    pdf.ln(2)
                                    
                                    # Answer
                                    pdf.set_font("Arial", size=10)
                                    pdf.set_text_color(30, 41, 59)  # Dark gray #1e293b
                                    answer_text = f"A{idx}: {qa['answer']}"
                                    answer_clean = answer_text.encode('latin-1', 'replace').decode('latin-1')
                                    pdf.multi_cell(0, 7, answer_clean, align='L')
                                    pdf.ln(5)
                            
                            buffer = BytesIO()
                            pdf_bytes = pdf.output(dest='S')
                            buffer.write(pdf_bytes.encode('latin-1'))
                            buffer.seek(0)
                            
                            st.download_button(
                                label="📄 Download PDF",
                                data=buffer.getvalue(),
                                file_name=f"{st.session_state.mission_story_title.replace(' ', '_')[:30]}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except ImportError:
                            st.info("📄 PDF generation will be available after libraries are installed. The app needs to redeploy with updated requirements.txt.")
                        except Exception as e:
                            st.warning(f"PDF generation error: {str(e)}")
                
                # Display story content
                st.markdown(f"""
                <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid var(--primary-color);'>
                    {st.session_state.mission_story.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Q&A Section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### 💬 Ask Questions About the Story")
                
                # Generate example question if not exists, story changed, or after Q&A (force refresh)
                # Get list of already asked questions to avoid repetition
                existing_questions = [qa['question'] for qa in st.session_state.story_qa_history] if st.session_state.story_qa_history else []
                
                if (not st.session_state.story_question_example or 
                    st.session_state.get('last_story_title') != st.session_state.mission_story_title or
                    st.session_state.get('refresh_question_example', False)):
                    st.session_state.story_question_example = generate_story_question_example(
                        st.session_state.mission_story_title,
                        st.session_state.mission_story,
                        existing_questions=existing_questions
                    )
                    st.session_state.last_story_title = st.session_state.mission_story_title
                    st.session_state.refresh_question_example = False
                
                # Question input form - use Q&A history length in key to force refresh when new answer is added
                qa_key_suffix = len(st.session_state.story_qa_history)
                with st.form("story_qa_form", clear_on_submit=True):
                    user_question = st.text_input(
                        "Ask a question about the story:",
                        placeholder=f"Example: {st.session_state.story_question_example}",
                        key=f"story_question_input_{qa_key_suffix}"
                    )
                    submit_question = st.form_submit_button("Ask", type="primary", use_container_width=True)
                    
                    if submit_question:
                        # Use example if question is blank
                        if not user_question or not user_question.strip():
                            user_question = st.session_state.story_question_example
                        
                        if user_question and user_question.strip():
                            try:
                                # Generate answer using OpenAI
                                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                                answer_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "You are a creative storyteller and teacher explaining children's stories to kids aged 5-10. Answer questions in a simple, friendly way using very simple words. IMPORTANT: You are NOT limited to just the story content. Be creative and build upon the story with additional details, background, character motivations, and fun nuances that enhance the story. Expand on the story world while staying consistent with what was told. Keep answers engaging and imaginative (2-4 sentences). Make it fun and easy to understand."},
                                        {"role": "user", "content": f"Story Title: {st.session_state.mission_story_title}\n\nStory:\n{st.session_state.mission_story}\n\nQuestion: {user_question}\n\nAnswer this question creatively, building upon and expanding the story with additional details and nuances. You can add background information, character motivations, fun facts, or imaginative details that enhance the story world. Keep it simple and child-friendly (2-4 short sentences)."}
                                    ],
                                    temperature=0.9,  # Higher temperature for more creativity
                                    max_tokens=200  # Increased to allow for more detailed, creative answers
                                )
                                answer = answer_response.choices[0].message.content.strip()
                                
                                # Add to Q&A history
                                st.session_state.story_qa_history.append({
                                    "question": user_question,
                                    "answer": answer
                                })
                                
                                # Regenerate example question after answer is generated
                                # Get updated list of asked questions (including the one just asked)
                                updated_existing_questions = [qa['question'] for qa in st.session_state.story_qa_history]
                                st.session_state.story_question_example = generate_story_question_example(
                                    st.session_state.mission_story_title,
                                    st.session_state.mission_story,
                                    existing_questions=updated_existing_questions
                                )
                                # Set flag to ensure example is refreshed on next render
                                st.session_state.refresh_question_example = True
                                
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
                
                # Display Q&A history
                if st.session_state.story_qa_history:
                    st.markdown("<br>", unsafe_allow_html=True)
                    for idx, qa in enumerate(st.session_state.story_qa_history):
                        st.markdown(f"""
                        <div style='background-color: #f9fafb; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6b46c1;'>
                            <div style='font-weight: bold; color: #553c9a; margin-bottom: 8px;'>
                                ❓ Question: {qa['question']}
                            </div>
                            <div style='color: #1e293b; padding-left: 10px;'>
                                💡 {qa['answer']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
