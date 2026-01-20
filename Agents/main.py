import streamlit as st
import os
import openai
from config import STREAMLIT_CONFIG, APP_VERSION, get_openai_api_key
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

def play_sound(sound_type):
    """Play a sound effect based on the event type using Web Audio API with better initialization"""
    # Comprehensive audio initialization that works across page reloads
    init_audio_script = """
    <script>
    console.log('🧪 AUDIO INIT SCRIPT STARTING...');
    (function() {
        console.log('🧪 Inside audio init IIFE');
        // Initialize audio system
        if (!window.denkenAudioSystem) {
            window.denkenAudioSystem = {
                context: null,
                initialized: false,
                userInteracted: false,
                enabled: false
            };
            
            // Function to initialize audio context
            function initAudioContext() {
                if (window.denkenAudioSystem.context) {
                    return window.denkenAudioSystem.context;
                }
                
                try {
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                    if (!AudioContextClass) {
                        console.log('Web Audio API not supported');
                        return null;
                    }
                    
                    window.denkenAudioSystem.context = new AudioContextClass();
                    console.log('Audio context created, state:', window.denkenAudioSystem.context.state);
                    return window.denkenAudioSystem.context;
                } catch(e) {
                    console.log('Audio context creation failed:', e);
                    return null;
                }
            }
            
            // Function to play a test sound
            function playTestSound(context) {
                try {
                    const oscillator = context.createOscillator();
                    const gainNode = context.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(context.destination);
                    
                    // Play a pleasant test tone
                    oscillator.frequency.value = 523.25; // C5 note
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.6, context.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.5);
                    
                    oscillator.start(context.currentTime);
                    oscillator.stop(context.currentTime + 0.5);
                    console.log('Test sound played successfully');
                    return true;
                } catch(e) {
                    console.error('Error playing test sound:', e);
                    return false;
                }
            }
            
            // Function to enable audio (called by user interaction)
            window.enableDenkenAudio = async function() {
                try {
                    console.log('enableDenkenAudio called');
                    window.denkenAudioSystem.enabled = true;
                    window.denkenAudioSystem.userInteracted = true;
                    const context = initAudioContext();
                    if (!context) {
                        console.error('Failed to create audio context');
                        throw new Error('Failed to create audio context');
                    }
                    console.log('Audio context state:', context.state);
                    if (context.state === 'suspended') {
                        await context.resume();
                        console.log('Audio enabled and context resumed, new state:', context.state);
                    } else {
                        console.log('Audio context already running');
                    }
                    
                    // Play a test sound to confirm it's working
                    setTimeout(() => {
                        playTestSound(context);
                    }, 100);
                    
                    return context;
                } catch(e) {
                    console.error('Error in enableDenkenAudio:', e);
                    throw e;
                }
            };
            
            // Function to ensure audio context is running
            async function ensureAudioReady() {
                // Check if audio is enabled
                if (!window.denkenAudioSystem.enabled) {
                    console.log('Audio not enabled yet');
                    return null;
                }
                
                let context = window.denkenAudioSystem.context || initAudioContext();
                if (!context) return null;
                
                // Resume if suspended
                if (context.state === 'suspended') {
                    try {
                        await context.resume();
                        console.log('Audio context resumed');
                    } catch(e) {
                        console.log('Failed to resume audio context:', e);
                        return null;
                    }
                }
                
                return context;
            }
            
            // Store function globally
            window.initDenkenAudio = initAudioContext;
            window.ensureDenkenAudioReady = ensureAudioReady;
            
            // Auto-enable on any user interaction (click, type, etc.)
            function enableAudio() {
                if (!window.denkenAudioSystem.userInteracted) {
                    window.enableDenkenAudio();
                }
            }
            
            // Listen for user interactions to enable audio
            ['click', 'touchstart', 'keydown', 'mousedown'].forEach(eventType => {
                document.addEventListener(eventType, enableAudio, { once: false, passive: true });
            });
        }
    })();
    </script>
    """
    
    sound_scripts = {
        'user_name': """
        <script>
        (function() {
            setTimeout(async () => {
                try {
                    const context = await window.ensureDenkenAudioReady();
                    if (!context) {
                        console.log('Audio context not available');
                        return;
                    }
                    
                    const oscillator = context.createOscillator();
                    const gainNode = context.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(context.destination);
                    
                    oscillator.frequency.value = 523.25; // C5 note
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.5, context.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.4);
                    
                    oscillator.start(context.currentTime);
                    oscillator.stop(context.currentTime + 0.4);
                    console.log('User name sound played');
                } catch(e) {
                    console.log('Audio playback error:', e);
                }
            }, 100);
        })();
        </script>
        """,
        'agent_created': """
        <script>
        (function() {
            setTimeout(async () => {
                try {
                    const context = await window.ensureDenkenAudioReady();
                    if (!context) {
                        console.log('Audio context not available');
                        return;
                    }
                    
                    // Play a pleasant ascending chord
                    const notes = [523.25, 659.25, 783.99]; // C5, E5, G5
                    notes.forEach((freq, index) => {
                        setTimeout(() => {
                            const osc = context.createOscillator();
                            const gain = context.createGain();
                            osc.connect(gain);
                            gain.connect(context.destination);
                            osc.frequency.value = freq;
                            osc.type = 'sine';
                            gain.gain.setValueAtTime(0.4, context.currentTime);
                            gain.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.5);
                            osc.start(context.currentTime);
                            osc.stop(context.currentTime + 0.5);
                        }, index * 120);
                    });
                    console.log('Agent created sound played');
                } catch(e) {
                    console.log('Audio playback error:', e);
                }
            }, 100);
        })();
        </script>
        """,
        'story_rendered': """
        <script>
        (function() {
            setTimeout(async () => {
                try {
                    const context = await window.ensureDenkenAudioReady();
                    if (!context) {
                        console.log('Audio context not available');
                        return;
                    }
                    
                    // Play a magical ascending melody
                    const melody = [
                        {freq: 392.00, time: 0},    // G4
                        {freq: 440.00, time: 0.1},  // A4
                        {freq: 493.88, time: 0.2},  // B4
                        {freq: 523.25, time: 0.3},  // C5
                        {freq: 587.33, time: 0.4},  // D5
                        {freq: 659.25, time: 0.5}   // E5
                    ];
                    
                    melody.forEach(note => {
                        setTimeout(() => {
                            const oscillator = context.createOscillator();
                            const gainNode = context.createGain();
                            oscillator.connect(gainNode);
                            gainNode.connect(context.destination);
                            oscillator.frequency.value = note.freq;
                            oscillator.type = 'sine';
                            gainNode.gain.setValueAtTime(0.4, context.currentTime);
                            gainNode.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.25);
                            oscillator.start(context.currentTime);
                            oscillator.stop(context.currentTime + 0.25);
                        }, note.time * 1000);
                    });
                    console.log('Story rendered sound played');
                } catch(e) {
                    console.log('Audio playback error:', e);
                }
            }, 100);
        })();
        </script>
        """,
        'answer_generated': """
        <script>
        (function() {
            setTimeout(async () => {
                try {
                    const context = await window.ensureDenkenAudioReady();
                    if (!context) {
                        console.log('Audio context not available');
                        return;
                    }
                    
                    // Play a cheerful two-tone chime
                    const oscillator = context.createOscillator();
                    const gainNode = context.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(context.destination);
                    
                    oscillator.frequency.value = 659.25; // E5
                    oscillator.type = 'sine';
                    gainNode.gain.setValueAtTime(0.5, context.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.2);
                    oscillator.start(context.currentTime);
                    oscillator.stop(context.currentTime + 0.2);
                    
                    setTimeout(() => {
                        const osc2 = context.createOscillator();
                        const gain2 = context.createGain();
                        osc2.connect(gain2);
                        gain2.connect(context.destination);
                        osc2.frequency.value = 783.99; // G5
                        osc2.type = 'sine';
                        gain2.gain.setValueAtTime(0.5, context.currentTime);
                        gain2.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.2);
                        osc2.start(context.currentTime);
                        osc2.stop(context.currentTime + 0.2);
                    }, 200);
                    console.log('Answer generated sound played');
                } catch(e) {
                    console.log('Audio playback error:', e);
                }
            }, 100);
        })();
        </script>
        """
    }
    
    # Inject initialization script (will run on every rerun but only initialize once)
    st.markdown(init_audio_script, unsafe_allow_html=True)
    
    # Only play sound if a valid sound type is provided
    if sound_type and sound_type in sound_scripts:
        st.markdown(sound_scripts[sound_type], unsafe_allow_html=True)

def generate_agent_example():
    """Generate a short random agent description example (around 10 words) inspired by Star Trek characters with extensive variety"""
    star_trek_roles = [
        "Captain", "Science Officer", "Chief Engineer", "Medical Officer", 
        "Security Officer", "Communications Officer", "Navigator", "Counselor",
        "Tactical Officer", "Operations Officer", "Helmsman", "First Officer",
        "Stellar Cartographer", "Exobiologist", "Quantum Physicist", "Xenolinguist",
        "Chief Medical Officer", "Engineering Specialist", "Weapons Officer", "Diplomatic Officer",
        "Astronomer", "Archaeologist", "Anthropologist", "Geologist", "Botanist", "Chemist",
        "Transporter Chief", "Shuttle Pilot", "Computer Specialist", "Research Scientist"
    ]
    star_trek_traits = [
        "bold and brave", "logical and analytical", "curious and scientific", 
        "caring and compassionate", "protective and vigilant", "diplomatic and friendly",
        "precise and focused", "wise and understanding", "strategic and tactical",
        "efficient and organized", "skilled and experienced", "loyal and trustworthy",
        "calm under pressure", "quick thinking", "adaptable and flexible", "intuitive and perceptive",
        "methodical and thorough", "creative problem solver", "strong communicator", "patient teacher",
        "detail oriented", "big picture thinker", "team player", "independent worker", "risk taker",
        "cautious planner", "innovative designer", "practical implementer", "visionary leader", "supportive teammate"
    ]
    star_trek_qualities = [
        "exploring new worlds", "solving complex problems", "helping others in need",
        "protecting the crew", "discovering new knowledge", "maintaining peace",
        "navigating through space", "healing and caring", "communicating with aliens",
        "analyzing data", "engineering solutions", "leading missions",
        "studying alien cultures", "mapping star systems", "repairing starships", "treating injuries",
        "negotiating treaties", "defending against threats", "researching anomalies", "teaching new skills",
        "managing resources", "coordinating operations", "breaking codes", "gathering intelligence",
        "planning expeditions", "documenting discoveries", "building alliances", "resolving conflicts",
        "improving systems", "optimizing performance", "developing strategies", "executing plans"
    ]
    
    # Track used combinations to avoid repetition
    if 'used_agent_examples' not in st.session_state:
        st.session_state.used_agent_examples = []
    
    # Try multiple combinations to find one not recently used
    max_attempts = 10
    for _ in range(max_attempts):
        role = random.choice(star_trek_roles)
        trait = random.choice(star_trek_traits)
        quality = random.choice(star_trek_qualities)
        
        # Generate different example formats for variety
        example_formats = [
            f"A {role} who is {trait} and loves {quality}.",
            f"An agent who is a {role}, {trait}, specializing in {quality}.",
            f"A {trait} {role} that enjoys {quality} and teamwork.",
            f"An {role} agent who is {trait} and excels at {quality}.",
            f"A {role} who is {trait} and dedicated to {quality}.",
            f"A {role} that is {trait}, always {quality}.",
            f"An agent as a {role}, known for being {trait} and {quality}.",
            f"A {role} agent who is {trait} with a passion for {quality}."
        ]
        example = random.choice(example_formats)
        
        # Check if this combination was recently used
        if example not in st.session_state.used_agent_examples[-30:]:
            st.session_state.used_agent_examples.append(example)
            # Keep only last 100 examples
            if len(st.session_state.used_agent_examples) > 100:
                st.session_state.used_agent_examples = st.session_state.used_agent_examples[-50:]
            return example
    
    # If all attempts failed, return anyway (shouldn't happen often)
    return example

def clean_agent_description(desc):
    """Clean and format agent description, handling dict-like structures."""
    if not desc:
        return ""
    
    # If it's already a string, check if it's a dict representation
    if isinstance(desc, str):
        # Check if it looks like a dict string
        if desc.strip().startswith('{') and desc.strip().endswith('}'):
            try:
                # Try to parse it as JSON
                desc_dict = json.loads(desc)
                if isinstance(desc_dict, dict):
                    # Format as readable text
                    desc_parts = []
                    if 'traits' in desc_dict:
                        traits = desc_dict['traits']
                        if isinstance(traits, list):
                            desc_parts.append(f"Traits: {', '.join(traits)}")
                        else:
                            desc_parts.append(f"Traits: {traits}")
                    if 'working_style' in desc_dict:
                        desc_parts.append(f"Working Style: {desc_dict['working_style']}")
                    if 'expertise' in desc_dict:
                        desc_parts.append(f"Expertise: {desc_dict['expertise']}")
                    if 'approach' in desc_dict:
                        desc_parts.append(f"Approach: {desc_dict['approach']}")
                    if desc_parts:
                        return ". ".join(desc_parts)
                    else:
                        return desc  # Return original if we can't parse it
            except (json.JSONDecodeError, ValueError):
                # If it's not valid JSON, return as-is
                pass
        return desc
    
    # If it's a dict, format it
    if isinstance(desc, dict):
        desc_parts = []
        if 'traits' in desc:
            traits = desc['traits']
            if isinstance(traits, list):
                desc_parts.append(f"Traits: {', '.join(traits)}")
            else:
                desc_parts.append(f"Traits: {traits}")
        if 'working_style' in desc:
            desc_parts.append(f"Working Style: {desc['working_style']}")
        if 'expertise' in desc:
            desc_parts.append(f"Expertise: {desc['expertise']}")
        if 'approach' in desc:
            desc_parts.append(f"Approach: {desc['approach']}")
        return ". ".join(desc_parts) if desc_parts else str(desc)
    
    return str(desc)

def generate_story_question_example(story_title, story_content, existing_questions=None):
    """Generate a relevant example question about the story with extensive variety"""
    # Extract key elements from the story for context
    story_preview = story_content[:500] if story_content else ""  # Increased preview length
    
    # Get list of already asked questions to avoid repetition
    asked_questions = existing_questions or []
    asked_questions_text = "\n".join([f"- {q}" for q in asked_questions]) if asked_questions else "None yet"
    
    # Extensive list of question types for variety
    question_types = [
        "about a specific character or agent and their role",
        "about what happened during a specific part of the mission",
        "about how the agents solved a particular problem",
        "about the outcome or ending of the story",
        "about a specific event or exciting moment",
        "about the teamwork or collaboration between agents",
        "about a challenge or obstacle they faced",
        "about what they discovered or learned",
        "about why something happened the way it did",
        "about what would happen next after the story",
        "about which agent was most important and why",
        "about a funny or interesting moment",
        "about how they used their special skills",
        "about what the mission taught them",
        "about the most dangerous part of the mission",
        "about how they became friends",
        "about what surprised them the most",
        "about the happiest moment in the story",
        "about how they worked as a team",
        "about what made the mission special"
    ]
    import random
    selected_type = random.choice(question_types)
    
    # Track all generated questions in session state for better tracking
    if 'all_generated_questions' not in st.session_state:
        st.session_state.all_generated_questions = []
    
    # Generate a question example using OpenAI with emphasis on variety
    try:
        client = openai.OpenAI(api_key=get_openai_api_key())
        
        # Build more comprehensive context including all previously generated questions
        all_previous = list(set(asked_questions + st.session_state.all_generated_questions[-20:]))
        all_previous_text = "\n".join([f"- {q}" for q in all_previous]) if all_previous else "None yet"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse, relevant questions about children's stories. Generate ONE simple question that a child (5-10 years old) might ask about the story. CRITICAL: Make each question COMPLETELY DIFFERENT from ALL previous questions. Vary the question type, focus, perspective, wording, and what aspect of the story it asks about. The question should reference specific elements from the story. Keep it simple, engaging, and age-appropriate. Return ONLY the question, nothing else."},
                {"role": "user", "content": f"Story Title: {story_title}\n\nStory Preview: {story_preview}\n\nALL previously asked/generated questions:\n{all_previous_text}\n\nGenerate a COMPLETELY NEW and DIFFERENT question that a child might ask about this story. Focus on: {selected_type}. The question must be unique and different from ALL previous questions in wording, focus, and perspective. Reference something specific from the story."}
            ],
            temperature=0.95,  # Higher temperature for more variety
            max_tokens=120
        )
        question = response.choices[0].message.content.strip()
        # Clean up question (remove quotes if present)
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1]
        if question.lower().startswith("question:"):
            question = question[9:].strip()
        
        # Track this question
        if question not in st.session_state.all_generated_questions:
            st.session_state.all_generated_questions.append(question)
            # Keep only last 100 questions
            if len(st.session_state.all_generated_questions) > 100:
                st.session_state.all_generated_questions = st.session_state.all_generated_questions[-50:]
        
        return question
    except Exception as e:
        # Extensive fallback examples with variety
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
            "What made the mission successful?",
            "Which agent was the bravest?",
            "What was the hardest part?",
            "How did they become friends?",
            "What did they learn?",
            "What happened first?",
            "What made them happy?",
            "How did they solve the problem?",
            "What was the funniest moment?",
            "Why did they need to work together?",
            "What would happen next?",
            "Which part was the most dangerous?",
            "How did each agent help?",
            "What was their goal?",
            "How did the mission start?",
            "What special thing did they do?",
            "Why was teamwork important?",
            "What was the best part?",
            "How did they celebrate?",
            "What obstacle did they overcome?",
            "What made them proud?"
        ]
        # Filter out already asked questions
        available_questions = [q for q in fallback_questions if q not in (asked_questions + st.session_state.get('all_generated_questions', []))[-20:]]
        if available_questions:
            selected = random.choice(available_questions)
            if selected not in st.session_state.get('all_generated_questions', []):
                if 'all_generated_questions' not in st.session_state:
                    st.session_state.all_generated_questions = []
                st.session_state.all_generated_questions.append(selected)
            return selected
        else:
            # If all fallbacks used, cycle through them
            return random.choice(fallback_questions)

def generate_creative_name_examples():
    """Generate fancy example names for creative input with extensive variety"""
    fancy_names = [
        # Space & Cosmic
        "Stellar Commander", "Cosmic Navigator", "Quantum Explorer", "Nebula Warrior", "Galaxy Seeker",
        "Starfire Guardian", "Aurora Whisper", "Solar Flare", "Lunar Eclipse", "Comet Rider",
        "Orbit Master", "Planet Walker", "Stardust Seeker", "Cosmic Sage", "Nova Phoenix",
        "Astro Captain", "Celestial Navigator", "Meteor Knight", "Eclipse Warrior", "Supernova Dreamer",
        
        # Elemental & Natural
        "Thunder Bolt", "Silver Shadow", "Crystal Knight", "Frost Walker", "Blaze Rider",
        "Ocean Sage", "Forest Whisper", "Mountain Peak", "Desert Storm", "River Flow",
        "Stone Guardian", "Wind Chaser", "Flame Seeker", "Ice Breaker", "Storm Bringer",
        "Earth Shaker", "Sky Dancer", "Wave Rider", "Leaf Whisper", "Thunder Roar",
        
        # Mythical & Fantasy
        "Neon Phoenix", "Golden Dragon", "Crimson Hawk", "Azure Dragon", "Midnight Spark",
        "Dawn Breaker", "Twilight Seeker", "Shadow Walker", "Light Bringer", "Dark Knight",
        "Phoenix Rising", "Dragon Rider", "Griffin Wing", "Unicorn Dream", "Mermaid Song",
        "Elf Warrior", "Dwarf Smith", "Wizard Sage", "Sorcerer King", "Enchantress Queen",
        
        # Action & Adventure
        "Swift Arrow", "Echo Runner", "Mystic Voyager", "Brave Heart", "Wild Spirit",
        "Bold Explorer", "Fearless Leader", "Courage Seeker", "Adventure Caller", "Journey Master",
        "Pathfinder", "Trail Blazer", "Quest Seeker", "Treasure Hunter", "Legend Maker",
        "Hero Rising", "Champion Born", "Warrior Soul", "Guardian Strong", "Protector Brave",
        
        # Colors & Characteristics
        "Crimson Flame", "Emerald Eye", "Sapphire Dream", "Amber Light", "Onyx Shadow",
        "Pearl Diver", "Ruby Heart", "Topaz Mind", "Jade Soul", "Diamond Spark",
        "Silver Tongue", "Golden Touch", "Copper Heart", "Bronze Will", "Steel Mind",
        
        # Abstract & Conceptual
        "Dream Weaver", "Hope Bringer", "Joy Seeker", "Peace Maker", "Wisdom Keeper",
        "Truth Finder", "Justice Seeker", "Freedom Caller", "Unity Builder", "Harmony Keeper",
        "Inspiration Source", "Creativity Flow", "Innovation Spark", "Genius Mind", "Brilliant Star"
    ]
    import random
    
    # Track used examples in session state to avoid repetition
    if 'used_creative_names' not in st.session_state:
        st.session_state.used_creative_names = []
    
    # Get available names (not recently used)
    available_names = [n for n in fancy_names if n not in st.session_state.used_creative_names[-20:]]
    
    # If we've used too many, reset tracking (keep last 10)
    if len(st.session_state.used_creative_names) > 80:
        st.session_state.used_creative_names = st.session_state.used_creative_names[-10:]
        available_names = fancy_names
    
    if not available_names:
        available_names = fancy_names
    
    # Select 3 random examples
    selected = random.sample(available_names, min(3, len(available_names)))
    
    # Track selected examples
    st.session_state.used_creative_names.extend(selected)
    
    return selected

def generate_mission_example():
    """Generate a random Star Trek-style mission example (around 20 words) with thousands of unique combinations"""
    # Expanded mission pool with hundreds of unique Star Trek-style missions
    star_trek_missions = [
        # Exploration Missions (50+)
        "explore an unknown planet and make contact with friendly alien civilizations",
        "investigate strange energy readings from a distant nebula and discover its source",
        "explore a new star system and map planets that might support life",
        "chart an unexplored sector of space and document all discoveries",
        "investigate an ancient alien structure found on a remote moon",
        "explore a binary star system to study its unique planetary formations",
        "discover new habitable worlds in the outer reaches of the galaxy",
        "investigate a mysterious void in space where stars seem to disappear",
        "explore a planet covered entirely in crystal formations",
        "discover a hidden asteroid belt filled with rare minerals",
        "explore a gas giant with floating cities in its atmosphere",
        "investigate a planet that orbits three suns simultaneously",
        "explore an ocean world where entire civilizations live underwater",
        "discover a space station abandoned for centuries",
        "explore a nebula that glows in colors never seen before",
        "investigate a planet where plants communicate through light",
        "explore a moon that orbits backwards around its planet",
        "discover a star system where all planets are connected by bridges",
        "explore a planet made entirely of ice with hidden oceans beneath",
        "investigate a region of space where gravity works in reverse",
        "explore a binary planet system where two worlds share an atmosphere",
        "discover a planet that rotates so fast days last only minutes",
        "explore a star system with a rainbow-colored asteroid field",
        "investigate a planet where sound creates visible patterns in the air",
        "explore a moon that changes color with the seasons",
        "discover a space anomaly that creates duplicate versions of objects",
        "explore a planet with floating islands in its sky",
        "investigate a star that pulses in a musical rhythm",
        "explore a planet where the ground is made of living crystal",
        "discover a nebula that contains memories of ancient civilizations",
        "explore a planet with six moons that create spectacular light shows",
        "investigate a region where time moves at different speeds",
        "explore a planet covered in giant mushrooms taller than mountains",
        "discover a space station that travels between dimensions",
        "explore a planet where the sky is filled with floating cities",
        "investigate a star system with planets that orbit in figure-eight patterns",
        "explore a planet with rivers of liquid light",
        "discover a moon that glows with its own inner light",
        "explore a planet where animals can communicate telepathically",
        "investigate a space anomaly that creates portals to other galaxies",
        "explore a planet with trees that grow in spiral patterns",
        "discover a star system where planets share a single atmosphere",
        "explore a planet covered in bioluminescent plants",
        "investigate a region of space where thoughts become reality",
        "explore a planet with mountains that float in the air",
        "discover a space station hidden inside a comet",
        "explore a planet where the ocean is made of liquid stardust",
        "investigate a star system with planets that sing as they orbit",
        "explore a planet with clouds that form geometric patterns",
        "discover a moon that orbits so close it creates tidal waves in space",
        "explore a planet where gravity pulls sideways instead of down",
        
        # Rescue Missions (40+)
        "rescue a stranded spaceship crew from a dangerous asteroid field",
        "rescue scientists trapped on a research outpost during a solar storm",
        "save a colony from an approaching comet by redirecting its path",
        "rescue an alien ship damaged by space debris",
        "evacuate a space station threatened by a plasma storm",
        "rescue a survey team lost on a dangerous ice planet",
        "save a space station from a black hole's gravitational pull",
        "rescue explorers trapped in a time loop on a distant planet",
        "evacuate a colony before their star goes supernova",
        "rescue a ship caught in a cosmic storm",
        "save a research team from a planet's rapidly changing climate",
        "rescue a cargo ship attacked by space pirates",
        "evacuate a moon base before it crashes into its planet",
        "rescue scientists from a lab that's sinking into a gas giant",
        "save a space station from colliding with a rogue planet",
        "rescue a diplomatic ship lost in an uncharted nebula",
        "evacuate a colony from a planet with increasing volcanic activity",
        "rescue a mining crew trapped in an asteroid mine collapse",
        "save a space station from a swarm of space debris",
        "rescue explorers from a planet where the ground is collapsing",
        "evacuate a research base from a planet with toxic atmosphere",
        "rescue a ship disabled by an electromagnetic pulse",
        "save a colony from a meteor shower",
        "rescue a team from a planet where time is accelerating",
        "evacuate a space station from a nearby star explosion",
        "rescue a ship caught in a gravitational anomaly",
        "save a research team from a planet with extreme weather",
        "rescue explorers from a moon that's breaking apart",
        "evacuate a colony from a planet with rising sea levels",
        "rescue a ship from a space vortex",
        "save a space station from a collision with an asteroid",
        "rescue a team from a planet where the atmosphere is disappearing",
        "evacuate a base from a planet with increasing radiation",
        "rescue a ship from a cosmic storm",
        "save a colony from a planet's magnetic field collapse",
        "rescue explorers from a space station losing power",
        "evacuate a research facility from a planet with earthquakes",
        "rescue a ship from a space anomaly",
        "save a team from a planet with extreme temperatures",
        "rescue a colony from a planet's core instability",
        
        # Diplomatic Missions (40+)
        "establish diplomatic relations with a newly discovered planet's inhabitants",
        "mediate a peace agreement between two warring alien civilizations",
        "negotiate a trade agreement with a friendly merchant species",
        "host a galactic summit to discuss inter-species cooperation",
        "resolve a territorial dispute between neighboring star systems",
        "facilitate cultural exchange between humans and a new alien species",
        "organize a peace conference between three conflicting species",
        "negotiate the release of hostages from a diplomatic crisis",
        "mediate a dispute over mining rights in a neutral zone",
        "establish a trade route between two distant civilizations",
        "facilitate an alliance between previously hostile species",
        "negotiate a scientific research agreement with a new species",
        "organize a cultural festival to celebrate inter-species friendship",
        "mediate a conflict over water rights on a shared planet",
        "establish diplomatic immunity protocols with a new civilization",
        "negotiate a non-aggression pact between rival factions",
        "facilitate technology exchange between advanced species",
        "organize a summit to address galactic environmental concerns",
        "mediate a dispute over ancient artifacts",
        "establish a joint exploration program with friendly aliens",
        "negotiate safe passage through a species' territory",
        "facilitate a prisoner exchange between warring factions",
        "organize a conference on inter-species communication",
        "mediate a conflict over a newly discovered resource",
        "establish a mutual defense agreement with allies",
        "negotiate access to a sacred alien site",
        "facilitate a student exchange program between species",
        "organize a peace ceremony to end a long conflict",
        "mediate a dispute over space station ownership",
        "establish a joint research facility with another species",
        "negotiate a treaty to protect endangered alien species",
        "facilitate a technology transfer to help a developing civilization",
        "organize a summit to address galactic migration issues",
        "mediate a conflict over a planet's natural resources",
        "establish a cultural preservation program with aliens",
        "negotiate a border agreement between star systems",
        "facilitate a medical exchange to help a species in need",
        "organize a conference on inter-species ethics",
        "mediate a dispute over historical claims to a planet",
        "establish a joint space exploration initiative",
        
        # Scientific Missions (60+)
        "study an unusual space anomaly that could reveal secrets about the universe",
        "discover the origin of ancient artifacts found floating in deep space",
        "investigate a time distortion that threatens the fabric of space",
        "analyze a new form of energy discovered in a stellar nursery",
        "study a planet where time moves differently than normal space",
        "investigate reports of mysterious signals coming from an abandoned space station",
        "research a phenomenon that could revolutionize faster-than-light travel",
        "study an ecosystem that exists entirely in zero gravity",
        "investigate a planet with multiple moons that create unique weather patterns",
        "analyze a star that's aging backwards",
        "study a planet where plants can think and communicate",
        "investigate a space phenomenon that creates matter from nothing",
        "research a planet with a core made of pure energy",
        "study a nebula that contains the building blocks of life",
        "investigate a planet where gravity changes based on emotions",
        "analyze a star system with planets that share a single consciousness",
        "study a planet with an atmosphere made of music",
        "investigate a space anomaly that preserves moments in time",
        "research a planet where the laws of physics work differently",
        "study a star that emits colors beyond the visible spectrum",
        "investigate a planet with a magnetic field that creates art",
        "analyze a nebula that contains memories of the universe's birth",
        "study a planet where sound creates physical structures",
        "investigate a space phenomenon that connects parallel universes",
        "research a planet with a core that pulses like a heartbeat",
        "study a star system with planets that orbit in perfect harmony",
        "investigate a planet where light moves in slow motion",
        "analyze a space anomaly that creates new elements",
        "study a planet with an ocean made of liquid time",
        "investigate a star that's actually a gateway to another dimension",
        "research a planet where thoughts become visible",
        "study a nebula that contains the genetic code of life",
        "investigate a planet with a core made of living crystal",
        "analyze a space phenomenon that reverses entropy",
        "study a planet where gravity creates beautiful patterns",
        "investigate a star system with planets that communicate through light",
        "research a planet with an atmosphere that changes color with emotions",
        "study a space anomaly that preserves ancient civilizations",
        "investigate a planet where the ground is made of compressed stardust",
        "analyze a nebula that contains the history of the universe",
        "study a planet with a magnetic field that creates music",
        "investigate a space phenomenon that creates life from energy",
        "research a planet with a core that contains a miniature universe",
        "study a star that pulses in a pattern that spells messages",
        "investigate a planet where time flows in multiple directions",
        "analyze a space anomaly that creates perfect geometric shapes",
        "study a planet with an ocean made of liquid light",
        "investigate a star system with planets that share memories",
        "research a planet where the atmosphere is made of living particles",
        "study a nebula that contains the consciousness of ancient beings",
        "investigate a planet with a core that generates new life forms",
        "analyze a space phenomenon that creates matter from sound",
        "study a planet where gravity creates floating islands",
        "investigate a star that's actually a sentient being",
        "research a planet with an atmosphere that changes with the seasons",
        "study a space anomaly that preserves moments of great importance",
        "investigate a planet where the ground is made of living matter",
        "analyze a nebula that contains the genetic memories of all life",
        "study a planet with a magnetic field that creates auroras of thought",
        "investigate a space phenomenon that connects all living things",
        "research a planet where time is stored in crystals",
        "study a star system with planets that orbit in a dance",
        
        # Protection Missions (40+)
        "help a peaceful alien species protect their homeworld from a natural disaster",
        "defend a space station from approaching meteoroids",
        "protect a rare species from poachers in deep space",
        "guard a sacred alien site from unauthorized visitors",
        "shield a developing civilization from dangerous cosmic radiation",
        "defend a colony from space pirates",
        "protect a research facility from hostile forces",
        "guard a diplomatic convoy through dangerous space",
        "shield a planet from an approaching asteroid",
        "defend a space station from a cyber attack",
        "protect a rare mineral deposit from illegal mining",
        "guard a cultural artifact from thieves",
        "shield a developing species from advanced technology",
        "defend a colony from a biological threat",
        "protect a space route from pirates",
        "guard a scientific discovery from exploitation",
        "shield a planet from solar flares",
        "defend a research team from dangerous wildlife",
        "protect a sacred grove from destruction",
        "guard a space station from sabotage",
        "shield a colony from electromagnetic storms",
        "defend a diplomatic mission from attack",
        "protect a rare ecosystem from contamination",
        "guard a historical site from vandals",
        "shield a planet from cosmic radiation",
        "defend a space station from a virus",
        "protect a trade route from raiders",
        "guard a scientific experiment from interference",
        "shield a colony from a meteor shower",
        "defend a research facility from intruders",
        "protect a rare species from extinction",
        "guard a diplomatic meeting from spies",
        "shield a planet from a supernova",
        "defend a space station from a mutiny",
        "protect a cultural festival from disruption",
        "guard a space route from blockades",
        "shield a colony from a plague",
        "defend a research team from danger",
        "protect a sacred site from desecration",
        "guard a space station from a breach",
        
        # Mystery Missions (50+)
        "solve the mystery of a starship that disappeared decades ago",
        "investigate why all electronic devices fail near a certain planet",
        "uncover the truth behind strange disappearances in a space corridor",
        "solve the puzzle of a planet with no day or night cycle",
        "investigate a signal that appears to be from Earth's past",
        "solve the mystery of a planet that appears and disappears",
        "investigate why time moves backwards on a certain moon",
        "uncover the truth behind a space station that's been abandoned",
        "solve the puzzle of a star that changes color",
        "investigate a signal that seems to be from the future",
        "solve the mystery of a planet where everyone looks identical",
        "investigate why gravity works upside down in a region",
        "uncover the truth behind a ship that travels through time",
        "solve the puzzle of a planet with no magnetic field",
        "investigate a signal that contains a message from aliens",
        "solve the mystery of a space station that's alive",
        "investigate why plants grow in geometric patterns",
        "uncover the truth behind a planet that's a perfect sphere",
        "solve the puzzle of a star that doesn't emit light",
        "investigate a signal that seems to be calling for help",
        "solve the mystery of a planet where shadows move independently",
        "investigate why the sky changes color every hour",
        "uncover the truth behind a space anomaly that creates duplicates",
        "solve the puzzle of a planet with no atmosphere but life exists",
        "investigate a signal that appears to be a countdown",
        "solve the mystery of a space station that's invisible",
        "investigate why animals can talk on a certain planet",
        "uncover the truth behind a planet that's actually a spaceship",
        "solve the puzzle of a star that pulses in Morse code",
        "investigate a signal that seems to be from another dimension",
        "solve the mystery of a planet where the ground is transparent",
        "investigate why the ocean is made of liquid crystal",
        "uncover the truth behind a space station that moves on its own",
        "solve the puzzle of a planet with three suns but no shadows",
        "investigate a signal that contains the genetic code of life",
        "solve the mystery of a planet where memories are visible",
        "investigate why the stars form patterns in the sky",
        "uncover the truth behind a space anomaly that preserves time",
        "solve the puzzle of a planet where the air is solid",
        "investigate a signal that seems to be a warning",
        "solve the mystery of a space station that's been duplicated",
        "investigate why the ground glows at night",
        "uncover the truth behind a planet that's shrinking",
        "solve the puzzle of a star that's actually two stars",
        "investigate a signal that appears to be a map",
        "solve the mystery of a planet where the ocean is upside down",
        "investigate why the clouds form words",
        "uncover the truth behind a space station that's aging backwards",
        "solve the puzzle of a planet with no gravity but things don't float",
        "investigate a signal that seems to be a song",
        
        # Cultural Missions (40+)
        "participate in an alien festival to learn about their traditions",
        "help preserve ancient knowledge from a dying alien civilization",
        "assist in relocating a species whose planet is becoming uninhabitable",
        "document the unique art and music of a newly contacted species",
        "participate in an alien coming-of-age ceremony",
        "help translate ancient alien texts before they're lost forever",
        "assist in preserving a species' cultural heritage",
        "document the unique architecture of an alien civilization",
        "participate in an inter-species cultural exchange program",
        "help restore a damaged alien monument",
        "assist in organizing a galactic cultural festival",
        "document the unique cooking traditions of an alien species",
        "participate in an alien wedding ceremony",
        "help preserve a species' oral history",
        "assist in creating a museum of alien artifacts",
        "document the unique dance traditions of a new species",
        "participate in an alien harvest festival",
        "help translate alien poetry into human languages",
        "assist in preserving a species' musical heritage",
        "document the unique storytelling traditions of aliens",
        "participate in an alien religious ceremony",
        "help preserve a species' traditional crafts",
        "assist in creating a library of alien literature",
        "document the unique fashion of an alien civilization",
        "participate in an alien sports competition",
        "help translate alien scientific texts",
        "assist in preserving a species' traditional medicine",
        "document the unique games played by aliens",
        "participate in an alien holiday celebration",
        "help preserve a species' traditional architecture",
        "assist in creating an archive of alien art",
        "document the unique rituals of an alien species",
        "participate in an alien naming ceremony",
        "help translate alien historical records",
        "assist in preserving a species' traditional music",
        "document the unique customs of an alien civilization",
        "participate in an alien festival of lights",
        "help preserve a species' traditional stories",
        "assist in creating a collection of alien artifacts",
        "document the unique celebrations of a new species",
        
        # Construction & Engineering Missions (30+)
        "build a bridge between two planets in a binary system",
        "construct a space station that orbits a black hole",
        "design a city that floats in a gas giant's atmosphere",
        "build a communication network across multiple star systems",
        "construct a research facility on a hostile planet",
        "design a ship that can travel through wormholes",
        "build a space elevator on a low-gravity moon",
        "construct a mining operation in an asteroid belt",
        "design a habitat for a species with unique needs",
        "build a spaceport that can handle thousands of ships",
        "construct a defense system for a vulnerable planet",
        "design a power station that harnesses star energy",
        "build a transportation network between space stations",
        "construct a research lab in a nebula",
        "design a ship that can survive in extreme conditions",
        "build a space station that rotates to create gravity",
        "construct a telescope that can see across galaxies",
        "design a city that adapts to changing environments",
        "build a communication relay in deep space",
        "construct a research facility on an ice planet",
        "design a ship that can phase through matter",
        "build a space station that's completely self-sufficient",
        "construct a mining facility on a lava planet",
        "design a habitat that floats in space",
        "build a defense network around a star system",
        "construct a research lab inside a comet",
        "design a ship that can transform into different shapes",
        "build a space station that travels between stars",
        "construct a power grid across multiple planets",
        "design a city that's built inside an asteroid",
        
        # Medical & Healing Missions (30+)
        "cure a plague that's spreading through a space station",
        "help a species recover from a devastating disease",
        "develop a vaccine for an alien virus",
        "provide medical aid to a colony in crisis",
        "heal a planet's ecosystem that's been damaged",
        "treat a species suffering from radiation poisoning",
        "develop medicine for an unknown alien illness",
        "help a colony recover from a natural disaster",
        "cure a genetic disease affecting an alien species",
        "provide emergency medical care during a crisis",
        "help a species adapt to a new environment",
        "develop treatments for space-related illnesses",
        "heal a planet's atmosphere that's been polluted",
        "treat a species affected by a cosmic anomaly",
        "develop a cure for a memory-wiping disease",
        "help a colony rebuild after a disaster",
        "cure a plague affecting multiple species",
        "provide medical research to help a dying species",
        "heal a planet's oceans that are dying",
        "treat a species suffering from time displacement",
        "develop medicine from alien plants",
        "help a colony recover from a war",
        "cure a disease that affects the mind",
        "provide medical aid to refugees",
        "heal a planet's forests that are disappearing",
        "treat a species affected by a space anomaly",
        "develop a treatment for aging",
        "help a colony recover from an attack",
        "cure a plague that affects technology",
        "provide medical care during a galactic crisis"
    ]
    
    # Expanded objectives pool (50+)
    mission_objectives = [
        "ensuring the safety of all crew members and new friends",
        "working together to solve complex problems and challenges",
        "using science, teamwork, and friendship to complete the mission",
        "gathering valuable information while respecting new cultures",
        "bringing peace and understanding to the galaxy",
        "learning and growing through new experiences together",
        "overcoming obstacles with creativity and cooperation",
        "making new friends across the stars",
        "proving that teamwork makes any mission possible",
        "demonstrating courage, wisdom, and kindness in action",
        "showing that different species can work together",
        "using each team member's unique skills and talents",
        "solving problems with creative thinking and innovation",
        "helping others while exploring the unknown",
        "building bridges between different civilizations",
        "protecting the innocent and helping those in need",
        "discovering new knowledge while being respectful",
        "working as a team to achieve impossible goals",
        "using technology and friendship to overcome challenges",
        "proving that kindness and cooperation always win",
        "learning from new cultures and sharing knowledge",
        "protecting life in all its forms",
        "using science to help others",
        "working together despite differences",
        "showing that everyone has something valuable to contribute",
        "overcoming fear with courage and friendship",
        "using creativity to solve impossible problems",
        "helping others without expecting anything in return",
        "proving that small actions can make big differences",
        "working together to create a better galaxy",
        "using each person's strengths to help the team",
        "showing that understanding is more powerful than force",
        "learning from mistakes and growing stronger",
        "protecting the weak and standing up for what's right",
        "using knowledge to help others",
        "working together to explore safely",
        "showing that friendship transcends species",
        "overcoming challenges with determination and teamwork",
        "using innovation to solve ancient problems",
        "helping others discover their own potential",
        "proving that cooperation beats competition",
        "working together to protect the galaxy",
        "using science and compassion together",
        "showing that everyone deserves respect",
        "learning that differences make us stronger",
        "protecting the environment and all life",
        "using teamwork to achieve the impossible",
        "helping others while learning about ourselves",
        "proving that unity creates strength",
        "working together to build a better future"
    ]
    
    # Track used mission examples to avoid repetition
    if 'used_mission_examples' not in st.session_state:
        st.session_state.used_mission_examples = []
    
    # Try multiple combinations to find one not recently used
    max_attempts = 20
    for _ in range(max_attempts):
        mission = random.choice(star_trek_missions)
        objective = random.choice(mission_objectives)
        
        # Expanded format variations for even more variety (20 formats)
        example_formats = [
            "Mission: {}, {}.",
            "Your mission is to {}, {}.",
            "The team's mission: {}, all while {}.",
            "Embark on a mission to {}, {}.",
            "Assigned mission: {}, {}.",
            "Your team must {}, always {}.",
            "The mission requires you to {}, while {}.",
            "Prepare for a mission where you will {}, {}.",
            "Your mission: {}, {}.",
            "The crew's mission is to {}, {}.",
            "Mission objective: {}, {}.",
            "You are assigned to {}, {}.",
            "The team needs to {}, {}.",
            "Your task: {}, {}.",
            "Mission brief: {}, {}.",
            "The assignment: {}, {}.",
            "Your goal is to {}, {}.",
            "Mission directive: {}, {}.",
            "The challenge: {}, {}.",
            "Your assignment: {}, {}."
        ]
        example = random.choice(example_formats).format(mission, objective)
        
        # Check if this combination was recently used (check last 100 instead of 30)
        if example not in st.session_state.used_mission_examples[-100:]:
            st.session_state.used_mission_examples.append(example)
            # Keep only last 500 examples (increased from 100)
            if len(st.session_state.used_mission_examples) > 500:
                st.session_state.used_mission_examples = st.session_state.used_mission_examples[-250:]
            return example
    
    # If all attempts failed, return a random one anyway
    mission = random.choice(star_trek_missions)
    objective = random.choice(mission_objectives)
    example_formats = [
        "Mission: {}, {}.",
        "Your mission is to {}, {}.",
        "The team's mission: {}, all while {}.",
        "Embark on a mission to {}, {}.",
        "Assigned mission: {}, {}.",
        "Your team must {}, always {}.",
        "The mission requires you to {}, while {}.",
        "Prepare for a mission where you will {}, {}.",
        "Your mission: {}, {}.",
        "The crew's mission is to {}, {}.",
        "Mission objective: {}, {}.",
        "You are assigned to {}, {}.",
        "The team needs to {}, {}.",
        "Your task: {}, {}.",
        "Mission brief: {}, {}.",
        "The assignment: {}, {}.",
        "Your goal is to {}, {}.",
        "Mission directive: {}, {}.",
        "The challenge: {}, {}.",
        "Your assignment: {}, {}."
    ]
    return random.choice(example_formats).format(mission, objective)

def main():
    # Set page config with logo as favicon
    config = STREAMLIT_CONFIG.copy()
    
    # Crop percentage: 0.0 = no crop, 0.5 = crop 50% from each side (shows only center)
    # Higher values = more zoom, less white background visible
    # Adjust this value to control how much of the logo is visible
    crop_percent = 0.185  # Crop 18.5% from each side = shows center 63% of image
    
    # Use relative path for logo (works locally and on Streamlit Cloud)
    # Prioritize Logo-DenkenLabs.png from Agents folder
    # Prioritize Logo-DenkenLabs.png (NOT Bot.png) for favicon
    # Try multiple locations for Logo-DenkenLabs.png, prioritizing agents subfolder
    # NOTE: favicon-logo.png is EXCLUDED - it contains a crown image, not the Denken Labs logo
    logo_paths = [
        # First priority: agents/Logo-DenkenLabs.png (in case main one is wrong)
        os.path.join(os.path.dirname(__file__), "agents", "Logo-DenkenLabs.png"),
        # Second priority: Logo-DenkenLabs.png from Agents folder root
        os.path.join(os.path.dirname(__file__), "Logo-DenkenLabs.png"),
        # Then try transparent background versions
        os.path.join(os.path.dirname(__file__), "Logo-DenkenLabs-transparent.png"),
        os.path.join(os.path.dirname(__file__), "Logo-DenkenLabs-trans.png"),
        os.path.join(os.path.dirname(__file__), "Logo-DenkenLabs-bg-transparent.png"),
        os.path.join(os.path.dirname(__file__), "agents", "Logo-DenkenLabs-transparent.png"),
        os.path.join(os.path.dirname(__file__), "agents", "Logo-DenkenLabs-trans.png"),
        # Other possible locations
        os.path.join(os.path.dirname(__file__), "logo_DenkenLabs.png"),
        os.path.join(os.path.dirname(__file__), "Agents", "Logo-DenkenLabs.png"),
        os.path.join(os.path.dirname(__file__), "Agents", "agents", "Logo-DenkenLabs.png"),
    ]
    
    logo_path = None
    # Find the first existing logo file (excluding favicon-logo.png and Bot.png)
    for path in logo_paths:
        if os.path.exists(path):
            # Skip favicon-logo.png (crown) and Bot.png
            if 'favicon-logo' not in path.lower() and ('bot' not in path.lower() or 'logo' in path.lower()):
                logo_path = path
                break
    
    # Final safety check: ensure we're NOT using Bot.png or favicon-logo.png (crown image)
    if logo_path and (('bot' in logo_path.lower() and 'logo' not in logo_path.lower()) or 'favicon-logo' in logo_path.lower()):
        # Force use of primary logo or set to None
        if os.path.exists(primary_logo):
            logo_path = primary_logo
        else:
            logo_path = None
    
    # NEW APPROACH: Create a cropped favicon file and use it directly with Streamlit's page_icon
    # This bypasses JavaScript and uses Streamlit's native favicon system
    cropped_favicon_path = None
    if logo_path and os.path.exists(logo_path):
        try:
            from PIL import Image
            
            # Open and process the logo
            img = Image.open(logo_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            width, height = img.size
            
            # Calculate crop coordinates (centered crop)
            crop_percent = min(crop_percent, 0.49)  # Cap at 49% to ensure valid coordinates
            left = int(width * crop_percent)
            top = int(height * crop_percent)
            right = int(width * (1 - crop_percent))
            bottom = int(height * (1 - crop_percent))
            
            # Validate crop coordinates
            if left >= right or top >= bottom:
                crop_percent = 0.25
                left = int(width * crop_percent)
                top = int(height * crop_percent)
                right = int(width * (1 - crop_percent))
                bottom = int(height * (1 - crop_percent))
            
            # Crop the image
            img = img.crop((left, top, right, bottom))
            
            # Validate dimensions
            width, height = img.size
            if width <= 0 or height <= 0:
                # Fallback to safe crop
                img = Image.open(logo_path)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                orig_width, orig_height = img.size
                crop_percent = 0.25
                left = int(orig_width * crop_percent)
                top = int(orig_height * crop_percent)
                right = int(orig_width * (1 - crop_percent))
                bottom = int(orig_height * (1 - crop_percent))
                img = img.crop((left, top, right, bottom))
                width, height = img.size
            
            # Make it square
            if width != height:
                size = max(width, height)
                square_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                paste_x = (size - width) // 2
                paste_y = (size - height) // 2
                square_img.paste(img, (paste_x, paste_y), img)
                img = square_img
            
            # Resize to 192x192 for better visibility
            favicon_size = 192
            img_resized = img.resize((favicon_size, favicon_size), Image.Resampling.LANCZOS)
            
            # Ensure transparency is preserved - convert to RGBA if not already
            if img_resized.mode != 'RGBA':
                img_resized = img_resized.convert('RGBA')
            
            # Save to a file in the same directory as the script
            # Use a filename that includes crop_percent so it changes when crop changes
            base_dir = os.path.dirname(__file__)
            crop_str = str(crop_percent).replace('.', '_')
            cropped_favicon_path = os.path.join(base_dir, f"favicon_cropped_{crop_str}.png")
            # Save PNG with transparency preserved - don't use optimize=True as it can affect transparency
            img_resized.save(cropped_favicon_path, format='PNG')
            
            # Use the cropped favicon file directly
            config['page_icon'] = cropped_favicon_path
            
        except Exception as e:
            # If cropping fails, fall back to original logo
            if logo_path and os.path.exists(logo_path) and 'favicon-logo' not in logo_path.lower():
                config['page_icon'] = logo_path
            else:
                config['page_icon'] = "🤖"  # Fallback emoji
    else:
        config['page_icon'] = "🤖"  # Fallback emoji
    
    st.set_page_config(**config)
    
    # Get the app URL for Open Graph image (for sharing previews)
    # For Streamlit Cloud, we need to construct the image URL
    import base64
    logo_url = None
    if logo_path and os.path.exists(logo_path):
        try:
            # Encode logo as base64 for embedding in meta tags
            with open(logo_path, "rb") as img_file:
                logo_base64 = base64.b64encode(img_file.read()).decode()
                logo_url = f"data:image/png;base64,{logo_base64}"
        except Exception:
            pass
    
    # Add meta tags for better sharing (title, description, and image)
    # Use JavaScript to inject meta tags into the head (runs immediately)
    # Note: Social media platforms cache link previews. After updating meta tags,
    # you may need to clear the cache using their debug tools:
    # - Facebook: https://developers.facebook.com/tools/debug/
    # - Twitter: https://cards-dev.twitter.com/validator
    # - LinkedIn: Use their post inspector
    meta_tags_js = """
    <script>
    (function() {
        // Update page title immediately
        document.title = "Denken Labs - Begin your space adventure";
        
        // Helper function to create or update meta tags
        function setMetaTag(attr, value, content) {
            let selector = `meta[${attr}="${value}"]`;
            let tag = document.querySelector(selector);
            if (!tag) {
                tag = document.createElement('meta');
                tag.setAttribute(attr, value);
                document.head.insertBefore(tag, document.head.firstChild);
            }
            tag.setAttribute('content', content);
        }
        
        // Set all Open Graph and Twitter Card meta tags
        setMetaTag('property', 'og:type', 'website');
        setMetaTag('property', 'og:title', 'Denken Labs');
        setMetaTag('property', 'og:description', 'Begin your space adventure');
        setMetaTag('property', 'og:site_name', 'Denken Labs');
        setMetaTag('property', 'og:url', window.location.href);
        setMetaTag('name', 'description', 'Begin your space adventure');
        setMetaTag('name', 'twitter:card', 'summary_large_image');
        setMetaTag('name', 'twitter:title', 'Denken Labs');
        setMetaTag('name', 'twitter:description', 'Begin your space adventure');
    """
    
    # Add image meta tags if logo is available
    if logo_url:
        meta_tags_js += f"""
        setMetaTag('property', 'og:image', '{logo_url}');
        setMetaTag('property', 'og:image:type', 'image/png');
        setMetaTag('property', 'og:image:width', '1200');
        setMetaTag('property', 'og:image:height', '630');
        setMetaTag('name', 'twitter:image', '{logo_url}');
        """
    
    meta_tags_js += """
    })();
    </script>
    """
    
    # Inject meta tags immediately using st.markdown
    st.markdown(meta_tags_js, unsafe_allow_html=True)
    
    # Favicon is now handled directly via Streamlit's page_icon using the cropped file
    # No need for JavaScript injection - the cropped favicon file is used directly
    # This is more reliable than JavaScript-based approach
    
    favicon_base64 = None
    if logo_path and os.path.exists(logo_path):
        # Use base64 encoding for favicon to ensure it loads and can be sized properly
        import base64
        try:
            # Crop image to remove white/transparent background and zoom into the logo content
            try:
                from PIL import Image
                
                img = Image.open(logo_path)
                
                # Convert to RGBA if needed (for transparency support)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Use crop percentage approach - simple and controllable
                # This crops a percentage from each edge, effectively zooming into the center
                width, height = img.size
                
                # Calculate crop coordinates (centered crop)
                # Ensure crop_percent is valid (0.0 to 0.5 max to avoid invalid coordinates)
                crop_percent = min(crop_percent, 0.49)  # Cap at 49% to ensure valid coordinates
                left = int(width * crop_percent)
                top = int(height * crop_percent)
                right = int(width * (1 - crop_percent))
                bottom = int(height * (1 - crop_percent))
                
                # Validate crop coordinates
                if left >= right or top >= bottom:
                    # Fallback to center 50% if crop is too aggressive
                    crop_percent = 0.25
                    left = int(width * crop_percent)
                    top = int(height * crop_percent)
                    right = int(width * (1 - crop_percent))
                    bottom = int(height * (1 - crop_percent))
                
                # Crop the image
                img = img.crop((left, top, right, bottom))
                
                # Validate the cropped image has valid dimensions
                width, height = img.size
                if width <= 0 or height <= 0:
                    # If crop resulted in invalid dimensions, use a safe default
                    crop_percent = 0.25
                    img = Image.open(logo_path)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    orig_width, orig_height = img.size
                    left = int(orig_width * crop_percent)
                    top = int(orig_height * crop_percent)
                    right = int(orig_width * (1 - crop_percent))
                    bottom = int(orig_height * (1 - crop_percent))
                    img = img.crop((left, top, right, bottom))
                    width, height = img.size
                
                # Make it square by taking the larger dimension and centering
                if width != height:
                    size = max(width, height)
                    # Create a new square image with transparent background
                    square_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                    # Paste the cropped image in the center
                    paste_x = (size - width) // 2
                    paste_y = (size - height) // 2
                    square_img.paste(img, (paste_x, paste_y), img)
                    img = square_img
                
                # Resize to 192x192 for maximum visibility - larger size means more detail
                # Browsers will scale it down but it will appear much larger and clearer
                favicon_size = 192
                img_resized = img.resize((favicon_size, favicon_size), Image.Resampling.LANCZOS)
                
                # Save to bytes
                from io import BytesIO
                img_bytes = BytesIO()
                img_resized.save(img_bytes, format='PNG', optimize=True)
                img_bytes.seek(0)
                favicon_base64 = base64.b64encode(img_bytes.read()).decode()
                
                # Verify we have valid base64 data
                if not favicon_base64 or len(favicon_base64) < 100:
                    favicon_base64 = None
            except (ImportError, Exception) as e:
                # If PIL fails, we can't crop, so don't use favicon_base64
                # This ensures we don't accidentally use the uncropped image
                favicon_base64 = None
                # Log the error for debugging (but don't show to user)
                import sys
                print(f"Favicon processing error: {e}", file=sys.stderr)
        except Exception:
            favicon_base64 = None
    
    # Use JavaScript to inject favicon and force browser to reload it
    # CRITICAL: Only use JavaScript favicon if we have a cropped version
    # If favicon_base64 is None, it means cropping failed, so we'll rely on Streamlit's native favicon
    if favicon_base64 and len(favicon_base64) > 100:  # Ensure we have actual base64 data
        import time
        import random
        # Include crop_percent in cache buster so browser reloads when crop changes
        # This ensures the browser fetches the new cropped image instead of using cached version
        crop_percent_str = str(crop_percent).replace('.', '_')
        cache_buster = f"crop_{crop_percent_str}_{int(time.time())}_{random.randint(1000, 9999)}_denkenlabs"
        try:
            st.markdown(f"""
                <script>
                (function() {{
                    // Function to remove ALL existing favicons - we'll replace with our own
                    function removeAllFavicons() {{
                        // Remove all favicon-related links EXCEPT our cropped one
                        var allLinks = document.querySelectorAll('link[rel*="icon"], link[rel="shortcut icon"], link[rel="apple-touch-icon"]');
                        allLinks.forEach(function(link) {{
                            var href = link.getAttribute('href') || '';
                            var id = link.getAttribute('id') || '';
                            // Keep only our cropped favicon (has our ID or base64 with our cache buster)
                            // Remove everything else, including Streamlit's native favicon
                            var isOurFavicon = (id && id.startsWith('denkenlabs-favicon')) || 
                                              (href.includes('data:image/png;base64') && href.includes('{cache_buster}'));
                            if (!isOurFavicon) {{
                                link.remove();
                            }}
                        }});
                    }}
                    
                    // Function to add our large, cropped favicon
                    function addLargeFavicon() {{
                        // First, remove all existing favicons
                        removeAllFavicons();
                        
                        // Force browser to clear favicon cache by creating a new link with unique ID
                        var faviconUrl = 'data:image/png;base64,{favicon_base64}?v={cache_buster}';
                        
                        // Add our favicon with multiple sizes - prioritize larger sizes
                        // 192x192 for maximum visibility (our processed, cropped image)
                        var link1 = document.createElement('link');
                        link1.id = 'denkenlabs-favicon-192';
                        link1.rel = 'icon';
                        link1.type = 'image/png';
                        link1.sizes = '192x192';
                        link1.href = faviconUrl;
                        // Force reload by cloning and replacing
                        var clone1 = link1.cloneNode(true);
                        document.head.insertBefore(clone1, document.head.firstChild);
                        
                        // 128x128 for high-DPI displays
                        var link2 = document.createElement('link');
                        link2.id = 'denkenlabs-favicon-128';
                        link2.rel = 'icon';
                        link2.type = 'image/png';
                        link2.sizes = '128x128';
                        link2.href = faviconUrl;
                        document.head.insertBefore(link2, document.head.firstChild);
                        
                        // 96x96 for standard high-DPI displays
                        var link3 = document.createElement('link');
                        link3.id = 'denkenlabs-favicon-96';
                        link3.rel = 'icon';
                        link3.type = 'image/png';
                        link3.sizes = '96x96';
                        link3.href = faviconUrl;
                        document.head.insertBefore(link3, document.head.firstChild);
                        
                        // 64x64 for standard displays
                        var link4 = document.createElement('link');
                        link4.id = 'denkenlabs-favicon-64';
                        link4.rel = 'icon';
                        link4.type = 'image/png';
                        link4.sizes = '64x64';
                        link4.href = faviconUrl;
                        document.head.insertBefore(link4, document.head.firstChild);
                        
                        // 32x32 standard favicon (browsers will use this as fallback)
                        var link5_32 = document.createElement('link');
                        link5_32.id = 'denkenlabs-favicon-32';
                        link5_32.rel = 'icon';
                        link5_32.type = 'image/png';
                        link5_32.sizes = '32x32';
                        link5_32.href = faviconUrl;
                        document.head.insertBefore(link5_32, document.head.firstChild);
                        
                        // Shortcut icon (legacy support) - place first for maximum priority
                        var link6 = document.createElement('link');
                        link6.id = 'denkenlabs-favicon-shortcut';
                        link6.rel = 'shortcut icon';
                        link6.type = 'image/png';
                        link6.href = faviconUrl;
                        document.head.insertBefore(link6, document.head.firstChild);
                        
                        // Apple touch icon for mobile
                        var link7 = document.createElement('link');
                        link7.id = 'denkenlabs-favicon-apple';
                        link7.rel = 'apple-touch-icon';
                        link7.sizes = '180x180';
                        link7.href = faviconUrl;
                        document.head.insertBefore(link7, document.head.firstChild);
                        
                        // Force browser to reload favicon by temporarily removing and re-adding
                        // This helps bypass aggressive browser caching
                        setTimeout(function() {{
                            var favLinks = document.querySelectorAll('link[id^="denkenlabs-favicon"]');
                            favLinks.forEach(function(link) {{
                                var oldHref = link.href;
                                link.href = '';
                                setTimeout(function() {{
                                    link.href = oldHref;
                                }}, 10);
                            }});
                        }}, 100);
                    }}
                    
                    // Run immediately - don't wait for anything
                    removeAllFavicons();
                    addLargeFavicon();
                    
                    // Also run on DOM load
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', function() {{
                            removeAllFavicons();
                            addLargeFavicon();
                        }});
                    }} else {{
                        // DOM already loaded, run immediately again
                        removeAllFavicons();
                        addLargeFavicon();
                    }}
                    
                    // Run multiple times aggressively to catch Streamlit's favicon additions
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 10);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 50);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 100);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 200);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 500);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 1000);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 2000);
                    setTimeout(function() {{ removeAllFavicons(); addLargeFavicon(); }}, 3000);
                    
                    // Watch for ANY new favicon links and replace them immediately
                    var observer = new MutationObserver(function(mutations) {{
                        var shouldReplace = false;
                        mutations.forEach(function(mutation) {{
                            mutation.addedNodes.forEach(function(node) {{
                                if (node.nodeName === 'LINK') {{
                                    var rel = node.getAttribute('rel') || '';
                                    var href = node.getAttribute('href') || '';
                                    if (rel.toLowerCase().includes('icon')) {{
                                        var id = node.getAttribute('id') || '';
                                        // If it's not our cropped favicon, remove it immediately
                                        var isOurFavicon = (id && id.startsWith('denkenlabs-favicon')) || 
                                                          (href.includes('data:image/png;base64') && href.includes('{cache_buster}'));
                                        if (!isOurFavicon) {{
                                            node.remove();
                                            shouldReplace = true;
                                        }}
                                    }}
                                }}
                            }});
                        }});
                        if (shouldReplace) {{
                            // Replace immediately when Streamlit adds its favicon
                            removeAllFavicons();
                            addLargeFavicon();
                        }}
                    }});
                    
                    // Observe the head for any changes
                    observer.observe(document.head, {{ childList: true, subtree: true }});
                    
                    // Also observe the document for any head additions
                    if (document.documentElement) {{
                        observer.observe(document.documentElement, {{ childList: true, subtree: true }});
                    }}
                }})();
                </script>
                """, unsafe_allow_html=True)
        except Exception as e:
            # Fallback: try to use base64 of original image without processing
            try:
                with open(logo_path, "rb") as f:
                    fallback_base64 = base64.b64encode(f.read()).decode()
                import time
                import random
                cache_buster = f"{int(time.time())}_{random.randint(1000, 9999)}_fallback"
                st.markdown(f"""
                <script>
                (function() {{
                    function setFavicon() {{
                        var link = document.querySelector('link[rel*="icon"]') || document.createElement('link');
                        link.rel = 'icon';
                        link.type = 'image/png';
                        link.href = 'data:image/png;base64,{fallback_base64}?v={cache_buster}';
                        if (!document.querySelector('link[rel*="icon"]')) {{
                            document.head.appendChild(link);
                        }}
                    }}
                    setFavicon();
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', setFavicon);
                    }}
                }})();
                </script>
                """, unsafe_allow_html=True)
            except Exception:
                pass  # If all else fails, Streamlit's native page_icon will be used
    
    # Mobile-responsive CSS with minimal spacing and logo-matching colors
    st.markdown("""
    <style>
    /* Aggressively remove white backgrounds from logo and all parent containers in dark mode */
    [data-theme="dark"] img[alt="Denken Labs Logo"],
    [data-theme="dark"] img[alt="Denken Labs Logo"] ~ *,
    [data-theme="dark"] .stImage img,
    [data-theme="dark"] .stImage > div,
    [data-theme="dark"] .stImage > div > img,
    [data-theme="dark"] div[data-testid="stImage"],
    [data-theme="dark"] div[data-testid="stImage"] > div,
    [data-theme="dark"] div[data-testid="stImage"] > div > img,
    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] .stMarkdown > div,
    [data-theme="dark"] .logo-container,
    [data-theme="dark"] .logo-container > div,
    [data-theme="dark"] .logo-container > div > div,
    [data-theme="dark"] div:has(img[alt="Denken Labs Logo"]),
    [data-theme="dark"] div:has(img[alt="Denken Labs Logo"]) > div,
    [data-theme="dark"] div:has(img[alt="Denken Labs Logo"]) > div > div {
        background: transparent !important;
        background-color: transparent !important;
        mix-blend-mode: normal !important;
    }
    
    /* Remove white backgrounds from Streamlit markdown containers that might wrap the logo */
    [data-theme="dark"] .element-container:has(img[alt="Denken Labs Logo"]),
    [data-theme="dark"] .element-container:has(img[alt="Denken Labs Logo"]) > div,
    [data-theme="dark"] .stMarkdown:has(img[alt="Denken Labs Logo"]),
    [data-theme="dark"] .stMarkdown:has(img[alt="Denken Labs Logo"]) > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Force transparent background on the logo image itself */
    img[alt="Denken Labs Logo"] {
        background: transparent !important;
        background-color: transparent !important;
        -webkit-background-clip: padding-box !important;
        background-clip: padding-box !important;
    }
    
    /* Remove any white/light backgrounds that might appear in dark mode */
    [data-theme="dark"] div:has(img[alt="Denken Labs Logo"]) {
        background: transparent !important;
        background-color: transparent !important;
    }
    </style>
    <style>
    /* Allow dark mode but ensure text is very light/bright for readability */
    /* Use comprehensive selectors to catch all text elements */
    [data-theme="dark"] * {
        color: #ffffff !important; /* Pure white for all text by default */
    }
    
    /* Specifically target all markdown and text elements */
    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] .stMarkdown *,
    [data-theme="dark"] .stMarkdown p,
    [data-theme="dark"] .stMarkdown strong,
    [data-theme="dark"] .stMarkdown em,
    [data-theme="dark"] .stMarkdown h1,
    [data-theme="dark"] .stMarkdown h2,
    [data-theme="dark"] .stMarkdown h3,
    [data-theme="dark"] .stMarkdown h4,
    [data-theme="dark"] .stMarkdown h5,
    [data-theme="dark"] .stMarkdown h6,
    [data-theme="dark"] .stMarkdown li,
    [data-theme="dark"] .stMarkdown span,
    [data-theme="dark"] .stMarkdown a,
    [data-theme="dark"] .stMarkdown div,
    [data-theme="dark"] .stMarkdown ul,
    [data-theme="dark"] .stMarkdown ol,
    [data-theme="dark"] .stText,
    [data-theme="dark"] .stText *,
    [data-theme="dark"] .stTextInput,
    [data-theme="dark"] .stTextInput *,
    [data-theme="dark"] .stTextInput > div > div > input,
    [data-theme="dark"] .stTextArea,
    [data-theme="dark"] .stTextArea *,
    [data-theme="dark"] .stTextArea > div > div > textarea,
    [data-theme="dark"] .stExpander,
    [data-theme="dark"] .stExpander *,
    [data-theme="dark"] .stExpander label,
    [data-theme="dark"] .stExpander .stMarkdown,
    [data-theme="dark"] .stExpander .stMarkdown *,
    [data-theme="dark"] .stExpander .stMarkdown p,
    [data-theme="dark"] .stInfo,
    [data-theme="dark"] .stInfo *,
    [data-theme="dark"] .stInfo .stMarkdown,
    [data-theme="dark"] .stInfo .stMarkdown *,
    [data-theme="dark"] .stSuccess,
    [data-theme="dark"] .stSuccess *,
    [data-theme="dark"] .stSuccess .stMarkdown,
    [data-theme="dark"] .stSuccess .stMarkdown *,
    [data-theme="dark"] .stWarning,
    [data-theme="dark"] .stWarning *,
    [data-theme="dark"] .stWarning .stMarkdown,
    [data-theme="dark"] .stWarning .stMarkdown *,
    [data-theme="dark"] .stError,
    [data-theme="dark"] .stError *,
    [data-theme="dark"] .stError .stMarkdown,
    [data-theme="dark"] .stError .stMarkdown *,
    [data-theme="dark"] .stSelectbox,
    [data-theme="dark"] .stSelectbox *,
    [data-theme="dark"] .stSelectbox label,
    [data-theme="dark"] .stSelectbox > div > div,
    [data-theme="dark"] .stCheckbox,
    [data-theme="dark"] .stCheckbox *,
    [data-theme="dark"] .stCheckbox label,
    [data-theme="dark"] .stRadio,
    [data-theme="dark"] .stRadio *,
    [data-theme="dark"] .stRadio label,
    [data-theme="dark"] .stLabel,
    [data-theme="dark"] .stLabel *,
    [data-theme="dark"] .stSubheader,
    [data-theme="dark"] .stTitle,
    [data-theme="dark"] .stHeader {
        color: #ffffff !important; /* Pure white text for maximum readability */
    }
    
    /* Ensure all headings are bright white in dark mode */
    [data-theme="dark"] h1,
    [data-theme="dark"] h2,
    [data-theme="dark"] h3,
    [data-theme="dark"] h4,
    [data-theme="dark"] h5,
    [data-theme="dark"] h6 {
        color: #ffffff !important; /* Pure white for headings */
    }
    
    /* Make input placeholders readable but distinguishable in dark mode */
    [data-theme="dark"] input::placeholder,
    [data-theme="dark"] textarea::placeholder {
        color: #e2e8f0 !important; /* Lighter gray placeholder text for better visibility */
        opacity: 1;
    }
    
    /* Ensure input text is pure white when typing in dark mode */
    [data-theme="dark"] input[type="text"],
    [data-theme="dark"] input[type="text"]:focus,
    [data-theme="dark"] input[type="email"],
    [data-theme="dark"] input[type="password"],
    [data-theme="dark"] textarea,
    [data-theme="dark"] textarea:focus {
        color: #ffffff !important; /* Pure white text in inputs */
    }
    
    /* Ensure buttons have bright readable text in dark mode */
    [data-theme="dark"] .stButton,
    [data-theme="dark"] .stButton *,
    [data-theme="dark"] .stButton > button {
        color: #ffffff !important;
    }
    
    /* Make sure custom styled elements are also bright */
    [data-theme="dark"] .main-content,
    [data-theme="dark"] .main-content *,
    [data-theme="dark"] .header-container,
    [data-theme="dark"] .header-container *,
    [data-theme="dark"] .logo-container,
    [data-theme="dark"] .logo-container * {
        color: #ffffff !important;
    }
    
    /* Specifically target descriptive/intro text in dark mode - make them light gray (not pure white) */
    [data-theme="dark"] .agent-description-intro,
    [data-theme="dark"] .agent-description-intro *,
    [data-theme="dark"] .agent-description-intro strong,
    [data-theme="dark"] .agent-description-intro p,
    [data-theme="dark"] .agent-description-intro span,
    [data-theme="dark"] .tagline-text,
    [data-theme="dark"] .tagline-text *,
    [data-theme="dark"] .tagline-text strong,
    [data-theme="dark"] .tagline-text span,
    [data-theme="dark"] .creative-name-intro,
    [data-theme="dark"] .creative-name-intro *,
    [data-theme="dark"] .creative-name-intro strong,
    [data-theme="dark"] .creative-name-intro span,
    [data-theme="dark"] .creative-name-intro em,
    [data-theme="dark"] .welcome-message,
    [data-theme="dark"] .welcome-message *,
    [data-theme="dark"] .welcome-message strong,
    [data-theme="dark"] .welcome-message span {
        color: #e2e8f0 !important; /* Light gray - lighter than default but not pure white */
        font-weight: 500 !important;
    }
    
    /* Story title and author in dark mode - make them visible */
    [data-theme="dark"] .story-title,
    [data-theme="dark"] .story-title * {
        color: #ffffff !important;
    }
    
    [data-theme="dark"] .story-author,
    [data-theme="dark"] .story-author * {
        color: #e2e8f0 !important;
    }
    
    /* Story content - use Streamlit default container styling (adapts to light/dark mode) */
    /* Keep purple border for visual consistency */
    .story-content {
        border-left-color: #6b46c1 !important; /* Purple border for visual consistency */
    }
    
    /* Light blue text for specific elements in dark mode - use ID for maximum specificity */
    [data-theme="dark"] #welcome-title-header,
    [data-theme="dark"] #welcome-title-header *,
    [data-theme="dark"] h2#welcome-title-header,
    [data-theme="dark"] h2#welcome-title-header *,
    [data-theme="dark"] .welcome-title#welcome-title-header,
    [data-theme="dark"] .welcome-title#welcome-title-header *,
    [data-theme="dark"] .welcome-title,
    [data-theme="dark"] .welcome-title * {
        color: #bfdbfe !important; /* Very light blue - lighter for better visibility */
    }
    
    /* Additional CSS to ensure the inline style is overridden in dark mode */
    [data-theme="dark"] h2[data-light-color] {
        color: #bfdbfe !important;
    }
    [data-theme="dark"] h2[data-light-color] * {
        color: #bfdbfe !important;
    }
    
    [data-theme="dark"] .version-number,
    [data-theme="dark"] .version-number * {
        color: #bfdbfe !important; /* Very light blue - lighter for better visibility */
    }
    
    /* Agent number and name - make them very light/white in dark mode */
    [data-theme="dark"] .agent-number,
    [data-theme="dark"] .agent-number *,
    [data-theme="dark"] .agent-number strong,
    [data-theme="dark"] .agent-number div,
    [data-theme="dark"] div.agent-number,
    [data-theme="dark"] div.agent-number *,
    [data-theme="dark"] div.agent-number strong {
        color: #dbeafe !important; /* Very light blue - even lighter for better visibility */
    }
    
    [data-theme="dark"] .agent-name,
    [data-theme="dark"] .agent-name *,
    [data-theme="dark"] .agent-name strong,
    [data-theme="dark"] .agent-name div,
    [data-theme="dark"] div.agent-name,
    [data-theme="dark"] div.agent-name *,
    [data-theme="dark"] div.agent-name strong {
        color: #dbeafe !important; /* Very light blue - even lighter for better visibility */
    }
    
    /* Override any Streamlit default dark mode text colors */
    [data-theme="dark"] p,
    [data-theme="dark"] span,
    [data-theme="dark"] div,
    [data-theme="dark"] label,
    [data-theme="dark"] li,
    [data-theme="dark"] td,
    [data-theme="dark"] th {
        color: #ffffff !important;
    }
    
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
        background-color: var(--background) !important;
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
    // Completely different approach: Use computed style override and requestAnimationFrame
    function brightenAgentNames() {
        var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        if (!isDark) return;
        
        // Target all agent names using multiple selector strategies
        var allSelectors = document.querySelectorAll(
            '[id^="agent-name-"], ' +
            '[data-agent-name="true"], ' +
            '.agent-name-bright, ' +
            '.agent-name, ' +
            'div.agent-name-bright, ' +
            'div.agent-name'
        );
        
        allSelectors.forEach(function(el) {
            // Use requestAnimationFrame for smooth updates
            requestAnimationFrame(function() {
                // Get computed style to see what's actually applied
                var computed = window.getComputedStyle(el);
                var currentColor = computed.color;
                
                // If not already white or very light, force it
                if (currentColor !== 'rgb(255, 255, 255)' && 
                    currentColor !== 'rgb(255, 255, 254)' &&
                    currentColor !== 'rgb(255, 255, 253)') {
                    
                    // Method 1: Direct style property with important
                    el.style.setProperty('color', '#ffffff', 'important');
                    
                    // Method 2: CSS variable override
                    el.style.setProperty('--agent-name-color', '#ffffff', 'important');
                    el.style.setProperty('color', 'var(--agent-name-color)', 'important');
                    
                    // Method 3: Direct color assignment
                    el.style.color = '#ffffff';
                    
                    // Method 4: Inline style manipulation
                    var inlineStyle = el.getAttribute('style') || '';
                    // Remove any color declarations
                    inlineStyle = inlineStyle.replace(/color\\s*:[^;]*;?/gi, '');
                    inlineStyle = inlineStyle.replace(/--agent-name-color\\s*:[^;]*;?/gi, '');
                    // Add our bright white color
                    inlineStyle += ' --agent-name-color: #ffffff !important;';
                    inlineStyle += ' color: #ffffff !important;';
                    el.setAttribute('style', inlineStyle);
                    
                    // Apply to all children with same intensity
                    var children = el.querySelectorAll('*');
                    children.forEach(function(child) {
                        requestAnimationFrame(function() {
                            child.style.setProperty('color', '#ffffff', 'important');
                            child.style.setProperty('--agent-name-color', '#ffffff', 'important');
                            child.style.color = '#ffffff';
                            var childStyle = child.getAttribute('style') || '';
                            childStyle = childStyle.replace(/color\\s*:[^;]*;?/gi, '');
                            childStyle += ' color: #ffffff !important;';
                            child.setAttribute('style', childStyle);
                        });
                    });
                }
            });
        });
    }
    
    // Run with multiple strategies
    brightenAgentNames();
    requestAnimationFrame(brightenAgentNames);
    setTimeout(brightenAgentNames, 0);
    setTimeout(brightenAgentNames, 5);
    setTimeout(brightenAgentNames, 10);
    setTimeout(brightenAgentNames, 25);
    
    // Continuous polling with requestAnimationFrame
    function pollAgentNames() {
        brightenAgentNames();
        requestAnimationFrame(pollAgentNames);
    }
    requestAnimationFrame(pollAgentNames);
    
    // Also use setInterval as backup
    setInterval(brightenAgentNames, 10); // Very frequent checking
    
    // Watch for theme changes
    var themeObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                requestAnimationFrame(brightenAgentNames);
                setTimeout(brightenAgentNames, 0);
                setTimeout(brightenAgentNames, 10);
            }
        });
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    
    // Watch for DOM additions
    var domObserver = new MutationObserver(function() {
        requestAnimationFrame(brightenAgentNames);
    });
    domObserver.observe(document.body, { childList: true, subtree: true });
    
    // Force "Welcome to Denken Labs" to be light in dark mode - target div with ID welcome-title-element
    function forceWelcomeTitleLight() {
        var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        
        // Target by ID first (most specific) - now using welcome-title-element div
        var welcomeTitle = document.getElementById('welcome-title-element');
        if (!welcomeTitle) {
            // Fallback: find by text content
            var allElements = document.querySelectorAll('div, h1, h2, h3, span');
            for (var i = 0; i < allElements.length; i++) {
                var text = (allElements[i].textContent || allElements[i].innerText || '').trim();
                if (text === 'Welcome to Denken Labs') {
                    welcomeTitle = allElements[i];
                    welcomeTitle.id = 'welcome-title-element';
                    break;
                }
            }
        }
        
        if (welcomeTitle && isDark) {
            // Apply light blue color with maximum priority
            welcomeTitle.style.color = '#bfdbfe';
            welcomeTitle.style.setProperty('color', '#bfdbfe', 'important');
            var currentStyle = welcomeTitle.getAttribute('style') || '';
            currentStyle = currentStyle.replace(/color[^;]*;?/gi, '');
            welcomeTitle.setAttribute('style', currentStyle + ' color: #bfdbfe !important;');
            
            // Apply to children
            var children = welcomeTitle.querySelectorAll('*');
            children.forEach(function(child) {
                child.style.setProperty('color', '#bfdbfe', 'important');
            });
        }
    }
    
    // Run immediately and continuously for welcome title
    forceWelcomeTitleLight();
    setInterval(forceWelcomeTitleLight, 50); // More frequent checking
    
    // Watch for theme changes
    var welcomeThemeObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                setTimeout(forceWelcomeTitleLight, 10);
            }
        });
    });
    welcomeThemeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    
    // Also watch for style attribute changes on the welcome title
    var welcomeElement = document.getElementById('welcome-title-element');
    if (welcomeElement) {
        var welcomeStyleObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    setTimeout(forceWelcomeTitleLight, 10);
                }
            });
        });
        welcomeStyleObserver.observe(welcomeElement, { attributes: true, attributeFilter: ['style'] });
    }
    
    // Story content now uses Streamlit default container styling (adapts to light/dark mode)
    // No need to force colors - let Streamlit handle theme adaptation
    
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
            
            // Purple for Build Your Own Adventure button
            if (text.includes('Build Your Own Adventure') || text.includes('🚀 Build Your Own Adventure')) {
                forcePurpleButton(btn);
            }
            
            // Purple for Create button - target ALL Create buttons unless in mission form
            if (text === 'Create' || text === 'Create Your Agent' || text === 'Keep Creating More Agents' || (text.includes('Create') && !text.includes('Mission') && !text.includes('Save'))) {
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
    
    // Use MutationObserver to catch dynamically added buttons and story content
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
                        // Also check for story content
                        var storyDivs = node.querySelectorAll('.story-content, div.story-content');
                        if (storyDivs.length > 0 || node.classList && node.classList.contains('story-content')) {
                            forceStoryTransparent();
                        }
                        // Check for welcome title by ID or text
                        if (node.querySelectorAll('#welcome-title-element, .welcome-title, h2, div[id*="welcome"]').length > 0) {
                            forceWelcomeTitleLight();
                        }
                        // Check for agent names
                        var agentNames = node.querySelectorAll('[id^="agent-name-"], .agent-name');
                        if (agentNames.length > 0) {
                            forceAgentNamesLight();
                        }
                    }
                    // Check if the node itself is story content
                    if (node.classList && node.classList.contains('story-content')) {
                        forceStoryTransparent();
                    }
                    // Check if the node itself is welcome title (by ID or text)
                    if ((node.id === 'welcome-title-element') || 
                        (node.classList && node.classList.contains('welcome-title')) || 
                        ((node.tagName === 'DIV' || node.tagName === 'H2') && node.textContent && node.textContent.includes('Welcome'))) {
                        forceWelcomeTitleLight();
                    }
                    // Check if the node itself is an agent name
                    if (node.id && node.id.startsWith('agent-name-') || 
                        (node.classList && node.classList.contains('agent-name'))) {
                        forceAgentNamesLight();
                    }
                }
            });
        });
        styleButtons();
        forceStoryTransparent();
        forceWelcomeTitleLight();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Listen for Streamlit's custom events
    window.addEventListener('streamlit:rerun', function() {
        setTimeout(function() {
            styleButtons();
            forceStoryTransparent();
            forceWelcomeTitleLight();
            forceAgentNamesLight();
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
            # Process logo to remove gray/white background and make it transparent
            import base64
            try:
                from PIL import Image
                from io import BytesIO
                
                # Open the logo image
                img = Image.open(logo_path)
                
                # Convert to RGBA if not already
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Get image data
                data = img.get_flattened_data()
                
                # Create new image data with transparent background
                # Remove gray/white backgrounds (RGB values close to gray/white)
                new_data = []
                for item in data:
                    r, g, b, a = item
                    # Check if pixel is gray/white (similar RGB values and high brightness)
                    # Gray typically has similar R, G, B values
                    rgb_avg = (r + g + b) / 3
                    is_gray = abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30
                    is_light = rgb_avg > 200  # Light gray/white
                    
                    # If it's a light gray/white pixel, make it transparent
                    if is_gray and is_light:
                        new_data.append((r, g, b, 0))  # Fully transparent
                    else:
                        new_data.append(item)  # Keep original
                
                # Apply the new data
                img.putdata(new_data)
                
                # Save to bytes
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                logo_base64 = base64.b64encode(img_bytes.read()).decode()
                
            except Exception:
                # Fallback: use original file if processing fails
                with open(logo_path, "rb") as img_file:
                    logo_base64 = base64.b64encode(img_file.read()).decode()
            
            st.markdown(f"""
                <div style="text-align: center; background: transparent !important; background-color: transparent !important; padding: 0; margin: 0;">
                    <img src="data:image/png;base64,{logo_base64}" 
                         style="max-width: 280px; width: 100%; height: auto; 
                                background: transparent !important; 
                                background-color: transparent !important;
                                display: block;
                                margin: 0 auto;
                                -webkit-background-clip: padding-box;
                                background-clip: padding-box;" 
                         alt="Denken Labs Logo" />
                </div>
                """, unsafe_allow_html=True)
            
            # Add JavaScript in a separate markdown call to prevent rendering issues
            st.markdown("""
                <script>
                (function() {
                    function removeWhiteBackgrounds() {
                        var logoImg = document.querySelector('img[alt="Denken Labs Logo"]');
                        if (logoImg) {
                            logoImg.style.background = 'transparent';
                            logoImg.style.backgroundColor = 'transparent';
                            var parent = logoImg.parentElement;
                            var depth = 0;
                            while (parent && parent !== document.body && depth < 20) {
                                var computedStyle = window.getComputedStyle(parent);
                                var bgColor = computedStyle.backgroundColor;
                                if (bgColor && (
                                    bgColor.includes('rgb(255') || 
                                    bgColor.includes('rgb(250') ||
                                    bgColor.includes('rgb(249') ||
                                    bgColor.includes('white') ||
                                    bgColor.includes('rgba(255') ||
                                    bgColor.includes('rgba(250') ||
                                    bgColor.includes('rgba(249')
                                )) {
                                    parent.style.background = 'transparent';
                                    parent.style.backgroundColor = 'transparent';
                                } else {
                                    if (document.documentElement.getAttribute('data-theme') === 'dark') {
                                        parent.style.background = 'transparent';
                                        parent.style.backgroundColor = 'transparent';
                                    }
                                }
                                parent = parent.parentElement;
                                depth++;
                            }
                        }
                    }
                    removeWhiteBackgrounds();
                    setTimeout(removeWhiteBackgrounds, 100);
                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', removeWhiteBackgrounds);
                    }
                    var observer = new MutationObserver(function(mutations) {
                        removeWhiteBackgrounds();
                    });
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['style', 'class']
                    });
                    var themeObserver = new MutationObserver(function(mutations) {
                        removeWhiteBackgrounds();
                    });
                    if (document.documentElement) {
                        themeObserver.observe(document.documentElement, {
                            attributes: true,
                            attributeFilter: ['data-theme']
                        });
                    }
                })();
                </script>
                """, unsafe_allow_html=True)
        else:
            # Debug: show which paths were checked
            st.info(f"Logo not found. Checked: {logo_paths}")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
    
    st.markdown(f"""
            <h2 class="version-number" style="color: var(--primary-color); margin: 0; font-size: 1.1rem; font-weight: 400; text-align: right;">v{APP_VERSION}</h2>
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
        # Use div instead of h2 to avoid Streamlit's heading styles, with inline !important
        st.markdown('<div id="welcome-title-element" style="font-size: 2.25rem; font-weight: 600; color: #bfdbfe !important; margin-bottom: 0.5rem;">Welcome to Denken Labs</div>', unsafe_allow_html=True)
        st.markdown('<div class="tagline-text">**Get ready for an exiting mission**</div>', unsafe_allow_html=True)
        
        # Test if JavaScript is working at all
        st.markdown("""
        <script>
        console.log('🧪 TEST: JavaScript is working!');
        alert('JavaScript test - if you see this, JS is working!');
        </script>
        """, unsafe_allow_html=True)
        
        # Inject audio initialization script FIRST (before button) so button can use it
        play_sound('')  # This injects the audio init script without playing a sound
        
        # Add enable sounds button if audio not enabled - using separate markdown calls
        st.markdown("""
        <div id="enable-audio-container" style="text-align: center; margin: 10px 0;">
            <button id="enable-audio-btn" 
                    style="background-color: #4CAF50; color: white; border: none; padding: 10px 20px; 
                           border-radius: 5px; cursor: pointer; font-size: 14px; font-weight: bold;">
                🔊 Enable Sounds
            </button>
            <div id="audio-status" style="margin-top: 5px; font-size: 12px; color: #666;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Separate script injection to ensure it executes
        st.markdown("""
        <script>
        (function() {
            console.log('=== BUTTON SCRIPT LOADING ===');
            
            function setupAudioButton() {
                console.log('Setting up audio button...');
                const btn = document.getElementById('enable-audio-btn');
                const status = document.getElementById('audio-status');
                
                if (!btn) {
                    console.error('Button not found!');
                    return;
                }
                
                console.log('Button found, checking audio system...');
                console.log('denkenAudioSystem exists:', !!window.denkenAudioSystem);
                console.log('enableDenkenAudio exists:', typeof window.enableDenkenAudio);
                
                // Hide button if audio is already enabled
                if (window.denkenAudioSystem && window.denkenAudioSystem.enabled) {
                    console.log('Audio already enabled, hiding button');
                    btn.style.display = 'none';
                    if (status) status.textContent = '✅ Sounds enabled';
                    return;
                }
                
                // Remove any existing listeners by cloning
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
                
                // Add click handler to new button
                newBtn.addEventListener('click', async function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🔘 BUTTON CLICKED!');
                    
                    try {
                        if (status) status.textContent = '🔄 Enabling sounds...';
                        newBtn.disabled = true;
                        newBtn.style.opacity = '0.6';
                        
                        let context = null;
                        
                        // Call enable function
                        if (window.enableDenkenAudio) {
                            console.log('Calling enableDenkenAudio...');
                            context = await window.enableDenkenAudio();
                            console.log('enableDenkenAudio returned:', context);
                        } else {
                            console.error('enableDenkenAudio function not found!');
                            if (status) status.textContent = '❌ Audio system not initialized. Please refresh the page.';
                            newBtn.disabled = false;
                            newBtn.style.opacity = '1';
                            return;
                        }
                        
                        if (context && context.state === 'running') {
                            if (status) status.textContent = '✅ Sounds enabled! You should hear a test sound.';
                            newBtn.textContent = '✅ Sounds Enabled';
                            setTimeout(() => {
                                if (status) status.textContent = '✅ Sounds are now enabled for all actions!';
                                setTimeout(() => {
                                    newBtn.style.display = 'none';
                                    if (status) status.style.display = 'none';
                                }, 2000);
                            }, 1000);
                        } else {
                            if (status) status.textContent = '⚠️ Audio enabled but context state: ' + (context ? context.state : 'null');
                            newBtn.disabled = false;
                            newBtn.style.opacity = '1';
                        }
                    } catch(e) {
                        console.error('❌ Error enabling audio:', e);
                        if (status) status.textContent = '❌ Error: ' + e.message;
                        newBtn.disabled = false;
                        newBtn.style.opacity = '1';
                    }
                });
                
                console.log('Button setup complete');
            }
            
            // Try to setup immediately
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupAudioButton);
            } else {
                setupAudioButton();
            }
            
            // Also try after delays (in case Streamlit hasn't rendered yet)
            setTimeout(setupAudioButton, 500);
            setTimeout(setupAudioButton, 1000);
            setTimeout(setupAudioButton, 2000);
            
            console.log('=== BUTTON SCRIPT COMPLETE ===');
        })();
        </script>
        """, unsafe_allow_html=True)
        
        # Build your own agent button - compact (purple)
        st.markdown("""
        <style>
        button[kind="primary"]:contains("Build Your Own Adventure"),
        div.stButton:has(> button:contains("Build Your Own Adventure")) button {
            background-color: #6b46c1 !important;
            border-color: #6b46c1 !important;
        }
        button[kind="primary"]:contains("Build Your Own Adventure"):hover,
        div.stButton:has(> button:contains("Build Your Own Adventure")) button:hover {
            background-color: #553c9a !important;
            border-color: #553c9a !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🚀 Build Your Own Adventure", type="primary", use_container_width=True, key="build_agent_btn"):
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
        
        # User's Creative Name section (first time only)
        if 'user_creative_name' not in st.session_state or not st.session_state.user_creative_name:
            st.markdown("### ✨ Your Creative Name")
            name_examples = generate_creative_name_examples()
            example_text = f"Examples: {', '.join(name_examples)}"
            st.markdown(f'<div class="creative-name-intro">**Enter your creative name!** *{example_text}*</div>', unsafe_allow_html=True)
            
            with st.form("user_name_form", clear_on_submit=True):
                # Add CSS to make name input slightly bigger
                st.markdown("""
                <style>
                input[data-testid*="user_name_input"] {
                    padding: 0.75rem 1rem !important;
                    font-size: 1rem !important;
                    min-height: 48px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                user_creative_name = st.text_input(
                    "Enter your creative name:",
                    placeholder=f"Example: {name_examples[0]}",
                    key="user_name_input"
                )
                name_submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
                
                if name_submitted:
                    # Use example name if nothing entered or only whitespace
                    if not user_creative_name or not user_creative_name.strip():
                        st.session_state.user_creative_name = name_examples[0]
                    else:
                        st.session_state.user_creative_name = user_creative_name.strip()
                    # Play sound for user name logged
                    play_sound('user_name')
                    st.rerun()
            return  # Stop here until user enters their name
        
        # Show user's name if already entered
        if st.session_state.user_creative_name:
            st.markdown(f'<div class="welcome-message">**Welcome, {st.session_state.user_creative_name}!** 🎉</div>', unsafe_allow_html=True)
            if st.button("Change Name", key="change_name_btn"):
                st.session_state.user_creative_name = ""
                st.rerun()
        
        # Display bot logo (only when creating agents)
        if bot_path:
            st.image(bot_path, width=100)
        else:
            st.info("🤖 Bot logo")
        
        # Agent description text input
        st.markdown("### Describe Your Agent")
        st.markdown('<div class="agent-description-intro">**Create individual AI agents with specific capabilities and personalities. When you have 2 or more agents, they form a team and can be assigned missions to work together.**</div>', unsafe_allow_html=True)
        
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
            
            # Create button (should be purple) - change text based on whether agents exist
            num_agents = len(st.session_state.get('created_bots', []))
            button_text = "Create Your Agent" if num_agents == 0 else "Keep Creating More Agents"
            submitted = st.form_submit_button(button_text, type="primary", use_container_width=True)
            
            if submitted:
                # Use example if description is blank
                if not agent_description or not agent_description.strip():
                    agent_description = st.session_state.agent_example
                
                if agent_description and agent_description.strip():
                    try:
                        # Generate bot name, description, and elaborate character using OpenAI
                        client = openai.OpenAI(api_key=get_openai_api_key())
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
                        
                        # Handle case where description might be a dict or contain dict-like structure
                        if isinstance(bot_desc, dict):
                            # If description is a dict, format it as readable text
                            desc_parts = []
                            if 'traits' in bot_desc:
                                desc_parts.append(f"Traits: {', '.join(bot_desc['traits']) if isinstance(bot_desc['traits'], list) else bot_desc['traits']}")
                            if 'working_style' in bot_desc:
                                desc_parts.append(f"Working Style: {bot_desc['working_style']}")
                            if 'expertise' in bot_desc:
                                desc_parts.append(f"Expertise: {bot_desc['expertise']}")
                            if 'approach' in bot_desc:
                                desc_parts.append(f"Approach: {bot_desc['approach']}")
                            bot_desc = ". ".join(desc_parts) if desc_parts else agent_description[:100]
                        elif isinstance(bot_desc, str) and (bot_desc.startswith('{') or bot_desc.startswith("'")):
                            # If description is a string representation of a dict, use fallback
                            bot_desc = agent_description[:100]
                        
                        # Handle case where character might be a dict
                        if isinstance(bot_character, dict):
                            char_parts = []
                            if 'traits' in bot_character:
                                char_parts.append(f"Traits: {', '.join(bot_character['traits']) if isinstance(bot_character['traits'], list) else bot_character['traits']}")
                            if 'working_style' in bot_character:
                                char_parts.append(f"Working Style: {bot_character['working_style']}")
                            if 'expertise' in bot_character:
                                char_parts.append(f"Expertise: {bot_character['expertise']}")
                            if 'approach' in bot_character:
                                char_parts.append(f"Approach: {bot_character['approach']}")
                            bot_character = ". ".join(char_parts) if char_parts else "A versatile AI agent ready to assist."
                        elif isinstance(bot_character, str) and (bot_character.startswith('{') or bot_character.startswith("'")):
                            # If character is a string representation of a dict, use fallback
                            bot_character = "A versatile AI agent ready to assist."
                        
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
                        
                        # Play sound for agent created
                        play_sound('agent_created')
                        
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
                        st.markdown(f'<div class="agent-number"><strong>Editing Agent #{bot_number}</strong></div>', unsafe_allow_html=True)
                        
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
                                    
                                    client = openai.OpenAI(api_key=get_openai_api_key())
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
                                    
                                    # Handle case where description might be a dict or contain dict-like structure
                                    if isinstance(new_desc, dict):
                                        # If description is a dict, format it as readable text
                                        desc_parts = []
                                        if 'traits' in new_desc:
                                            desc_parts.append(f"Traits: {', '.join(new_desc['traits']) if isinstance(new_desc['traits'], list) else new_desc['traits']}")
                                        if 'working_style' in new_desc:
                                            desc_parts.append(f"Working Style: {new_desc['working_style']}")
                                        if 'expertise' in new_desc:
                                            desc_parts.append(f"Expertise: {new_desc['expertise']}")
                                        if 'approach' in new_desc:
                                            desc_parts.append(f"Approach: {new_desc['approach']}")
                                        new_desc = ". ".join(desc_parts) if desc_parts else edited_description[:100]
                                    elif isinstance(new_desc, str) and (new_desc.startswith('{') or new_desc.startswith("'")):
                                        # If description is a string representation of a dict, use fallback
                                        new_desc = edited_description[:100]
                                    
                                    # Handle case where character might be a dict
                                    if isinstance(new_character, dict):
                                        char_parts = []
                                        if 'traits' in new_character:
                                            char_parts.append(f"Traits: {', '.join(new_character['traits']) if isinstance(new_character['traits'], list) else new_character['traits']}")
                                        if 'working_style' in new_character:
                                            char_parts.append(f"Working Style: {new_character['working_style']}")
                                        if 'expertise' in new_character:
                                            char_parts.append(f"Expertise: {new_character['expertise']}")
                                        if 'approach' in new_character:
                                            char_parts.append(f"Approach: {new_character['approach']}")
                                        new_character = ". ".join(char_parts) if char_parts else "A versatile AI agent ready to assist."
                                    elif isinstance(new_character, str) and (new_character.startswith('{') or new_character.startswith("'")):
                                        # If character is a string representation of a dict, use fallback
                                        new_character = "A versatile AI agent ready to assist."
                                    
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
                            st.markdown(f"<div class='agent-number' style='text-align: left; margin-top: 5px;'><strong>#{bot.get('number', bot_number)}</strong></div>", unsafe_allow_html=True)
                        
                        with col2:
                            # Use inline style with ID and data attribute for maximum targeting
                            # Use CSS variable and direct computed style override approach
                            agent_name_id = f"agent-name-{bot['id']}"
                            st.markdown(f'<div id="{agent_name_id}" class="agent-name-bright" data-agent-name="true" style="--agent-name-color: #ffffff; color: var(--agent-name-color) !important;"><strong style="color: var(--agent-name-color) !important;">{bot["name"]}</strong></div>', unsafe_allow_html=True)
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
                            
                            client = openai.OpenAI(api_key=get_openai_api_key())
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
                                
                                # Play sound for story rendered
                                play_sound('story_rendered')
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
                                    
                                    # Clear Q&A history for the new mission
                                    st.session_state.story_qa_history = []
                                    st.session_state.story_question_example = ""
                                    
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
                    # Display story title and author
                    if st.session_state.mission_story_title:
                        st.markdown(f"""
                        <div class="story-title" style='text-align: center; font-size: 1.5rem; font-weight: bold; color: var(--primary-color); margin-bottom: 10px;'>
                            {st.session_state.mission_story_title}
                        </div>
                        """, unsafe_allow_html=True)
                        # Display author name if available
                        if st.session_state.get('user_creative_name'):
                            st.markdown(f"""
                            <div class="story-author" style='text-align: center; font-size: 1rem; color: var(--text-secondary); margin-bottom: 15px; font-style: italic;'>
                                by {st.session_state.user_creative_name}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
                with col_pdf:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    # Generate PDF using reportlab
                    # Use the module-level imports (they should work if reportlab is installed)
                    if REPORTLAB_AVAILABLE:
                        try:
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
                                story.append(Spacer(1, 0.2*inch))
                                
                                # Add author if available
                                if st.session_state.get('user_creative_name'):
                                    author_style = ParagraphStyle(
                                        'Author',
                                        parent=getSampleStyleSheet()['Normal'],
                                        fontSize=12,
                                        textColor='#475569',
                                        spaceAfter=20,
                                        alignment=TA_CENTER,
                                        fontName='Helvetica-Oblique'
                                    )
                                    story.append(Paragraph(f"by {st.session_state.user_creative_name}", author_style))
                                    story.append(Spacer(1, 0.2*inch))
                                else:
                                    story.append(Spacer(1, 0.2*inch))
                            
                            # Add mission if available
                            if st.session_state.team_mission:
                                mission_style = ParagraphStyle(
                                    'Mission',
                                    parent=getSampleStyleSheet()['Heading2'],
                                    fontSize=14,
                                    textColor='#2563eb',
                                    spaceAfter=15,
                                    spaceBefore=10
                                )
                                story.append(Paragraph("🎯 The Mission", mission_style))
                                story.append(Paragraph(st.session_state.team_mission.replace('\n', '<br/>'), body_style))
                                story.append(Spacer(1, 0.3*inch))
                            
                            # Add agent descriptions section before story
                            if st.session_state.created_bots:
                                story.append(Spacer(1, 0.3*inch))
                                
                                # Agent descriptions section title
                                agents_title_style = ParagraphStyle(
                                    'AgentsTitle',
                                    parent=getSampleStyleSheet()['Heading2'],
                                    fontSize=14,
                                    textColor='#2563eb',
                                    spaceAfter=15,
                                    spaceBefore=10
                                )
                                story.append(Paragraph("🤖 The Team", agents_title_style))
                                story.append(Spacer(1, 0.15*inch))
                                
                                # Agent description style
                                agent_name_style = ParagraphStyle(
                                    'AgentName',
                                    parent=getSampleStyleSheet()['Normal'],
                                    fontSize=11,
                                    textColor='#1e40af',
                                    spaceAfter=5,
                                    leftIndent=0,
                                    fontName='Helvetica-Bold'
                                )
                                
                                agent_desc_style = ParagraphStyle(
                                    'AgentDesc',
                                    parent=getSampleStyleSheet()['Normal'],
                                    fontSize=10,
                                    textColor='#475569',
                                    spaceAfter=10,
                                    leftIndent=15
                                )
                                
                                # Add each agent's description
                                for bot in st.session_state.created_bots:
                                    agent_name = bot.get('name', 'Agent')
                                    agent_number = bot.get('number', '')
                                    agent_desc = clean_agent_description(bot.get('description', ''))
                                    agent_character = clean_agent_description(bot.get('character', ''))
                                    
                                    # Agent name with number
                                    name_text = f"Agent #{agent_number}: {agent_name}" if agent_number else agent_name
                                    story.append(Paragraph(name_text.replace('\n', '<br/>'), agent_name_style))
                                    
                                    # Agent description
                                    if agent_desc:
                                        story.append(Paragraph(agent_desc.replace('\n', '<br/>'), agent_desc_style))
                                    
                                    # Agent character profile
                                    if agent_character:
                                        character_text = f"<i>{agent_character}</i>"
                                        story.append(Paragraph(character_text.replace('\n', '<br/>'), agent_desc_style))
                                    
                                    story.append(Spacer(1, 0.1*inch))
                                
                                story.append(Spacer(1, 0.3*inch))
                            
                            # Add story section title
                            story_title_style = ParagraphStyle(
                                'StoryTitle',
                                parent=getSampleStyleSheet()['Heading2'],
                                fontSize=14,
                                textColor='#2563eb',
                                spaceAfter=15,
                                spaceBefore=10
                            )
                            story.append(Paragraph("📚 The Story", story_title_style))
                            story.append(Spacer(1, 0.15*inch))
                            
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
                            st.error(f"📄 PDF generation error: {str(e)}")
                            st.info("📄 PDF generation requires the 'reportlab' library. Please ensure 'reportlab>=4.0.0' is in requirements.txt and the app has been redeployed on Streamlit Cloud.")
                    else:
                        st.error("📄 PDF generation is not available.")
                        st.info("📄 The 'reportlab' library is not installed. Please ensure 'reportlab>=4.0.0' is in requirements.txt and the app has been redeployed on Streamlit Cloud.")
                
                # Display mission elaboration above the story
                if st.session_state.team_mission:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🎯 The Mission")
                    with st.container():
                        st.markdown(f"""
                        <div class="mission-elaboration" style='padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2563eb; background-color: var(--background-color);'>
                            <div style='padding-left: 10px; font-size: 1.05rem; line-height: 1.6; color: var(--text-color);'>
                                {st.session_state.team_mission}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display story content - use Streamlit container styling like agent descriptions
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📚 The Story")
                # Convert story text to HTML - handle both single and double newlines
                story_text = st.session_state.mission_story
                # Replace double newlines with paragraph breaks
                story_paragraphs = story_text.split('\n\n')
                story_html_parts = []
                for para in story_paragraphs:
                    if para.strip():
                        # Replace single newlines within paragraphs with line breaks
                        para = para.replace('\n', '<br>')
                        # Wrap each paragraph in a div
                        story_html_parts.append(f'{para}<br><br>')
                story_html = ''.join(story_html_parts).rstrip('<br><br>')
                
                # Use Streamlit container styling (same as agent descriptions) - adapts to light/dark mode
                with st.container():
                    st.markdown(f"""
                    <div class="story-content" style='padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6b46c1;'>
                        <div style='padding-left: 10px;'>
                            {story_html}
                        </div>
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
                                client = openai.OpenAI(api_key=get_openai_api_key())
                                # Get user's name for personalization
                                user_name = st.session_state.get('user_creative_name', 'friend')
                                
                                answer_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": f"You are a creative storyteller and teacher explaining children's stories to kids aged 5-10. Answer questions in a simple, friendly way using very simple words. IMPORTANT: You are NOT limited to just the story content. Be creative and build upon the story with additional details, background, character motivations, and fun nuances that enhance the story. Expand on the story world while staying consistent with what was told. Keep answers engaging and imaginative (2-4 sentences). Make it fun and easy to understand. Address the reader by their name '{user_name}' to make it personal and friendly."},
                                        {"role": "user", "content": f"Story Title: {st.session_state.mission_story_title}\n\nStory:\n{st.session_state.mission_story}\n\nQuestion from {user_name}: {user_question}\n\nAnswer this question creatively for {user_name}, building upon and expanding the story with additional details and nuances. You can add background information, character motivations, fun facts, or imaginative details that enhance the story world. Address {user_name} by name to make it personal. Keep it simple and child-friendly (2-4 short sentences)."}
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
                                
                                # Play sound for answer generated
                                play_sound('answer_generated')
                                
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
                
                # Display Q&A history - use Streamlit container styling like agent descriptions
                if st.session_state.story_qa_history:
                    st.markdown("<br>", unsafe_allow_html=True)
                    for idx, qa in enumerate(st.session_state.story_qa_history):
                        with st.container():
                            st.markdown(f"""
                            <div style='padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6b46c1;'>
                                <div style='font-weight: bold; margin-bottom: 8px;'>
                                    ❓ Question: {qa['question']}
                                </div>
                                <div style='padding-left: 10px;'>
                                    💡 {qa['answer']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
