import streamlit as st
import os
import re
from openai import OpenAI

# ----------------------------
# Initialize OpenAI client
# ----------------------------
# API key should be loaded from environment variable or secrets
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Streamlit App Debugger", page_icon="🛠️", layout="centered")
st.title("🛠️ Streamlit App Debugger & Modifier")

# ----------------------------
# Helper to extract code safely
# ----------------------------
def extract_code(ai_output):
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", ai_output, re.DOTALL)
    code = code_blocks[0].strip() if code_blocks else ai_output.strip()
    # Remove dangling '('
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens > close_parens:
        lines = code.split('\n')
        while lines and lines[-1].count('(') > lines[-1].count(')'):
            lines.pop()
        code = '\n'.join(lines)
    return code

# ----------------------------
# Session state
# ----------------------------
if "app_path" not in st.session_state:
    st.session_state.app_path = ""
if "last_result_message" not in st.session_state:
    st.session_state.last_result_message = ""

# ----------------------------
# Function to apply modification
# ----------------------------
def apply_modification(app_path, intent):
    if not app_path or not os.path.exists(app_path):
        return False, "❌ Invalid app path. Please enter a valid .py file."
    if not intent.strip():
        return False, "❌ Modification intent is empty."

    with open(app_path, "r") as f:
        original_code = f.read()

    try:
        st.info("💡 Sending code + intent to GPT for modification...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert Python and Streamlit code modifier."},
                {"role": "user", "content": f"Current app code:\n\n{original_code}"},
                {"role": "user", "content": f"Modify/debug based on intent:\n\n{intent}"}
            ],
        )

        raw_output = response.choices[0].message.content
        modified_code = extract_code(raw_output)

        if not modified_code or "import streamlit" not in modified_code:
            return False, "⚠️ AI cannot execute the task. Manual intervention required."

        # Backup original
        backup_path = app_path + ".bak"
        with open(backup_path, "w") as backup:
            backup.write(original_code)

        # Write modified code
        with open(app_path, "w") as f:
            f.write(modified_code)

        return True, (
            f"✅ Changes applied! Backup saved as {backup_path}.\n\n"
            f"⚠️ Please manually restart your Streamlit app to check if the changes work:\n"
            f"`streamlit run {app_path}`"
        )

    except Exception as e:
        return False, f"⚠️ Error during modification: {e}\nManual intervention required."

# ----------------------------
# Single prompt (always visible)
# ----------------------------
st.subheader("Provide Modification Request")
st.session_state.app_path = st.text_input(
    "Enter the path of the Streamlit app (.py file):",
    value=st.session_state.app_path
)

current_intent = st.text_area(
    "Enter your modification/debugging request:",
    key="single_prompt"
)

if st.button("Apply Changes"):
    if st.session_state.app_path and current_intent.strip():
        success, message = apply_modification(st.session_state.app_path, current_intent)
        st.session_state.last_result_message = message
        if success:
            st.success(message)
        else:
            st.error(message)

# ----------------------------
# Show result of last execution
# ----------------------------
if st.session_state.last_result_message:
    st.markdown("---")
    st.subheader("Result of Last Modification Request")
    st.markdown(st.session_state.last_result_message)
