import os
import re
import asyncio
import tempfile
import edge_tts
import pypdf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server compatibility
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Universal Multilingual AI Visual Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Modern Glassmorphism CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Radial Gradient Background */
    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b, #0f172a 60%, #020617 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Gradient Header Typography */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Chat Messages Glass Container */
    div[data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# 3. Sidebar Configuration & API Setup
# ---------------------------------------------------------
final_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if not final_api_key:
    st.error("⚠️ Gemini API Key missing! Please set `GEMINI_API_KEY` in Streamlit Cloud Secrets.")
    st.stop()

client = genai.Client(api_key=final_api_key)

with st.sidebar:
    st.markdown("### ⚙️ Workspace Control")
    st.markdown("---")
    
    target_language = st.selectbox(
        "🌐 Reply Language Mode:",
        ["Auto-detect / Universal Match", "Bangla (বাংলা)", "English", "Urdu (اردو)", "Spanish (Español)", "French (Français)", "German (Deutsch)"]
    )

    enable_audio = st.checkbox("🔊 Voice Assistant Response", value=True)

# ---------------------------------------------------------
# 4. Strict System Instruction & Format Constraints
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are an elite AI Academic Visual Tutor focused on 100% FACTUAL AND MATHEMATICAL ACCURACY.\n\n"
    "CRITICAL RULE FOR VISUALS & ANIMATIONS:\n"
    "- NEVER state or claim that you cannot generate images, plots, or animations.\n"
    "- ALWAYS generate executable Python code when asked to draw, plot, visualize, or animate.\n"
    "- For static plots, output ONLY Python code inside ```python_plot ... ```.\n"
    "- For animations/videos, output ONLY Python code inside ```python_video ... ```.\n\n"
    "ACCURACY & REASONING RULES:\n"
    "- Double-check all formulas and calculations step-by-step before answering.\n"
    "- Never make assumptions. State clear step-by-step logic.\n\n"
    "CRITICAL LANGUAGE RULES:\n"
    f"- Target Language: {target_language}.\n"
    "- If set to 'Auto-detect / Universal Match', respond in the user's exact input language.\n\n"
    "ANIMATION CODE STRUCTURE REQUIREMENT:\n"
    "You MUST import PillowWriter and save the animation as 'output_animation.gif':\n"
    "```python_video\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.animation as animation\n"
    "from matplotlib.animation import PillowWriter\n\n"
    "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
    "fig.patch.set_facecolor('#0f172a')\n"
    "ax.set_facecolor('#1e293b')\n"
    "ax.tick_params(colors='#94a3b8')\n"
    "ax.grid(True, color='#334155', linestyle='--')\n\n"
    "x = np.linspace(0, 2*np.pi, 100)\n"
    "line, = ax.plot(x, np.sin(x), color='#38bdf8', lw=2)\n\n"
    "def update(frame):\n"
    "    line.set_ydata(np.sin(x + frame / 10.0))\n"
    "    return line,\n\n"
    "anim = animation.FuncAnimation(fig, update, frames=40, interval=50, blit=True)\n"
    "writer = PillowWriter(fps=20)\n"
    "anim.save('output_animation.gif', writer=writer)\n"
    "```"
)

# ---------------------------------------------------------
# 5. Multilingual Neural Audio Engine
# ---------------------------------------------------------
def detect_language_script_and_get_voice(text: str) -> str:
    if re.search(r'[\u0980-\u09FF]', text):
        return "bn-BD-NabanitaNeural"
    elif re.search(r'[\u0600-\u06FF]', text):
        return "ur-PK-UzmaNeural"
    elif re.search(r'[\u0900-\u097F]', text):
        return "hi-IN-SwaraNeural"
    else:
        return "en-US-AvaNeural"

def sanitize_text_for_tts(text: str) -> str:
    text = re.sub(r'```python_video.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```python_plot.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'[\$\*\_~`>\+\=\-\\/]', ' ', text)
    return text.strip()

async def generate_neural_audio(text: str) -> str:
    voice = detect_language_script_and_get_voice(text)
    cleaned_text = sanitize_text_for_tts(text[:600])
    
    if not cleaned_text:
        return None
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    communicate = edge_tts.Communicate(cleaned_text, voice)
    await communicate.save(temp_file.name)
    return temp_file.name

# ---------------------------------------------------------
# 6. Dynamic Code Execution Engines (With Auto-Correction)
# ---------------------------------------------------------
def sanitize_python_code(code_str: str) -> str:
    # Auto-correct common Matplotlib method casing mistakes from LLM output
    code_str = re.sub(r'\bset_Ylim\b', 'set_ylim', code_str)
    code_str = re.sub(r'\bset_Xlim\b', 'set_xlim', code_str)
    code_str = re.sub(r'\bset_Xlabel\b', 'set_xlabel', code_str)
    code_str = re.sub(r'\bset_Ylabel\b', 'set_ylabel', code_str)
    code_str = re.sub(r'\bset_Title\b', 'set_title', code_str)
    return code_str

def execute_and_render_plot(code_str: str):
    try:
        code_str = sanitize_python_code(code_str)
        fig = plt.figure(figsize=(6, 3.5), facecolor='#0f172a')
        local_scope = {"plt": plt, "np": np, "fig": fig}
        exec(code_str, {}, local_scope)
        
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            if "fig" in local_scope:
                st.pyplot(local_scope["fig"], use_container_width=False)
            else:
                st.pyplot(plt.gcf(), use_container_width=False)
        plt.close('all')
    except Exception as e:
        st.warning(f"Could not render plot: {str(e)}")

def execute_and_render_video(code_str: str):
    try:
        code_str = sanitize_python_code(code_str)
        output_file = 'output_animation.gif'
        if os.path.exists(output_file):
            os.remove(output_file)

        execution_namespace = {
            "plt": plt,
            "np": np,
            "animation": animation,
            "PillowWriter": PillowWriter
        }

        exec(code_str, execution_namespace, execution_namespace)

        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            if os.path.exists(output_file):
                st.image(output_file, caption="🎬 Visual Concept Animation", use_container_width=False, width=520)
            else:
                st.warning("Animation file could not be generated on disk.")
        plt.close('all')
    except Exception as e:
        st.warning(f"Could not render video animation: {str(e)}")

# ---------------------------------------------------------
# 7. Unified Message Renderer
# ---------------------------------------------------------
def render_assistant_message(content: str, audio_path: str = None):
    clean_text = re.sub(r'```python_video.*?```', '', content, flags=re.DOTALL)
    clean_text = re.sub(r'```python_plot.*?```', '', clean_text, flags=re.DOTALL)
    
    st.markdown(clean_text)

    plot_matches = re.findall(r'```python_plot(.*?)```', content, re.DOTALL)
    for plot_code in plot_matches:
        execute_and_render_plot(plot_code.strip())

    video_matches = re.findall(r'```python_video(.*?)```', content, re.DOTALL)
    for video_code in video_matches:
        execute_and_render_video(video_code.strip())

    if audio_path:
        st.audio(audio_path, format="audio/mp3")

# ---------------------------------------------------------
# 8. Main App UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Universal AI Visual Tutor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurate Explanations with Voice & Animated Concepts</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message["content"], message.get("audio"))
        else:
            st.markdown(message["content"])

# ---------------------------------------------------------
# 9. User Input Handling
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload Image or PDF (Optional):", type=["jpg", "jpeg", "png", "pdf"])

col_input, col_mic = st.columns([8.5, 1.5])
with col_input:
    user_input = st.chat_input("Ask a question in any language...")

with col_mic:
    mic_audio = st.audio_input("🎙️")

# ---------------------------------------------------------
# 10. Execution Engine
# ---------------------------------------------------------
if user_input or uploaded_file or mic_audio:
    contents = []

    if mic_audio:
        contents.append(
            types.Part.from_bytes(
                data=mic_audio.read(),
                mime_type="audio/wav"
            )
        )

    if uploaded_file:
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            contents.append(Image.open(uploaded_file))
        elif uploaded_file.type == "application/pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            pdf_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            contents.append(f"PDF Content:\n{pdf_text[:10000]}")

    if user_input:
        contents.append(user_input)
    elif mic_audio and not user_input:
        contents.append("Respond to my voice input in the language I spoke.")

    with st.chat_message("user"):
        if user_input:
            st.markdown(user_input)
        if mic_audio:
            st.audio(mic_audio)

    st.session_state.messages.append({"role": "user", "content": user_input or "Voice Input"})

    with st.chat_message("assistant"):
        with st.spinner("🤖 Solving step-by-step..."):
            models_to_try = [
                "gemini-3.1-pro-preview",
                "gemini-3.5-flash-lite"
            ]
            response = None
            last_error = None

            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            temperature=0.0
                        )
                    )
                    if response and response.text:
                        break
                except Exception as err:
                    last_error = err
                    continue

            if not response or not response.text:
                st.error(f"API Error: {str(last_error)}")
                st.stop()

            response_text = response.text

            audio_file_path = None
            if enable_audio and response_text:
                try:
                    audio_file_path = asyncio.run(generate_neural_audio(response_text))
                except Exception:
                    pass

            render_assistant_message(response_text, audio_file_path)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audio": audio_file_path
            })
