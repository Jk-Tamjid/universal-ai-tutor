import os
import re
import asyncio
import tempfile
import edge_tts
import pypdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
# 2. Advanced Premium CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Gradient Background */
    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b, #0f172a 60%, #020617 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Glassmorphism Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Gradient Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
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

    /* Custom Input and Audio Control */
    div[data-testid="stAudioInput"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    div[data-testid="stAudioInput"] button {
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        background-color: #38bdf8 !important;
        color: #0f172a !important;
    }

    /* Custom File Uploader Styling */
    div[data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px stroke rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# 3. Sidebar Configuration & API Setup (Key Input Removed)
# ---------------------------------------------------------
final_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if not final_api_key:
    st.error("⚠️ Gemini API Key পাওয়া যায়নি! দয়া করে `.streamlit/secrets.toml` বা Environment Variable-এ `GEMINI_API_KEY` সেট করুন।")
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
# Accurate System Instruction & Formatting
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are an elite, highly precise AI Academic Visual Tutor. Your primary directive is 100% FACTUAL AND MATHEMATICAL ACCURACY.\n\n"
    "ACCURACY & REASONING RULES:\n"
    "- Double-check all mathematical formulas, calculations, and facts step-by-step before producing final answers.\n"
    "- Never make assumptions or guesses. If details are ambiguous, clearly state assumptions or ask for clarification.\n"
    "- Provide clear, concise, logically sound academic explanations without errors.\n\n"
    "CRITICAL LANGUAGE RULES:\n"
    f"- Preferred Language Selection: {target_language}.\n"
    "- If set to 'Auto-detect / Universal Match', ALWAYS respond in the exact language the user used to ask the question.\n"
    "- If requested in a specific language, write 100% of your explanation in that language.\n"
    "- Graph titles, axis labels, and comments must match the output language.\n\n"
    "GRAPH & VIDEO GENERATION RULES:\n"
    "- DO NOT generate graphs/plots/videos for casual chat, greetings, or non-visual questions.\n"
    "- ONLY generate a visual plot if explicitly requested to draw, plot, visualize, or graph.\n"
    "- Output static plot code inside ```python_plot ... ```.\n"
    "- ONLY generate an animated video/simulation if explicitly requested for motion, animation, or video.\n"
    "- Output animation code inside ```python_video ... ```.\n\n"
    "PYTHON ANIMATION STRUCTURE RULE:\n"
    "All variables modified inside update function MUST be defined globally first.\n"
    "Example:\n"
    "```python_video\n"
    "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
    "fig.patch.set_facecolor('#0f172a')\n"
    "ax.set_facecolor('#1e293b')\n"
    "ax.tick_params(colors='#94a3b8')\n"
    "ax.xaxis.label.set_color('#94a3b8')\n"
    "ax.yaxis.label.set_color('#94a3b8')\n"
    "ax.title.set_color('#38bdf8')\n"
    "ax.grid(True, color='#334155', linestyle='--')\n\n"
    "x = np.linspace(0, 2*np.pi, 100)\n"
    "line, = ax.plot(x, np.sin(x), color='#38bdf8', lw=2)\n\n"
    "def update(frame):\n"
    "    line.set_ydata(np.sin(x + frame / 10.0))\n"
    "    return line,\n\n"
    "anim = animation.FuncAnimation(fig, update, frames=50, interval=50, blit=True)\n"
    "anim.save('output_animation.gif', writer='pillow', fps=20)\n"
    "```"
)

# ---------------------------------------------------------
# 4. Universal Multilingual Voice Engine
# ---------------------------------------------------------
def detect_language_script_and_get_voice(text: str) -> str:
    if re.search(r'[\u0980-\u09FF]', text):
        return "bn-BD-NabanitaNeural"
    elif re.search(r'[\u0600-\u06FF]', text):
        return "ur-PK-UzmaNeural"
    elif re.search(r'[\u0900-\u097F]', text):
        return "hi-IN-SwaraNeural"
    elif re.search(r'\b(el|la|los|las|un|una|es|por|para|con|como|gracias|hola)\b', text, re.IGNORECASE):
        return "es-ES-ElviraNeural"
    elif re.search(r'\b(le|la|les|un|une|est|dans|pour|avec|bonjour|merci)\b', text, re.IGNORECASE):
        return "fr-FR-DeniseNeural"
    elif re.search(r'\b(der|die|das|und|ist|nicht|mit|fuer|hallo|danke)\b', text, re.IGNORECASE):
        return "de-DE-KatjaNeural"
    else:
        return "en-US-AvaNeural"

def sanitize_text_for_tts(text: str, voice_code: str) -> str:
    text = re.sub(r'```python_video.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```python_plot.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\$\$|\$', '', text)
    text = re.sub(r'[#\*_~`>\+\=\-\\/]', ' ', text)

    if "bn-BD" in voice_code:
        text = re.sub(r'\\sin', 'সাইন', text)
        text = re.sub(r'\\cos', 'কস', text)
        text = re.sub(r'\\omega', 'ওমেগা', text)
        text = re.sub(r'\\delta', 'ডেল্টা', text)
        text = re.sub(r'[a-zA-Z]+', '', text)
        
    return text.strip()

async def generate_neural_audio(text: str) -> str:
    voice = detect_language_script_and_get_voice(text)
    cleaned_text = sanitize_text_for_tts(text[:700], voice)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    communicate = edge_tts.Communicate(cleaned_text, voice)
    await communicate.save(temp_file.name)
    return temp_file.name

# ---------------------------------------------------------
# 5. Dynamic Code Execution Engine (Plots & Videos)
# ---------------------------------------------------------
def execute_and_render_plot(code_str: str):
    try:
        local_scope = {"plt": plt, "np": np}
        exec(code_str, {}, local_scope)
        if "fig" in local_scope:
            st.pyplot(local_scope["fig"])
        else:
            st.pyplot(plt.gcf())
        plt.close('all')
    except Exception as e:
        st.warning(f"Could not render plot: {str(e)}")

def execute_and_render_video(code_str: str):
    try:
        if os.path.exists('output_animation.gif'):
            os.remove('output_animation.gif')

        execution_namespace = {
            "plt": plt,
            "np": np,
            "animation": animation
        }

        exec(code_str, execution_namespace, execution_namespace)

        if os.path.exists('output_animation.gif'):
            st.image('output_animation.gif', caption="🎬 Video Concept Animation")
        else:
            st.warning("Animation could not be saved to disk.")
        plt.close('all')
    except Exception as e:
        st.warning(f"Could not render video animation: {str(e)}")

# ---------------------------------------------------------
# 6. Unified Rendering Helper
# ---------------------------------------------------------
def render_assistant_message(content: str, audio_path: str = None):
    clean_display_text = re.sub(r'```python_video.*?```', '', content, flags=re.DOTALL)
    clean_display_text = re.sub(r'```python_plot.*?```', '', clean_display_text, flags=re.DOTALL)
    
    st.markdown(clean_display_text)

    # Static Plot Rendering
    plot_matches = re.findall(r'```python_plot(.*?)```', content, re.DOTALL)
    for plot_code in plot_matches:
        execute_and_render_plot(plot_code.strip())

    # Video Animation Rendering
    video_matches = re.findall(r'```python_video(.*?)```', content, re.DOTALL)
    for video_code in video_matches:
        execute_and_render_video(video_code.strip())

    # Voice Audio Output
    if audio_path:
        st.audio(audio_path, format="audio/mp3")

# ---------------------------------------------------------
# 7. Main UI Layout & History
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Universal AI Visual Tutor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MathGPT-style Accurate Explanations with Voice & Animations</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message["content"], message.get("audio"))
        else:
            st.markdown(message["content"])

# ---------------------------------------------------------
# 8. Chat Inputs
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload Book Image or PDF (Optional):", type=["jpg", "jpeg", "png", "pdf"])

col_input, col_mic = st.columns([8.5, 1.5])
with col_input:
    user_input = st.chat_input("Ask anything in any language (e.g. English, Bangla, Urdu, Spanish, French)...")

with col_mic:
    mic_audio = st.audio_input("🎙️")

# ---------------------------------------------------------
# 9. Request Handler & Execution Engine
# ---------------------------------------------------------
if user_input or uploaded_file or mic_audio:
    contents = []

    if mic_audio:
        audio_bytes = mic_audio.read()
        contents.append(
            types.Part.from_bytes(
                data=audio_bytes,
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
        contents.append("Please respond to my voice input in the exact language I spoke. Only draw a graph or generate a video if requested.")

    with st.chat_message("user"):
        if user_input:
            st.markdown(user_input)
        if mic_audio:
            st.audio(mic_audio)

    st.session_state.messages.append({"role": "user", "content": user_input or "Voice Input"})

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking & verifying accurate solution..."):
            # Highest accuracy model prioritized first
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
                            temperature=0.0  # Zero temperature forces deterministic & factual answers
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

            # Generate TTS Audio before rendering
            audio_file_path = None
            if enable_audio and response_text:
                try:
                    audio_file_path = asyncio.run(generate_neural_audio(response_text))
                except Exception:
                    pass

            # Render message content
            render_assistant_message(response_text, audio_file_path)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audio": audio_file_path
            })