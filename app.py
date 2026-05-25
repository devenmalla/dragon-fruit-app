import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import base64

# ==========================
# BUILD FAVICON AS PIL IMAGE
# Streamlit accepts a PIL Image for page_icon —
# this is the only reliable way to use a custom icon.
# ==========================

def _make_dragon_fruit_icon(size=128):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    tips = [
        [(cx, cy-52), (cx-12, cy-38), (cx+12, cy-38)],
        [(cx-50, cy-26), (cx-36, cy-18), (cx-34, cy-2)],
        [(cx+50, cy-26), (cx+36, cy-18), (cx+34, cy-2)],
        [(cx-54, cy+10), (cx-40, cy+2), (cx-36, cy+18)],
        [(cx+54, cy+10), (cx+40, cy+2), (cx+36, cy+18)],
        [(cx-44, cy+38), (cx-30, cy+28), (cx-24, cy+44)],
        [(cx+44, cy+38), (cx+30, cy+28), (cx+24, cy+44)],
        [(cx, cy+56), (cx-12, cy+42), (cx+12, cy+42)],
    ]
    scale_colors = [
        "#4a9e5c","#3d8a4e","#5ab56c","#3d8a4e",
        "#5ab56c","#4a9e5c","#5ab56c","#4a9e5c",
    ]
    for tip, color in zip(tips, scale_colors):
        draw.polygon(tip, fill=color)
    draw.ellipse([cx-38, cy-40, cx+38, cy+40], fill="#e8547a")
    draw.ellipse([cx-28, cy-34, cx-8, cy-12], fill="#f07a96")
    for sx, sy in [
        (cx-6,cy-10),(cx+8,cy-4),(cx-4,cy+8),
        (cx+8,cy+4),(cx-8,cy+6),(cx+4,cy+16),(cx-2,cy+20),
    ]:
        draw.ellipse([sx-2,sy-3,sx+2,sy+3], fill="#1a0a0a")
    return img

_favicon = _make_dragon_fruit_icon()

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Dragon Fruit Disease Detection",
    page_icon=_favicon,
    layout="centered"
)

# ==========================
# DRAGON FRUIT SVG ICON
# A hand-drawn SVG of a dragon fruit (pitaya) — pink oval body,
# green scaly tips, white flesh cross-section hint.
# Used as both the favicon injection and the hero illustration.
# ==========================

DRAGON_FRUIT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <!-- Body: pink oval -->
  <ellipse cx="50" cy="54" rx="32" ry="36" fill="#e8547a"/>
  <!-- Skin highlight -->
  <ellipse cx="40" cy="40" rx="10" ry="14" fill="#f07a96" opacity="0.5"/>
  <!-- Scales / leaf tips — arranged around the oval -->
  <!-- Top -->
  <path d="M50 20 C44 8 36 4 34 12 C38 16 44 18 50 20Z" fill="#4a9e5c"/>
  <path d="M50 20 C56 8 64 4 66 12 C62 16 56 18 50 20Z" fill="#5ab56c"/>
  <!-- Upper left -->
  <path d="M22 38 C10 30 6 20 14 18 C18 24 20 32 22 38Z" fill="#4a9e5c"/>
  <!-- Upper right -->
  <path d="M78 38 C90 30 94 20 86 18 C82 24 80 32 78 38Z" fill="#5ab56c"/>
  <!-- Mid left -->
  <path d="M18 60 C4 56 2 44 10 44 C14 50 16 56 18 60Z" fill="#4a9e5c"/>
  <!-- Mid right -->
  <path d="M82 60 C96 56 98 44 90 44 C86 50 84 56 82 60Z" fill="#5ab56c"/>
  <!-- Lower left -->
  <path d="M26 80 C14 82 8 72 16 68 C20 72 24 76 26 80Z" fill="#4a9e5c"/>
  <!-- Lower right -->
  <path d="M74 80 C86 82 92 72 84 68 C80 72 76 76 74 80Z" fill="#5ab56c"/>
  <!-- Bottom tip -->
  <path d="M50 90 C44 100 36 102 36 94 C42 90 46 90 50 90Z" fill="#4a9e5c"/>
  <path d="M50 90 C56 100 64 102 64 94 C58 90 54 90 50 90Z" fill="#5ab56c"/>
  <!-- Cross-section hint: white oval flesh -->
  <ellipse cx="50" cy="54" rx="20" ry="23" fill="#fff8f5" opacity="0.15"/>
  <!-- Seeds: tiny dark dots -->
  <circle cx="46" cy="48" r="1.2" fill="#2a1a1a" opacity="0.5"/>
  <circle cx="55" cy="52" r="1.2" fill="#2a1a1a" opacity="0.5"/>
  <circle cx="48" cy="60" r="1.2" fill="#2a1a1a" opacity="0.5"/>
  <circle cx="54" cy="44" r="1.0" fill="#2a1a1a" opacity="0.5"/>
  <circle cx="44" cy="58" r="1.0" fill="#2a1a1a" opacity="0.5"/>
  <circle cx="57" cy="63" r="1.0" fill="#2a1a1a" opacity="0.5"/>
</svg>"""

# ==========================
# CUSTOM CSS + FAVICON
# ==========================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Outfit:wght@300;400;500;600&display=swap');

:root {{
    --bg:       #fdf6ee;
    --surface:  #fff9f2;
    --border:   #e8d9c5;
    --accent:   #c0392b;
    --green:    #2d6a4f;
    --yellow:   #e9b44c;
    --pink:     #e8547a;
    --text:     #1c1410;
    --muted:    #8a7060;
    --card:     #ffffff;
    --shadow:   0 2px 24px rgba(60,30,10,0.08);
}}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container {{
    background-color: var(--bg) !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: none !important;
}}

.main .block-container {{
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 680px !important;
}}

*, *::before, *::after {{
    font-family: 'Outfit', sans-serif;
    box-sizing: border-box;
}}

.hero-wrap {{
    text-align: center;
    padding: 3rem 0 2.2rem;
}}

.hero-icon-svg {{
    width: 80px;
    height: 80px;
    margin: 0 auto 1.4rem;
    display: block;
}}

.hero-title {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
    margin: 0 0 1rem !important;
    letter-spacing: -0.01em;
}}

.hero-title em {{
    color: var(--green);
    font-style: italic;
}}

.hero-sub {{
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    line-height: 1.65;
    margin: 0 auto;
    text-align: center;
}}

.rule {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0;
}}
.rule::before, .rule::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}}
.rule-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--yellow);
    flex-shrink: 0;
}}

.upload-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    display: block;
}}

[data-testid="stFileUploader"] {{
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 20px !important;
    padding: 0.25rem !important;
    box-shadow: var(--shadow) !important;
    transition: border-color 0.2s !important;
}}

[data-testid="stFileUploader"]:focus-within,
[data-testid="stFileUploader"]:hover {{
    border-color: var(--green) !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: transparent !important;
    border: none !important;
    padding: 1.5rem !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: var(--muted) !important;
    font-family: 'Outfit', sans-serif !important;
}}

[data-testid="stFileUploader"] label {{
    display: none !important;
}}

[data-testid="stImage"] img {{
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    width: 100% !important;
}}

[data-testid="stImage"] > div > p {{
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    text-align: center !important;
    margin-top: 0.4rem !important;
}}

[data-testid="stButton"] > button {{
    width: 100% !important;
    background: var(--green) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.5rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    margin-top: 0.8rem !important;
    transition: filter 0.15s, transform 0.1s !important;
    box-shadow: 0 4px 16px rgba(45,106,79,0.25) !important;
}}

[data-testid="stButton"] > button:hover {{
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
}}

[data-testid="stButton"] > button:active {{
    transform: translateY(0) !important;
}}

[data-testid="stAlert"] {{
    display: none !important;
}}

.result-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem 1.4rem;
    margin-top: 1rem;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    gap: 1.4rem;
}}

.result-icon-wrap {{
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    flex-shrink: 0;
}}

.result-icon-wrap.bad  {{ background: #fff0ef; }}
.result-icon-wrap.good {{ background: #edf7f2; }}

.result-body {{ flex: 1; min-width: 0; }}

.result-tag {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.2rem;
}}

.result-name {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.7rem;
}}
.result-name.bad  {{ color: var(--accent); }}
.result-name.good {{ color: var(--green);  }}

.bar-track {{
    background: #ede8e2;
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
}}

.bar-fill {{
    height: 100%;
    border-radius: 99px;
}}

.bar-fill.bad  {{ background: var(--accent); }}
.bar-fill.good {{ background: var(--green);  }}

.conf-text {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.4rem;
}}

[data-testid="stSpinner"] p {{
    color: var(--muted) !important;
    font-family: 'Outfit', sans-serif !important;
}}

.footer {{
    text-align: center;
    padding: 3rem 0 0.5rem;
    font-size: 0.72rem;
    color: #c8b8a8;
    letter-spacing: 0.08em;
}}
</style>

""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mobilenetv2_model.h5")
    return model

model = load_model()

# ==========================
# CLASS NAMES
# IMPORTANT:
# Order must match training
# ==========================

class_names = [
    'Bad fruit',
    'Bad leaf',
    'Good fruit',
    'Good leaf'
]

# ==========================
# HERO
# ==========================

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-icon-svg">{DRAGON_FRUIT_SVG}</div>
    <h1 class="hero-title">Dragon Fruit<br><em>Disease Detection</em></h1>
    <p class="hero-sub">Upload a photo of a fruit or leaf.<br>The model will classify its health instantly.</p>
</div>
<div class="rule"><div class="rule-dot"></div></div>
""", unsafe_allow_html=True)

# ==========================
# IMAGE UPLOAD
# ==========================

st.markdown('<span class="upload-label">Upload image</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # ==========================
    # PREPROCESS IMAGE
    # ==========================

    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ==========================
    # PREDICTION
    # ==========================

    if st.button("Analyse Image"):

        with st.spinner("Running analysis..."):
            prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = class_names[predicted_class]

        is_bad = result.startswith("Bad")
        cls = "bad" if is_bad else "good"
        icon = "⚠️" if is_bad else "✅"

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-icon-wrap {cls}">{icon}</div>
            <div class="result-body">
                <div class="result-tag">Result</div>
                <div class="result-name {cls}">{result}</div>
                <div class="bar-track">
                    <div class="bar-fill {cls}" style="width:{confidence:.1f}%"></div>
                </div>
                <div class="conf-text">Confidence: {confidence:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================

st.markdown('<div class="footer">Dragon Fruit Disease Detection</div>', unsafe_allow_html=True)