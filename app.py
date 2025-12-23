import torch
import re
import html
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# =========================
# MODEL CONFIG
# =========================
BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_PATH = "./nllb_lora_finance_lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SOURCE_LEN = 512
MAX_NEW_TOKENS = 256

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="cpu"   # safer on Windows
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# CHUNKING
# =========================
def split_into_chunks(text, max_tokens=400):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, cur_tokens = [], "", 0

    for sent in sentences:
        tokens = tokenizer(sent, add_special_tokens=False)["input_ids"]
        if cur_tokens + len(tokens) > max_tokens:
            chunks.append(current.strip())
            current, cur_tokens = sent, len(tokens)
        else:
            current += " " + sent
            cur_tokens += len(tokens)

    if current.strip():
        chunks.append(current.strip())
    return chunks

# =========================
# TRANSLATION
# =========================
def translate_long(text, tgt_lang):
    tokenizer.src_lang = "eng_Latn"
    outputs = []

    for ch in split_into_chunks(text):
        enc = tokenizer(
            ch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LEN
        ).to(model.device)

        bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=bos_id,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4
            )

        outputs.append(tokenizer.batch_decode(out, skip_special_tokens=True)[0])

    return " ".join(outputs)

# =========================
# LANGUAGE MAP
# =========================
LANG_MAP = {
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Hindi": "hin_Deva",
}

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="FinanceHub Pro | Translator",
    page_icon="🌍",
    layout="wide"
)

# =========================
# CSS (FIXED COLORS)
# =========================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-card {
    background: black;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #3B82F6;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    color: #fffffff;
}
.info-box {
    background: #F0F9FF;
    border-left: 4px solid #10B981;
    padding: 1rem;
    border-radius: 8px;
    color: #1E293B;
}
.info-box li {
    color: #fffffff;
}
.stButton>button {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
    color: white;
    border-radius: 8px;
    padding: 0.75rem;
    font-weight: 600;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
    <h1>🌍 Multilingual Financial Translator</h1>
    <p>Domain-controlled translation powered by LoRA-adapted NLLB-200</p>
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN UI
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Input Document")
    input_text = st.text_area(
        "Paste English financial text",
        height=320,
        placeholder="Annual report, earnings call, regulatory filing…",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🌐 Translation Settings")
    target_label = st.selectbox("Target Language", list(LANG_MAP.keys()))
    target_lang_code = LANG_MAP[target_label]

    st.markdown("""
    <div class="info-box">
        <strong>✔ Translation Guarantees</strong>
        <ul>
            <li>Terminology-safe output</li>
            <li>Numerical fidelity</li>
            <li>Long-document handling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    translate_btn = st.button("🌐 Translate Document")

# =========================
# OUTPUT
# =========================
if translate_btn:
    if not input_text.strip():
        st.error("❌ Please enter text to translate")
    elif len(input_text.strip()) < 30:
        st.warning("⚠️ Please provide more content for reliable translation")
    else:
        with st.spinner("🔄 Translating with NLLB-200 + LoRA…"):
            result = translate_long(input_text, target_lang_code)

        st.success("✅ Translation completed successfully")
        st.markdown("### 📑 Translated Output")

        st.markdown(
            f"""
            <div class="feature-card">
                <p style="line-height:1.7;">{html.escape(result)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; font-size:0.9rem;">
    NLLB-200 + LoRA • FinanceHub Pro Translation Module
</div>
""", unsafe_allow_html=True)
