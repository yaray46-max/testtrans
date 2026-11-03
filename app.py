"""다국어 번역기 - Streamlit + OpenAI."""

import logging
import os

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-6-astra"
MAX_CHARS = 5000

LANGUAGES = {
    "en": {"label": "EN · 영어", "name": "English"},
    "ja": {"label": "JA · 일본어", "name": "Japanese"},
    "zh": {"label": "ZH · 중국어", "name": "Simplified Chinese"},
}

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {target_language}. "
    "Preserve the original meaning, tone, formatting, and line breaks. "
    "Output only the translated text without any explanations."
)


def load_config() -> tuple[str | None, str]:
    """.env에서 API 키와 모델명을 읽어 반환한다."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or None
    model = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    return api_key, model


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def translate(text: str, target_language: str) -> str:
    """text를 target_language로 번역해 반환한다."""
    api_key, model = load_config()
    if not api_key:
        raise RuntimeError("missing_api_key")
    response = get_client(api_key).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(target_language=target_language)},
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def error_message(exc: Exception) -> str:
    """예외를 사용자용 한국어 메시지로 변환한다."""
    if isinstance(exc, RuntimeError) and str(exc) == "missing_api_key":
        return "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
    if isinstance(exc, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다."
    if isinstance(exc, openai.RateLimitError):
        return "요청이 많습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.NotFoundError):
        return "모델을 찾을 수 없습니다. .env의 OPENAI_MODEL을 확인하세요."
    if isinstance(exc, openai.APIConnectionError):
        return "네트워크 연결을 확인해 주세요."
    return "번역 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


def set_error(level: str, message: str) -> None:
    """오류를 저장하고, 현재 입력과 맞지 않는 이전 결과는 지운다."""
    st.session_state["error"] = (level, message)
    st.session_state.pop("result", None)
    st.session_state.pop("result_lang", None)


def run_translation(text: str, code: str) -> None:
    """입력 검증 후 번역하고 결과를 session_state에 저장한다."""
    st.session_state.pop("error", None)
    if not text.strip():
        set_error("warning", "번역할 내용을 입력해 주세요.")
        return
    if len(text) > MAX_CHARS:
        set_error("warning", f"최대 {MAX_CHARS:,}자까지 번역할 수 있습니다.")
        return
    try:
        with st.spinner("번역 중..."):
            result = translate(text, LANGUAGES[code]["name"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Translation failed")
        set_error("error", error_message(exc))
        return
    st.session_state["result"] = result
    st.session_state["result_lang"] = code


CUSTOM_CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 3rem; }

[data-testid="stMarkdownContainer"] p.app-title { font-size: 2rem; font-weight: 700; margin: 0 0 .25rem; letter-spacing: -0.02em; }
.app-subtitle { opacity: .7; margin: 0 0 1.25rem; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    background: rgba(79, 70, 229, 0.035);
}

.stTextArea textarea { border-radius: 10px; font-size: 1rem; line-height: 1.6; }

.char-count { text-align: right; font-size: .85rem; opacity: .65; margin-top: -.25rem; }
.char-count.over { color: #dc2626; opacity: 1; font-weight: 600; }

.stButton > button {
    border-radius: 10px;
    padding: .6rem .5rem;
    font-weight: 600;
    transition: background-color .15s ease, box-shadow .15s ease, transform .15s ease;
}
.stButton > button:hover {
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
    transform: translateY(-1px);
}

[data-testid="stMarkdownContainer"] p.result-title { font-size: 1.1rem; font-weight: 600; margin: 0 0 .5rem; }
[data-testid="stCode"] pre { border-radius: 10px; font-size: 1rem; line-height: 1.6; }
</style>
"""


def clear_all() -> None:
    """입력과 결과를 모두 초기화한다."""
    st.session_state["input_text"] = ""
    for key in ("result", "result_lang", "error"):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<p class="app-title">🌐 다국어 번역기</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">글을 입력하고 번역할 언어 버튼을 누르세요.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        text = st.text_area(
            "번역할 내용",
            placeholder="번역할 내용을 입력하세요.",
            key="input_text",
            height=200,
            label_visibility="collapsed",
        )
        over = " over" if len(text) > MAX_CHARS else ""
        st.markdown(
            f'<div class="char-count{over}">{len(text):,} / {MAX_CHARS:,}자</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3)
        for col, (code, lang) in zip(cols, LANGUAGES.items()):
            if col.button(lang["label"], key=f"btn_{code}", type="primary", use_container_width=True):
                run_translation(text, code)

        st.button("🧹 지우기", key="btn_clear", on_click=clear_all, use_container_width=True)

    if "error" in st.session_state:
        level, msg = st.session_state["error"]
        getattr(st, level)(msg)

    if "result" in st.session_state:
        with st.container(border=True):
            label = LANGUAGES[st.session_state["result_lang"]]["label"]
            st.markdown(f'<p class="result-title">{label} 번역 결과</p>', unsafe_allow_html=True)
            st.code(st.session_state["result"], language=None, wrap_lines=True)


if __name__ == "__main__":
    main()
