import streamlit as st
import logging
from summarizer_service import SummarizerService

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def init_service():
    """初始化摘要服務（使用 cache_resource 避免重複初始化）"""
    return SummarizerService()

summarizer_service = init_service()

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if "summary_output" not in st.session_state:
    st.session_state.summary_output = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "summary_type" not in st.session_state:
    st.session_state.summary_type = None  # "pdf" 或 "url"


def stream_summary(generator, status_container, result_placeholder=None):
    summary_text = ""

    try:
        if isinstance(generator, str):
            status_container.update(label=f"生成摘要中... ({len(generator)} 字元)")
            if result_placeholder is not None:
                result_placeholder.markdown(generator)
            return generator

        for chunk in generator:
            if chunk:
                summary_text += chunk
                status_container.update(label=f"生成摘要中... ({len(summary_text)} 字元)")
                if result_placeholder is not None:
                    result_placeholder.markdown(summary_text)

        return summary_text

    except Exception as e:
        error_msg = f"摘要生成失敗: {str(e)}"
        status_container.update(label="處理失敗", state="error")
        logging.error(f"Stream summary error: {e}")
        return error_msg

st.title("PDF & Web Summarizer")
st.markdown("支援 PDF 文件上傳、網頁 URL 摘要和 YouTube 影片摘要")

st.header("📁 PDF 文件")

uploaded_file = st.file_uploader(
    label="上傳 PDF 檔案",
    type=["pdf"],
    accept_multiple_files=False,
    help="上傳要進行摘要的 PDF 文件",
    key="pdf_uploader"
)

# 檢查是否有新的 PDF 上傳
if uploaded_file != st.session_state.last_uploaded:
    st.session_state.last_uploaded = uploaded_file
    st.session_state.summary_output = None
    st.session_state.summary_type = None

pending_pdf = (
    uploaded_file is not None and
    st.session_state.summary_output is None and
    uploaded_file == st.session_state.last_uploaded and
    not st.session_state.is_processing
)
pending_url = None

st.divider()

st.header("🌐 網頁")

url_input = st.text_input(
    label="請輸入網址",
    placeholder="https://danluu.com/car-safety/ 或 https://www.youtube.com/watch?v=...",
    help="輸入要進行摘要的網頁網址、PDF 連結或 YouTube 影片網址",
    key="url_input"
)

if st.button(
    "開始摘要網頁",
    type="primary",
    key="url_summary_button",
    disabled=st.session_state.is_processing
):
    if url_input.strip():
        st.session_state.summary_output = None
        st.session_state.summary_type = None
        pending_url = url_input.strip()
    else:
        st.warning("請輸入有效的網址")

st.divider()

st.subheader("摘要結果")
summary_result_placeholder = st.empty()
if st.session_state.summary_output:
    summary_result_placeholder.markdown(st.session_state.summary_output)

if pending_pdf:
    st.session_state.is_processing = True
    st.session_state.summary_output = None

    with st.status("正在處理 PDF...", expanded=True) as status:
        def report(msg):
            status.update(label=msg)

        try:
            summary_generator = summarizer_service.summarize_upload_stream(
                uploaded_file,
                on_progress=report,
            )
            summary = stream_summary(summary_generator, status, summary_result_placeholder)

            st.session_state.summary_output = summary
            st.session_state.summary_type = "PDF"
            status.update(label="完成", state="complete")

        except Exception as e:
            error_msg = f"PDF 處理失敗: {str(e)}"
            st.session_state.summary_output = error_msg
            st.session_state.summary_type = "PDF"
            status.update(label="處理失敗", state="error")
            logging.error(f"PDF processing error: {e}")

    st.session_state.is_processing = False

if pending_url:
    st.session_state.is_processing = True

    with st.status("正在分析網頁...", expanded=True) as status:
        try:
            summary_generator = summarizer_service.summarize_stream(pending_url)
            summary = stream_summary(summary_generator, status, summary_result_placeholder)

            st.session_state.summary_output = summary
            st.session_state.summary_type = "網頁"
            status.update(label="完成", state="complete")

        except Exception as e:
            error_msg = f"網頁處理失敗: {str(e)}"
            st.session_state.summary_output = error_msg
            st.session_state.summary_type = "網頁"
            status.update(label="處理失敗", state="error")
            logging.error(f"URL processing error: {e}")

    st.session_state.is_processing = False
