import streamlit as st
import logging
import asyncio
import sys
import time
from pdf_summarizer import PDFSummarizer
from html_summarizer import WebArticleSummarizer

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def init_summarizers():
    """初始化摘要器（使用 cache_resource 避免重複初始化）"""
    pdf_summarizer = PDFSummarizer(max_chunk_length=2000)
    web_summarizer = WebArticleSummarizer(
        token_limit=3000,
        overlap_ratio=0.15
    )
    return pdf_summarizer, web_summarizer

pdf_summarizer, web_summarizer = init_summarizers()

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
        for chunk in generator:
            if chunk:
                summary_text += chunk
                status_container.update(label=f"生成摘要中... ({len(summary_text)} 字元)")
                result_placeholder.markdown(summary_text)

        return summary_text

    except Exception as e:
        error_msg = f"摘要生成失敗: {str(e)}"
        status_container.update(label="處理失敗", state="error")
        logging.error(f"Stream summary error: {e}")
        return error_msg

def process_url_with_fallback(url):
    """處理 URL 摘要，包含容錯機制"""
    try:
        # 嘗試標準摘要流程
        for chunk in web_summarizer.get_summary(url):
            yield chunk
    except Exception as e:
        logging.error(f"Standard summary failed: {e}")
        try:
            # 備用方案：手動獲取內容並生成簡化摘要
            is_pdf = web_summarizer.detect_content_type(url)
            content = web_summarizer.fetch_content_sync(url, is_pdf)
            if content:
                yield "正在生成摘要..."
                summary = web_summarizer.generate_simple_summary(content)
                yield summary
            else:
                yield f"無法獲取內容: {str(e)}"
        except Exception as e2:
            logging.error(f"Fallback also failed: {e2}")
            yield f"摘要生成失敗: {str(e2)}"

st.title("PDF & Web Summarizer")
st.markdown("支援 PDF 文件上傳和網頁 URL 摘要")

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

# 處理 PDF 摘要
if (uploaded_file is not None and
    st.session_state.summary_output is None and
    uploaded_file == st.session_state.last_uploaded and
    not st.session_state.is_processing):

    st.session_state.is_processing = True
    st.session_state.summary_output = None

    with st.status("正在處理 PDF...", expanded=True) as status:
        def report(msg):
            status.update(label=msg)

        try:
            summary_generator = pdf_summarizer.get_summary(
                source=uploaded_file,
                on_progress=report
            )

            if hasattr(summary_generator, '__iter__'):
                summary = stream_summary(summary_generator, status, st.session_state.summary_placeholder)
            else:
                summary = summary_generator

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

st.divider()

st.header("🌐 網頁")

url_input = st.text_input(
    label="請輸入網址",
    placeholder="https://danluu.com/car-safety/",
    help="輸入要進行摘要的網頁網址或 PDF 連結",
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
        st.session_state.is_processing = True

        with st.status("正在分析網頁...", expanded=True) as status:
            try:
                summary_generator = process_url_with_fallback(url_input.strip())
                summary = stream_summary(summary_generator, status, st.session_state.summary_placeholder)

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
    else:
        st.warning("請輸入有效的網址")

st.divider()

st.subheader("摘要結果")
summary_result_placeholder = st.empty()
st.session_state.summary_placeholder = summary_result_placeholder

if st.session_state.summary_output:
    st.markdown(st.session_state.summary_output)

