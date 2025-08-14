import requests
import json
import io
import sys
import PyPDF2
from typing import Union, Callable, Optional
from llm import DocumentSummarizer

class PDFSummarizer:
    def __init__(self, max_chunk_length: int = 2000):
        """
        初始化 PDF 摘要器

        Args:
            max_chunk_length: 每個文字塊的最大長度
        """
        self.max_chunk_length = max_chunk_length
        self.summarizer = DocumentSummarizer()

    def extract_text_from_pdf(self, source: Union[str, io.BytesIO]) -> str:
        """
        從 PDF 擷取所有文字，可接受：
        - 檔案路徑（字串）
        - URL（字串）
        - BytesIO 或 UploadedFile 類型
        """
        text = ""
        pdf_stream = None

        try:
            if isinstance(source, str):
                if source.startswith("http://") or source.startswith("https://"):
                    response = requests.get(source)
                    response.raise_for_status()
                    pdf_stream = io.BytesIO(response.content)
                else:
                    pdf_stream = open(source, "rb")
            else:
                # 已經是 BytesIO（例如 Streamlit 的 uploaded_file）
                pdf_stream = source

            reader = PyPDF2.PdfReader(pdf_stream)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

        except Exception as e:
            print(f"PDF 讀取錯誤: {e}", file=sys.stderr)
            raise e
        finally:
            # 只有當我們開啟本地檔案時才需要關閉
            if isinstance(source, str) and not source.startswith("http") and pdf_stream:
                pdf_stream.close()

        return text

    def split_text(self, text: str) -> list[str]:
        """
        將文字分割成不超過 max_length 的塊，
        盡量尊重段落邊界。
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.max_chunk_length:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())

                if len(para) > self.max_chunk_length:
                    # 將長段落分割成子塊
                    for i in range(0, len(para), self.max_chunk_length):
                        chunks.append(para[i:i + self.max_chunk_length].strip())
                    current = ""
                else:
                    current = para + "\n\n"

        if current:
            chunks.append(current.strip())

        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        try:
            summary = self.summarizer.summarize_content(chunk)
            return summary if summary else ""
        except Exception as e:
            print(f"摘要錯誤: {e}", file=sys.stderr)
            return f"摘要失敗: {str(e)}"

    def get_summary(self,
                   source: Union[str, io.BytesIO],
                   on_progress: Optional[Callable[[str], None]] = None) -> str:
        """
        獲取 PDF 的完整摘要

        Args:
            source: PDF 來源（檔案路徑、URL 或 BytesIO）
            on_progress: 進度回調函數

        Returns:
            str: 摘要結果
        """
        if on_progress is None:
            on_progress = lambda x: None  # 空函數

        try:
            # 步驟 1: 提取文字
            on_progress("正在從 PDF 提取文字...")
            full_text = self.extract_text_from_pdf(source)
            on_progress(f"提取的文字長度: {len(full_text)} 字元")

            if len(full_text) == 0:
                print("警告：沒有提取到任何文字")
                return "無法從 PDF 中提取文字內容"

            if len(full_text) > 0:
                print(f"文字前100字: {full_text[:100]}")

            # 步驟 2: 分割文字
            on_progress("正在將文字分割成塊...")
            chunks = self.split_text(full_text)
            on_progress(f"總共分割成 {len(chunks)} 個區塊")

            # 步驟 3: 對每個塊進行摘要
            summaries = []
            for idx, chunk in enumerate(chunks, start=1):
                on_progress(f"正在摘要第 {idx}/{len(chunks)} 個區塊...")
                summary = self.summarize_chunk(chunk)
                if summary:
                    summaries.append(summary)

            # 步驟 4: 處理摘要結果
            if len(summaries) == 0:
                return "摘要過程中發生錯誤，無法生成摘要"
            elif len(summaries) == 1:
                return summaries[0]
            else:
                # 多個摘要需要整合
                on_progress("正在整合最終摘要...")
                final_summary = self.summarizer.merge_summaries(summaries)
                return final_summary if final_summary else "\n\n".join(summaries)

        except Exception as e:
            error_msg = f"摘要過程中發生錯誤: {str(e)}"
            print(error_msg, file=sys.stderr)
            return error_msg

    def reset_summarizer(self):
        """重置摘要器的對話狀態"""
        self.summarizer.reset_conversation()

    def set_chunk_length(self, length: int):
        """設置文字塊的最大長度"""
        self.max_chunk_length = max(500, length)  # 最小 500 字元

