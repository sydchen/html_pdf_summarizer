import asyncio
import httpx
import tempfile
import os
import PyPDF2
from typing import List, Dict, Generator, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from llm import DocumentSummarizer

class WebArticleSummarizer:
    def __init__(self,
                 token_limit: int = 3000,
                 overlap_ratio: float = 0.1):
        """
        初始化網頁文章摘要器（支援 HTML 和 PDF）

        Args:
            model: Ollama 模型名稱
            token_limit: Token 數量限制
            overlap_ratio: 切分時的重疊比例
        """
        self.token_limit = token_limit
        self.overlap_ratio = overlap_ratio
        self.summarizer = DocumentSummarizer()

        # 根據不同任務類型調整建議值
        self.recommended_limits = {
            "short_summary": 1500,
            "long_summary": 3000,
            "detailed_analysis": 4000,
            "academic_paper": 6000
        }

    def detect_content_type(self, url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.head(url, headers=headers, follow_redirects=True)
                content_type = response.headers.get('content-type', '').lower()

                print(content_type)

                if any(pdf_type in content_type for pdf_type in [
                    'application/pdf',
                    'application/x-pdf',
                    'application/acrobat',
                    'applications/vnd.pdf',
                    'text/pdf',
                    'text/x-pdf'
                ]):
                    return True
                return False


        except Exception as e:
            print(f"Content-Type 檢測失敗: {e}")
            # 如果 HEAD 請求失敗，回退到 URL 檢查
            if url.lower().endswith('.pdf'):
                return True
            return False  # 預設為 HTML


    async def download_pdf(self, url: str) -> str:
        """下載 PDF 文件到臨時位置"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            print(f"正在下載 PDF: {url}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                # 創建臨時 PDF 文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(response.content)
                    print(f"PDF 下載完成，大小: {len(response.content)} bytes")
                    return temp_file.name

        except httpx.RequestError as e:
            raise ValueError(f"下載 PDF 錯誤: {e}")
        except Exception as e:
            raise ValueError(f"PDF 下載失敗: {e}")

    def extract_pdf_text(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                print(f"PDF 共 {total_pages} 頁，開始提取文字...")

                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # 只加入有內容的頁面
                        text += f"\n--- 第 {i+1} 頁 ---\n{page_text}\n"

                    # 顯示進度
                    if (i + 1) % 10 == 0 or i == total_pages - 1:
                        print(f"已處理 {i+1}/{total_pages} 頁")

        except Exception as e:
            raise ValueError(f"PDF 文字提取失敗: {e}")

        # 清理文字
        return self.clean_pdf_text(text)

    def clean_pdf_text(self, text: str) -> str:
        """清理從 PDF 提取的文字"""
        if not text:
            return ""

        # 移除多餘的空白和換行
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:  # 跳過空行
                lines.append(line)

        # 合併被切斷的行（啟發式方法）
        cleaned_lines = []
        i = 0
        while i < len(lines):
            current_line = lines[i]

            # 如果當前行很短且下一行存在，可能是被切斷的
            if (i + 1 < len(lines) and
                len(current_line) < 80 and
                not current_line.endswith(('.', '!', '?', ':', ';')) and
                not lines[i + 1].startswith(('第', '---', '•', '-', '1.', '2.', '3.'))):

                # 合併到下一行
                current_line += " " + lines[i + 1]
                i += 2
            else:
                i += 1

            cleaned_lines.append(current_line)

        return "\n\n".join(cleaned_lines)

    async def fetch_html_text(self, url: str) -> str:
        """獲取 HTML 網頁文字"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # 移除不需要的標籤
                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()

                # 尋找文章內容
                article = (soup.find("article") or
                          soup.find("main") or
                          soup.find("div", class_=lambda x: x and any(cls in str(x).lower() for cls in ["content", "article", "post"])) or
                          soup.find("body"))

                if not article:
                    raise ValueError("找不到文章內容區塊")

                # 提取段落文字
                paragraphs = article.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

                if not text:
                    raise ValueError("無法提取文章文字內容")

                return text

        except httpx.RequestError as e:
            raise ValueError(f"網路請求錯誤: {e}")
        except Exception as e:
            raise ValueError(f"文章提取錯誤: {e}")

    async def fetch_content(self, url: str, is_pdf: bool) -> str:
        if is_pdf:
            temp_pdf_path = await self.download_pdf(url)
            try:
                return self.extract_pdf_text(temp_pdf_path)
            finally:
                try:
                    os.unlink(temp_pdf_path)
                    print("臨時 PDF 文件已清理")
                except:
                    pass
        else:
            return await self.fetch_html_text(url)

    def fetch_content_sync(self, url: str, is_pdf: bool) -> str:
        return asyncio.run(self.fetch_content(url, is_pdf))

    def count_tokens(self, text: str) -> int:
        """計算文字的 token 數量（使用字符估算）"""
        # 這裡使用保守估計：3 字符 = 1 token
        return len(text) // 3

    def count_tokens_batch(self, texts: List[str]) -> int:
        """計算文字列表的總 token 數"""
        return sum(self.count_tokens(text) for text in texts)

    def get_recommended_token_limit(self, text_length: int, task_type: str = "long_summary") -> int:
        """根據文章長度和任務類型推薦合適的 token 限制"""
        base_limit = self.recommended_limits.get(task_type, 3000)

        if text_length < 5000:
            return min(base_limit, 1500)
        elif text_length < 15000:
            return base_limit
        else:
            return min(base_limit * 1.5, 6000)

    def split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """將長文字按 token 數量切分成多個部分"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_tokens = self.count_tokens(paragraph)

            # 如果單個段落就超過限制，需要進一步切分
            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # 按句子切分段落
                sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)

                    if current_tokens + sent_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sent_tokens
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens += sent_tokens
            else:
                if current_tokens + para_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                    current_tokens = para_tokens
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    current_tokens += para_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def split_texts_by_tokens(self, texts: List[str], max_tokens: int) -> List[List[str]]:
        """將文字列表分割成符合 token 限制的群組"""
        groups = []
        current_group = []
        current_tokens = 0

        for text in texts:
            text_tokens = self.count_tokens(text)

            if current_tokens + text_tokens > max_tokens and current_group:
                groups.append(current_group)
                current_group = [text]
                current_tokens = text_tokens
            else:
                current_group.append(text)
                current_tokens += text_tokens

        if current_group:
            groups.append(current_group)

        return groups

    def generate_summary(self, content: str, current: Optional[int] = None, total: Optional[int] = None) -> str:
        """生成單個內容的摘要"""
        if current is not None and total is not None:
            print(f"正在處理摘要 {current} / {total}")

        return self.summarizer.summarize_content(content)

    def reduce_summaries(self, summaries: List[str], current: Optional[int] = None, total: Optional[int] = None) -> str:
        """合併多個摘要"""
        if current is not None and total is not None:
            print(f"正在合併摘要 {current} / {total}")

        return self.summarizer.merge_summaries(summaries)

    def reduce_summaries_stream(self, summaries: List[str]) -> Generator[str, None, None]:
        """串流方式合併摘要"""
        for chunk in self.summarizer.merge_summaries_stream(summaries):
            yield chunk

    def summarize_texts_stream(self, texts: List[str]) -> Generator[str, None, None]:
        """串流方式摘要文字列表"""
        if not texts:
            yield "沒有文字可供摘要"
            return

        try:
            # 第一階段：對每個文字生成摘要（不輸出進度訊息）
            summaries = []
            for i, text in enumerate(texts):
                summary = self.generate_summary(text, i + 1, len(texts))
                if summary:
                    summaries.append(summary)

            if not summaries:
                yield "無法生成摘要"
                return

            # 迭代合併直到符合 token 限制（不輸出進度訊息）
            while self.count_tokens_batch(summaries) > self.token_limit:
                chunks = self.split_texts_by_tokens(summaries, self.token_limit)

                new_summaries = []
                for i, chunk in enumerate(chunks):
                    summary = self.reduce_summaries(chunk, i + 1, len(chunks))
                    if summary:
                        new_summaries.append(summary)

                if not new_summaries:
                    break

                summaries = new_summaries

            # 最終合併 - 只輸出摘要內容
            for chunk in self.reduce_summaries_stream(summaries):
                yield chunk

        except Exception as e:
            yield f"摘要過程中發生錯誤: {str(e)}"

    def get_summary(self, url: str, task_type: str = "long_summary") -> Generator[str, None, None]:
        """獲取文件摘要（支援 HTML 和 PDF）"""
        try:
            is_pdf = self.detect_content_type(url)
            file_type = "PDF" if is_pdf else "網頁"
            print(f"正在獲取{file_type}內容...")

            content = self.fetch_content_sync(url, is_pdf)

            if not content:
                yield f"無法獲取{file_type}內容"
                return

            print(f"{file_type}長度: {len(content)} 字元")

            # 計算文件的 token 數
            total_tokens = self.count_tokens(content)
            print(f"{file_type} token 數: {total_tokens}")

            # 動態調整 token 限制
            recommended_limit = self.get_recommended_token_limit(len(content), task_type)
            actual_limit = min(self.token_limit, recommended_limit)
            print(f"建議 token 限制: {recommended_limit}, 實際使用: {actual_limit}")

            # 如果文件太長，先切分再處理
            if total_tokens > actual_limit:
                print(f"{file_type}超過 token 限制 ({actual_limit})，正在切分...")
                text_chunks = self.split_text_by_tokens(content, actual_limit)
                print(f"切分成 {len(text_chunks)} 個部分")
            else:
                print(f"{file_type}長度適中，直接處理")
                text_chunks = [content]

            print("開始生成摘要...")
            for chunk in self.summarize_texts_stream(text_chunks):
                yield chunk

        except Exception as e:
            yield f"處理過程中發生錯誤: {str(e)}"

    def generate_simple_summary(self, content: str) -> str:
        """簡化版摘要生成（用於錯誤恢復）"""
        try:
            # 如果內容太長，先截取前面部分
            max_length = 5000
            if len(content) > max_length:
                content = content[:max_length] + "..."

            return self.summarizer.summarize_content(content)
        except Exception as e:
            return f"摘要生成失敗: {str(e)}"

    async def fetch_content_safe(self, url: str) -> tuple[str, str]:
        """安全的內容獲取（返回內容和類型）"""
        try:
            content_type = await self.detect_content_type(url)
            content = await self.fetch_content(url)
            return content, content_type
        except Exception as e:
            raise Exception(f"無法獲取內容: {str(e)}")

    def get_summary_sync(self, url: str, task_type: str = "long_summary") -> str:
        """同步版本的摘要獲取（用於 Streamlit）"""
        try:
            summary_parts = []
            for chunk in self.get_summary(url, task_type):
                summary_parts.append(chunk)
            return "".join(summary_parts)
        except Exception as e:
            return f"摘要生成失敗: {str(e)}"

if __name__ == "__main__":
    summarizer = WebArticleSummarizer(
        token_limit=3000,
        overlap_ratio=0.15
    )

    html_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    for chunk in summarizer.get_summary(html_url):
        print(chunk, end='', flush=True)
