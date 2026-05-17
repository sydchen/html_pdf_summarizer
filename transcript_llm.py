import os
import argparse
import sys
import re
from pathlib import Path
from typing import Generator

# This script does not use Prefect. Disable Prefect telemetry in case a shared
# environment or transitive import starts Prefect's background services.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY", "false")

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

try:
    import ollama
except ImportError:
    ollama = None

class TranscriptSummarizer:
    """Specialized summarizer for video transcripts"""

    DEFAULT_MODEL = "gemma4:e4b"

    LANGUAGE_INSTRUCTION = (
        "語言要求：請使用台灣常用繁體中文輸出。"
        "除非是原文專有名詞、英文術語、程式碼或網址，否則不要使用英文作為主要敘述語言。"
        "禁止使用簡體中文用字與中國大陸慣用詞。"
        "輸出前請自行檢查並轉換所有簡體字，例如："
        "视频→影片、内容→內容、重点→重點、数据→資料、质量→品質、"
        "逻辑→邏輯、发现→發現、实现→實作、问题→問題、通过→透過。"
    )

    FORMAT_INSTRUCTION = (
        "格式要求：可以使用章節標題、編號與項目符號，但不要使用 Markdown 粗體或斜體。"
        "禁止輸出 **文字**、__文字__、*文字* 這類強調語法。"
        "結論段落也請使用一般純文字標題，例如「3. 【核心觀點與結論】」，不要把標題或關鍵句加粗。"
    )

    def __init__(self, model_name=None):
        load_dotenv()
        self.model = model_name or os.getenv('TRANSCRIPT_MODEL') or self.DEFAULT_MODEL

        # 專門為逐字稿設計的系統提示詞
        # self.system_prompt = (
        #     "你是一個專業的影片逐字稿摘要助手。用戶會提供影片的逐字稿內容，請對這些內容進行結構化摘要。\n\n"
        #     "摘要規則：\n"
        #     "• 識別並提取影片的主要主題和核心觀點\n"
        #     "• 依照時間順序或邏輯順序組織內容\n"
        #     "• 保留重要的數據、案例、引用和關鍵論述\n"
        #     "• 去除口語化重複（如「嗯」、「那個」等填充詞的痕跡）\n"
        #     "• 將零散的口語表達整理成清晰的書面語\n"
        #     "• 使用標題和分段來組織不同主題\n"
        #     "• 摘要長度控制在原文的 20-30%\n"
        #     "• 以結構化的段落形式呈現（可使用標題、項目符號等）\n\n"
        #     "輸出格式建議：\n"
        #     "1. 簡短概述（1-2句話）\n"
        #     "2. 主要內容（分段或分點說明）\n"
        #     "3. 關鍵要點或結論\n\n"
        #     "重要：只能使用繁體中文或英文回應，不可使用簡體中文。直接開始摘要，不要詢問內容在哪裡。"
        # )

        self.system_prompt = (
            """你是一位專業的「影片逐字稿摘要與知識結構化助手」。

            使用者將提供一份長篇逐字稿（約 4～5 萬字，可能分段提供）。  
            你的任務是：在不遺漏重要內容的前提下，產生一份具層次、邏輯清晰且內容完整的摘要。

            --- 摘要規則 ---
            1. 結構化摘要：
               • 優先依「主題章節」組織（可輔以時間順序）  
               • 每個章節包含：
                 - 小標題（簡短，反映核心主題）
                 - 條列重點（摘要主要觀點、事件或論點）
                 - 補充說明（保留數據、案例、引用、具體說明）

            2. 語氣與格式：
               • 使用中性、專業、書面化語氣  
               • 去除口語化重複與贅詞（如「嗯」、「那個」等）  
               • 以清晰段落與標題格式組織文本  
               • 可使用項目符號、縮排強化層次  
               • 不要使用 Markdown 粗體或斜體，例如 **文字**、__文字__、*文字*

            3. 範圍與長度：
               • 目標摘要長度約為原文的 25–35%  
               • 保留重要資訊與邏輯脈絡，不要壓縮得太短  

            4. 多段輸入策略（若有）：
               • 若逐字稿分為多個部分提供，請：
                 - 為每段生成局部摘要（Partial Summary）
                 - 在最後合併所有摘要，產出完整整合版（Final Summary）
               • 確保整合後內容具前後銜接與一致邏輯  

            5. 語言規範：
               • 使用台灣常用繁體中文輸出  
               • 不可使用簡體中文或中國大陸慣用詞  
               • 專有名詞、英文術語、程式碼與網址可保留原文  
               • 輸出前請自行檢查並將所有簡體字轉為繁體字  

            --- 輸出格式 ---
            1. 【整體概述】  
               （1–2 句話，簡述影片主題與主要內容）

            2. 【主題摘要】  
               （分章節呈現，每段包含小標題、條列重點與補充說明）

            3. 【核心觀點與結論】  
               （歸納影片主要結論、啟示或行動方向）

            --- 特別注意 ---
            請直接開始摘要，不要詢問逐字稿內容在哪裡。
            若接收到多段輸入，請暫存前段的摘要內容，最後再整合。"""
        )
        self.system_prompt = (
            f"{self.system_prompt}\n\n"
            f"{self.LANGUAGE_INSTRUCTION}\n\n"
            f"{self.FORMAT_INSTRUCTION}"
        )

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_input):
        """Non-streaming chat method"""
        if ollama is None:
            print("錯誤: 找不到 ollama 套件，請先安裝或切換到正確的 Python 環境")
            return None

        self.messages.append({"role": "user", "content": user_input})

        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=False
            )

            assistant_response = response['message']['content']
            self.messages.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            print(f"錯誤: {e}")
            return None

    def chat_stream(self, user_input) -> Generator[str, None, None]:
        """Streaming chat method"""
        if ollama is None:
            yield "串流錯誤: 找不到 ollama 套件，請先安裝或切換到正確的 Python 環境"
            return

        self.messages.append({"role": "user", "content": user_input})

        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True
            )

            assistant_response = ""
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    assistant_response += content
                    yield content

            self.messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            print(f"串流錯誤: {e}")
            yield f"串流錯誤: {str(e)}"

    def reset_conversation(self):
        """Reset conversation history"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def summarize_transcript(self, transcript: str) -> str:
        """Generate summary for transcript (non-streaming)"""
        prompt = (
            f"{self.LANGUAGE_INSTRUCTION}\n\n"
            f"{self.FORMAT_INSTRUCTION}\n\n"
            "請為以下影片逐字稿撰寫結構化摘要。"
            "所有標題、條列與說明都必須使用台灣繁體中文，不可混入簡體中文。\n\n"
            f"{transcript}"
        )
        self.reset_conversation()
        return self.chat(prompt)

    def summarize_transcript_stream(self, transcript: str) -> Generator[str, None, None]:
        """Generate summary for transcript (streaming)"""
        prompt = (
            f"{self.LANGUAGE_INSTRUCTION}\n\n"
            f"{self.FORMAT_INSTRUCTION}\n\n"
            "請為以下影片逐字稿撰寫結構化摘要。"
            "所有標題、條列與說明都必須使用台灣繁體中文，不可混入簡體中文。\n\n"
            f"{transcript}"
        )
        self.reset_conversation()
        for chunk in self.chat_stream(prompt):
            yield chunk

    @staticmethod
    def split_transcript_into_chunks(
        transcript: str,
        chunk_size: int = 8000,
        overlap_words: int = 200
    ) -> list[str]:
        """
        Split English transcript text on sentence boundaries with word overlap.

        Args:
            transcript: The full transcript text
            chunk_size: Target maximum characters per chunk
            overlap_words: Number of trailing words to repeat in the next chunk

        Returns:
            Transcript chunks in source order
        """
        normalized = re.sub(r"\s+", " ", transcript).strip()
        if not normalized:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence) + 1
            if current_sentences and current_length + sentence_length > chunk_size:
                current_text = " ".join(current_sentences).strip()
                chunks.append(current_text)

                overlap = " ".join(current_text.split()[-overlap_words:])
                current_sentences = [overlap, sentence] if overlap else [sentence]
                current_length = len(" ".join(current_sentences))
            else:
                current_sentences.append(sentence)
                current_length += sentence_length

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return chunks

    def summarize_transcript_part(self, transcript: str, part_number: int, total_parts: int) -> str:
        """Generate a structured summary for one transcript chunk."""
        prompt = (
            f"{self.LANGUAGE_INSTRUCTION}\n\n"
            f"{self.FORMAT_INSTRUCTION}\n\n"
            f"以下是影片逐字稿的 Part {part_number}/{total_parts}。"
            "請只摘要此片段，並保留足夠資訊供最後整合。"
            "輸出必須標示 Part 編號，並包含：主要主題、關鍵論點、重要例子/人物/數字、與前後文相關的銜接資訊。"
            "請使用台灣繁體中文，不可混入簡體中文。\n\n"
            f"{transcript}"
        )
        self.reset_conversation()
        return self.chat(prompt)

    def build_final_merge_prompt(self, summaries: list[str]) -> str:
        """Build the final merge prompt for ordered chunk summaries."""
        merged = "\n\n".join(
            f"[Part {index}/{len(summaries)}]\n{summary}"
            for index, summary in enumerate(summaries, 1)
        )
        return (
            f"{self.LANGUAGE_INSTRUCTION}\n\n"
            f"{self.FORMAT_INSTRUCTION}\n\n"
            "以下是影片逐字稿依原始順序產生的多段摘要。"
            "請依 Part 順序整合成一份完整、連貫的最終摘要。"
            "整合時請去除重複內容、保留原影片的論述脈絡與重要細節，"
            "不要新增逐字稿或各 Part 摘要中沒有的資訊。"
            "最終輸出必須使用台灣繁體中文，並修正任何簡體中文用字。\n\n"
            f"{merged}"
        )

    def chunk_and_summarize(
        self,
        transcript: str,
        chunk_size: int = 8000,
        overlap_words: int = 200
    ) -> str:
        """
        For very long transcripts, split into chunks and summarize

        Args:
            transcript: The full transcript text
            chunk_size: Target maximum characters per chunk
            overlap_words: Number of trailing words to repeat in the next chunk

        Returns:
            Final merged summary
        """
        # If transcript is short enough, summarize directly
        if len(transcript) <= chunk_size:
            return self.summarize_transcript(transcript)

        chunks = self.split_transcript_into_chunks(transcript, chunk_size, overlap_words)
        if not chunks:
            return ""

        # Summarize each chunk
        print(f"逐字稿過長，分成 {len(chunks)} 個部分處理...")
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"正在處理第 {i}/{len(chunks)} 部分...")
            summary = self.summarize_transcript_part(chunk, i, len(chunks))
            if summary:
                summaries.append(summary)
            else:
                raise RuntimeError(f"第 {i}/{len(chunks)} 部分摘要失敗")

        # Merge all summaries
        if len(summaries) == 1:
            return summaries[0]
        else:
            prompt = self.build_final_merge_prompt(summaries)
            self.reset_conversation()
            return self.chat(prompt)

    def chunk_and_summarize_stream(
        self,
        transcript: str,
        chunk_size: int = 8000,
        overlap_words: int = 200
    ) -> Generator[str, None, None]:
        """
        Streaming version of chunk_and_summarize

        Args:
            transcript: The full transcript text
            chunk_size: Target maximum characters per chunk
            overlap_words: Number of trailing words to repeat in the next chunk

        Yields:
            Summary chunks as they are generated
        """
        # If transcript is short enough, summarize directly
        if len(transcript) <= chunk_size:
            for chunk in self.summarize_transcript_stream(transcript):
                yield chunk
            return

        chunks = self.split_transcript_into_chunks(transcript, chunk_size, overlap_words)
        if not chunks:
            return

        # Summarize each chunk
        print(f"逐字稿過長，分成 {len(chunks)} 個部分處理...")
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"正在處理第 {i}/{len(chunks)} 部分...")
            summary = self.summarize_transcript_part(chunk, i, len(chunks))
            if summary:
                summaries.append(summary)
            else:
                yield f"處理失敗：第 {i}/{len(chunks)} 部分摘要失敗"
                return

        # Stream the merged summary
        if len(summaries) == 1:
            yield summaries[0]
        else:
            prompt = self.build_final_merge_prompt(summaries)
            self.reset_conversation()
            for chunk in self.chat_stream(prompt):
                yield chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="影片逐字稿摘要測試工具")
    parser.add_argument("transcript", nargs="?", help="逐字稿檔案路徑（.srt 或 .txt）")
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama 模型名稱。未指定時使用 TRANSCRIPT_MODEL 環境變數，否則使用 gemma4:e4b。"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        help="長逐字稿分段字元數，預設 8000。"
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=200,
        help="長逐字稿分段重疊字數，預設 200。"
    )
    args = parser.parse_args()

    if args.transcript:
        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            print(f"錯誤: 逐字稿檔案不存在: {args.transcript}", file=sys.stderr)
            sys.exit(1)

        transcript = transcript_path.read_text(encoding="utf-8")
        if transcript_path.suffix.lower() == ".srt":
            lines = transcript.splitlines()
            transcript = " ".join(
                line.strip()
                for line in lines
                if line.strip() and not line.strip().isdigit() and "-->" not in line
            )
            transcript = re.sub(r"\s+", " ", transcript).strip()
    else:
        transcript = "這是一段測試逐字稿..."

    summarizer = TranscriptSummarizer(model_name=args.model)
    print(f"使用模型: {summarizer.model}")
    summary = summarizer.chunk_and_summarize(
        transcript,
        chunk_size=args.chunk_size,
        overlap_words=args.overlap_words
    )
    print("\n摘要結果:")
    print(summary)
