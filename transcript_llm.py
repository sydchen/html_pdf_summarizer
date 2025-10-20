import ollama
import os
from typing import Generator
from dotenv import load_dotenv

class TranscriptSummarizer:
    """Specialized summarizer for video transcripts"""

    def __init__(self, model_name=None):
        load_dotenv()
        self.model = model_name or os.getenv('MODEL') or 'gemma3:4b'

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

            3. 範圍與長度：
               • 目標摘要長度約為原文的 25–35%  
               • 保留重要資訊與邏輯脈絡，不要壓縮得太短  

            4. 多段輸入策略（若有）：
               • 若逐字稿分為多個部分提供，請：
                 - 為每段生成局部摘要（Partial Summary）
                 - 在最後合併所有摘要，產出完整整合版（Final Summary）
               • 確保整合後內容具前後銜接與一致邏輯  

            5. 語言規範：
               • 僅能使用繁體中文或英文  
               • 不可使用簡體中文  

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

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_input):
        """Non-streaming chat method"""
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
        prompt = f"請為以下影片逐字稿撰寫結構化的摘要：\n\n{transcript}"
        self.reset_conversation()
        return self.chat(prompt)

    def summarize_transcript_stream(self, transcript: str) -> Generator[str, None, None]:
        """Generate summary for transcript (streaming)"""
        prompt = f"請為以下影片逐字稿撰寫結構化的摘要：\n\n{transcript}"
        self.reset_conversation()
        for chunk in self.chat_stream(prompt):
            yield chunk

    def chunk_and_summarize(self, transcript: str, chunk_size: int = 8000) -> str:
        """
        For very long transcripts, split into chunks and summarize

        Args:
            transcript: The full transcript text
            chunk_size: Maximum characters per chunk

        Returns:
            Final merged summary
        """
        # If transcript is short enough, summarize directly
        if len(transcript) <= chunk_size:
            return self.summarize_transcript(transcript)

        # Split into chunks
        chunks = []
        words = transcript.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Summarize each chunk
        print(f"逐字稿過長，分成 {len(chunks)} 個部分處理...")
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"正在處理第 {i}/{len(chunks)} 部分...")
            summary = self.summarize_transcript(chunk)
            if summary:
                summaries.append(summary)

        # Merge all summaries
        if len(summaries) == 1:
            return summaries[0]
        else:
            merged = "\n\n".join(summaries)
            prompt = f"以下是影片不同部分的摘要，請將它們整合成一個完整、連貫的最終摘要：\n\n{merged}"
            self.reset_conversation()
            return self.chat(prompt)

    def chunk_and_summarize_stream(self, transcript: str, chunk_size: int = 8000) -> Generator[str, None, None]:
        """
        Streaming version of chunk_and_summarize

        Args:
            transcript: The full transcript text
            chunk_size: Maximum characters per chunk

        Yields:
            Summary chunks as they are generated
        """
        # If transcript is short enough, summarize directly
        if len(transcript) <= chunk_size:
            for chunk in self.summarize_transcript_stream(transcript):
                yield chunk
            return

        # Split into chunks
        chunks = []
        words = transcript.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Summarize each chunk
        print(f"逐字稿過長，分成 {len(chunks)} 個部分處理...")
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"正在處理第 {i}/{len(chunks)} 部分...")
            summary = self.summarize_transcript(chunk)
            if summary:
                summaries.append(summary)

        # Stream the merged summary
        if len(summaries) == 1:
            yield summaries[0]
        else:
            merged = "\n\n".join(summaries)
            prompt = f"以下是影片不同部分的摘要，請將它們整合成一個完整、連貫的最終摘要：\n\n{merged}"
            self.reset_conversation()
            for chunk in self.chat_stream(prompt):
                yield chunk


if __name__ == "__main__":
    # Test the transcript summarizer
    summarizer = TranscriptSummarizer()
    sample_transcript = "這是一段測試逐字稿..."
    summary = summarizer.summarize_transcript(sample_transcript)
    print("摘要結果:", summary)
