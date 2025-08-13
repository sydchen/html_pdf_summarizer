import ollama
import os
from typing import Generator
from dotenv import load_dotenv

class DocumentSummarizer:
    def __init__(self, model_name=None):
        load_dotenv()
        self.model = model_name or os.getenv('MODEL') or 'gemma3:4b'

        self.system_prompt = (
            "你是一個專業的文件摘要助手。用戶會提供文字內容，請直接對這些內容進行摘要。\n\n"
            "摘要規則：\n"
            "• 提取核心主題和主要論點\n"
            "• 保留重要數據和關鍵事實\n"
            "• 保持邏輯結構清晰\n"
            "• 長度控制在原文的 15-25%\n"
            "• 以段落形式呈現，避免冗餘\n\n"
            "重要：只能使用繁體中文或英文回應，不可使用簡體中文。直接開始摘要，不要詢問內容在哪裡。"
        )

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=False
            )

            assistant_response = response['message']['content']

            # 將助手回應添加到對話歷史
            self.messages.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            print(f"錯誤: {e}")
            return None

    def chat_stream(self, user_input) -> Generator[str, None, None]:
        """串流模式的對話方法"""
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

            # 將完整的助手回應添加到對話歷史
            self.messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            print(f"串流錯誤: {e}")
            yield f"串流錯誤: {str(e)}"

    def reset_conversation(self):
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def get_conversation_history(self):
        """獲取完整對話歷史"""
        return self.messages.copy()

    def summarize_content(self, content: str) -> str:
        """生成內容摘要"""
        prompt = f"請為以下內容撰寫簡潔的中文摘要：\n\n{content}"
        self.reset_conversation()
        return self.chat(prompt)

    def merge_summaries(self, summaries: list[str]) -> str:
        """合併多個摘要"""
        merged = "\n\n".join(summaries)
        prompt = f"以下是一系列摘要內容，請將它們整合成一個完整、連貫的最終摘要：\n\n{merged}"
        self.reset_conversation()
        return self.chat(prompt)

    def merge_summaries_stream(self, summaries: list[str]) -> Generator[str, None, None]:
        """串流模式合併多個摘要"""
        merged = "\n\n".join(summaries)
        prompt = f"以下是一系列摘要內容，請將它們整合成一個完整、連貫的最終摘要：\n\n{merged}"
        self.reset_conversation()
        for chunk in self.chat_stream(prompt):
            yield chunk

if __name__ == "__main__":
    summarizer = DocumentSummarizer()
    summary1 = summarizer.chat("這是要摘要的文件內容...")
    print("類別方法結果:", summary1)

