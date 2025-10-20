import os
import re
import subprocess
from pathlib import Path
from typing import Generator, Optional
from prefect import flow, task
from config import Config
from transcript_llm import TranscriptSummarizer


class YouTubeSummarizer:
    """Summarizer for YouTube videos using Prefect workflow"""

    def __init__(self):
        self.config = Config
        self.transcript_summarizer = TranscriptSummarizer(model_name = "qwen2.5:7b")

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if URL is a YouTube URL"""
        youtube_patterns = [
            r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(https?://)?(www\.)?youtu\.be/[\w-]+',
            r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'youtu\.be\/([0-9A-Za-z_-]{11})',
            r'embed\/([0-9A-Za-z_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    @task(retries=2, retry_delay_seconds=10, name="下載 YouTube 音訊")
    def download_youtube_audio(url: str, output_dir: str, video_id: str) -> str:
        """
        Download YouTube video audio using yt-dlp

        Args:
            url: YouTube video URL
            output_dir: Output directory for downloaded file
            video_id: YouTube video ID

        Returns:
            Path to downloaded WAV file
        """
        output_path = os.path.join(output_dir, f"{video_id}_original.wav")

        # 檢查快取：如果檔案已存在，直接返回
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"使用快取的音訊檔案: {output_path} ({file_size} bytes)")
            return output_path

        print(f"正在下載 YouTube 音訊: {url}")

        try:
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "wav",
                "-o", output_path,
                url
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"音訊下載完成: {output_path} ({file_size} bytes)")
                return output_path
            else:
                raise FileNotFoundError(f"下載的音訊檔案不存在: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"yt-dlp 錯誤: {e.stderr}")
            raise Exception("無法下載 YouTube 影片，請確認網址是否正確")
        except Exception as e:
            print(f"下載錯誤: {str(e)}")
            raise Exception("下載 YouTube 音訊時發生錯誤")

    @staticmethod
    @task(retries=2, retry_delay_seconds=10, name="轉換音訊格式")
    def convert_audio_format(input_path: str, output_dir: str, video_id: str) -> str:
        """
        Convert audio to 16kHz mono WAV using ffmpeg

        Args:
            input_path: Path to input WAV file
            output_dir: Output directory
            video_id: YouTube video ID

        Returns:
            Path to converted WAV file
        """
        output_path = os.path.join(output_dir, f"{video_id}.wav")

        # 檢查快取：如果檔案已存在，直接返回
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"使用快取的轉換音訊: {output_path} ({file_size} bytes)")
            return output_path

        print(f"正在轉換音訊格式: {input_path} -> {output_path}")

        try:
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-ar", str(Config.AUDIO_SAMPLE_RATE),  # Sample rate
                "-ac", str(Config.AUDIO_CHANNELS),      # Channels (mono)
                "-y",  # Overwrite output file
                output_path
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"音訊轉換完成: {output_path} ({file_size} bytes)")
                return output_path
            else:
                raise FileNotFoundError(f"轉換後的音訊檔案不存在: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"ffmpeg 錯誤: {e.stderr}")
            raise Exception("音訊格式轉換失敗，請確認 ffmpeg 是否已安裝")
        except Exception as e:
            print(f"轉換錯誤: {str(e)}")
            raise Exception("轉換音訊格式時發生錯誤")

    @staticmethod
    @task(retries=2, retry_delay_seconds=10, name="偵測影片語言")
    def detect_video_language(url: str) -> str:
        """
        Detect video language using yt-dlp metadata

        Args:
            url: YouTube video URL

        Returns:
            Language code (e.g., 'en', 'zh', 'ja') or 'auto' if detection fails
        """
        print(f"正在偵測影片語言: {url}")

        try:
            cmd = [
                "yt-dlp",
                "--dump-json",  # Get metadata in JSON format
                "--skip-download",  # Don't download the video
                url
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            import json
            metadata = json.loads(result.stdout)

            # Try to get language from various metadata fields
            language = None

            # First, try the 'language' field
            if 'language' in metadata and metadata['language']:
                language = metadata['language']
                print(f"從 'language' 欄位偵測到語言: {language}")

            # Second, try subtitles/automatic_captions
            elif 'subtitles' in metadata and metadata['subtitles']:
                # Get the first available subtitle language
                language = list(metadata['subtitles'].keys())[0]
                print(f"從字幕偵測到語言: {language}")

            elif 'automatic_captions' in metadata and metadata['automatic_captions']:
                # Get the first available auto-caption language
                language = list(metadata['automatic_captions'].keys())[0]
                print(f"從自動字幕偵測到語言: {language}")

            # Map to Whisper-supported language code
            if language:
                # Extract base language code (e.g., 'zh-CN' -> 'zh')
                base_lang = language.split('-')[0].lower()

                # Map to Whisper language code
                whisper_lang = Config.LANGUAGE_MAP.get(base_lang, base_lang)
                print(f"映射到 Whisper 語言代碼: {whisper_lang}")
                return whisper_lang

            # If no language detected, use auto-detection
            print("無法偵測影片語言，將使用 Whisper 自動偵測")
            return 'auto'

        except subprocess.CalledProcessError as e:
            print(f"yt-dlp 語言偵測失敗: {e.stderr}")
            print("將使用 Whisper 自動偵測")
            return 'auto'
        except Exception as e:
            print(f"語言偵測錯誤: {str(e)}")
            print("將使用 Whisper 自動偵測")
            return 'auto'

    @staticmethod
    @task(retries=2, retry_delay_seconds=10, name="轉錄音訊")
    def transcribe_audio(audio_path: str, language: str = 'auto') -> str:
        """
        Transcribe audio using Whisper

        Args:
            audio_path: Path to audio file
            language: Language code for transcription ('auto' for auto-detection)

        Returns:
            Path to generated SRT file
        """
        srt_path = f"{audio_path}.srt"

        # 檢查快取：如果 SRT 檔案已存在，直接返回
        if os.path.exists(srt_path):
            file_size = os.path.getsize(srt_path)
            print(f"使用快取的逐字稿: {srt_path} ({file_size} bytes)")
            return srt_path

        print(f"正在轉錄音訊: {audio_path}")
        print(f"使用 Whisper 模型: {Config.WHISPER_MODEL_PATH}")
        print(f"使用語言: {language}")

        try:
            cmd = [
                Config.WHISPER_BINARY_PATH,
                "-m", Config.WHISPER_MODEL_PATH,
                "-osrt",  # Output SRT format
                "-f", audio_path
            ]

            # Add language parameter if not auto-detection
            if language != 'auto':
                cmd.extend(["-l", language])
            else:
                print("使用 Whisper 自動語言偵測")

            print(f"執行指令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            print(f"Whisper 輸出:\n{result.stdout}")

            if os.path.exists(srt_path):
                file_size = os.path.getsize(srt_path)
                print(f"轉錄完成: {srt_path} ({file_size} bytes)")
                return srt_path
            else:
                raise FileNotFoundError(f"SRT 檔案未生成: {srt_path}")

        except subprocess.CalledProcessError as e:
            print(f"Whisper 錯誤 (stdout): {e.stdout}")
            print(f"Whisper 錯誤 (stderr): {e.stderr}")
            raise Exception("音訊轉錄失敗，請檢查 Whisper 設定")
        except Exception as e:
            print(f"轉錄錯誤: {str(e)}")
            raise Exception("轉錄音訊時發生錯誤，請檢查 Whisper 是否正確安裝")

    @staticmethod
    @task(name="清理逐字稿")
    def clean_transcript(srt_path: str) -> str:
        """
        Remove timestamps from SRT file and extract clean text

        Args:
            srt_path: Path to SRT file

        Returns:
            Clean transcript text
        """
        print(f"正在清理逐字稿: {srt_path}")

        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse SRT format
            # SRT format:
            # 1
            # 00:00:00,000 --> 00:00:02,000
            # Text line 1
            # Text line 2
            #
            # 2
            # 00:00:02,000 --> 00:00:04,000
            # ...

            lines = content.split('\n')
            transcript_lines = []

            for line in lines:
                line = line.strip()
                # Skip empty lines, numbers, and timestamp lines
                if (line and
                    not line.isdigit() and
                    '-->' not in line):
                    transcript_lines.append(line)

            transcript = ' '.join(transcript_lines)

            # Clean up extra spaces
            transcript = re.sub(r'\s+', ' ', transcript).strip()

            print(f"清理完成，逐字稿長度: {len(transcript)} 字元")

            return transcript

        except Exception as e:
            print(f"清理逐字稿錯誤: {str(e)}")
            raise Exception("清理逐字稿時發生錯誤")

    @task(name="生成摘要")
    def summarize_transcript_task(self, transcript: str) -> str:
        """
        Generate summary from transcript

        Args:
            transcript: Clean transcript text

        Returns:
            Summary text
        """
        print(f"正在生成摘要，逐字稿長度: {len(transcript)} 字元")

        try:
            # Use non-streaming version for Prefect task
            summary = self.transcript_summarizer.chunk_and_summarize(transcript)

            print(f"摘要生成完成，長度: {len(summary)} 字元")

            return summary

        except Exception as e:
            print(f"摘要生成錯誤: {str(e)}")
            raise Exception("生成摘要時發生錯誤")

    @flow(name="YouTube 影片摘要流程")
    def youtube_summary_flow(self, url: str) -> str:
        """
        Main Prefect flow for YouTube video summarization

        Args:
            url: YouTube video URL

        Returns:
            Summary text
        """
        print(f"開始處理 YouTube 影片: {url}")

        # Extract video ID
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"無法從 URL 提取影片 ID: {url}")

        print(f"影片 ID: {video_id}")

        # Ensure output directory exists
        output_dir = Config.ensure_output_dir()
        print(f"輸出目錄: {output_dir}")

        # Step 1: Detect video language
        detected_language = self.detect_video_language(url)

        # Step 2: Download audio
        original_audio_path = self.download_youtube_audio(url, output_dir, video_id)

        # Step 3: Convert audio format
        converted_audio_path = self.convert_audio_format(
            original_audio_path,
            output_dir,
            video_id
        )

        # Step 4: Transcribe audio with detected language
        srt_path = self.transcribe_audio(converted_audio_path, detected_language)

        # Step 5: Clean transcript
        transcript = self.clean_transcript(srt_path)

        # Step 6: Generate summary
        summary = self.summarize_transcript_task(transcript)

        print("YouTube 影片摘要流程完成")

        return summary

    def get_summary(self, url: str) -> str:
        """
        Get summary for YouTube video (synchronous)

        Args:
            url: YouTube video URL

        Returns:
            Summary text
        """
        if not self.is_youtube_url(url):
            raise ValueError(f"不是有效的 YouTube URL: {url}")

        return self.youtube_summary_flow(url)

    def get_summary_stream(self, url: str) -> Generator[str, None, None]:
        """
        Get summary for YouTube video (streaming)

        Args:
            url: YouTube video URL

        Yields:
            Summary chunks
        """
        if not self.is_youtube_url(url):
            raise ValueError(f"不是有效的 YouTube URL: {url}")

        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                yield "處理失敗：無法識別 YouTube 影片 ID"
                return

            yield f"影片 ID: {video_id}\n"

            # Ensure output directory exists
            output_dir = Config.ensure_output_dir()

            # Step 1: Detect video language
            try:
                yield "正在偵測影片語言...\n"
                detected_language = self.detect_video_language(url)
                yield f"✓ 偵測到語言: {detected_language}\n"
            except Exception as e:
                print(f"語言偵測失敗詳細錯誤: {str(e)}")
                detected_language = 'auto'
                yield "⚠ 語言偵測失敗，將使用自動偵測\n"

            # Step 2: Download audio
            try:
                yield "正在下載音訊...\n"
                original_audio_path = self.download_youtube_audio(url, output_dir, video_id)
                yield "✓ 音訊下載完成\n"
            except Exception as e:
                print(f"下載失敗詳細錯誤: {str(e)}")
                yield f"\n處理失敗：{str(e)}"
                return

            # Step 3: Convert audio format
            try:
                yield "正在轉換音訊格式...\n"
                converted_audio_path = self.convert_audio_format(
                    original_audio_path,
                    output_dir,
                    video_id
                )
                yield "✓ 音訊格式轉換完成\n"
            except Exception as e:
                print(f"轉換失敗詳細錯誤: {str(e)}")
                yield f"\n處理失敗：{str(e)}"
                return

            # Step 4: Transcribe audio
            try:
                yield f"正在轉錄音訊（語言: {detected_language}，這可能需要幾分鐘）...\n"
                srt_path = self.transcribe_audio(converted_audio_path, detected_language)
                yield "✓ 音訊轉錄完成\n"
            except Exception as e:
                print(f"轉錄失敗詳細錯誤: {str(e)}")
                yield f"\n處理失敗：{str(e)}"
                return

            # Step 5: Clean transcript
            try:
                yield "正在清理逐字稿...\n"
                transcript = self.clean_transcript(srt_path)
                yield f"✓ 逐字稿清理完成（{len(transcript)} 字元）\n\n"
            except Exception as e:
                print(f"清理失敗詳細錯誤: {str(e)}")
                yield f"\n處理失敗：{str(e)}"
                return

            # Step 6: Generate summary (streaming)
            try:
                yield "正在生成摘要...\n\n"
                for chunk in self.transcript_summarizer.chunk_and_summarize_stream(transcript):
                    yield chunk
            except Exception as e:
                print(f"摘要生成失敗詳細錯誤: {str(e)}")
                yield f"\n\n摘要生成失敗：{str(e)}"
                return

        except Exception as e:
            print(f"未預期的錯誤: {str(e)}")
            yield f"\n\n處理過程中發生未預期的錯誤，請稍後再試"


if __name__ == "__main__":
    summarizer = YouTubeSummarizer()
    test_url = "https://www.youtube.com/watch?v=NgrCQcU0Sbg"

    print(f"測試 YouTube 摘要器: {test_url}")
    try:
        summary = summarizer.get_summary(test_url)
        print("\n摘要結果:")
        print(summary)
    except Exception as e:
        print(f"錯誤: {e}")
