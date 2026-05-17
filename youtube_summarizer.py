import os
import re
import sys
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
        self.transcript_summarizer = TranscriptSummarizer(model_name = "gemma4:e4b")

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
    def is_transcript_file(input_str: str) -> bool:
        """Check if input is a transcript file path"""
        # Check if it's a file path (not a URL) and has supported extension
        if not input_str.startswith(('http://', 'https://', 'www.')):
            path = Path(input_str)
            return path.suffix.lower() in ['.srt', '.txt']
        return False

    @staticmethod
    def is_video_file(input_str: str) -> bool:
        """Check if input is a local video file path"""
        if not input_str.startswith(('http://', 'https://', 'www.')):
            path = Path(input_str)
            return path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm']
        return False

    @staticmethod
    def load_transcript_from_file(file_path: str) -> str:
        """
        Load transcript from file (supports .srt and .txt)

        Args:
            file_path: Path to transcript file

        Returns:
            Clean transcript text
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"逐字稿檔案不存在: {file_path}")

        print(f"正在讀取逐字稿檔案: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # If it's an SRT file, clean it using the existing method
            if file_path_obj.suffix.lower() == '.srt':
                # Parse SRT format to extract text
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
            else:
                # For .txt files, use content as-is (with basic cleanup)
                transcript = re.sub(r'\s+', ' ', content).strip()

            print(f"逐字稿載入完成，長度: {len(transcript)} 字元")
            return transcript

        except Exception as e:
            print(f"讀取逐字稿檔案錯誤: {str(e)}")
            raise Exception(f"無法讀取逐字稿檔案: {str(e)}")

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
    @task(retries=2, retry_delay_seconds=10, name="下載 YouTube 影片")
    def download_youtube_video(url: str, output_dir: str, video_id: str) -> str:
        """
        Download YouTube video as MP4 using yt-dlp

        Args:
            url: YouTube video URL
            output_dir: Output directory for downloaded file
            video_id: YouTube video ID

        Returns:
            Path to downloaded MP4 file
        """
        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

        # 檢查快取：如果檔案已存在，直接返回
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"使用快取的影片檔案: {output_path} ({file_size} bytes)")
            return output_path

        print(f"正在下載 YouTube 影片: {url}")

        try:
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--no-playlist",
                "-o", output_template,
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
                print(f"影片下載完成: {output_path} ({file_size} bytes)")
                return output_path
            else:
                raise FileNotFoundError(f"下載的影片檔案不存在: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"yt-dlp 錯誤: {e.stderr}")
            raise Exception("無法下載 YouTube 影片，請確認網址是否正確")
        except Exception as e:
            print(f"下載錯誤: {str(e)}")
            raise Exception("下載 YouTube 影片時發生錯誤")

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

                # Only remove legacy intermediate audio files. Keep video
                # inputs such as mp4 files.
                input_file = Path(input_path)
                if input_file.name.endswith("_original.wav"):
                    try:
                        if input_file.exists():
                            input_file.unlink()
                            print(f"已刪除原始音訊檔案: {input_path}")
                    except Exception as e:
                        print(f"警告: 無法刪除原始音訊檔案: {str(e)}")
                        # Don't raise exception, conversion was successful

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
        # Use language-specific SRT path to avoid cache collision between languages
        if language != 'auto':
            srt_path = f"{audio_path}.{language}.srt"
        else:
            srt_path = f"{audio_path}.srt"
        whisper_default_srt = f"{audio_path}.srt"

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

            # whisper.cpp always outputs to audio_path.srt; rename to language-specific path
            if language != 'auto' and os.path.exists(whisper_default_srt):
                os.rename(whisper_default_srt, srt_path)

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

        # Step 2: Download video and keep the MP4 for reuse
        video_path = self.download_youtube_video(url, output_dir, video_id)

        # Step 3: Convert audio format
        converted_audio_path = self.convert_audio_format(
            video_path,
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

    @flow(name="本地影片摘要流程")
    def video_summary_flow(self, file_path: str, language: str = 'auto') -> str:
        """
        Prefect flow for local video file summarization

        Args:
            file_path: Path to local video file (.mp4, .mov, etc.)
            language: Language code for transcription ('auto' for auto-detection)

        Returns:
            Summary text
        """
        print(f"開始處理本地影片: {file_path}")
        print(f"使用語言: {language}")

        video_path = Path(file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"影片檔案不存在: {file_path}")

        # Use filename stem as identifier
        video_id = video_path.stem
        output_dir = Config.ensure_output_dir()

        # Step 1: Convert video to audio (ffmpeg handles mp4 directly)
        converted_audio_path = self.convert_audio_format(file_path, output_dir, video_id)

        # Step 2: Transcribe audio
        srt_path = self.transcribe_audio(converted_audio_path, language)

        # Step 3: Clean transcript
        transcript = self.clean_transcript(srt_path)

        # Step 4: Generate summary
        summary = self.summarize_transcript_task(transcript)

        print("本地影片摘要流程完成")

        return summary

    @flow(name="逐字稿摘要流程")
    def transcript_summary_flow(self, file_path: str) -> str:
        """
        Prefect flow for transcript file summarization

        Args:
            file_path: Path to transcript file (.srt or .txt)

        Returns:
            Summary text
        """
        print(f"開始處理逐字稿檔案: {file_path}")

        # Load transcript from file
        transcript = self.load_transcript_from_file(file_path)

        # Generate summary
        summary = self.summarize_transcript_task(transcript)

        print("逐字稿摘要流程完成")

        return summary

    def get_summary(self, input_source: str, language: str = 'auto') -> str:
        """
        Get summary for YouTube video, local video, or transcript file (synchronous)

        Args:
            input_source: YouTube URL, local video file path (.mp4 etc.), or transcript file path (.srt/.txt)
            language: Language code for transcription, only applies to local video files ('auto' for auto-detection)

        Returns:
            Summary text
        """
        if self.is_transcript_file(input_source):
            return self.transcript_summary_flow(input_source)
        elif self.is_video_file(input_source):
            return self.video_summary_flow(input_source, language)
        elif self.is_youtube_url(input_source):
            return self.youtube_summary_flow(input_source)
        else:
            raise ValueError(
                f"無效的輸入: {input_source}\n"
                f"請提供 YouTube URL、本地影片檔案 (.mp4/.mov 等) 或逐字稿檔案 (.srt/.txt)"
            )

    def get_summary_stream(self, input_source: str, language: str = 'auto') -> Generator[str, None, None]:
        """
        Get summary for YouTube video, local video, or transcript file (streaming)

        Args:
            input_source: YouTube URL, local video file path (.mp4 etc.), or transcript file path (.srt/.txt)
            language: Language code for transcription, only applies to local video files ('auto' for auto-detection)

        Yields:
            Summary chunks
        """
        # Handle transcript file
        if self.is_transcript_file(input_source):
            try:
                yield f"正在讀取逐字稿檔案: {input_source}\n"
                transcript = self.load_transcript_from_file(input_source)
                yield f"✓ 逐字稿載入完成（{len(transcript)} 字元）\n\n"

                yield "正在生成摘要...\n\n"
                for chunk in self.transcript_summarizer.chunk_and_summarize_stream(transcript):
                    yield chunk

            except Exception as e:
                print(f"處理逐字稿檔案失敗: {str(e)}")
                yield f"\n\n處理失敗：{str(e)}"
            return

        # Handle local video file
        if self.is_video_file(input_source):
            try:
                video_path = Path(input_source)
                if not video_path.exists():
                    yield f"處理失敗：影片檔案不存在: {input_source}"
                    return

                yield f"正在處理本地影片: {video_path.name}\n"
                if language != 'auto':
                    yield f"指定語言: {language}\n"
                video_id = video_path.stem
                output_dir = Config.ensure_output_dir()

                try:
                    yield "正在從影片提取並轉換音訊...\n"
                    converted_audio_path = self.convert_audio_format(input_source, output_dir, video_id)
                    yield "✓ 音訊提取完成\n"
                except Exception as e:
                    yield f"\n處理失敗：{str(e)}"
                    return

                try:
                    lang_display = language if language != 'auto' else '自動偵測'
                    yield f"正在轉錄音訊（語言: {lang_display}，這可能需要幾分鐘）...\n"
                    srt_path = self.transcribe_audio(converted_audio_path, language)
                    yield "✓ 音訊轉錄完成\n"
                except Exception as e:
                    yield f"\n處理失敗：{str(e)}"
                    return

                try:
                    yield "正在清理逐字稿...\n"
                    transcript = self.clean_transcript(srt_path)
                    yield f"✓ 逐字稿清理完成（{len(transcript)} 字元）\n\n"
                except Exception as e:
                    yield f"\n處理失敗：{str(e)}"
                    return

                yield "正在生成摘要...\n\n"
                for chunk in self.transcript_summarizer.chunk_and_summarize_stream(transcript):
                    yield chunk

            except Exception as e:
                print(f"處理本地影片失敗: {str(e)}")
                yield f"\n\n處理失敗：{str(e)}"
            return

        # Handle YouTube URL
        if not self.is_youtube_url(input_source):
            yield f"無效的輸入: {input_source}\n請提供 YouTube URL、本地影片檔案 (.mp4/.mov 等) 或逐字稿檔案 (.srt/.txt)"
            return

        try:
            url = input_source
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

            # Step 2: Download video
            try:
                yield "正在下載影片...\n"
                video_path = self.download_youtube_video(url, output_dir, video_id)
                yield "✓ 影片下載完成\n"
            except Exception as e:
                print(f"下載失敗詳細錯誤: {str(e)}")
                yield f"\n處理失敗：{str(e)}"
                return

            # Step 3: Convert audio format
            try:
                yield "正在轉換音訊格式...\n"
                converted_audio_path = self.convert_audio_format(
                    video_path,
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


def print_usage():
    """Print usage information"""
    print("YouTube 影片/本地影片/逐字稿摘要工具")
    print("\n使用方式:")
    print("  python youtube_summarizer.py <URL_或_檔案路徑>")
    print("\n範例:")
    print("  # YouTube 影片")
    print("  python youtube_summarizer.py https://www.youtube.com/watch?v=VIDEO_ID")
    print("\n  # 本地 MP4 影片")
    print("  python youtube_summarizer.py ./video.mp4")
    print("\n  # SRT 逐字稿檔案")
    print("  python youtube_summarizer.py ./transcripts/lecture.srt")
    print("\n  # TXT 文字檔案")
    print("  python youtube_summarizer.py ./transcripts/speech.txt")
    print("\n支援格式:")
    print("  - YouTube URL (http/https 開頭)")
    print("  - 本地影片檔案 (.mp4, .mov, .avi, .mkv, .m4v, .webm)")
    print("  - SRT 逐字稿檔案 (.srt)")
    print("  - 純文字檔案 (.txt)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YouTube 影片/本地影片/逐字稿摘要工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python youtube_summarizer.py https://www.youtube.com/watch?v=VIDEO_ID\n"
            "  python youtube_summarizer.py ./video.mp4\n"
            "  python youtube_summarizer.py ./video.mp4 --lang zh\n"
            "  python youtube_summarizer.py ./video.mp4 --lang ja\n"
            "  python youtube_summarizer.py ./lecture.srt\n"
        )
    )
    parser.add_argument("input", help="YouTube URL、本地影片路徑或逐字稿路徑")
    parser.add_argument(
        "--lang",
        default="auto",
        metavar="LANG",
        help="指定語言代碼（僅適用本地影片），例如: zh, ja, en, ko。預設為自動偵測"
    )

    args = parser.parse_args()
    input_source = args.input
    language = args.lang

    # Validate input
    if input_source.startswith(('http://', 'https://')):
        print(f"處理 YouTube 影片: {input_source}\n")
    else:
        file_path = Path(input_source)
        if not file_path.exists():
            print(f"錯誤: 檔案不存在: {input_source}")
            sys.exit(1)

        supported_extensions = ['.srt', '.txt', '.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm']
        if file_path.suffix.lower() not in supported_extensions:
            print(f"錯誤: 不支援的檔案格式: {file_path.suffix}")
            print(f"支援的格式: {', '.join(supported_extensions)}")
            sys.exit(1)

        if file_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm']:
            lang_display = language if language != 'auto' else '自動偵測'
            print(f"處理本地影片: {input_source}（語言: {lang_display}）\n")
        else:
            print(f"處理逐字稿檔案: {input_source}\n")

    # Process the input
    summarizer = YouTubeSummarizer()

    try:
        print("開始處理...\n")
        summary = summarizer.get_summary(input_source, language=language)
        print("\n" + "="*60)
        print("摘要結果:")
        print("="*60)
        print(summary)
        print("="*60)
    except Exception as e:
        print(f"\n處理失敗: {e}")
        sys.exit(1)
