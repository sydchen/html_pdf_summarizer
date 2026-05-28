import argparse
import sys
from pathlib import Path

from config import Config
from youtube_summarizer import YouTubeSummarizer


def transcribe_youtube(url: str, language: str = "auto", write_txt: bool = True) -> str:
    summarizer = YouTubeSummarizer()

    video_id = summarizer.extract_video_id(url)
    if not video_id:
        raise ValueError(f"無法從 URL 提取 YouTube 影片 ID: {url}")

    output_dir = Config.ensure_output_dir()
    print(f"影片 ID: {video_id}")
    print(f"輸出目錄: {output_dir}")

    if language == "auto":
        detected_language = summarizer.detect_video_language.fn(url)
    else:
        detected_language = language
        print(f"使用指定語言: {detected_language}")

    video_path = summarizer.download_youtube_video.fn(url, output_dir, video_id)
    audio_path = summarizer.convert_audio_format.fn(video_path, output_dir, video_id)
    srt_path = summarizer.transcribe_audio.fn(audio_path, detected_language)

    print(f"SRT 逐字稿: {srt_path}")

    if write_txt:
        transcript = summarizer.clean_transcript.fn(srt_path)
        txt_path = str(Path(srt_path).with_suffix(".txt"))
        Path(txt_path).write_text(transcript, encoding="utf-8")
        print(f"純文字逐字稿: {txt_path}")

    return srt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YouTube 轉逐字稿工具，不產生摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python youtube_transcriber.py https://www.youtube.com/watch?v=VIDEO_ID\n"
            "  python youtube_transcriber.py https://www.youtube.com/watch?v=VIDEO_ID --lang en\n"
            "  python youtube_transcriber.py https://www.youtube.com/watch?v=VIDEO_ID --no-txt\n"
        ),
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--lang",
        default="auto",
        metavar="LANG",
        help="指定 Whisper 語言代碼，例如: zh, ja, en, ko。預設會嘗試自動偵測。",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="只輸出 SRT，不另外輸出清理後的 TXT。",
    )

    args = parser.parse_args()

    try:
        transcribe_youtube(args.url, language=args.lang, write_txt=not args.no_txt)
        return 0
    except Exception as exc:
        print(f"錯誤: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
