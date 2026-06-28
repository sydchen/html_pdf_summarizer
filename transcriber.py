import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from config import Config
from youtube_summarizer import YouTubeSummarizer


def safe_file_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-") or "media"


def get_apple_podcast_filename(url: str) -> str:
    parsed_url = urlparse(url)
    path_parts = [part for part in parsed_url.path.split("/") if part]
    if "podcast" in path_parts:
        podcast_index = path_parts.index("podcast")
        if podcast_index + 1 < len(path_parts):
            return safe_file_id(path_parts[podcast_index + 1])

    path_match = re.search(r"/id(\d+)", parsed_url.path)
    query_id = parse_qs(parsed_url.query).get("i", [None])[0]
    return safe_file_id(query_id or (path_match.group(1) if path_match else parsed_url.path))


def get_apple_podcast_id(url: str) -> str:
    return get_apple_podcast_filename(url)


def is_apple_podcast_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.netloc.lower() == "podcasts.apple.com"


def transcript_language_suffix(language: str) -> str:
    return safe_file_id(language or "auto")


def standard_transcript_path(output_dir: str, media_id: str, language: str) -> Path:
    return Path(output_dir) / f"{media_id}.{transcript_language_suffix(language)}.srt"


def normalize_transcript_filename(srt_path: str, output_dir: str, media_id: str, language: str) -> str:
    source_path = Path(srt_path)
    target_path = standard_transcript_path(output_dir, media_id, language)

    if source_path == target_path:
        return str(target_path)
    if target_path.exists():
        print(f"使用標準命名的逐字稿: {target_path} ({target_path.stat().st_size} bytes)")
        return str(target_path)

    source_path.replace(target_path)
    print(f"逐字稿重新命名: {source_path} -> {target_path}")
    return str(target_path)


def cleanup_source_audio(downloaded_path: str, keep_source_audio: bool = False) -> None:
    source_path = Path(downloaded_path)
    if keep_source_audio or source_path.suffix.lower() != ".mp3":
        return
    try:
        source_path.unlink()
        print(f"已刪除原始 MP3: {source_path}")
    except FileNotFoundError:
        return


def download_media_audio(url: str, output_dir: str, media_id: str) -> str:
    output_path = Path(output_dir) / f"{media_id}.%(ext)s"
    expected_files = sorted(Path(output_dir).glob(f"{media_id}.*"))
    for existing_file in expected_files:
        if existing_file.suffix.lower() in {".m4a", ".mp3", ".mp4", ".webm", ".opus", ".aac", ".wav"}:
            print(f"使用快取的媒體檔案: {existing_file} ({existing_file.stat().st_size} bytes)")
            return str(existing_file)

    print(f"正在下載媒體音訊: {url}")

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "-o",
        str(output_path),
        url,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"yt-dlp 錯誤: {exc.stderr}")
        raise Exception("無法下載媒體音訊，請確認網址是否正確或 yt-dlp 是否支援此連結")

    downloaded_files = sorted(Path(output_dir).glob(f"{media_id}.*"))
    for downloaded_file in downloaded_files:
        if downloaded_file.suffix.lower() in {".m4a", ".mp3", ".mp4", ".webm", ".opus", ".aac", ".wav"}:
            print(f"媒體下載完成: {downloaded_file} ({downloaded_file.stat().st_size} bytes)")
            return str(downloaded_file)

    raise FileNotFoundError(f"下載的媒體檔案不存在: {output_path}")


def transcribe_media(
    url: str,
    language: str = "auto",
    write_txt: bool = True,
    keep_source_audio: bool = False,
) -> str:
    summarizer = YouTubeSummarizer()

    output_dir = Config.ensure_output_dir()
    print(f"輸出目錄: {output_dir}")

    if is_apple_podcast_url(url):
        media_id = get_apple_podcast_filename(url)
        print(f"Apple Podcasts filename: {media_id}")
        downloaded_path = download_media_audio(url, output_dir, media_id)
        detected_language = language
        if language == "auto":
            print("Apple Podcasts 連結未做語言 metadata 偵測，交由 Whisper 自動偵測")
    else:
        media_id = summarizer.extract_video_id(url)
        if not media_id:
            raise ValueError(f"目前只支援 YouTube 或 Apple Podcasts URL: {url}")

        print(f"YouTube 影片 ID: {media_id}")
        downloaded_path = summarizer.download_youtube_video.fn(url, output_dir, media_id)

    if language == "auto" and not is_apple_podcast_url(url):
        detected_language = summarizer.detect_video_language.fn(url)
    else:
        detected_language = language
        if language != "auto":
            print(f"使用指定語言: {detected_language}")

    audio_path = summarizer.convert_audio_format.fn(downloaded_path, output_dir, media_id)
    cleanup_source_audio(downloaded_path, keep_source_audio=keep_source_audio)
    standard_srt_path = standard_transcript_path(output_dir, media_id, detected_language)
    if standard_srt_path.exists():
        srt_path = str(standard_srt_path)
        print(f"使用快取的逐字稿: {standard_srt_path} ({standard_srt_path.stat().st_size} bytes)")
    else:
        srt_path = summarizer.transcribe_audio.fn(audio_path, detected_language)
        srt_path = normalize_transcript_filename(srt_path, output_dir, media_id, detected_language)

    print(f"SRT 逐字稿: {srt_path}")

    if write_txt:
        transcript = summarizer.clean_transcript.fn(srt_path)
        txt_path = str(Path(srt_path).with_suffix(".txt"))
        Path(txt_path).write_text(transcript, encoding="utf-8")
        print(f"純文字逐字稿: {txt_path}")

    return srt_path


def transcribe_youtube(
    url: str,
    language: str = "auto",
    write_txt: bool = True,
    keep_source_audio: bool = False,
) -> str:
    return transcribe_media(
        url,
        language=language,
        write_txt=write_txt,
        keep_source_audio=keep_source_audio,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YouTube / Apple Podcasts 轉逐字稿工具，不產生摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python transcriber.py https://www.youtube.com/watch?v=VIDEO_ID\n"
            "  python transcriber.py https://www.youtube.com/watch?v=VIDEO_ID --lang en\n"
            "  python transcriber.py https://podcasts.apple.com/tw/podcast/.../id123456789?i=987654321\n"
            "  python transcriber.py https://www.youtube.com/watch?v=VIDEO_ID --no-txt\n"
            "  python transcriber.py https://podcasts.apple.com/tw/podcast/.../id123456789?i=987654321 --keep-source-audio\n"
        ),
    )
    parser.add_argument("url", help="YouTube 或 Apple Podcasts URL")
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
    parser.add_argument(
        "--keep-source-audio",
        action="store_true",
        help="保留 yt-dlp 下載的原始音訊檔。預設會在轉成 WAV 後刪除 MP3。",
    )

    args = parser.parse_args()

    try:
        transcribe_media(
            args.url,
            language=args.lang,
            write_txt=not args.no_txt,
            keep_source_audio=args.keep_source_audio,
        )
        return 0
    except Exception as exc:
        print(f"錯誤: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
