import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_subtitle(audio_path: Path) -> Optional[Path]:
    candidates = [
        audio_path.with_suffix(".srt"),
        Path(str(audio_path) + ".srt"),
        audio_path.with_name(f"{audio_path.stem}.bilingual.srt"),
        audio_path.with_name(f"{audio_path.stem}.en.srt"),
        audio_path.with_name(f"{audio_path.stem}.zh-TW.srt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_ffmpeg_command(
    audio_path: Path,
    output_path: Path,
    subtitle_path: Optional[Path],
    resolution: str,
    fps: int,
    overwrite: bool,
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={resolution}:r={fps}",
        "-i",
        str(audio_path),
    ]

    if subtitle_path:
        cmd.extend(["-i", str(subtitle_path)])

    cmd.extend([
        "-map",
        "0:v",
        "-map",
        "1:a",
    ])

    if subtitle_path:
        cmd.extend(["-map", "2:s"])

    cmd.extend([
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
    ])

    if subtitle_path:
        cmd.extend([
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
        ])

    cmd.extend(["-shortest", str(output_path)])
    return cmd


def wav_to_mp4(
    audio_path: Path,
    output_path: Optional[Path] = None,
    subtitle_path: Optional[Path] = None,
    auto_subtitles: bool = True,
    resolution: str = "1280x720",
    fps: int = 30,
    overwrite: bool = False,
) -> Path:
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"輸入檔案必須是 .wav: {audio_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"WAV 檔案不存在: {audio_path}")

    output_path = output_path or audio_path.with_suffix(".mp4")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"輸出檔案已存在，請加 --overwrite 或指定 --output: {output_path}")

    if subtitle_path and not subtitle_path.exists():
        raise FileNotFoundError(f"字幕檔案不存在: {subtitle_path}")
    if subtitle_path is None and auto_subtitles:
        subtitle_path = find_subtitle(audio_path)

    if subtitle_path:
        print(f"使用軟字幕: {subtitle_path}")
    else:
        print("未加入字幕")

    cmd = build_ffmpeg_command(
        audio_path=audio_path,
        output_path=output_path,
        subtitle_path=subtitle_path,
        resolution=resolution,
        fps=fps,
        overwrite=overwrite,
    )

    print(f"輸出 MP4: {output_path}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffmpeg 轉換失敗，請確認 ffmpeg 已安裝且輸入檔案可讀取") from exc

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="將 WAV 音訊轉成黑底 MP4，可選擇加入 SRT 軟字幕",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python wav_to_mp4.py youtube_downloads/apple_podcast_1000768911278.wav\n"
            "  python wav_to_mp4.py youtube_downloads/apple_podcast_1000768911278.wav --subtitle youtube_downloads/apple_podcast_1000768911278.bilingual.srt\n"
            "  python wav_to_mp4.py youtube_downloads/apple_podcast_1000768911278.wav --no-subtitles --output output.mp4\n"
            "  python wav_to_mp4.py youtube_downloads/apple_podcast_1000768911278.wav --overwrite\n"
        ),
    )
    parser.add_argument("wav", help="輸入 WAV 檔案")
    parser.add_argument("--output", "-o", help="輸出 MP4 路徑，預設為同名 .mp4")
    parser.add_argument("--subtitle", "-s", help="指定 SRT 字幕檔，會以軟字幕加入 MP4")
    parser.add_argument("--no-subtitles", action="store_true", help="不要自動搜尋或加入字幕")
    parser.add_argument("--resolution", default="1280x720", help="影片解析度，預設 1280x720")
    parser.add_argument("--fps", type=int, default=30, help="影片 FPS，預設 30")
    parser.add_argument("--overwrite", action="store_true", help="允許覆蓋既有輸出檔案")

    args = parser.parse_args()

    try:
        wav_to_mp4(
            audio_path=Path(args.wav),
            output_path=Path(args.output) if args.output else None,
            subtitle_path=Path(args.subtitle) if args.subtitle else None,
            auto_subtitles=not args.no_subtitles,
            resolution=args.resolution,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        print(f"錯誤: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
