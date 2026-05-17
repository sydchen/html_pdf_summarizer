#!/usr/bin/env python3
"""
逐字稿清理工具

從 SRT 逐字稿檔案中移除編號和時間訊息，輸出純文字內容。
"""

import sys
import re
from pathlib import Path
from typing import Optional


TIMESTAMP_PATTERN = re.compile(
    r'^\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}'
)


def is_timestamp_line(line: str) -> bool:
    """Return True if a line looks like an SRT timestamp range."""
    return bool(TIMESTAMP_PATTERN.match(line.strip()))


def clean_srt_file(input_file: str, output_file: Optional[str] = None, auto_output: bool = True) -> str:
    """
    清理 SRT 逐字稿檔案，移除編號和時間戳記

    Args:
        input_file: 輸入的 SRT 檔案路徑
        output_file: 輸出檔案路徑（可選）
        auto_output: 是否自動在同目錄產生輸出檔案（預設 True）

    Returns:
        清理後的文字內容
    """
    input_path = Path(input_file)

    # 檢查檔案是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"檔案不存在: {input_file}")

    # 檢查檔案格式
    if input_path.suffix.lower() != '.srt':
        raise ValueError(f"不支援的檔案格式: {input_path.suffix}，請提供 .srt 檔案")

    # 如果沒有指定輸出檔案且 auto_output 為 True，自動產生輸出檔案名
    if output_file is None and auto_output:
        # 在同一目錄下產生 原檔名_clean.txt
        output_file = input_path.parent / f"{input_path.stem}_clean.txt"

    # 讀取檔案內容
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # 解析 SRT 格式
    # SRT 格式範例:
    # 1
    # 00:00:00,000 --> 00:00:02,000
    # 這是第一行文字
    # 這是第二行文字
    #
    # 2
    # 00:00:02,000 --> 00:00:04,000
    # 下一段文字

    lines = content.split('\n')
    transcript_blocks = []  # 儲存每個字幕塊
    current_block = []  # 當前字幕塊的文字行

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        next_line_is_timestamp = i + 1 < len(lines) and is_timestamp_line(lines[i + 1])

        # 如果是字幕編號且下一行是時間軸，開始新的字幕塊
        if line.isdigit() and next_line_is_timestamp:
            # 如果有累積的文字，先保存當前塊
            if current_block:
                block_text = ' '.join(current_block)
                block_text = re.sub(r'\s+', ' ', block_text).strip()
                if block_text:
                    transcript_blocks.append(block_text)
                current_block = []

            # 跳過編號行
            i += 1

            # 跳過時間戳記行
            if i < len(lines) and is_timestamp_line(lines[i]):
                i += 1

            # 收集字幕內容（直到空行或下一個編號）
            while i < len(lines):
                text_line = lines[i].strip()

                # 遇到空行，結束當前塊
                if not text_line:
                    break

                next_text_line_is_timestamp = i + 1 < len(lines) and is_timestamp_line(lines[i + 1])

                # 遇到下一個字幕編號，結束當前塊
                if text_line.isdigit() and next_text_line_is_timestamp:
                    break

                # 不是時間戳記行，就是字幕內容
                if not is_timestamp_line(text_line):
                    current_block.append(text_line)

                i += 1
        else:
            i += 1

    # 處理最後一個塊
    if current_block:
        block_text = ' '.join(current_block)
        block_text = re.sub(r'\s+', ' ', block_text).strip()
        if block_text:
            transcript_blocks.append(block_text)

    # 用換行符連接所有字幕塊
    transcript = '\n'.join(transcript_blocks)

    # 寫入輸出檔案
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"✓ 清理完成")
        print(f"  輸入檔案: {input_file}")
        print(f"  輸出檔案: {output_file}")
        print(f"  段落數量: {len(transcript_blocks)}")
        print(f"  總字元數: {len(transcript)} 字元")
    else:
        # 輸出到 stdout（當 auto_output=False 且無指定輸出檔案時）
        print(transcript)

    return transcript


def print_usage():
    """顯示使用說明"""
    print("逐字稿清理工具")
    print("\n用途:")
    print("  從 SRT 逐字稿檔案中移除編號和時間訊息，輸出純文字內容")
    print("  每個字幕段落會保留為一行，根據原始編號斷行")
    print("\n使用方式:")
    print("  python clean_transcript.py <輸入檔案.srt> [輸出檔案.txt]")
    print("\n範例:")
    print("  # 自動在同目錄產生 檔名_clean.txt")
    print("  python clean_transcript.py lecture.srt")
    print("  # 產生: lecture_clean.txt")
    print("\n  # 指定輸出檔案名稱")
    print("  python clean_transcript.py lecture.srt output.txt")
    print("\n  # 輸出到標準輸出（使用 --stdout）")
    print("  python clean_transcript.py lecture.srt --stdout")


def main():
    """主程式"""
    # 檢查參數
    if len(sys.argv) < 2:
        print("錯誤: 請提供 SRT 檔案路徑\n")
        print_usage()
        sys.exit(1)

    if sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)

    input_file = sys.argv[1]

    # 檢查是否使用 --stdout 選項
    if len(sys.argv) > 2 and sys.argv[2] == '--stdout':
        # 輸出到標準輸出
        output_file = None
        auto_output = False
    elif len(sys.argv) > 2:
        # 指定輸出檔案
        output_file = sys.argv[2]
        auto_output = False
    else:
        # 預設：自動在同目錄產生檔案
        output_file = None
        auto_output = True

    try:
        clean_srt_file(input_file, output_file, auto_output)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"錯誤: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"處理失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
