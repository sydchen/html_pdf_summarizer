# YouTube 影片摘要功能實作說明

## 概述

已成功實作 YouTube 影片摘要功能，使用 Prefect 工作流程管理整個處理流程。

## 新增檔案

### 1. `config.py`
配置管理檔案，包含：
- Whisper 模型路徑和執行檔路徑
- YouTube 下載設定
- 音訊處理參數（取樣率 16kHz，單聲道）
- 所有設定可透過環境變數覆蓋

### 2. `transcript_llm.py`
專門用於逐字稿的摘要器：
- `TranscriptSummarizer` 類別，使用特製的系統提示詞
- 優化處理長逐字稿（自動分段處理）
- 支援串流和非串流模式
- 獨立於現有的 `DocumentSummarizer`

### 3. `youtube_summarizer.py`
YouTube 影片處理主模組：
- 使用 Prefect 管理工作流程
- 五個主要任務（每個都有重試機制）：
  1. **下載影片**: 使用 `yt-dlp` 下載 YouTube 影片為 MP4
  2. **轉換格式**: 使用 `ffmpeg` 轉換為 16kHz 單聲道
  3. **轉錄音訊**: 使用 Whisper CLI 生成 SRT 逐字稿
  4. **清理逐字稿**: 移除時間戳記，提取純文字
  5. **生成摘要**: 使用 Ollama 生成結構化摘要

### 4. 更新 `frontend.py`
- 整合 YouTube 摘要器
- 自動偵測 YouTube URL（支援 youtube.com 和 youtu.be）
- 在現有 URL 輸入欄位中自動路由到對應的摘要器
- 即時顯示處理進度

### 5. 更新 `requirements.txt`
新增依賴：
- `prefect==3.1.14` - 工作流程管理

注意：`yt-dlp`、`ffmpeg` 和 whisper.cpp 使用系統指令，不透過 `requirements.txt` 安裝。

## 使用方式

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 確認系統工具

確保系統已有以下指令：
- `ffmpeg` - 音訊轉換
- `yt-dlp` - YouTube 下載（使用系統指令）
- Whisper (whisper.cpp) - 音訊轉錄

### 3. 設定環境變數（選擇性）

建立 `.env` 檔案：

```env
# Whisper 設定（如使用非預設路徑）
WHISPER_MODEL_PATH=/path/to/whisper/model.bin
WHISPER_BINARY_PATH=/path/to/whisper-cli
WHISPER_LANGUAGE=auto

# 輸出目錄
YOUTUBE_OUTPUT_DIR=./youtube_downloads

# 檔案保留設定
KEEP_AUDIO_FILES=true
KEEP_TRANSCRIPT_FILES=true
```

### 4. 執行應用程式

```bash
streamlit run frontend.py
```

### 5. 使用 YouTube 功能

1. 在「🌐 網頁」區塊輸入 YouTube URL
2. 例如：`https://www.youtube.com/watch?v=NgrCQcU0Sbg`
3. 點擊「開始摘要網頁」按鈕
4. 系統會自動偵測為 YouTube URL 並執行完整流程
5. 即時顯示處理進度和最終摘要

## 工作流程

```
YouTube URL
    ↓
[1] 下載影片 (yt-dlp)
    ↓ {video_id}.mp4
[2] 轉換格式 (ffmpeg)
    ↓ {video_id}.wav (16kHz, mono)
[3] 轉錄音訊 (Whisper)
    ↓ {video_id}.wav.srt
[4] 清理逐字稿
    ↓ 純文字逐字稿
[5] 生成摘要 (Ollama)
    ↓ 結構化摘要
```

## 檔案存放

所有處理過程的檔案都會保存在 `youtube_downloads/` 目錄：
- `{video_id}.mp4` - 下載的影片
- `{video_id}.wav` - 轉換後的音訊（16kHz mono）
- `{video_id}.wav.srt` 或 `{video_id}.wav.{language}.srt` - SRT 格式逐字稿

## 容錯機制

- 所有 Prefect 任務都有重試機制（2 次重試，間隔 10 秒）
- 詳細的錯誤訊息和日誌輸出
- 自動處理長逐字稿（分段處理避免 token 限制）

## 特色功能

1. **自動偵測**: 輸入 URL 後自動識別 YouTube 影片
2. **進度追蹤**: 即時顯示每個步驟的處理狀態
3. **智慧摘要**: 針對逐字稿特性優化的摘要提示詞
4. **檔案保留**: 保留所有中間檔案供後續使用
5. **可擴展**: 基於 Prefect 的工作流程易於修改和擴展

## 注意事項

- 首次使用請確認 Whisper 模型路徑正確
- 處理長影片可能需要較長時間（特別是轉錄步驟）
- 確保有足夠的磁碟空間存放下載的音訊檔案
- YouTube 下載受 yt-dlp 和 YouTube 政策限制

## 疑難排解

### Whisper 找不到模型
```bash
# 確認路徑是否正確
ls -la /Users/sydchen/projects/asr/whisper.cpp/models/ggml-medium.bin
```

### ffmpeg 未安裝
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### yt-dlp 下載失敗
```bash
# 更新 yt-dlp
python3 -m pip install --upgrade yt-dlp
```

## 未來改進方向

- [x] 支援語言偵測與本地影片語言參數
- [ ] 批次處理多個 YouTube URL
- [ ] 支援 YouTube 播放清單
- [ ] 加入影片資訊擷取（標題、描述等）
- [ ] 提供摘要品質選項（簡短/詳細）
