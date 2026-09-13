# Movie-trans — 影片翻译 & 实时语音翻译系统

一个端到端的视频配音翻译系统：视频 → 人声分离 → 说话人分离 → ASR → LLM 翻译 → TTS 配音 → 回填视频。
TTS 采用 **IndexTTS-2.5**（支持中/英/日/西/阿拉伯语，音色克隆 + 情绪控制），另附实时翻译 Demo。

## 功能特性

### 文件翻译流水线（主功能）
- **视频处理**：从视频提取音频，时间戳全程对齐
- **音频增强**：UVR5 人声分离 / 降噪
- **说话人分离**：pyannote 识别并拆分多说话人
- **语音识别（ASR）**：FunASR（中文）/ Faster-Whisper（多语言）
- **机器翻译**：DeepSeek API（目标语言可选中/英/日）
- **TTS 配音**：IndexTTS-2.5 语音克隆，支持 5 种语言（`ZH / EN / JA / ES / AR`）
  - **音色与情绪解耦**：音色用每说话人固定参考（embedding 缓存命中，快且稳定），情绪默认跟随原片切片的韵律表演
  - 情绪来源可选：原片参考音频 / QwenEmotion 文本推断 / 8 维情绪向量
  - 语速控制（`duration_factor`）、日语 G2P（fugashi，汉字自动注音）
- **整合回填**：翻译语音与原视频合成，可选字幕
- **Web UI**：Gradio 界面串起全流程，TTS 日志实时滚动

### 实时翻译（Demo）
- WASAPI 环回捕获 → 流式 ASR → 实时翻译 → 同步 TTS + 双语字幕（Windows，.NET 客户端 + Python 后端）

## 项目结构

```
Movie-trans/
├── main.py                  # 主流水线（Gradio UI）
├── env_config.py            # .env 密钥读取（无第三方依赖）
├── .env.example             # 密钥模板（复制为 .env 并填写）
├── Download_indextts25.py   # IndexTTS-2.5 权重下载（ModelScope，国内直连）
├── Download_models.py       # ASR / pyannote / 辅助模型下载（HuggingFace）
├── tools/                   # 流水线各环节脚本
│   ├── process_video.py     #   视频转音频
│   ├── denoise.py           #   UVR5 人声分离降噪
│   ├── speaker_diarization.py  # 说话人分离
│   ├── merge_speaker_segments.py  # 合并相邻片段
│   ├── test_clips.py        #   切分参考音频片段
│   ├── asr.py               #   ASR 处理
│   ├── translate.py         #   LLM 翻译
│   ├── batch_tts.py         #   批量 TTS（语言/情绪/音色策略可配）
│   ├── merge_tts_video_improved.py  # 语音回填视频
│   └── annotate.py          #   Web 标注界面
├── asr/                     # ASR 封装（FunASR / Faster-Whisper）
├── uvr5/                    # UVR5 人声分离（权重需单独下载）
├── index-tts/               # IndexTTS-2.5 源码（含本地定制），权重需单独下载
├── real-time/               # 实时翻译 Demo（.NET 客户端 + Python 后端）
├── temp/                    # 中间产物（运行时生成）
└── results/                 # 输出目录
```

## 安装

### 先决条件
- Python 3.10+（建议 3.12）
- CUDA GPU（强烈建议，TTS/ASR 均可 GPU 加速）
- .NET 9.0 SDK（仅实时 Demo 需要）
- ffmpeg

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入：
#   DEEPSEEK_API_KEY  — 翻译用（https://platform.deepseek.com/）
#   HF_TOKEN          — 下载 pyannote 模型用（https://huggingface.co/settings/tokens）
```

### 3. 下载模型

**IndexTTS-2.5 权重**（走 ModelScope，国内速度快，约 5.5GB）：

```bash
python Download_indextts25.py
```

脚本带字节校验，中断重跑会续传/覆盖不完整文件。

**ASR / pyannote 模型**（走 HuggingFace，pyannote 需先在 `.env` 配好 `HF_TOKEN`）：

```bash
python Download_models.py
```

UVR5 权重已内置 `HP2-all-vocals`（`uvr5/uvr5_weights/`）；FunASR 模型首次运行会自动下载。

## 使用

### 文件翻译

```bash
python main.py
```

浏览器会打开 Gradio 界面，按流程操作：上传视频 → 降噪/人声分离 → ASR + 说话人分离 → 标注校对 → 翻译 → TTS 合成 → 回填视频。

TTS 环节的选项已集成到 UI（语言 / 推理器版本 / 情绪来源 / 语速）；也可以直接跑 `tools/batch_tts.py` 并用环境变量精细控制：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `TTS_LANG` | `ZH` | 合成语言：`ZH / EN / JA / ES / AR` |
| `INDEXTTS_VERSION` | `2.5` | 推理器版本（`2` 可回退旧版） |
| `SPK_REF_MODE` | `fixed` | 音色参考：`fixed` 每说话人固定一段（缓存命中，推荐）/ `segment` 每段各自当参考 |
| `TTS_EMO_MODE` | `ref` | 情绪来源：`ref` 跟随原片切片表演（推荐配音）/ `text` QwenEmotion 文本推断（适合有声书）/ `vector` 固定向量 / `none` |
| `TTS_EMO_ALPHA` | `1.0` | 情绪强度系数 |
| `TTS_EMO_VECTOR` | — | `vector` 模式的 8 维情绪向量，逗号分隔 |
| `TTS_DURATION_FACTOR` | `1.0` | 语速：>1 变慢，<1 变快 |
| `USE_QWEN_EMO` | `false` | 加载 Qwen 情感模型（`text` 模式需要） |

### 实时翻译（Demo）

```bash
cd real-time
python model_server_streaming.py   # 终端 1：Python 后端
dotnet run                         # 终端 2：.NET 客户端
```

## 演示示例

`examples/` 内含演示视频：

- `English-raw.mp4` — 原片片段
- `Chinese-result.mp4` — 翻译配音结果
- `live-demo.mp4` — 实时翻译 Demo 录屏

## 技术细节

- **ASR**：FunASR Paraformer（中，VAD+标点）/ Faster-Whisper large-v3（多语言）
- **说话人分离**：pyannote segmentation-3.0 + wespeaker
- **人声分离**：UVR5（MDX-Net / BS-Roformer）
- **翻译**：DeepSeek Chat API
- **TTS**：[IndexTTS-2.5](https://github.com/index-tts/index-tts)（IndexTeam），BigVGAN 声码器；分词器为多语言 tiktoken，日语经 fugashi G2P 转读音

## 常见问题

- **TTS 报 transformers 相关 ImportError**：本仓库锁定 `transformers==4.52.1` + `huggingface_hub==0.34.4`，请勿随意升级（5.x 移除了 IndexTTS 依赖的 API）。
- **环境里有 tensorflow 导致启动崩溃**：本项目不需要 TensorFlow，直接 `pip uninstall tensorflow`。
- **显存不足**：2.5 推理器默认 bf16；Qwen 情绪模型（约 1.2GB）按需开启。

## 许可证

MIT License，详见 [LICENSE](LICENSE)。`index-tts/` 子目录遵循上游 IndexTTS 的许可条款。

## 致谢

- [IndexTTS](https://github.com/index-tts/index-tts)（B 站 IndexTeam）— 语音合成
- [FunASR](https://github.com/modelscope/FunASR)、[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — 语音识别
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — 说话人分离
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) — 人声分离
- [DeepSeek](https://www.deepseek.com/) — 翻译 API
- [ModelScope](https://www.modelscope.cn/) — 模型分发
