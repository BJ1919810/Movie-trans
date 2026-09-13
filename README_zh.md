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

浏览器会打开 Gradio 界面，四个页签按流程走：

① 抽音频 & 降噪 → ② ASR & 切段 → ③ 翻译 & 标注 → ④ TTS & 合片。
每个页签底部的 **⚙️ 折叠区是高级参数**（降噪模型/激进程度、说话人聚类阈值、切段粒度、
ASR 设备与精度、TTS 音色策略/情绪强度/长度上限、合片按段对齐等），都已按最优值预设，
不确定就别动；常用的都摆在明面上。

**音频标注页**（③ 页签里启动）支持：原文 `raw_text` 与译文 `result_text` **双栏编辑**、
与已合成的 TTS **对照试听**、每段显示说话人/起止/时长、切分与合并片段。提交或翻页会自动写回 JSON。

TTS 环节的选项已集成到 UI（语言 / 情绪来源 / 语速）；也可以直接跑 `tools/batch_tts.py` 并用环境变量精细控制：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `TTS_LANG` | `ZH` | 合成语言：`ZH / EN / JA / ES / AR` |
| `SPK_REF_MODE` | `segment` | 音色参考：`segment` 每段用自身原片切片（同声配音正确解，推荐）/ `fixed` 每说话人固定一段（更快但音色情绪会漂移） |
| `TTS_EMO_MODE` | `ref` | 情绪来源：`ref` 跟随原片切片表演（推荐配音）/ `text` QwenEmotion 文本推断（适合有声书）/ `vector` 固定向量 / `none` |
| `TTS_EMO_ALPHA` | `1.0` | 情绪强度系数 |
| `TTS_EMO_VECTOR` | — | `vector` 模式的 8 维情绪向量，逗号分隔 |
| `TTS_DURATION_FACTOR` | `1.0` | 语速：>1 变慢，<1 变快 |
| `TTS_MAX_MEL_TOKENS` | `1815` | 单段生成长度上限（1815 = 2.5 上限）。调小会**静默截断**较长台词 |
| `USE_QWEN_EMO` | `false` | 加载 Qwen 情感模型（`text` 模式需要） |

回填视频（`tools/merge_tts_video_improved.py`）的**按段对齐**选项——TTS 时长很少正好等于原段时长，
不对齐就会留空档或盖到下一句上，所以默认用 atempo（保音高）把每段拉到原段时长：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ALIGN_TTS` | `true` | 是否启用按段时长对齐 |
| `ALIGN_MAX_RATE` | `1.25` | 最大伸缩倍率（语速最多变 ±25%，防止为对齐把语速拉变形） |
| `ALIGN_MIN_DEV` | `0.05` | 时长偏差阈值，5% 以内不动 |
| `ALIGN_TRIM_OVERFLOW` | `false` | 对齐后仍超长的段是否裁到段长；`false` 时只在日志汇总告警 |
| `TTS_FADE_MS` | `15` | 每段淡入淡出毫秒数，避免硬切爆音 |
| `TTS_LOUDNESS_MATCH` | `true` | 逐段把 TTS 响度**对齐到该段原声切片的 RMS**（峰值≠响度：旧的按峰值归一化会让"闷响/密度高"的段平白响 ~10 dB） |
| `TTS_LOUDNESS_OFFSET` | `0.0` | 对齐后再叠加的偏移 dB，想整体更响就调正 |
| `TTS_PEAK_CEIL` | `-1.0` | 峰值上限 dBFS，超过就整体下压（防叠加削顶） |
| `TTS_MIN_RMS` | `-30.0` | 目标响度下限 dBFS，防止原声极轻的段被对齐到听不见 |
| `OUTPUT_SR` | `44100` | 成片音频采样率。**别删也别乱改**：ffmpeg 的 loudnorm 不指定 `-ar` 会把输出落成 192kHz，AAC 编码器上限 96kHz，成片就会变成非常规的 96k 音轨（实测同码率 SNR 低 3.1 dB） |

全片响度归一化（`loudnorm I=-16:TP=-1.5:LRA=11`）只在合片前做**一次**，封装时不再重复，避免二次压缩动态。

### 实时翻译（Demo）

```bash
cd real-time
python model_server_streaming.py   # 终端 1：Python 后端
dotnet run                         # 终端 2：.NET 客户端
```

## 技术细节

- **ASR**：FunASR Paraformer（中，VAD+标点）/ Faster-Whisper large-v3（多语言）
- **说话人分离**：pyannote segmentation-3.0 + wespeaker
- **人声分离**：UVR5（MDX-Net / BS-Roformer）
- **翻译**：DeepSeek Chat API
- **TTS**：[IndexTTS-2.5](https://github.com/index-tts/index-tts)（IndexTeam），BigVGAN 声码器；分词器为多语言 tiktoken，日语经 fugashi G2P 转读音

## 常见问题

- **TTS 输出是噪声 / 又轻又糊（最坑的一个）**：多半是**同时跑了两个 TTS 进程**。
  8GB 显卡上并发抢显存**不会**抛 CUDA OOM，而是静默降级——输出电平比正常低约 20dB、
  部分分段直接是数字静音，听感就是"严重失真的噪声"（同一批文本单进程重跑即 24/24 正常）。
  所以 TTS 一律**单进程线性跑**：批量合成前先确认没有别的实例在占显存
  （本脚本、`main.py` 的 UI、`real-time/` 服务三者互斥）。
- **参考音频怎么选**：本项目是同声配音，默认 `SPK_REF_MODE=segment`——
  每段译文用**它自己那段原片切片**同时作为音色与情绪参考，逐段跟随原声的麦位/音量/表演。
  改成 `fixed`（每说话人固定一段）只是提速近似，音色与情绪会偏离该段原声。
- **bigvgan CUDA 内核加载失败（提示 Ninja is required）**：正常现象，会自动回退到
  torch 实现，音质不受影响（仅慢一点点）。
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
