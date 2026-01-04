# Movie-trans - 实时与文件式语音翻译系统

一个功能丰富的翻译系统，支持基于文件的电影翻译和实时语音翻译（Demo形式），具有先进的AI功能，包括说话人分离、自动语音识别、大语言模型翻译和文本转语音合成。

## 功能特性

### 文件翻译流水线
- **视频处理**：从视频文件中提取音频并保持时间戳同步
- **音频增强**：使用UVR5技术对音频进行降噪和人声分离
- **说话人分离**：识别和分离不同的说话人
- **自动语音识别(ASR)**：支持多种语言（中文、英文、日语等）
- **机器翻译**：使用DeepSeek API提供高质量文本翻译
- **文本转语音(TTS)**：自然语音合成，保留原始说话人特征
- **标注界面**：基于Web的用户界面，用于审查和编辑翻译结果

### 实时翻译（demo）
- **实时音频捕获**：使用WASAPI环回捕获进行实时音频处理
- **流式ASR**：低延迟语音识别
- **即时翻译**：实时文本翻译
- **同步TTS**：实时语音合成，匹配原始说话人的语气
- **双语字幕**：实时显示原文和翻译文本的双语字幕

## 项目结构

```
Movie-trans/
├── main.py              # 带Gradio UI的主要文件翻译流水线
├── real-time/           # 实时翻译演示
│   ├── Program.cs       # .NET音频捕获和流客户端
│   └── model_server_streaming.py  # 实时处理的Python后端
├── asr/                 # 自动语音识别模型和工具
├── tools/               # 流水线处理工具
│   ├── process_video.py    # 视频转音频转换
│   ├── denoise.py          # 音频降噪
│   ├── speaker_diarization.py  # 说话人分离
│   ├── merge_speaker_segments.py  # 合并相邻片段
│   ├── test_clips.py      # 从片段创建音频剪辑
│   ├── asr.py            # ASR处理
│   └── annotate.py       # 基于Web的标注界面
├── index-tts/           # 带语音克隆功能的文本转语音系统
├── temp/                # 临时处理目录
└── results/             # 输出目录
```

## 安装

### 先决条件
- Python 3.8+
- .NET 9.0.300+ SDK（用于实时演示）
- 建议使用支持CUDA的GPU（以提高处理速度）

### 依赖安装

1. 安装Python依赖：
```bash
pip install -r requirements.txt
```

2. 对于实时翻译演示：
```bash
# 从 https://dotnet.microsoft.com/download/dotnet/9.0 下载并安装 .NET SDK 9.0.300
# 安装后验证是否正常工作：
dotnet --version

# 然后恢复 .NET 项目依赖
cd real-time
dotnet restore
```

3. 下载所需模型（**注意**：如下载失败，请检查网络连接或代理设置）：
   - **ASR模型**：首次运行时自动下载
   - **index-tts模型**：
     1. 从GitHub克隆项目到指定目录：
        ```bash
        cd d:\0Coding\pythonworkspace\My_program\Movie-trans
        git clone https://github.com/index-tts/index-tts.git
        ```
     2. 仔细阅读index-tts官网说明，从Hugging Face等平台下载所需模型文件
   - **pyannote模型**：
     1. 访问模型页面并接受授权：
        - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
        - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
     2. 登录Hugging Face并创建具有模型读取权限的访问令牌
     3. 修改`tools/speaker_diarization.py`文件中的以下内容：
        ```python
        # 设置HF访问令牌为环境变量
        HF_TOKEN = "YOUR_HF_TOKEN"  # 替换为你的Hugging Face令牌
        os.environ["HF_TOKEN"] = HF_TOKEN
        
        # 注释掉离线模式设置
        # os.environ["HF_HUB_OFFLINE"] = "1"
        ```
     4. 首次运行后可重新启用离线模式

## 使用方法

### 文件翻译

1. 启动主应用程序：
```bash
python main.py
```

2. Gradio Web界面将在浏览器中打开。

3. 按照流水线步骤操作：
   - **上传视频**：选择要处理的视频文件
   - **音频降噪**：提高音频质量并分离人声
   - **ASR处理**：识别语音并执行说话人分离
   - **标注**：审查和编辑转录内容（可选）
   - **翻译**：将文本翻译为目标语言
   - **TTS合成**：生成翻译后的语音
   - **整合视频**：将翻译后的语音与原始视频合并，可选择是否添加字幕及调整字幕形式

### 实时翻译（Demo）

1. 启动Python后端服务器：
```bash
cd real-time
python model_server_streaming.py
```

2. 在另一个终端中，启动.NET客户端：
```bash
cd real-time
dotnet run
```

3. 系统将自动捕获音频，实时处理，并输出翻译后的语音。

## 技术细节

### ASR模型
- **FunASR**：带VAD和标点的中文语音识别
- **Faster-Whisper**：多语言支持（英语、日语等）

### 音频处理
- **UVR5**：AI驱动的人声隔离
- **Librosa**：音频重采样和处理

### 翻译
- **DeepSeek API**：高质量神经机器翻译

### TTS
- **IndexTTS**：具有语音克隆功能的高级文本转语音
- **BigVGAN**：高保真音频合成

## 配置

### API密钥
在`main.py`和`real-time/model_server_streaming.py`中编辑API密钥：
```python
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```
如果你不想使用deepseek的api或者想要使用其他大模型的api，你可以在`main.py`和`real-time/model_server_streaming.py`中修改相关代码。

### 模型路径
在相应的配置文件中配置自定义模型路径：
- ASR模型：`asr/models/`
- TTS模型：`index-tts/checkpoints/`

## 性能考虑

- **GPU加速**：启用CUDA可显著提高速度
- **批量处理**：文件翻译以批处理方式处理片段以提高效率
- **实时延迟**：演示模式的延迟可能因硬件而异

## 限制

- 实时翻译目前处于演示模式，准确性有限
- 处理时间取决于视频长度和硬件能力
- 需要互联网连接以使用DeepSeek API进行翻译

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 贡献

欢迎贡献！请随时提交Pull Request。

## 致谢

- PyAnnote（说话人分离）
- FunASR和Whisper（语音识别）
- DeepSeek（翻译API）
- IndexTTS（语音合成）
- UVR5（人声隔离）
