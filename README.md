# Movie-trans - Real-time and File-based Speech Translation System

A versatile translation system supporting both file-based movie translation and real-time voice translation (demo mode), with advanced AI-powered features including speaker diarization, automatic speech recognition, large language model translation, and text-to-speech synthesis.

## Features

### File Translation Pipeline
- **Video Processing**: Extract audio from video files with synchronized timestamps
- **Audio Enhancement**: Denoise and isolate vocals using UVR5 technology
- **Speaker Diarization**: Identify and separate different speakers
- **Automatic Speech Recognition (ASR)**: Support for multiple languages (Chinese, English, Japanese, etc.)
- **Machine Translation**: High-quality text translation using DeepSeek API
- **Text-to-Speech (TTS)**: Natural-sounding voice synthesis preserving original speaker characteristics
- **Annotation Interface**: Web-based UI for reviewing and editing translation results

### Real-time Translation (Demo)
- **Live Audio Capture**: Real-time audio processing using WASAPI loopback capture
- **Streaming ASR**: Low-latency speech recognition
- **Instant Translation**: Real-time text translation
- **Synchronized TTS**: Real-time voice synthesis matching original speaker's tone

## Project Structure

```
Movie-trans/
├── main.py              # Main file translation pipeline with Gradio UI
├── real-time/           # Real-time translation demo
│   ├── Program.cs       # .NET audio capture and streaming client
│   └── model_server_streaming.py  # Python backend for real-time processing
├── asr/                 # Automatic Speech Recognition models and utilities
├── tools/               # Pipeline processing tools
│   ├── process_video.py    # Video to audio conversion
│   ├── denoise.py          # Audio denoising
│   ├── speaker_diarization.py  # Speaker separation
│   ├── merge_speaker_segments.py  # Merge adjacent segments
│   ├── test_clips.py      # Create audio clips from segments
│   ├── asr.py            # ASR processing
│   └── annotate.py       # Web-based annotation interface
├── index-tts/           # Text-to-Speech system with voice cloning
├── temp/                # Temporary processing directory
└── results/             # Output directory
```

## Installation

### Prerequisites
- Python 3.8+
- .NET Core 5.0+ (for real-time demo)
- CUDA-capable GPU recommended (for faster processing)

### Dependencies Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. For real-time translation demo:
```bash
# Install .NET dependencies
cd real-time
dotnet restore
```

3. Download required models (**Note**: If downloads fail, check network connection or proxy settings):
   - **ASR models**: Automatically downloaded on first run
   - **index-tts models**:
     1. Clone the project from GitHub to the specified directory:
        ```bash
        cd d:\0Coding\pythonworkspace\My_program\Movie-trans
        git clone https://github.com/index-tts/index-tts.git
        ```
     2. Carefully read the index-tts official instructions and download required model files from Hugging Face or other platforms
   - **pyannote models**:
     1. Visit model pages and accept authorization:
        - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
        - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
     2. Log in to Hugging Face and create an access token with model read permissions
     3. Modify the following content in `tools/speaker_diarization.py`:
        ```python
        # Set HF access token as environment variable
        HF_TOKEN = "YOUR_HF_TOKEN"  # Replace with your Hugging Face token
        os.environ["HF_TOKEN"] = HF_TOKEN
        
        # Comment out offline mode setting
        # os.environ["HF_HUB_OFFLINE"] = "1"
        ```
     4. You can re-enable offline mode after first run

## Usage

### File Translation

1. Start the main application:
```bash
python main.py
```

2. The Gradio web interface will open in your browser.

3. Follow the pipeline steps:
   - **Upload Video**: Select a video file to process
   - **Denoise Audio**: Enhance audio quality and isolate vocals
   - **ASR Processing**: Recognize speech and perform speaker diarization
   - **Annotation**: Review and edit transcriptions (optional)
   - **Translation**: Translate text to target language
   - **TTS Synthesis**: Generate translated speech
   - **Integrate Video**: Merge the translated voice with the original video, and choose whether to add subtitles and adjust the format

### Real-time Translation (Demo)

1. Start the Python backend server:
```bash
cd real-time
python model_server_streaming.py
```

2. In another terminal, start the .NET client:
```bash
cd real-time
dotnet run
```

3. The system will automatically capture audio, process it in real-time, and output translated speech.

## Technical Details

### ASR Models
- **FunASR**: Chinese speech recognition with VAD and punctuation
- **Faster-Whisper**: Multi-language support (English, Japanese, etc.)

### Audio Processing
- **UVR5**: AI-powered vocal isolation
- **Librosa**: Audio resampling and manipulation

### Translation
- **DeepSeek API**: High-quality neural machine translation

### TTS
- **IndexTTS**: Advanced text-to-speech with voice cloning capabilities
- **BigVGAN**: High-fidelity audio synthesis

## Configuration

### API Keys
Edit the API key in `main.py` and `real-time/model_server_streaming.py`:
```python
DEEPSEEK_API_KEY = "your-api-key-here"
```
If you do not want to use DeepSeek's API or wish to utilize the API of other large language models, you can modify the relevant code in `main.py` and `real-time/model_server_streaming.py`.

### Model Paths
Configure custom model paths in the respective configuration files:
- ASR models: `asr/models/`
- TTS models: `index-tts/checkpoints/`

## Performance Considerations

- **GPU Acceleration**: Enable CUDA for significant speed improvements
- **Batch Processing**: File translation processes segments in batches for efficiency
- **Real-time Latency**: Demo mode may have noticeable latency depending on hardware

## Limitations

- Real-time translation is currently in demo mode with limited accuracy
- Processing time depends on video length and hardware capabilities
- Internet connection required for DeepSeek API translation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyAnnote for speaker diarization
- FunASR and Whisper for speech recognition
- DeepSeek for translation API
- IndexTTS for voice synthesis
- UVR5 for vocal isolation
