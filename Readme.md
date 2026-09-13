# Movie-trans — Movie Dubbing Translation & Real-time Speech Translation

An end-to-end video dubbing pipeline: video → vocal isolation → speaker diarization → ASR → LLM translation → TTS dubbing → remux back into video.
Powered by **IndexTTS-2.5** (voice cloning + emotion control, supporting ZH/EN/JA/ES/AR), with a real-time translation demo included.

## Features

### File Translation Pipeline (main)
- **Video processing**: audio extraction with timestamp alignment throughout
- **Audio enhancement**: UVR5 vocal isolation / denoising
- **Speaker diarization**: multi-speaker separation with pyannote
- **ASR**: FunASR (Chinese) / Faster-Whisper (multilingual)
- **Machine translation**: DeepSeek API (target language: zh / en / ja)
- **TTS dubbing**: IndexTTS-2.5 voice cloning, 5 languages (`ZH / EN / JA / ES / AR`)
  - **Timbre / emotion decoupling**: fixed per-speaker reference audio (embedding cache hits, fast and stable); emotion follows the original clip's prosody by default
  - Emotion sources: original-audio reference / QwenEmotion text inference / 8-dim emotion vector
  - Speaking-rate control (`duration_factor`), Japanese G2P via fugashi
- **Remux**: merge dubbed speech back into the original video, optional subtitles
- **Web UI**: Gradio app wiring the whole pipeline together, with live-streaming TTS logs

### Real-time Translation (Demo)
- WASAPI loopback capture → streaming ASR → live translation → synchronized TTS + bilingual subtitles (Windows, .NET client + Python backend)

## Project Structure

```
Movie-trans/
├── main.py                  # Main pipeline (Gradio UI)
├── env_config.py            # .env loader (no third-party deps)
├── .env.example             # API key template (copy to .env)
├── Download_indextts25.py   # IndexTTS-2.5 weights downloader (ModelScope)
├── Download_models.py       # ASR / pyannote / auxiliary models (HuggingFace)
├── tools/                   # Pipeline stage scripts
│   ├── process_video.py     #   video → audio
│   ├── denoise.py           #   UVR5 vocal isolation
│   ├── speaker_diarization.py  # speaker separation
│   ├── merge_speaker_segments.py  # merge adjacent segments
│   ├── test_clips.py        #   reference-clip extraction
│   ├── asr.py               #   ASR
│   ├── translate.py         #   LLM translation
│   ├── batch_tts.py         #   batch TTS (language/emotion/timbre policies)
│   ├── merge_tts_video_improved.py  # remux dubbed audio into video
│   └── annotate.py          #   web annotation UI
├── asr/                     # ASR wrappers (FunASR / Faster-Whisper)
├── uvr5/                    # UVR5 vocal separation (weights downloaded separately)
├── index-tts/               # IndexTTS-2.5 source (vendored with local patches)
├── real-time/               # Real-time demo (.NET client + Python backend)
├── temp/                    # Intermediate artifacts (generated at runtime)
└── results/                 # Output directory
```

## Installation

### Prerequisites
- Python 3.10+ (3.12 recommended)
- CUDA GPU (strongly recommended)
- .NET 9 SDK (real-time demo only)
- ffmpeg

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env:
#   DEEPSEEK_API_KEY  — translation (https://platform.deepseek.com/)
#   HF_TOKEN          — pyannote model download (https://huggingface.co/settings/tokens)
```

### 3. Download models

**IndexTTS-2.5 weights** (via ModelScope, ~5.5 GB):

```bash
python Download_indextts25.py
```

The script verifies file sizes, so re-running it after an interruption resumes/repairs incomplete downloads.

**ASR / pyannote models** (via HuggingFace; set `HF_TOKEN` in `.env` first):

```bash
python Download_models.py
```

UVR5 weights ship with the repo (`uvr5/uvr5_weights/`, HP2-all-vocals); FunASR models auto-download on first run.

## Usage

### File translation

```bash
python main.py
```

A Gradio UI opens in your browser: upload video → denoise / vocal isolation → ASR + diarization → annotate → translate → TTS → remux.

TTS options are built into the UI (language / inference version / emotion source / speaking rate). You can also run `tools/batch_tts.py` directly with environment variables:

| Variable | Default | Description |
|---|---|---|
| `TTS_LANG` | `ZH` | Synthesis language: `ZH / EN / JA / ES / AR` |
| `INDEXTTS_VERSION` | `2.5` | Inference version (`2` = legacy) |
| `SPK_REF_MODE` | `fixed` | Timbre reference: `fixed` per-speaker clip (cache-friendly, recommended) / `segment` per-segment |
| `TTS_EMO_MODE` | `ref` | Emotion source: `ref` follow original performance (dubbing) / `text` QwenEmotion inference (audiobooks) / `vector` / `none` |
| `TTS_EMO_ALPHA` | `1.0` | Emotion intensity |
| `TTS_EMO_VECTOR` | — | 8-dim emotion vector for `vector` mode (comma-separated) |
| `TTS_DURATION_FACTOR` | `1.0` | Speaking rate: >1 slower, <1 faster |
| `USE_QWEN_EMO` | `false` | Load the Qwen emotion model (required by `text` mode) |

### Real-time translation (Demo)

```bash
cd real-time
python model_server_streaming.py   # terminal 1: Python backend
dotnet run                         # terminal 2: .NET client
```

## Technical Details

- **ASR**: FunASR Paraformer (zh, VAD + punctuation) / Faster-Whisper large-v3 (multilingual)
- **Diarization**: pyannote segmentation-3.0 + wespeaker
- **Vocal isolation**: UVR5 (MDX-Net / BS-Roformer)
- **Translation**: DeepSeek Chat API
- **TTS**: [IndexTTS-2.5](https://github.com/index-tts/index-tts) (IndexTeam) with BigVGAN vocoder; multilingual tiktoken tokenizer, Japanese readings via fugashi G2P

## Troubleshooting

- **transformers ImportError in TTS**: this repo pins `transformers==4.52.1` + `huggingface_hub==0.34.4` — do not upgrade (5.x removed APIs IndexTTS depends on).
- **Crash on startup with tensorflow installed**: this project does not need TensorFlow; run `pip uninstall tensorflow`.
- **Out of VRAM**: the 2.5 inferencer defaults to bf16; enable the Qwen emotion model (~1.2 GB) only when needed.

## License

MIT License — see [LICENSE](LICENSE). The `index-tts/` subdirectory follows the upstream IndexTTS license.

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) (IndexTeam @ Bilibili) — speech synthesis
- [FunASR](https://github.com/modelscope/FunASR), [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — ASR
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) — vocal isolation
- [DeepSeek](https://www.deepseek.com/) — translation API
- [ModelScope](https://www.modelscope.cn/) — model hosting
