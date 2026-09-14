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

**ASR / pyannote models + IndexTTS auxiliary models** (via HuggingFace; set `HF_TOKEN` in `.env` first):

```bash
python Download_models.py
```

#### Where models live (single canonical location — do not change)

| Content | Location |
|---|---|
| IndexTTS-2.5 main weights (`gpt.pth` / `s2mel.pth` / `codec.pth` / `config.yaml` …) | `index-tts/checkpoints/` |
| IndexTTS **auxiliary models**, flat layout (`w2v-bert-2.0/`, `bigvgan/`, `campplus_cn_common.bin`, `semantic_codec_model.safetensors`) | `index-tts/checkpoints/hf_cache/` |
| ASR / pyannote / Whisper / FunASR | `asr/models/` |
| Vocal-separation weights | `uvr5/uvr5_weights/` |

Two rules:

1. The **only** canonical location for auxiliary models is `index-tts/checkpoints/hf_cache/` — that is
   `{model_dir}/hf_cache` as resolved by `ensure_models_available(model_dir)`, and it is what the runtime
   actually reads. Both download scripts write there and skip anything already present.
2. **Never download into `<project-root>/checkpoints/`.** That was a legacy second location holding a
   duplicate copy of the same models (3.56 GB), removed on 2026-09-14 after a per-file `sha256` comparison
   confirmed byte-identical content. No runtime code points there any more.

UVR5 weights must be provided in `uvr5/uvr5_weights/`: the default `roformer` engine needs
BS-Roformer (see below), the fallback `uvr5` engine needs `HP2_all_vocals.pth`.
FunASR models auto-download on first run.

### Vocal separation: two engines (BS-Roformer by default)

`tools/denoise.py` supports two engines with identical output filenames, so nothing downstream changes:

| `--engine` | Model | Weights | Quality / speed |
|---|---|---|---|
| **`roformer` (default)** | **BS-Roformer** `model_bs_roformer_ep_317_sdr_12.9755` | 610 MB (see below) | ~53 s for 121 s of audio on an RTX 3070. **Best in A/B listening**: least high-frequency residue, sibilance and breath preserved |
| `uvr5` | VR arch `HP2_all_vocals` | 60 MB | Faster and smaller, but noticeably more residue on dialogue over loud BGM |

```bash
# roformer is the default — just run it
python tools/denoise.py --input temp/output_audio.wav
# switch back to the old engine
python tools/denoise.py --input temp/output_audio.wav --engine uvr5 --model-name HP2_all_vocals
```

**Where to put the weights**: `uvr5/uvr5_weights/`, with filenames matching the default model names in
`tools/denoise.py` (the roformer weight must contain `bs_roformer` or `mel_band_roformer` to be auto-detected).

**Both engines' weights are fetched by `Download_models.py`** (too large for git, so **not committed**):

```bash
python Download_models.py    # downloads into uvr5/uvr5_weights/ (skips when present and sha256 matches)
```

| Engine | Files | Size |
|---|---|---|
| `roformer` (default) | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` + same-named `.yaml` | 610 MB + 2 KB |
| `uvr5` (fallback) | `HP2_all_vocals.pth` | 60 MB |

Manual download (these are the exact URLs the script uses; `huggingface.co` is not directly reachable, use the mirror):

```bash
# BS-Roformer weights, 610MB
curl -L -o uvr5/uvr5_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  "https://hf-mirror.com/Sucial/MSST-WebUI/resolve/main/All_Models/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# config
curl -L -o uvr5/uvr5_weights/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
# VR-arch HP2 (uvr5 engine), 60MB
curl -L -o uvr5/uvr5_weights/HP2_all_vocals.pth \
  "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth"
```

Checksums (`sha256`):
- BS-Roformer `.ckpt`: `5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa`
- BS-Roformer `.yaml`: `2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f`
- `HP2_all_vocals.pth`: `39796caa5db18d7f9382d8ac997ac967bfd85f7761014bb807d2543cc844ef05`

**Note**: the roformer engine **requires CUDA** (`uvr5/bsroformer.py` hardcodes `torch.amp.autocast("cuda")`)
and defaults to half precision (`--fp32` disables it). `--agg` only applies to the uvr5 engine.

> ⚠️ **Do not add "denoise / de-reverb / high-frequency cleanup" post-processing** (settled 2026-09-14).
> Reason 1: denoise / de-reverb do **full-band reconstruction**, repainting the voice's harmonic tails and
> room reverb together → the result sounds **dry**.
> Reason 2: even a pure spectral-threshold variant ("only cut broadband residue exceeding the voice
> envelope") fails — the `HF(3-10k)/voice-core` ratio **cannot separate birdsong from ordinary sibilance**
> (the distributions of normal and residue-heavy regions almost overlap), so it cut 8–11 dB of
> high frequency across the whole film, sounding just as dull. That code has been removed.

## Usage

### File translation

```bash
python main.py
```

A Gradio UI opens in your browser with four tabs following the pipeline:

① Extract & Denoise → ② ASR & Segmentation → ③ Translate & Annotate → ④ TTS & Mux.
Each tab has a collapsed **⚙️ advanced section** (denoise model/aggressiveness, diarization
clustering, segment granularity, ASR device/precision, TTS voice strategy / emotion strength /
length cap, per-segment alignment, ...). All are pre-set to the recommended values — leave them
alone unless you know why you are changing them. The parameters you actually touch are up front.

The **annotation page** (launched from tab ③) lets you edit both `raw_text` and `result_text`
side by side, A/B listen against the synthesized TTS of that segment, see speaker / start / end /
duration per row, and split or merge segments. Submitting or paging writes back to the JSON automatically.

TTS options are built into the UI (language / emotion source / speaking rate). You can also run `tools/batch_tts.py` directly with environment variables:

| Variable | Default | Description |
|---|---|---|
| `TTS_LANG` | `ZH` | Synthesis language: `ZH / EN / JA / ES / AR` |
| `SPK_REF_MODE` | `segment` | Timbre reference: `segment` per-segment original clip (correct for dubbing, recommended) / `fixed` one clip per speaker (faster but timbre/emotion drift) |
| `TTS_EMO_MODE` | `ref` | Emotion source: `ref` follow original performance (dubbing) / `text` QwenEmotion inference (audiobooks) / `vector` / `none` |
| `TTS_EMO_ALPHA` | `1.0` | Emotion intensity |
| `TTS_EMO_VECTOR` | — | 8-dim emotion vector for `vector` mode (comma-separated) |
| `TTS_DURATION_FACTOR` | `1.0` | Speaking rate: >1 slower, <1 faster |
| `TTS_MAX_MEL_TOKENS` | `1815` | Max generation length per segment (1815 = the 2.5 ceiling). Lower values **silently truncate** longer lines |
| `TTS_USE_CUDA_KERNEL` | `false` | BigVGAN's custom CUDA kernel. **Leave it off**: `ninja` is not installed here, and enabling it makes model loading **hang silently** (symptom: one `GPT2InferenceModel has generative capabilities...` line, then nothing — the UI looks idle). The torch fallback is **numerically equivalent with no quality loss**, just a slightly slower vocoder (measured: 15 s init, 2.8 s per line) |
| `USE_QWEN_EMO` | `false` | Load the Qwen emotion model (required by `text` mode) |

**Per-segment alignment** options for the remux step (`tools/merge_tts_video_improved.py`) — TTS duration
rarely matches the original segment exactly, which leaves gaps or bleeds into the next line, so each segment
is time-stretched to its original length with atempo (pitch preserved):

| Variable | Default | Description |
|---|---|---|
| `ALIGN_TTS` | `true` | Enable per-segment duration alignment |
| `ALIGN_MAX_RATE` | `1.25` | Max stretch factor (speaking rate changes at most ±25%) |
| `ALIGN_MIN_DEV` | `0.05` | Deviation threshold; anything under 5% is left untouched |
| `ALIGN_TRIM_OVERFLOW` | `false` | Trim segments that are still too long; when `false` they are only reported in a summary |
| `TTS_FADE_MS` | `15` | Fade in/out per segment (ms) to avoid hard-cut clicks |
| `BG_PAD_MS` | `200` | When the original voice is replaced by the instrumental, extend each segment by this many ms on both sides — this **covers the original sentence-tail breath that falls outside the segment** (otherwise the dub ends and the original's breath follows). Neighbouring segments are protected (half the gap); `0` restores the old behaviour |
| `TTS_LOUDNESS_MATCH` | `true` | Match each segment's loudness to **the RMS of its own original clip** (peak ≠ loudness: the old peak-normalize made dense/boomy segments ~10 dB louder for free) |
| `TTS_LOUDNESS_OFFSET` | `0.0` | Extra offset (dB) applied after matching; use a positive value for an overall louder dub |
| `TTS_PEAK_CEIL` | `-1.0` | Peak ceiling (dBFS); anything above is pulled down to avoid mix clipping |
| `TTS_MIN_RMS` | `-30.0` | Lower bound for the target loudness (dBFS), so very quiet originals don't become inaudible |
| `OUTPUT_SR` | `44100` | Output audio sample rate. **Do not remove or change lightly**: without an explicit `-ar`, ffmpeg's loudnorm writes its output at 192 kHz, and since the AAC encoder caps at 96 kHz the film ends up with an unusual 96 kHz track (measured 3.1 dB worse SNR at the same bitrate) |
| `LOUDNESS_TARGET` | `source` | Target loudness of the whole film. `source` = **follow the original's measured loudness** (recommended; the film ends up at the original's level); a number pins it, e.g. `-16` or `-23` |
| `LOUDNESS_TP` | `-1.5` | True-peak ceiling (dBTP) for the whole-film normalization |
| `LOUDNESS_LRA` | `11` | Loudness range (in `linear` mode it only participates in measurement, it does not reshape dynamics) |

Whole-film loudness normalization uses **two-pass `linear`** (the first pass measures, the second applies a
**single constant gain**), with the target defaulting to **the original's measured loudness**. It runs
**once** before muxing and is not repeated during muxing.

> **Do not revert to single-pass `loudnorm=I=-16:TP=-1.5:LRA=11`**: single-pass runs in `dynamic` mode and
> applies a **time-varying** gain — the quiet opening (pure BGM, no dialogue) gets pushed ~10 dB further,
> while the fixed −16 LUFS target sits far above the material itself (the original measures only −29.3 LUFS),
> pumping +13.9 dB into the whole film. It sounds like "the dub's BGM is far louder than the original,
> especially at the start", with the voice going loud too.
> Measured on a 121 s clip (metric = opening−global loudness ratio): old single-pass **−0.18 dB** →
> new two-pass **−11.36 dB** (original baseline **−9.83 dB**), with the global level at −33.65 dBFS
> matching the original's −33.64 dBFS.

### Reference clip health check (ASR stage, **diagnostics only**)

The TTS reference audio is simply **each segment's own original clip** (`SPK_REF_MODE=segment`, the correct
choice for simultaneous dubbing). If a clip carries original BGM residue or noise, the model learns those as
part of the timbre and performance (audible as **gasping delivery, occasional blow-ups, muddy tone**). After
ASR you can print a health table (CPU only, seconds, **read-only**) that flags **which segments' clips are dirty**:

| Metric | Meaning |
|---|---|
| `pause drop` | Level difference between the loudest 20% frames and the "pause band" ← **key metric**. Clean speech drops deep in pauses (measured 37–54 dB in this film); BGM residue only drops 20–32 dB |
| `speech ratio` | Share of frames more than 6 dB above the pause band (too low = mostly silence/noise) |
| `band` | Share of energy in the 0.3–4 kHz speech band |
| `duration` / clipping | Clips outside 0.6–8 s, or clipping, are unsuitable references |

```bash
python tools/asr.py --only-ref-check     # standalone: no ASR model, existing text untouched
python tools/asr.py --ref-check          # run right after ASR
```

The check performs **no** selection, fallback or cleaning, and writes nothing to the JSON (the file's `md5`
is unchanged before and after). Two reasons (settled 2026-09-14):

1. **Selection/fallback must pool clips by speaker label, and pyannote's multi-speaker labelling is not
   trustworthy** (tested on Chinese, English and Japanese; it even merged male and female voices into one
   speaker). Once a label is wrong, the fallback amplifies the error — e.g. the female lead's timbre applied
   to a male segment. The risk outweighs the benefit.
2. **Reference cleaning lost an A/B listening test — the original reference won on all three samples**: the
   −6 dB bell at 120 Hz cuts the male fundamental (i.e. the timbre itself), `afftdn` leaves metallic artefacts
   the model picks up, and windowing throws away the performance arc the emotion reference needs.

> To actually fix dirty references, go after **vocal separation quality** (e.g. switch to a SOTA model such
> as BS-Roformer / Mel-Band Roformer), or delete/merge those segments in the annotation UI and re-synthesize
> them individually.

| Variable | Default | Description |
|---|---|---|
| `REF_PAUSE_DROP_MIN` | `30` | Minimum pause drop (dB); below this a clip is flagged dirty |
| `REF_SPEECH_RATIO_MIN` | `0.5` | Minimum speech-frame ratio |
| `REF_MIN_SEC` / `REF_MAX_SEC` | `0.6` / `8` | Acceptable clip length (seconds) |

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

- **TTS output is noise / very quiet and muddy (the nasty one)**: you almost certainly have **two TTS processes running at once**. On an 8 GB GPU, concurrent runs do **not** raise a CUDA OOM — they silently degrade: output sits ~20 dB below normal and some segments come out as digital silence, which sounds like "severe distortion". Re-running the same text as a single process produces 24/24 normal segments. So always run TTS **as a single, sequential process**: before batch synthesis, make sure nothing else is holding VRAM (`batch_tts`, the `main.py` UI, and the `real-time/` service are mutually exclusive).
- **Which reference clip is used**: this project does simultaneous dubbing, so `SPK_REF_MODE=segment` by default — every segment uses **its own original clip** for both timbre and emotion, tracking that segment's mic position, loudness and performance. `fixed` (one clip per speaker) is a speed-only approximation that drifts from the original.
- **bigvgan CUDA kernel fails to load (`Ninja is required`)**: expected — it falls back to the torch implementation with no quality loss (just slightly slower).
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
