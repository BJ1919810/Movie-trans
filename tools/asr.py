#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
import json
import re
import time
import traceback
from pathlib import Path
import torch
from faster_whisper import WhisperModel
from funasr import AutoModel
from modelscope import snapshot_download
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Windows 编码兜底：stdout 一旦被重定向 / 被上级进程用管道接走，Python 会退回
# 本地编码（中文系统 = GBK），脚本里打印的 emoji（⚠️ ✅ 🔄 …）会直接抛
# UnicodeEncodeError 把整条流水线打断（已踩过一次：主 UI 调本脚本时崩在 ⚠️）。
# 这里只放宽错误处理、**不改编码** —— 编不出的字符降级成 '?'，
# 中文与上级进程的解码方式都保持原样，不会产生乱码。
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:   # noqa: BLE001  （老版本 / 非常规流：忽略）
        pass

# 获取项目根目录（tools目录的父目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置模型路径（使用相对路径）
MODEL_DIR = os.path.join(project_root, "asr", "models")
os.environ["HF_HOME"] = MODEL_DIR

# 解决Xet存储后端问题的环境变量设置
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# FunASR模型缓存
funasr_models = {}


def create_funasr_model(language="zh"):
    """创建 FunASR 模型用于中文识别，支持本地缺失时自动下载至 MODEL_DIR"""
    # 定义模型 ID 与本地路径映射
    model_configs = {
        "zh": {
            "asr": {
                "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "local_name": "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            },
            "vad": {
                "model_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "local_name": "speech_fsmn_vad_zh-cn-16k-common-pytorch"
            },
            "punc": {
                "model_id": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                "local_name": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
            }
        }
    }

    if language not in model_configs:
        raise ValueError(f"FunASR 不支持该语言: {language}")

    config = model_configs[language]
    revision = "v2.0.4"

    # 构建本地路径 & 确保 MODEL_DIR 存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    def ensure_model_downloaded(model_key, model_info):
        local_path = os.path.join(MODEL_DIR, model_info["local_name"])
        if not os.path.exists(local_path):
            print(f"[Downloading] {model_key.upper()} Model -> {local_path}")
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    snapshot_download(
                        model_info["model_id"],
                        revision=revision,
                        cache_dir=MODEL_DIR,
                        local_files_only=False
                    )
                    # snapshot_download 默认会建 `MODEL_DIR/model_id/...`，但 FunASR 期望直接是模型目录
                    # 实际下载后路径为：MODEL_DIR/model_id/ → 我们 rename 为期望的 local_name
                    downloaded_dir = os.path.join(MODEL_DIR, model_info["model_id"].replace("/", "--"))
                    if os.path.exists(downloaded_dir):
                        os.rename(downloaded_dir, local_path)
                    else:
                        # fallback: 可能因 modelscope 版本差异直接下到 local_name？
                        pass
                    print(f"[Success] {model_key.upper()} Model Download Completed <- {local_path}")
                    break
                except Exception as e:
                    print(f"[Warning] Attempt {attempt + 1} to download {model_key.upper()} model failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"[Retry] Performing attempt {attempt + 2}...")
                        time.sleep(5)  # 等待5秒后重试
                    else:
                        raise RuntimeError(
                            f"[X] {model_key.upper()} 模型下载失败！请检查网络或 ModelScope Token。\n"
                            f"Model ID: {model_info['model_id']}, Revision: {revision}\n"
                            f"Error: {e}"
                        )
        else:
            print(f"[Ready] {model_key.upper()} Model <- {local_path}")
        return local_path

    # 检查并下载三模型
    path_asr = ensure_model_downloaded("asr", config["asr"])
    path_vad = ensure_model_downloaded("vad", config["vad"])
    path_punc = ensure_model_downloaded("punc", config["punc"])

    # 从缓存加载 or 新建
    if language in funasr_models:
        print(f"[Reuse] FunASR Model Already Loaded: {language.upper()}")
        return funasr_models[language]
    else:
        model = AutoModel(
            model=path_asr,
            model_revision=revision,
            vad_model=path_vad,
            vad_model_revision=revision,
            punc_model=path_punc,
            punc_model_revision=revision,
        )
        print(f"[Complete] FunASR Model Successfully Loaded: {language.upper()}")

        funasr_models[language] = model
        return model


def transcribe_with_funasr(audio_file, language="zh"):
    """使用FunASR进行中文语音识别"""
    try:
        model = create_funasr_model(language)
        result = model.generate(input=audio_file)
        return result[0]["text"] if result else ""
    except Exception as e:
        print(f"FunASR recognition error: {e}")
        traceback.print_exc()
        return ""


def transcribe_with_faster_whisper(audio_file, model, language=None):
    """使用Faster-Whisper进行多语言语音识别"""
    try:
        if language == "zh":
            print("User specified Chinese text, processed by FunASR")
            text = transcribe_with_funasr(audio_file, language="zh")
        else:
            segments, info = model.transcribe(
                audio=audio_file,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language,
            )
            text = ""
            # 若检测到的语言是中文，则使用FunASR进行识别
            if info.language == "zh" and language == None:
                print("Detected Chinese text, switching to FunASR processing")
                text = transcribe_with_funasr(audio_file, language="zh")
            # 如果FunASR没有返回结果或其他语言，使用Faster-Whisper
            else:
                for segment in segments:
                    text += segment.text
                
        return text
    except Exception as e:
        print(f"Faster-Whisper recognition error: {e}")
        traceback.print_exc()
        return ""


CLIP_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)")
CLIP_SPK_RE = re.compile(r"clip_(\w+?)_\d+_")


def get_clip_info_from_filename(filename):
    """从文件名中提取片段信息

    格式: clip_00_001_14.73-15.29.wav（兼容带尾部后缀的命名，如 clip_00_001_14.73-15.29_00.wav）
    早期实现用 split("_") 取 parts[3]，遇到后缀就会解析错；改用正则。
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    spk_m = CLIP_SPK_RE.search(stem)
    time_m = CLIP_TIME_RE.search(stem)
    if not spk_m or not time_m:
        return None
    return {
        "speaker": f"SPEAKER_{spk_m.group(1)}",
        "start": float(time_m.group(1)),
        "end": float(time_m.group(2)),
    }


def find_matching_segment(segments, clip_info, tolerance=0.05):
    """在 segments 中查找匹配的片段

    注意：JSON 里的 start/end 可能被人工标注工具改过小数位，
    早期实现用 float 精确相等比较，一改就全段匹配失败（表现为"全部识别失败"）。
    这里改成按说话人 + 数值容差匹配。
    """
    best, best_err = None, None
    for segment in segments:
        if segment["speaker"] != clip_info["speaker"]:
            continue
        err = abs(segment["start"] - clip_info["start"]) + abs(segment["end"] - clip_info["end"])
        if err <= tolerance * 2:
            if best_err is None or err < best_err:
                best, best_err = segment, err
    return best


def process_clips(clips_dir, diarization_file, model, language=None):
    """处理所有音频片段并更新识别结果"""
    # 读取说话人分离结果
    with open(diarization_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # 清空所有片段的raw_text、result_text字段，确保完全重新识别
    for segment in segments:
        segment.pop('raw_text', None)
        segment.pop('result_text', None)
    
    print(f"Loaded {len(segments)} segments")
    
    # 处理每个说话人的片段
    speaker_dirs = [d for d in Path(clips_dir).iterdir() if d.is_dir()]
    
    # 用于跟踪处理状态
    total_processed = 0
    total_errors = 0
    
    for speaker_dir in speaker_dirs:
        speaker_name = speaker_dir.name
        print(f"\nProcessing segments for {speaker_name}...")
        
        # 获取该说话人的所有音频文件
        audio_files = list(speaker_dir.glob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        # 为每个说话人创建单独的进度条
        pbar = tqdm(audio_files, desc=f"识别 {speaker_name}")
        
        for audio_file in pbar:
            try:
                # 从文件名提取信息
                clip_info = get_clip_info_from_filename(audio_file.name)
                if not clip_info:
                    print(f"Cannot parse filename: {audio_file.name}")
                    total_errors += 1
                    continue
                
                # 在segments中找到匹配的片段
                matching_segment = find_matching_segment(segments, clip_info)
                if not matching_segment:
                    print(f"No matching segment found: {audio_file.name}")
                    total_errors += 1
                    continue
                
                # 进行语音识别
                print(f"\nRecognizing: {audio_file.name}")
                transcription = transcribe_with_faster_whisper(str(audio_file), model, language)
                
                # 保存识别结果
                matching_segment['raw_text'] = transcription
                print(f"Transcription result: {transcription}")
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing file {audio_file.name}: {str(e)}")
                total_errors += 1
                # 继续处理下一个文件而不是中断整个过程
                continue
        
        # 更新进度条描述信息
        pbar.set_postfix({"已处理": total_processed, "错误": total_errors})
    
    # 过滤掉raw_text为空的条目（识别失败 / 纯音乐段）
    # 注意：这些段会从主 JSON 移除，但**必须留痕**——早期版本直接删掉，
    # 表现是"某几句台词凭空消失"，无法追溯。这里额外写审计文件 + 打印时间范围。
    filtered_segments = [segment for segment in segments if segment.get('raw_text', '').strip()]
    dropped_segments = [segment for segment in segments if not segment.get('raw_text', '').strip()]
    removed_count = len(dropped_segments)

    if dropped_segments:
        dropped_path = os.path.join(os.path.dirname(os.path.abspath(diarization_file)), "asr_dropped_segments.json")
        try:
            with open(dropped_path, "w", encoding="utf-8") as f:
                json.dump(dropped_segments, f, ensure_ascii=False, indent=2)
            print(f"!! 有 {removed_count} 段识别结果为空，已从主 JSON 移除，明细写入: {dropped_path}")
            for seg in dropped_segments:
                print(f"   - {seg.get('speaker')} {seg.get('start')}-{seg.get('end')}s （原文本为空，需人工补录）")
        except Exception as e:  # noqa: BLE001
            print(f"!! 写入丢段审计文件失败: {e}")
            for seg in dropped_segments:
                print(f"   - 丢段: {seg.get('speaker')} {seg.get('start')}-{seg.get('end')}s")

    # 保存更新后的结果
    with open(diarization_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_segments, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completion statistics:")
    print(f"- Total processed: {total_processed} files")
    print(f"- Errors: {total_errors} files")
    print(f"- Removed empty entries: {removed_count}")
    print(f"- Final saved: {len(filtered_segments)} entries")
    
    if total_errors > 0:
        print(f"\nWarning: {total_errors} files failed to process, please check the error messages above")
    
    print(f"\nAll recognition results saved to: {diarization_file}")


# ===========================================================================
# 参考切片体检（**只诊断**：不改合成行为、不写 JSON、不产出任何文件）
# ---------------------------------------------------------------------------
# 为什么需要它：
#   同声配音时，每段的"音色/情绪参考"就是该段自己的原片切片。切片里一旦混入原片
#   BGM 残留或噪声，模型会把它当成"音色 + 表演方式"学过去（听感：大喘气、容易炸、
#   音质糊）。这个体检表用**纯音频指标**指出哪些段的切片是脏的。
#
# ⚠️ 它为什么**不做**"择优 / 回退 / 净化"（2026-09-14 的结论，别再往回加）：
#   1) 任何"同说话人择优/回退"都依赖 pyannote 的 speaker 标签，而主人的实测结论是
#      **pyannote 多说话人标定不可信**（中日英都试过，甚至把男女声并成一个 speaker）。
#      标签一错，回退只会把错放大（女主的参考被套到男主段上）。
#   2) 参考净化（削低频/去噪/裁窗口）做过 A/B，主人听下来**三个样本都是原始参考更好**
#      —— 120Hz 削低频削掉的正是音色本体，afftdn 会上金属味伪影。
#   所以这里只输出诊断；要改质量应该从**人声分离**下手。
#
# 用法（都很快，不加载 ASR 模型）：
#   python tools/asr.py --only-ref-check        # 单独跑体检
#   python tools/asr.py --ref-check             # 跟在 ASR 之后顺带跑
#
# 指标（纯 CPU，零新依赖）：
#   pause_drop   最响 20% 帧 与 "停顿档" 帧 的电平差  ← **核心指标**
#                干净语音：停顿里只剩本底噪声，降得很深（本片实测 37~54dB）
#                混了 BGM：停顿里还有音乐在响，只降 20~32dB
#                （不用绝对噪声底：切片首尾若有数字静音，噪声底会变 -inf、SNR 虚高到 100+）
#   ratio        能量高于停顿档 6dB 的帧占比（太低说明大半是静音/噪声）
#   band         语音频段(0.3–4kHz) 能量占比
#   dur / clip   时长是否落在 0.6~8s、有无削顶
#
# 环境变量（都有默认值，一般不用动）：
#   REF_PAUSE_DROP_MIN=30  REF_SPEECH_RATIO_MIN=0.5  REF_MIN_SEC=0.6  REF_MAX_SEC=8
# ===========================================================================
try:
    import soundfile as _sf
    import numpy as _np
except ImportError:  # 理论上项目环境都有
    _sf = None
    _np = None

REF_PAUSE_DROP_MIN = float(os.environ.get("REF_PAUSE_DROP_MIN", "30"))
REF_SPEECH_RATIO_MIN = float(os.environ.get("REF_SPEECH_RATIO_MIN", "0.5"))
REF_MIN_SEC = float(os.environ.get("REF_MIN_SEC", "0.6"))
REF_MAX_SEC = float(os.environ.get("REF_MAX_SEC", "8"))


def _db(v):
    return 20 * _np.log10(max(float(v), 1e-12))


def _load_mono(path):
    """读成单声道 float64（切片是 44.1k 立体声，体检用不上双声道）"""
    x, sr = _sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def _frame_bands(x, sr, frame_ms=20):
    """逐帧算 低频(80-300) / 语音频段(300-4000) / 总能量，返回三个数组"""
    fl = max(16, int(sr * frame_ms / 1000))
    n_frames = max(0, (len(x) - fl) // fl)
    low, high, tot = [], [], []
    for i in range(n_frames):
        seg = x[i * fl:(i + 1) * fl]
        spec = _np.abs(_np.fft.rfft(seg * _np.hanning(len(seg)))) ** 2
        freq = _np.fft.rfftfreq(len(seg), 1 / sr)
        lo = float(spec[(freq >= 80) & (freq < 300)].sum())
        hi = float(spec[(freq >= 300) & (freq < 4000)].sum())
        low.append(lo)
        high.append(hi)
        tot.append(lo + hi)
    return _np.array(low), _np.array(high), _np.array(tot)


def analyze_clip(path):
    """单条切片的体检指标；读不到或太短返回 None"""
    try:
        x, sr = _load_mono(path)
    except Exception:  # noqa: BLE001
        return None
    dur = len(x) / sr
    if dur < 0.15:
        return None
    low, high, tot = _frame_bands(x, sr)
    if len(tot) < 3:
        return None
    order = _np.argsort(tot)
    n_loud = max(2, len(order) // 5)
    loud = order[-n_loud:]                                  # 最响 = 说话
    # 取 20%~45% 分位那一档当"停顿"：既不是数字静音，也不是说话
    lo_i, hi_i = max(1, len(order) // 5), max(2, int(len(order) * 0.45))
    quiet = order[lo_i:hi_i] if hi_i > lo_i else order[:n_loud]
    speech_level = float(tot[loud].mean())
    pause_level = float(tot[quiet].mean())
    pause_drop = _db(speech_level) - _db(pause_level)
    ratio = float((tot > max(pause_level, 1e-12) * 4).mean())
    band = 100.0 * float(high.sum()) / max(float(high.sum() + low.sum()), 1e-30)
    clip_cnt = int((_np.abs(x) >= 0.999).sum())

    ok = (REF_MIN_SEC <= dur <= REF_MAX_SEC
          and pause_drop >= REF_PAUSE_DROP_MIN
          and ratio >= REF_SPEECH_RATIO_MIN
          and clip_cnt == 0)
    return {
        "dur": dur, "pause_drop": pause_drop, "ratio": ratio,
        "band": band, "clip": clip_cnt, "ok": bool(ok),
    }


def _clip_path_for_segment(segment, clips_dir):
    """按 speaker/start/end 找该段的切片（文件名两位小数，JSON 里可能不是 → 容差匹配）"""
    speaker = str(segment.get("speaker", ""))
    start = float(segment.get("start", 0) or 0)
    end = float(segment.get("end", 0) or 0)
    folder = os.path.join(clips_dir, speaker)
    hits = sorted(Path(folder).glob(f"clip_*_{start:.2f}-{end:.2f}.wav")) if os.path.isdir(folder) else []
    if hits:
        return str(hits[0])
    for cand in sorted(Path(folder).glob("clip_*.wav")) if os.path.isdir(folder) else []:
        info = get_clip_info_from_filename(cand.name)
        if info and abs(info["start"] - start) <= 0.01 and abs(info["end"] - end) <= 0.01:
            return str(cand)
    return None


def run_ref_check(diarization_file, clips_dir=None):
    """逐段打印参考切片体检表（只读；返回 (检查段数, 统计)）"""
    if _sf is None:
        print("!! 缺少 soundfile/numpy，跳过参考体检")
        return 0, {}
    clips_dir = clips_dir or os.path.join(project_root, "temp", "clips")
    if not os.path.isdir(clips_dir):
        print(f"!! 找不到切片目录 {clips_dir}，跳过参考体检")
        return 0, {}

    with open(diarization_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    cache = {}

    def metrics_of(p):
        if p not in cache:
            cache[p] = analyze_clip(p)
        return cache[p]

    stats = {"clean": 0, "dirty": 0, "no_clip": 0}
    dirty_rows = []
    print()
    print("  参考切片体检（只诊断，不改变合成行为）")
    print("  段                        说话人        时长  停顿降幅  语音占比   频段  判定")
    for seg in segments:
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", 0) or 0)
        spk = str(seg.get("speaker", "?"))
        label = f"{start:>6.2f}-{end:<6.2f}"
        clip = _clip_path_for_segment(seg, clips_dir)
        m = metrics_of(clip) if clip else None
        if m is None:
            stats["no_clip"] += 1
            print(f"  {label} {spk:<12s}    ——      切片缺失/过短，无法体检")
            continue
        if m["ok"]:
            stats["clean"] += 1
            verdict = "OK"
        else:
            stats["dirty"] += 1
            dirty_rows.append((seg, m, clip))
            verdict = "脏"
        print(f"  {label} {spk:<12s} {m['dur']:5.2f} {m['pause_drop']:8.1f} "
              f"{m['ratio'] * 100:8.0f}% {m['band']:5.0f}%   {verdict}")

    print()
    print(f"体检汇总: 干净 {stats['clean']} / 脏 {stats['dirty']} / 无切片 {stats['no_clip']}")
    print(f"判定阈值: 停顿降幅≥{REF_PAUSE_DROP_MIN:.0f}dB、语音占比≥{REF_SPEECH_RATIO_MIN * 100:.0f}%、"
          f"时长 {REF_MIN_SEC:.1f}~{REF_MAX_SEC:.1f}s、无削顶")
    if dirty_rows:
        print()
        print(f"[!] {len(dirty_rows)} 段的参考切片混有连续噪声/BGM 残留——这些段的合成最容易出问题：")
        for seg, m, clip in dirty_rows[:20]:
            print(f"     {float(seg.get('start', 0) or 0):>6.2f}-{float(seg.get('end', 0) or 0):<6.2f} "
                  f"{str(seg.get('speaker', '?')):<12s} 停顿仅降 {m['pause_drop']:5.1f}dB  "
                  f"{os.path.basename(clip)}")
        if len(dirty_rows) > 20:
            print(f"     ... 另有 {len(dirty_rows) - 20} 段")
        print("   → 根源是**人声分离质量**（后半段 BGM 起来时尤其明显）。")
        print("     可选做法：换 SOTA 分离模型（BS-Roformer / Mel-Band Roformer 等）、")
        print("     在标注页删掉/合并这些段、或对它们单独重新合成。")
    print()
    print("（本体检只读：未写入 JSON、未产出净化文件、未改变任何合成参数）")
    return len(segments), stats


def main():
    # 导入torch以检查CUDA可用性
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    parser = argparse.ArgumentParser(description="ASR处理脚本")
    parser.add_argument("--model_size", type=str, default="large-v3", help="Whisper模型大小")
    parser.add_argument("--device", type=str, default="cuda" if cuda_available else "cpu", help="运行设备")
    parser.add_argument("--compute_type", type=str, default="float16" if cuda_available else "int8", 
                       help="计算类型")
    parser.add_argument("--language", type=str, default=None, help="音频语言")
    # 参考切片体检（只诊断：不写 JSON、不产文件、不改合成参数）
    parser.add_argument("--ref-check", action="store_true",
                        help="ASR 结束后顺带打印参考切片体检表")
    parser.add_argument("--only-ref-check", action="store_true",
                        help="只打印参考切片体检表，跳过 ASR（不加载模型、不清空已有文本）")
    parser.add_argument("--clips-dir", type=str,
                        default=os.path.join(project_root, "temp", "clips"),
                        help="切片目录（默认 temp/clips）")
    parser.add_argument("--diarization-json", type=str,
                        default=os.path.join(project_root, "results", "speaker_diarization.json"),
                        help="说话人 JSON（默认 results/speaker_diarization.json）")
    
    args = parser.parse_args()

    # 只做体检：不加载 ASR 模型、不动 raw_text/result_text，秒级完成
    if args.only_ref_check:
        run_ref_check(args.diarization_json, args.clips_dir)
        return
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 检查CUDA兼容性，如果CUDA版本不匹配则使用CPU
    if args.device == "cuda":
        try:
            # 尝试初始化一个简单的CUDA操作来检查兼容性
            import torch
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1).cuda()
                print("CUDA environment check passed")
            else:
                print("CUDA unavailable, will use CPU")
                args.device = "cpu"
                args.compute_type = "int8"
        except Exception as e:
            print(f"CUDA environment check failed: {e}")
            print("Will use CPU for inference")
            args.device = "cpu"
            args.compute_type = "int8"
    
    # 初始化Faster-Whisper模型
    print("Loading Faster-Whisper model...")
    # 检查本地是否存在模型models--Systran--faster-whisper-large-v3
    model_path = f"{MODEL_DIR}"
    print(f"Checking model path: {os.path.join(model_path, f'models--Systran--faster-whisper-{args.model_size}')}")
    if os.path.exists(os.path.join(model_path, f"models--Systran--faster-whisper-{args.model_size}")):
        print("Found local model directory")
        # 查找snapshots目录中的实际模型版本
        snapshots_dir = os.path.join(model_path, f"models--Systran--faster-whisper-{args.model_size}", "snapshots")
        print(f"Checking snapshots directory: {snapshots_dir}")
        if os.path.exists(snapshots_dir):
            print("Found snapshots directory")
            # 获取第一个快照目录（通常只有一个）
            snapshot_dirs = os.listdir(snapshots_dir)
            print(f"Snapshot directory list: {snapshot_dirs}")
            if snapshot_dirs:
                actual_model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                print(f"Using local model: {actual_model_path}")
                try:
                    model = WhisperModel(actual_model_path, device=args.device, compute_type=args.compute_type)
                except Exception as e:
                    print(f"GPU model loading failed: {e}")
                    print("Attempting to load model with CPU...")
                    # 在这里也确保使用CPU设备和兼容的计算类型
                    args.device = "cpu"
                    args.compute_type = "int8"
                    model = WhisperModel(actual_model_path, device=args.device, compute_type=args.compute_type)
            else:
                print("No model snapshot directory found, re-downloading from HuggingFace...")
                # 添加重试机制
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"Attempt {attempt + 1} to download Faster-Whisper model...")
                        model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type, download_root=model_path)
                        print("Model download completed!")
                        break
                    except Exception as e:
                        print(f"[! Warning] Attempt {attempt + 1} to download Faster-Whisper model failed: {str(e)}")
                        if attempt < max_retries - 1:
                            print(f"[Retry] Waiting 5 seconds before attempt {attempt + 2}...")
                            time.sleep(5)
                        else:
                            raise
        else:
            print("Snapshots directory not found, re-downloading from HuggingFace...")
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Attempt {attempt + 1} to download Faster-Whisper model...")
                    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type, download_root=model_path)
                    print("Model download completed!")
                    break
                except Exception as e:
                    print(f"[Warning] Attempt {attempt + 1} to download Faster-Whisper model failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"[Retry] Waiting 5 seconds before attempt {attempt + 2}...")
                        time.sleep(5)
                    else:
                        raise
    else:
        # 如果本地模型不存在，从HuggingFace下载并保存到指定目录
        print(f"Model not found locally, will download from HuggingFace to: {model_path}")
        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} to download Faster-Whisper model...")
                model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type, download_root=model_path)
                print("Model download completed!")
                break
            except Exception as e:
                print(f"[! Warning] Attempt {attempt + 1} to download Faster-Whisper model failed: {str(e)}")
                if "cublas64_12.dll" in str(e):
                    print("Detected CUDA library issue, switching to CPU mode")
                    args.device = "cpu"
                    args.compute_type = "int8"
                    try:
                        model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type, download_root=model_path)
                        print("Model loaded successfully in CPU mode!")
                        break
                    except Exception as cpu_e:
                        print(f"Failed to load model in CPU mode as well: {str(cpu_e)}")
                        if attempt < max_retries - 1:
                            print(f"[Retry] Waiting 5 seconds before attempt {attempt + 2}...")
                            time.sleep(5)
                        else:
                            raise
                elif attempt < max_retries - 1:
                    print(f"[Retry] Waiting 5 seconds before attempt {attempt + 2}...")
                    time.sleep(5)
                else:
                    raise
    print("Model loading completed!")
    
    # 设置默认路径（使用相对路径）
    clips_dir = os.path.join(project_root, "temp", "clips")
    diarization_file = os.path.join(project_root, "results", "speaker_diarization.json")
    
    # 处理音频片段
    process_clips(clips_dir, diarization_file, model, args.language)

    # 参考切片体检（只诊断；不做选择/回退/净化——理由见 run_ref_check 上方注释）
    if args.ref_check:
        print()
        run_ref_check(diarization_file, clips_dir)

    # -----------------------------------------------------------------
    # Windows 退出兜底（2026-09-15）：本进程同时持有 torch(CUDA) +
    # funasr(onnxruntime/sentencepiece) + ctranslate2 三套 C 运行时，
    # 解释器 shutdown 阶段销毁它们会在 C 层 abort（Fatal Python error:
    # Aborted，无堆栈无消息）→ 子进程 returncode!=0，UI 误报"识别失败"，
    # 但识别结果和 JSON 其实早已完整写出。这里在 main() 末尾显式 flush
    # 后直接退出，跳过 shutdown 清理（显存/内存由操作系统回收）。
    # 注意：--only-ref-check 分支在上方已 return，不经过这里（它不崩）。
    # -----------------------------------------------------------------
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()