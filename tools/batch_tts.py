#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index-TTS 批量语音合成脚本

注意：此版本已修改为优先使用本地模型文件，避免重复网络下载并支持离线使用。

本地模型文件应保存在 index-tts/checkpoints 目录下的相应子目录中。
如果本地模型文件缺失或损坏，请运行 download_models.py 脚本下载所有必需的模型文件到本地。

---------------------------------------------------------------------------
音色 vs 情绪（重要）
---------------------------------------------------------------------------
IndexTTS-2.5 里这两件事是**两条独立通道**：

  * spk_audio_prompt  -> 只决定"谁在说话"（音色 embedding），可安全复用/缓存
  * emo_vector / emo_text / emo_audio_prompt -> 决定"用什么情绪说"

源码保证（indextts/infer_v2_5.py）：只要传了 emo_vector 或 use_emo_text=True，
就会强制 emo_audio_prompt=None，情绪完全由情绪通道决定。
所以"复用同一段参考音频"**不会**把上一句的情绪带过来 —— 情绪靠
TTS_EMO_MODE=text 时按本句文本实时推断，每句都可以不一样。

环境变量：
  SPK_REF_MODE=segment|fixed   音色参考音频策略（默认 segment）
      segment : 每段用自己那段原片切片 —— 同声配音的正确解（推荐）
                同一说话人在全片的麦位/音量/表演状态都在变，
                音色与情绪都必须逐段跟随原声，否则整体跑偏。
      fixed   : 每个说话人固定用一段参考音频（缓存命中、更快），
                代价是音色/情绪偏离该段原声，仅在素材质量均匀时作提速近似。
  TTS_SORT_BY_SPEAKER=true|false  按说话人分组处理（默认 true）
      仅对 fixed 模式有缓存收益（segment 模式下每段参考都不同）。
  TTS_EMO_MODE=text|ref|vector|none  情绪策略（默认 ref）
      ref    : 用当段原切片作为情绪参考（情绪跟着原片演员的表演走）
               *** 电影配音的推荐模式 ***
               台词文本承载不了"表演方式"——面无表情说狠话、反讽、笑着哭，
               这些只存在于原声韵律里。文本推断会把"生气的台词"识别成
               "生气地说"，丢失表演意图。分离后的人声切片恰好保留了
               演员的韵律/能量/语速，情绪 embedding 对轻微失真不敏感。
      text   : 用 QwenEmotion 按本句文本推断情绪（适合有声书/播客等
               "文本即表演"的场景；电影配音会丢失表演方式，慎用）
      vector : 固定情绪向量，见 TTS_EMO_VECTOR
      none   : 不控制情绪（等同 ref，走参考音频自带的情绪）
  TTS_EMO_ALPHA=0.0~1.0   情绪强度（默认 1.0）
  TTS_EMO_VECTOR=0,0,0,0,0,0,0,1   TTS_EMO_MODE=vector 时的 8 维向量
  TTS_LANG=ZH|EN|JA|ES|AR  合成语言（默认 ZH）
  TTS_DURATION_FACTOR=0.5~2.0  语速，>1 变慢、<1 变快（默认 1.0）
  TTS_MAX_MEL_TOKENS=1~1815  单段生成长度上限（默认 1815 = 2.5 上限）
      调小会**静默截断**较长台词（话说一半），除非明确知道原因否则别动。
"""

import os
import sys
import glob
import json
import re

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目根目录到Python路径
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'index-tts'))

# 国内网络：强制走 ModelScope 下载缺失的辅助模型（无需手动代理判断）
os.environ.setdefault("USE_MODELSCOPE", "true")
# 环境里装了 tensorflow 2.16 且与 numpy 2.x 不兼容，会让 transformers 导入直接崩，
# 这里显式关掉 TF/JAX 后端检测（本项目只用 torch）。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

# ---------------------------------------------------------------------------
# 语言选项（IndexTTS-2.5 支持 5 种语言 + 跨语言音色克隆）
#   ZH 中文 | EN 英文 | JA 日语 | ES 西班牙语 | AR 阿拉伯语
# 通过环境变量 TTS_LANG 覆盖，例如：set TTS_LANG=JA
# ---------------------------------------------------------------------------
SUPPORTED_LANGS = {"ZH", "EN", "JA", "ES", "AR"}
TTS_LANG = os.environ.get("TTS_LANG", "ZH").upper()
if TTS_LANG not in SUPPORTED_LANGS:
    print(f"!! 不支持的语言 {TTS_LANG}，可选：{sorted(SUPPORTED_LANGS)}，回退到 ZH")
    TTS_LANG = "ZH"

def _env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

# --- 音色参考音频策略 -----------------------------------------------------
# 本项目是**同声配音**：每段译文都对应一段原片切片，音色与情绪都应当参考
# "那一段自己"——同一说话人在全片里的麦位、音量、表演状态都在变，
# 用固定 clip 会同时偏离该段的音色与情绪。
#   segment（默认，配音正确解）: 每段用自己那段切片
#   fixed  （仅提速的近似解）  : 每个说话人固定一段，缓存命中率高但音色/情绪会漂移
SPK_REF_MODE = os.environ.get("SPK_REF_MODE", "segment").strip().lower()
if SPK_REF_MODE not in ("fixed", "segment"):
    print(f"!! 未知的 SPK_REF_MODE={SPK_REF_MODE}，回退到 segment")
    SPK_REF_MODE = "segment"

# 按说话人分组处理，最大化 spk embedding 缓存命中
SORT_BY_SPEAKER = _env_bool("TTS_SORT_BY_SPEAKER", True)

# --- 情绪策略（与音色解耦，任何模式下情绪都不会被"上一句"污染）---------------
# 默认 ref：情绪跟着原片演员的表演走（台词文本承载不了表演方式，文本推断会丢演技）
TTS_EMO_MODE = os.environ.get("TTS_EMO_MODE", "ref").strip().lower()
if TTS_EMO_MODE not in ("text", "ref", "vector", "none"):
    print(f"!! 未知的 TTS_EMO_MODE={TTS_EMO_MODE}，回退到 ref")
    TTS_EMO_MODE = "ref"
TTS_EMO_ALPHA = float(os.environ.get("TTS_EMO_ALPHA", "1.0"))
EMO_VECTOR = None
if TTS_EMO_MODE == "vector":
    raw = os.environ.get("TTS_EMO_VECTOR", "")
    try:
        EMO_VECTOR = [float(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        print(f"!! TTS_EMO_VECTOR 解析失败：{raw!r}，回退到 ref 模式")
        TTS_EMO_MODE, EMO_VECTOR = "ref", None
    if EMO_VECTOR and len(EMO_VECTOR) != 8:
        print(f"!! 情绪向量应为 8 维，当前 {len(EMO_VECTOR)} 维，回退到 ref 模式")
        TTS_EMO_MODE, EMO_VECTOR = "ref", None
    elif not EMO_VECTOR:
        print("!! 未设置 TTS_EMO_VECTOR，回退到 ref 模式")
        TTS_EMO_MODE = "ref"

# --- 单段音频长度上限 -------------------------------------------------------
# max_mel_tokens 决定单次推理最多生成多少个 mel token（约等于音频长度上限）。
# 2.5 的实现上限是 1815；用 1000 时稍长的台词会被**静默截断**（话说一半没了），
# 所以这里默认拉到上限 1815，正常句子根本到不了，长句才有救。
try:
    TTS_MAX_MEL_TOKENS = int(os.environ.get("TTS_MAX_MEL_TOKENS", "1815"))
except ValueError:
    print("!! TTS_MAX_MEL_TOKENS 不是整数，回退到 1815")
    TTS_MAX_MEL_TOKENS = 1815
TTS_MAX_MEL_TOKENS = max(1, min(TTS_MAX_MEL_TOKENS, 1815))  # 超上限会被模型拒绝

# IndexTTS-2.5（多语言）；v2 推理器已淘汰，本项目只走 infer_v2_5
from indextts.infer_v2_5 import IndexTTS2

# ---------------------------------------------------------------------------
# 线性合成：本脚本必须**单进程**运行。
# 8GB 卡上同时跑两个 TTS 进程不会抛 CUDA OOM，而是静默降级——
# 输出电平比正常低约 20dB、部分分段是数字静音，听感即"严重失真的噪声"。
# 因此运行前请确认没有其他实例（本脚本 / webui / real-time 服务）在占用显存。
# ---------------------------------------------------------------------------

# 初始化模型，启用多种性能优化选项
# 强制使用CUDA，不回退到CPU模式
init_kwargs = dict(
    cfg_path=os.path.join(project_root, "index-tts", "checkpoints", "config.yaml"),
    model_dir=os.path.join(project_root, "index-tts", "checkpoints"),
    use_cuda_kernel=True,   # 使用自定义的CUDA内核来加速BigVGAN的推理（加载失败会自动回退 torch 实现）
    use_deepspeed=False,    # 暂时不启用DeepSpeed，因为它可能在某些系统上导致性能下降
    use_accel=False,         # 启用加速引擎来优化GPT模型的推理
    use_torch_compile=False, # 使用torch.compile来进一步优化模型执行
    device="cuda:0",         # 强制使用CUDA设备
    # 2.5 默认使用 bf16（显存更省、数值更稳）
    use_bf16=True,
    # Qwen 情感模型：TTS_EMO_MODE=text 时必须加载，否则 use_emo_text 会直接 RuntimeError
    use_qwen_emo=_env_bool("USE_QWEN_EMO", False) or TTS_EMO_MODE == "text",
)

tts = IndexTTS2(**init_kwargs)
print("Model initialized with CUDA kernel support. (IndexTTS-2.5, lang=%s)" % TTS_LANG)

# 输出目录（使用相对路径）
output_dir = os.path.join(project_root, "results", "tts_output")

# 在每次运行时清空输出目录
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# 读取说话人分割结果
with open(os.path.join(project_root, "results", "speaker_diarization.json"), "r", encoding="utf-8") as f:
    segments = json.load(f)

# 从文件名提取时间段的正则：兼容 clip_00_005_44.47-59.58_00.wav 这类带尾部后缀的命名
CLIP_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)")

# 获取所有参考音频文件
ref_audio_files = {}
# 动态获取所有说话人目录
clips_dir = os.path.join(project_root, "temp", "clips")
speaker_dirs = [d for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d))]

for speaker in speaker_dirs:
    pattern = os.path.join(project_root, "temp", "clips", f"{speaker}/*.wav")
    files = glob.glob(pattern)
    for file_path in files:
        # 从文件名中提取时间信息
        stem = os.path.splitext(os.path.basename(file_path))[0]
        m = CLIP_TIME_RE.search(stem)
        if m:
            start_s, end_s = float(m.group(1)), float(m.group(2))
            ref_audio_files[f"{speaker}_{start_s:.2f}-{end_s:.2f}"] = file_path

# ---------------------------------------------------------------------------
# 音色参考音频（SPK_REF_MODE）
#   segment（默认）: 音色参考=本段自己的原片切片 —— 同声配音的正确解。
#                    同一说话人在全片的麦位/音量/表演状态都在变，
#                    逐段跟随才不会让音色和情绪跑偏。
#   fixed（可选）  : 每说话人挑一段固定参考，缓存命中、更快，
#                    代价是音色与情绪偏离该段原声（仅作提速近似）。
# 挑选策略（fixed 时；推理侧 _load_and_cut_audio 会把参考截到 15 秒，超长片段也合法）：
#   1) 3~15s 里最接近 8 秒的（理想区间）
#   2) >=15s 的（截断后等效 15s，信息量最大）
#   3) 0~15s 里最长的
#   4) 都不行就随便拿一段
# ---------------------------------------------------------------------------
TARGET_REF_SEC = 8.0
speaker_fixed_ref = {}


def _clip_duration(path):
    """从 clip 文件名解析时长（秒）；无法解析返回 -1。"""
    m = CLIP_TIME_RE.search(os.path.splitext(os.path.basename(path))[0])
    if not m:
        return -1.0
    return max(0.0, float(m.group(2)) - float(m.group(1)))


if SPK_REF_MODE == "fixed":
    for speaker in speaker_dirs:
        cand = [p for p in glob.glob(os.path.join(project_root, "temp", "clips", f"{speaker}/*.wav"))]
        if not cand:
            continue
        ideal = [p for p in cand if 3.0 <= _clip_duration(p) <= 15.0]
        if ideal:
            best = min(ideal, key=lambda p: abs(_clip_duration(p) - TARGET_REF_SEC))
        else:
            longs = [p for p in cand if _clip_duration(p) >= 15.0]
            shorts = [p for p in cand if 0 < _clip_duration(p) < 15.0]
            if longs:
                best = min(longs, key=lambda p: _clip_duration(p))  # 越接近 15s 越好
            elif shorts:
                best = max(shorts, key=_clip_duration)
            else:
                best = cand[0]
        speaker_fixed_ref[speaker] = best
        print(f"[音色参考·fixed] {speaker} -> {os.path.basename(best)} ({_clip_duration(best):.2f}s)")
else:
    print("[音色参考·segment] 每段使用自身原片切片（同声配音：音色+情绪逐段跟随原声）")

# ---------------------------------------------------------------------------
# 情绪参数构造（与音色参考音频完全解耦）
# ---------------------------------------------------------------------------
def build_emo_kwargs(segment_ref_path):
    """按 TTS_EMO_MODE 生成传给 tts.infer 的情绪参数。"""
    if TTS_EMO_MODE == "text":
        # emo_text=None -> 直接用本句待合成文本推断情绪，每句都重新算
        return dict(use_emo_text=True, emo_text=None, emo_alpha=TTS_EMO_ALPHA)
    if TTS_EMO_MODE == "vector":
        return dict(emo_vector=EMO_VECTOR, emo_alpha=TTS_EMO_ALPHA)
    if TTS_EMO_MODE == "ref":
        # 情绪跟着原片走：用当段切片作为情绪参考音频
        return dict(emo_audio_prompt=segment_ref_path, emo_alpha=TTS_EMO_ALPHA)
    return dict()  # none：不显式控制，走默认（参考音频自带情绪）


# 按说话人分组（仅 fixed 模式对缓存有意义；segment 模式下每段参考都不同）
if SORT_BY_SPEAKER:
    order = sorted(range(len(segments)), key=lambda i: segments[i]["speaker"])
else:
    order = range(len(segments))

# 为每个片段生成语音
for i in order:
    segment = segments[i]
    # 获取片段信息
    speaker = segment["speaker"]
    start = segment["start"]
    end = segment["end"]
    text = segment["result_text"]

    # 构建参考音频键
    time_key = f"{speaker}_{start:.2f}-{end:.2f}"

    # 检查参考音频是否存在
    if time_key not in ref_audio_files:
        print(f"Reference audio does not exist: {time_key}")
        continue

    seg_ref_path = ref_audio_files[time_key]

    # 音色参考：fixed 模式下用说话人固定片段（缓存命中），失败则回退到本段切片
    if SPK_REF_MODE == "fixed":
        ref_audio_path = speaker_fixed_ref.get(speaker) or seg_ref_path
    else:
        ref_audio_path = seg_ref_path

    # 构建输出文件名
    output_filename = f"result_{speaker[-2:]}_{start:.2f}-{end:.2f}.wav"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Generating speech: {output_filename}")
    print(f"Text: {text}")
    print(f"Reference audio: {ref_audio_path}")

    # 执行推理
    infer_kwargs = dict(
        text=text,
        spk_audio_prompt=ref_audio_path,
        output_path=output_path,
        verbose=True,
        # 性能优化参数
        max_mel_tokens=TTS_MAX_MEL_TOKENS,  # 2.5 上限 1815（默认），避免长句被截断
        do_sample=True,
        top_p=0.8,
        top_k=15,
        temperature=0.8,
        num_beams=3,
        repetition_penalty=10.0,
        max_text_tokens_per_segment=120
    )
    infer_kwargs.update(build_emo_kwargs(seg_ref_path))
    # 2.5 需要显式指定语言（日语用 JA）；duration_factor 控制语速，>1 变慢、<1 变快
    infer_kwargs["lang"] = TTS_LANG
    infer_kwargs["duration_factor"] = float(os.environ.get("TTS_DURATION_FACTOR", "1.0"))
    tts.infer(**infer_kwargs)

    print(f"Speech saved to: {output_path}\n")

print("All speech segments generated successfully!")