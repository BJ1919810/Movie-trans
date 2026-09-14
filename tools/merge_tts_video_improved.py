import os
import sys
import re
import glob
import json
import argparse
import subprocess
from pydub import AudioSegment

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目根目录到Python路径
sys.path.append(project_root)


# ===========================================================================
# 同声配音：按段对齐（重要）
#   TTS 的实际时长由模型决定，很少正好等于原段时长（end-start）。
#   不对齐的话：短了留空档、长了直接盖到下一句上，口型/节奏全乱。
#   这里用 ffmpeg atempo 做**保音高**的时间伸缩，把每段拉到原段时长；
#   仅在偏差超过阈值时处理，且限制最大倍率，避免为了对齐把语速拉变形。
# 环境变量：
#   ALIGN_TTS=true|false     总开关（默认 true）
#   ALIGN_MAX_RATE=1.25      最大加速/减速倍率（默认 1.25，即语速最多变 ±25%）
#   ALIGN_MIN_DEV=0.05       时长偏差阈值，低于 5% 不动（默认 0.05）
#   ALIGN_TRIM_OVERFLOW=false 对齐后仍超长的段是否裁剪到段长（默认 false，只告警）
#   TTS_FADE_MS=15           每段淡入淡出毫秒数，避免硬切爆音（默认 15）
#   BG_PAD_MS=200            把原声替换成伴奏时，段边界向前后各多扩多少毫秒
#                            （盖掉段外的原声句末气声/尾音；相邻段按间距一半保护；0 = 旧行为）
#
# 同声配音：逐段响度对齐（重要，别改回"按峰值归一化"）
#   以前用 normalize(-3dBFS) 是**按峰值**归一化，而峰值不等于响度：
#   归一化后每段响度 = -3 - 波峰因数，故"密度高/闷响"的段（波峰因数低）
#   天然比别的段响 ~10 dB —— 实测 24 段的段间响度差达 9.9 dB，全程无人修正。
#   现在改成：把每段的响度**对齐到它自己那段原声切片的 RMS**（与音色/情绪
#   用自身切片是同一个道理）——既消除模型输出的电平漂移，又保留原片表演的
#   强弱起伏（喊的仍然响、轻声仍然轻），最后再做峰值保护防削顶。
# 环境变量：
#   TTS_LOUDNESS_MATCH=true|false  逐段响度对齐总开关（默认 true）
#   TTS_LOUDNESS_OFFSET=0.0        对齐后再叠加的偏移 dB（默认 0，想整体更响就调正）
#   TTS_PEAK_CEIL=-1.0             峰值上限 dBFS，超过就整体下压（默认 -1）
#   TTS_MIN_RMS=-30.0              目标响度下限 dBFS，防止原声极轻的段被对齐到听不见
# ===========================================================================
def _env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


ALIGN_TTS = _env_bool("ALIGN_TTS", True)
ALIGN_MAX_RATE = float(os.environ.get("ALIGN_MAX_RATE", "1.25"))
ALIGN_MIN_DEV = float(os.environ.get("ALIGN_MIN_DEV", "0.05"))
ALIGN_TRIM_OVERFLOW = _env_bool("ALIGN_TRIM_OVERFLOW", False)
TTS_FADE_MS = int(os.environ.get("TTS_FADE_MS", "15"))

# 把原声替换成纯伴奏时，段边界向前后各多扩多少毫秒（默认 200）。
# 目的：盖掉落在 diarization 边界之外的**原声句末气声/尾音**（如日配句末的「はぁ」），
# 否则成片会出现"中文念完紧跟一声原声尾气"。相邻段之间按间距的一半做重叠保护。
# 设 0 可回到旧行为。
BG_PAD_MS = int(os.environ.get("BG_PAD_MS", "200"))

# --- 逐段响度对齐（见文件头说明）---------------------------------------------
LOUDNESS_MATCH = _env_bool("TTS_LOUDNESS_MATCH", True)
TTS_LOUDNESS_OFFSET = float(os.environ.get("TTS_LOUDNESS_OFFSET", "0"))
TTS_PEAK_CEIL = float(os.environ.get("TTS_PEAK_CEIL", "-1.0"))
TTS_MIN_RMS = float(os.environ.get("TTS_MIN_RMS", "-30.0"))

# 切片文件名形如 clip_00_009_34.80-35.20.wav
CLIP_NAME_RE = re.compile(r"^clip_\d+_\d+_([\d.]+)-([\d.]+)\.wav$", re.IGNORECASE)


def resolve_clip_path(segment):
    """找该段对应的原片切片（文件名是两位小数，JSON 里可能是一位小数，所以按数值容差匹配）"""
    speaker = str(segment.get("speaker", ""))
    start = float(segment.get("start", 0) or 0)
    end = float(segment.get("end", 0) or 0)
    clips_dir = os.path.join(project_root, "temp", "clips", speaker)

    matches = sorted(glob.glob(os.path.join(clips_dir, f"clip_*_{start:.2f}-{end:.2f}.wav")))
    if matches:
        return matches[0]
    for candidate in sorted(glob.glob(os.path.join(clips_dir, "clip_*.wav"))):
        m = CLIP_NAME_RE.match(os.path.basename(candidate))
        if m and abs(float(m.group(1)) - start) <= 0.01 and abs(float(m.group(2)) - end) <= 0.01:
            return candidate
    return None


def _match_loudness(tts_audio, ref_path):
    """把 TTS 段的响度对齐到该段原声切片的 RMS（保留表演动态），再做峰值保护。

    拿不到原声切片时回退到旧的 normalize(-3dBFS) 并告警。
    """
    fallback = tts_audio.normalize(-3.0)
    if not LOUDNESS_MATCH or not ref_path or not os.path.exists(ref_path):
        return fallback

    try:
        ref = AudioSegment.from_wav(ref_path)
    except Exception as e:  # noqa: BLE001
        print(f"   ! 读不到原声切片（{e}），回退到按峰值 -3dBFS")
        return fallback

    if len(ref) < 100 or len(tts_audio) < 100:
        return fallback

    ref_rms = ref.dBFS            # pydub 的 dBFS 就是 RMS dBFS
    tts_rms = tts_audio.dBFS
    if ref_rms == float("-inf") or tts_rms == float("-inf"):
        return fallback

    target = max(ref_rms + TTS_LOUDNESS_OFFSET, TTS_MIN_RMS)
    gain = max(-30.0, min(30.0, target - tts_rms))   # 限幅，避免异常段被放大/压死
    out = tts_audio.apply_gain(gain)

    # 峰值保护：叠到背景上之前留 headroom，避免求和削顶
    over = out.max_dBFS - TTS_PEAK_CEIL
    if over > 0:
        out = out.apply_gain(-over)

    print(f"   响度对齐: 原声 {ref_rms:.1f} / TTS {tts_rms:.1f} → {out.dBFS:.1f} dBFS "
          f"(增益 {gain:+.1f}dB{', 峰值保护 -%.1f' % over if over > 0 else ''})")
    return out

# ---------------------------------------------------------------------------
# 输出采样率（重要，别删）
#   ffmpeg 的 loudnorm 滤镜内部按 192kHz 处理，**不显式指定 -ar 就会把输出落成 192000Hz**，
#   接着 AAC 编码器上限是 96kHz → ffmpeg 自动再插一次重采样，成片音轨就变成 96k AAC
#   （非常规、部分播放器不认、同码率下高频位被浪费）。
#   源音频是 44.1kHz（bg/TTS 混音后也是 44.1kHz），所以这里统一钉死 44.1k，
#   整条链 44.1k → 44.1k，不再有 192k 中转。
# ---------------------------------------------------------------------------
OUTPUT_SR = int(os.environ.get("OUTPUT_SR", "44100"))


def probe_video_size(video_path, fallback=(852, 480)):
    """用 ffprobe 探测视频分辨率（字幕 PlayRes 必须与视频一致，否则字幕错位/大小失真）"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            w, h = out.stdout.strip().split("x")[:2]
            w, h = int(w), int(h)
            if w > 0 and h > 0:
                return w, h
    except Exception as e:  # noqa: BLE001
        print(f"Warning: ffprobe 探测分辨率失败（{e}），回退到 {fallback[0]}x{fallback[1]}")
    return fallback


def align_audio_to_duration(tts_filepath, target_ms):
    """用 atempo（保音高）把 TTS 音频伸缩到目标时长；返回新的 AudioSegment。

    偏差在阈值内 / 倍率超限时返回原始音频。
    """
    audio = AudioSegment.from_wav(tts_filepath)
    src_ms = len(audio)
    if not ALIGN_TTS or target_ms <= 0 or src_ms <= 0:
        return audio

    ratio = src_ms / target_ms  # >1 说明 TTS 比原段长，需要加速
    if abs(ratio - 1.0) < ALIGN_MIN_DEV:
        return audio

    tempo = min(max(ratio, 1.0 / ALIGN_MAX_RATE), ALIGN_MAX_RATE)
    if abs(tempo - 1.0) < 0.01:
        return audio

    tmp_path = os.path.join(project_root, "temp", f"_align_tmp_{os.getpid()}.wav")
    try:
        cmd = ["ffmpeg", "-y", "-i", tts_filepath, "-filter:a", f"atempo={tempo:.4f}",
               "-c:a", "pcm_s16le", tmp_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            print(f"   ! 对齐失败（保留原速）: {res.stderr.strip().splitlines()[-1] if res.stderr else 'unknown'}")
            return audio
        return AudioSegment.from_wav(tmp_path)
    except Exception as e:  # noqa: BLE001
        print(f"   ! 对齐异常（保留原速）: {e}")
        return audio
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def fit_tts_audio(tts_filepath, target_ms, ref_path=None):
    """加载 TTS 音频 -> 按段对齐 -> 淡入淡出 -> **按原声响度对齐**，返回 (audio, 溢出毫秒)

    ref_path 是该段原片切片：用它当响度基准（拿不到就回退旧的按峰值 -3dBFS）。
    """
    audio = align_audio_to_duration(tts_filepath, target_ms)

    if TTS_FADE_MS > 0 and len(audio) > TTS_FADE_MS * 2:
        audio = audio.fade_in(TTS_FADE_MS).fade_out(TTS_FADE_MS)

    overflow_ms = 0
    if target_ms > 0 and len(audio) > target_ms:
        overflow_ms = len(audio) - target_ms
        if ALIGN_TRIM_OVERFLOW:
            audio = audio[:target_ms].fade_out(min(TTS_FADE_MS * 2, 60))

    # 逐段响度对齐到该段原声（不是按峰值！见文件头说明）+ 峰值保护
    return _match_loudness(audio, ref_path), overflow_ms


def _run_ffmpeg(args, desc=""):
    """执行 ffmpeg 命令。

    用列表传参（不拼 shell 字符串）：路径含空格/中文/括号都不会炸；
    并检查 returncode，失败直接抛错，避免像 os.system 那样静默"成功"。
    """
    cmd = ["ffmpeg"] + [str(a) for a in args]
    print("  $ " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tail = "\n".join((res.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg 执行失败（{desc}，returncode={res.returncode}）:\n{tail}")
    return res


def get_existing_audio():
    """获取已存在的音频文件"""
    existing_audio_path = os.path.join(project_root, "temp", "output_audio.wav")
    
    if os.path.exists(existing_audio_path):
        print(f"Using existing audio file: {existing_audio_path}")
        return existing_audio_path
    else:
        # 如果不存在，则从原始视频中提取音频
        print("Existing audio file not found, extracting audio from original video...")
        video_path = os.path.join(project_root, "1.mp4")
        extracted_audio_path = os.path.join(project_root, "temp", "output_audio.wav")
        
        print("Extracting audio from original video...")
        _run_ffmpeg(["-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "44100", "-ac", "2", extracted_audio_path],
                    "从原视频抽取音频")
        print(f"Audio extracted to: {extracted_audio_path}")
        
        return extracted_audio_path

def insert_background_audio(original_audio_path, bg_audio_path, segments):
    """根据时间戳把原声替换成纯伴奏（背景音）

    为什么要在段边界外**多扩一点**（BG_PAD_MS）：
        diarization/VAD 给出的是"语音段"边界，而原片在句末往往还有一小截
        气声/尾音/呼吸（例如日配句末的「はぁ」）。它落在 [start, end] 之外，
        原来的实现只替换 [start, end]，于是这段**原声气声被原样保留**，
        成片听起来就是"中文台词念完，紧跟着一声日语尾气"。
        这里按 BG_PAD_MS 向前后各扩一点，把尾巴盖掉；
        并按相邻段的间距做**重叠保护**（最多扩到间距的一半），不会吃掉邻段开头。
    """
    pad_ms = BG_PAD_MS
    print("Loading original audio...")
    original_audio = AudioSegment.from_wav(original_audio_path)
    
    print("Loading background audio...")
    bg_audio = AudioSegment.from_wav(bg_audio_path)
    
    print("Inserting background audio into specified positions...")
    if pad_ms > 0:
        print(f"段边界外扩 {pad_ms}ms（相邻段按间距一半做重叠保护）")
    # 分别保存原始音频和背景音频，便于后续独立处理
    audio_with_bg = original_audio
    bg_segments_positions = []
    
    # 先按时间正序算好每段"实际要替换的区间"（扩边界需要看邻居，倒序算不方便）
    ordered = sorted(segments, key=lambda x: x['start'])
    spans = []
    for i, seg in enumerate(ordered):
        start_ms = int(round(seg['start'] * 1000))
        end_ms = int(round(seg['end'] * 1000))
        pad_start = pad_end = pad_ms
        if pad_ms > 0:
            if i > 0:
                gap = start_ms - int(round(ordered[i - 1]['end'] * 1000))
                pad_start = max(0, min(pad_ms, gap // 2))
            if i < len(ordered) - 1:
                gap = int(round(ordered[i + 1]['start'] * 1000)) - end_ms
                pad_end = max(0, min(pad_ms, gap // 2))
        spans.append((start_ms - pad_start, end_ms + pad_end, seg))

    # 按时间倒序处理，避免位置偏移
    for start_position, end_position, _segment in sorted(spans, key=lambda x: x[0], reverse=True):
        start_position = max(0, start_position)
        end_position = max(start_position, end_position)
        
        # 从背景音频中提取对应片段
        bg_segment = bg_audio[start_position:end_position]
        
        # 直接将背景音频片段插入到指定位置
        before_insert = audio_with_bg[:start_position]
        after_insert = audio_with_bg[end_position:]
        audio_with_bg = before_insert + bg_segment + after_insert
        
        # 保存背景音频片段的位置信息，便于后续处理
        bg_segments_positions.append({
            'start': start_position,
            'end': end_position,
            'segment': bg_segment
        })
        
        print(f"Background audio inserted at {start_position / 1000:.2f}s - {end_position / 1000:.2f}s position")
    
    return audio_with_bg, bg_segments_positions

def overlay_tts_audio(final_audio, tts_output_dir, segments):
    """将TTS音频叠加到对应位置，同时保持背景音乐音量

    每段先经 fit_tts_audio() 处理（按原段时长 atempo 对齐 -> 淡入淡出 -> 归一化），
    再按 start 叠加。这样短了不留空档、长了不会盖到下一句上。
    未能对齐到段长（超长且未开裁剪）的段会在末尾汇总告警。
    """
    print("Overlaying TTS audio to corresponding positions...")
    
    overflow_segments = []   # (文件名, 目标毫秒, 实际毫秒, 超出毫秒) —— 超长且未裁剪
    missing_files = []       # 找不到 TTS 文件的段
    
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        text = segment.get('result_text', '')
        
        # 目标时长 = 原段时长（毫秒）；同声配音要让每段 TTS 正好填满它自己那一段
        target_ms = int(round((end_time - start_time) * 1000))
        
        # 构造TTS音频文件名
        speaker_num = speaker.split('_')[1]
        start_str = f"{start_time:.2f}"
        end_str = f"{end_time:.2f}"
        
        tts_filename = f"result_{speaker_num}_{start_str}-{end_str}.wav"
        tts_filepath = os.path.join(tts_output_dir, tts_filename)
        
        # 检查TTS音频文件是否存在
        if not os.path.exists(tts_filepath):
            print(f"Warning: Cannot find TTS audio file {tts_filepath}")
            missing_files.append(tts_filename)
            continue
        
        print(f"Processing: {tts_filename}")
        print(f"Text: {text}")
        
        # 对齐（atempo 保音高）-> 淡入淡出 -> 按该段原声响度对齐 + 峰值保护
        ref_clip = resolve_clip_path(segment)
        tts_audio, overflow_ms = fit_tts_audio(tts_filepath, target_ms, ref_clip)
        if overflow_ms > 0:
            if ALIGN_TRIM_OVERFLOW:
                print(f"   ! 超出段长 {overflow_ms / 1000:.2f}s，已裁剪到段长")
            else:
                overflow_segments.append((tts_filename, target_ms, len(tts_audio), overflow_ms))
        
        # 计算叠加位置（毫秒）
        overlay_position = int(start_time * 1000)
        
        # 将TTS音频叠加到最终音频的指定位置（直接叠加，不处理原音频）
        final_audio = final_audio.overlay(tts_audio, position=overlay_position)
        
        print(f"TTS audio overlaid at {start_time} seconds position "
              f"(时长 {len(tts_audio) / 1000:.2f}s / 目标 {target_ms / 1000:.2f}s)")
    
    if overflow_segments:
        print("")
        print(f"Warning: {len(overflow_segments)} 段 TTS 对齐后仍长于原段（未裁剪，会盖到后一句上）：")
        for name, target_ms, actual_ms, overflow_ms in overflow_segments:
            print(f"   - {name}: 实际 {actual_ms / 1000:.2f}s vs 目标 {target_ms / 1000:.2f}s"
                  f"（超出 {overflow_ms / 1000:.2f}s）")
        print("   可选：调大 ALIGN_MAX_RATE 让它被拉伸得更狠，"
              "或设 ALIGN_TRIM_OVERFLOW=true 自动裁到段长。")
    if missing_files:
        print(f"Warning: {len(missing_files)} 段找不到 TTS 音频，未叠加：{', '.join(missing_files)}")
    
    return final_audio

def generate_ass_subtitle(segments, output_ass_path, video_width=852, video_height=480):
    """生成ASS字幕文件，包含原始文本和翻译文本

    PlayResX / PlayResY 必须与视频真实分辨率一致，否则字幕会错位、大小失真。
    字号/描边/边距按 480p（原始设计基准）等比缩放，保证不同分辨率下观感一致。
    """
    print(f"Generating ASS subtitle file... (PlayRes {video_width}x{video_height})")
    
    # 以 480p 设计稿为基准等比缩放
    scale = (video_height / 480.0) if video_height else 1.0
    fs_result = max(1, round(27 * scale))    # 译文（主字幕）
    fs_raw = max(1, round(18 * scale))       # 原文（副字幕）
    outline = max(1, round(2 * scale))
    shadow = max(1, round(1 * scale))
    mv_result = max(1, round(10 * scale))
    mv_raw = max(1, round(30 * scale))
    
    # ASS文件头部信息
    ass_header = f"""[Script Info]
Title: Bilingual Subtitles
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ResultText,Arial,{fs_result},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{mv_result},1
Style: RawText,Arial,{fs_raw},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{mv_raw},1
Style: Default,Arial,{fs_raw},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{mv_result},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            raw_text = segment.get('raw_text', '')
            result_text = segment.get('result_text', '')
            
            # 将秒数转换为ASS时间格式 (HH:MM:SS.cc)
            def seconds_to_ass_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                centiseconds = int((seconds * 100) % 100)
                return f"{hours:01d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
            
            start_ass_time = seconds_to_ass_time(start_time)
            end_ass_time = seconds_to_ass_time(end_time)
            
            # 双行字幕：上面是翻译文本(result_text)，下面是原始文本(raw_text)
            # 使用不同样式分别设置字体大小
            # 处理换行符，确保文本可以正确换行显示
            raw_text_escaped = raw_text.replace('\n', '\\N').replace('\r', '')
            result_text_escaped = result_text.replace('\n', '\\N').replace('\r', '')
            subtitle_text = f"{{\\rResultText}}{result_text_escaped}\\N{{\\rRawText}}{raw_text_escaped}"
            
            f.write(f"Dialogue: 0,{start_ass_time},{end_ass_time},Default,,0,0,0,,{subtitle_text}\n")
    
    print(f"ASS subtitle file saved to: {output_ass_path}")

def merge_audio_with_video(video_path, final_audio_path, output_video_path, subtitle_path=None, burn_subtitles=False):
    """将处理后的音频与视频合并，根据需要封装软字幕或硬烧录字幕。

    响度归一化**只在 main() 里做一次**（第 ⑨ 步 loudnorm），
    这里不再加 `-af loudnorm`，避免二次整形动态。
    """
    print("Merging processed audio with video...")
    
    # 获取输出文件扩展名以确定格式
    output_extension = os.path.splitext(output_video_path)[1].lower()
    
    if subtitle_path and os.path.exists(subtitle_path):
        if burn_subtitles:
            # 硬烧录字幕到视频帧中（ass 滤镜渲染到画面）
            print("Burning subtitles into video frames...")
            # ffmpeg 滤镜内的路径需转义：反斜杠成对、盘符冒号加反斜杠
            escaped_subtitle_path = subtitle_path.replace('\\', '\\\\').replace(':', '\\:')
            ass_filter = f"ass='{escaped_subtitle_path}'"
            
            # 如果输出是MP4格式，先生成MKV再转码为MP4
            # 【保留】mp4 直接硬烧会失败，实测必须先出 mkv 再转 mp4，不要"优化"成一步
            if output_extension == ".mp4":
                temp_mkv_path = output_video_path.replace(".mp4", "_temp.mkv")
                try:
                    _run_ffmpeg([
                        "-y",
                        "-i", video_path,
                        "-i", final_audio_path,
                        "-map", "0:v", "-map", "1:a",
                        "-vf", ass_filter,
                        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                        # 中间 mkv 的音轨用**无损 PCM**（反正是临时文件，几分钟后就删）：
                        # 这样音频只在「mkv 转 mp4」那步编码一次 AAC，
                        # 避免 aac->aac 两次有损编码带来的代际损失。
                        # 视频链路完全不动（该烧的字幕仍在 mkv 里烧好）。
                        "-c:a", "pcm_s16le", "-ar", str(OUTPUT_SR),
                        temp_mkv_path,
                    ], "硬烧字幕 -> mkv 中间文件")
                    
                    # 将MKV转码为MP4
                    print("Converting MKV to MP4 format...")
                    _run_ffmpeg([
                        "-y",
                        "-i", temp_mkv_path,
                        "-map", "0:v", "-map", "0:a",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                        "-c:a", "aac", "-ar", str(OUTPUT_SR), "-b:a", "192k",
                        output_video_path,
                    ], "mkv 转 mp4")
                finally:
                    # 删除临时MKV文件
                    if os.path.exists(temp_mkv_path):
                        os.remove(temp_mkv_path)
                
                print(f"MP4 video with burned-in subtitles saved to: {output_video_path}")
            else:
                # 直接生成MKV格式
                _run_ffmpeg([
                    "-y",
                    "-i", video_path,
                    "-i", final_audio_path,
                    "-map", "0:v", "-map", "1:a",
                    "-vf", ass_filter,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-c:a", "aac", "-ar", str(OUTPUT_SR),
                    output_video_path,
                ], "硬烧字幕 -> mkv")
                print(f"MKV video with burned-in subtitles saved to: {output_video_path}")
        else:
            # 软字幕封装
            if output_extension == ".mkv":
                print("Embedding soft subtitles into MKV video...")
                # MKV格式支持完整的ASS字幕，无需转换编码
                _run_ffmpeg([
                    "-y",
                    "-i", video_path,
                    "-i", final_audio_path,
                    "-i", subtitle_path,
                    "-c:v", "copy", "-c:a", "aac", "-ar", str(OUTPUT_SR),
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                    "-c:s", "copy",
                    output_video_path,
                ], "软字幕封装 -> mkv")
                print(f"MKV video with soft subtitles saved to: {output_video_path}")
            else:
                print("Embedding soft subtitles into MP4 video...")
                # MP4格式使用mov_text编码封装字幕
                _run_ffmpeg([
                    "-y",
                    "-i", video_path,
                    "-i", final_audio_path,
                    "-i", subtitle_path,
                    "-c:v", "copy", "-c:a", "aac", "-ar", str(OUTPUT_SR),
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                    "-c:s", "mov_text",
                    output_video_path,
                ], "软字幕封装 -> mp4")
                print(f"MP4 video with soft subtitles saved to: {output_video_path}")
    else:
        # 不使用字幕，直接合并音频和视频
        _run_ffmpeg([
            "-y",
            "-i", video_path,
            "-i", final_audio_path,
            "-c:v", "copy", "-c:a", "aac", "-ar", str(OUTPUT_SR),
            "-map", "0:v:0", "-map", "1:a:0",
            output_video_path,
        ], "合并音视频（无字幕）")
        if output_extension == ".mkv":
            print(f"MKV video saved to: {output_video_path}")
        else:
            print(f"MP4 video saved to: {output_video_path}")

def load_subtitle_data(json_path):
    """加载字幕数据"""
    if not os.path.exists(json_path):
        print(f"Warning: Subtitle data file does not exist: {json_path}")
        return None
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 检查数据格式
    if isinstance(data, dict) and 'segments' in data:
        segments = data['segments']
    elif isinstance(data, list):
        segments = data
    else:
        print("Warning: Subtitle data format is incorrect")
        return None
        
    # 验证每个片段是否包含所需字段
    for i, segment in enumerate(segments):
        if 'start' not in segment or 'end' not in segment:
            print(f"Warning: Segment {i} is missing start or end field")
            return None
            
    return segments

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Merge TTS audio with video')
    parser.add_argument('--enable-subtitles', action='store_true', help='Enable bilingual subtitles (raw_text and result_text)')
    parser.add_argument('--burn-subtitles', action='store_true', help='Burn subtitles directly into the video frames (hardsub)')
    parser.add_argument('--output-format', choices=['mp4', 'mkv'], default='mkv', help='Output video format (default: mkv)')
    args = parser.parse_args()
    
    # 定义路径（使用项目根目录）
    bg_audio_path = os.path.join(project_root, "temp", "bg_1_44100.wav")
    tts_output_dir = os.path.join(project_root, "results", "tts_output")
    diarization_file = os.path.join(project_root, "results", "speaker_diarization.json")
    video_path = os.path.join(project_root, "1.mp4")
    
    # 根据参数决定输出格式
    output_extension = "." + args.output_format
    output_video_path = os.path.join(project_root, "results", f"output_improved{output_extension}")
    final_audio_path = os.path.join(project_root, "temp", "final_audio_improved.wav")
    
    # 初始化字幕路径
    subtitle_path = None
    
    # 如果启用了字幕功能，则加载字幕数据
    if args.enable_subtitles:
        segments_data = load_subtitle_data(diarization_file)
        
        if segments_data:
            # 生成ASS字幕文件，统一命名为subtitle.ass，并产出到results目录中
            subtitle_path = os.path.join(project_root, "results", "subtitle.ass")
            # 字幕 PlayRes 必须与视频真实分辨率一致，否则字幕错位/大小失真
            video_width, video_height = probe_video_size(video_path)
            generate_ass_subtitle(segments_data, subtitle_path, video_width, video_height)
        else:
            print("Warning: Unable to load subtitle data, will continue processing without subtitles")
    
    # 读取说话人分割信息
    print("Reading speaker diarization information...")
    if not os.path.exists(diarization_file):
        print(f"Error: Speaker diarization file does not exist: {diarization_file}")
        return
        
    with open(diarization_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # 1. 获取原始音频文件
    original_audio_path = get_existing_audio()
    
    # 2. 将背景音频插入到指定位置
    audio_with_bg, bg_segments_positions = insert_background_audio(original_audio_path, bg_audio_path, segments)
    
    # 3. 将TTS音频叠加到对应位置
    final_audio = overlay_tts_audio(audio_with_bg, tts_output_dir, segments)
    
    # 对最终音频进行响度归一化处理
    print("Performing loudness normalization on final audio...")
    # 先保存为临时文件
    temp_audio_path = os.path.join(project_root, "temp", "temp_final_audio.wav")
    final_audio.export(temp_audio_path, format="wav")
    
    # 使用ffmpeg进行loudnorm处理（全片唯一一次响度归一化，合视频时不再重复）
    # -ar 必须给：否则 loudnorm 会把输出落成 192kHz（见文件头 OUTPUT_SR 的说明）
    normalized_audio_path = final_audio_path
    _run_ffmpeg(["-y", "-i", temp_audio_path,
                 "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                 "-ar", str(OUTPUT_SR),
                 normalized_audio_path], "全片响度归一化 loudnorm")
    
    # 删除临时文件
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    print(f"Loudness normalized audio saved to: {normalized_audio_path}")
    
    # 5. 将处理后的音频与视频合并
    merge_audio_with_video(video_path, final_audio_path, output_video_path, subtitle_path, args.burn_subtitles)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()