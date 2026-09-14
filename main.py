#!/usr/bin/env python3
"""
Main entry point for the Movie-trans Gradio interface.
This script provides a web UI for the video processing pipeline.
"""

import os
import sys
import subprocess
import json
import shutil
import gradio as gr
from typing import Optional

# ---------------------------------------------------------------------------
# Windows 编码兜底：stdout 被重定向 / 被管道接走时，Python 会退回本地编码
# （中文系统 = GBK），界面/日志里的 emoji（⚠️ ✅ 🎬 …）会抛 UnicodeEncodeError。
# 只放宽错误处理、**不改编码**，编不出的字符降级成 '?'，中文与子进程解码都不受影响。
# 注意：子进程（tools/*.py）在"被管道接走"时也会遇到同样问题，所以那些脚本里
# 有 emoji 的（asr.py / annotate.py）各自加了同一段兜底。
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:   # noqa: BLE001
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(PROJECT_DIR, "temp")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Ensure standard dirs exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add the project root to the Python path
sys.path.append(PROJECT_DIR)

# 全局：标注进程
ANNOTATE_PROCESS: Optional[subprocess.Popen] = None
ANNOTATE_PORT = 9871

# API Key 统一从项目根 .env 读取（环境变量优先），不再硬编码在代码里
from env_config import get_api_key

DEEPSEEK_API_KEY = get_api_key("DEEPSEEK_API_KEY")

# ==================== Helper Functions ====================
def safe_copy(src, dst):
    """Copy src to dst only if they are different files."""
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs != dst_abs:
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        shutil.copy2(src_abs, dst_abs)
    return dst_abs

def _find_nonempty_file(candidates):
    """Find first non-empty file in candidate paths."""
    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None

# ==================== Pipeline Functions ====================
def process_video(video_file_path, output_dir=None):
    try:
        output_dir = output_dir or TEMP_DIR
        os.makedirs(output_dir, exist_ok=True)

        process_script = os.path.join(PROJECT_DIR, "tools", "process_video.py")
        target_video_path = os.path.join(PROJECT_DIR, "1.mp4")
        expected_audio_path = os.path.join(output_dir, "output_audio.wav")

        safe_copy(video_file_path, target_video_path)

        cmd = [sys.executable, process_script, target_video_path, "--output-dir", output_dir]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=600
        )

        if result.returncode != 0:
            return None, f"❌ Error processing video:\n{result.stderr}"

        candidates = [
            expected_audio_path,
            os.path.join(PROJECT_DIR, "output_audio.wav"),
        ]
        real_path = _find_nonempty_file(candidates)

        if not real_path:
            return None, "⚠️ Audio file not found or empty."

        if real_path != expected_audio_path:
            safe_copy(real_path, expected_audio_path)
            real_path = expected_audio_path

        return real_path, f"✅ Video processed!\nAudio: {real_path}"

    except subprocess.TimeoutExpired:
        return None, "⏱️ Timeout: Video processing took too long."
    except Exception as e:
        return None, f"💥 Error: {type(e).__name__}: {str(e)}"

def denoise_audio(input_audio_path, model_name=None, agg=None, fp16=False, engine="roformer"):
    try:
        if not input_audio_path or not os.path.exists(input_audio_path):
            return None, None, None, "❌ Input audio file not found."

        os.makedirs(TEMP_DIR, exist_ok=True)

        target_input = os.path.join(TEMP_DIR, "output_audio.wav")
        safe_copy(input_audio_path, target_input)

        denoise_script = os.path.join(PROJECT_DIR, "tools", "denoise.py")
        cmd = [sys.executable, denoise_script, "--input", target_input, "--output-dir", TEMP_DIR]
        if engine:
            cmd += ["--engine", str(engine)]
        if model_name:
            cmd += ["--model-name", str(model_name)]
        if agg is not None:
            cmd += ["--agg", str(int(agg))]
        if fp16:
            cmd.append("--fp16")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=600
        )

        if result.returncode != 0:
            return None, None, None, f"❌ Denoise failed:\n{result.stderr}"

        vocal_16k = _find_nonempty_file([os.path.join(TEMP_DIR, "vocal_1_16000.wav")])
        vocal_44k = _find_nonempty_file([os.path.join(TEMP_DIR, "vocal_1_44100.wav")])
        bg = _find_nonempty_file([os.path.join(TEMP_DIR, "bg_1_44100.wav")])

        msgs = []
        if vocal_16k: msgs.append("16kHz vocal ✅")
        if vocal_44k: msgs.append("44.1kHz vocal ✅")
        if bg: msgs.append("Background ✅")

        if not msgs:
            return None, None, None, "⚠️ No denoised files found."

        return vocal_16k, vocal_44k, bg, "✅ Audio denoised!\n" + ", ".join(msgs)

    except subprocess.TimeoutExpired:
        return None, None, None, "⏱️ Timeout: Denoising took too long."
    except Exception as e:
        return None, None, None, f"💥 Error: {type(e).__name__}: {str(e)}"

def run_speaker_diarization(audio_file_path, expected_json=None,
                            cluster_threshold=None, min_cluster_size=None,
                            min_duration_off=None):
    try:
        if not audio_file_path or not os.path.exists(audio_file_path):
            return None, "❌ Input audio not found."

        if expected_json is None:
            expected_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        target_audio = os.path.join(TEMP_DIR, "vocal_1_16000.wav")
        safe_copy(audio_file_path, target_audio)

        diar_script = os.path.join(PROJECT_DIR, "tools", "speaker_diarization.py")
        cmd = [sys.executable, diar_script, "--audio", target_audio, "--output", expected_json]
        if cluster_threshold is not None:
            cmd += ["--threshold", str(cluster_threshold)]
        if min_cluster_size is not None:
            cmd += ["--min-cluster-size", str(int(min_cluster_size))]
        if min_duration_off is not None:
            cmd += ["--min-duration-off", str(min_duration_off)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=os.path.dirname(diar_script)
        )
        
        if result.returncode != 0:
            error_msg = f"❌ Diarization failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            print(error_msg)  # 打印错误信息以便调试
            return None, error_msg

        if os.path.exists(expected_json) and os.path.getsize(expected_json) > 0:
            try:
                with open(expected_json, 'r', encoding='utf-8') as f:
                    json.load(f)
                return expected_json, f"✅ Diarization done!\nJSON: {expected_json}"
            except json.JSONDecodeError as e:
                return None, f"⚠️ Invalid JSON: {str(e)}"
        else:
            return None, "⚠️ JSON not created or empty."

    except subprocess.TimeoutExpired:
        return None, "⏱️ Timeout: Diarization took too long."
    except Exception as e:
        return None, f"💥 Error: {type(e).__name__}: {str(e)}"

def run_merge_speaker_segments(json_file_path, max_gap=None, min_duration=None, max_duration=None):
    try:
        if not json_file_path or not os.path.exists(json_file_path):
            return None, "❌ Input JSON not found."

        target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
        safe_copy(json_file_path, target_json)

        merge_script = os.path.join(PROJECT_DIR, "tools", "merge_speaker_segments.py")
        cmd = [sys.executable, merge_script, "--input", target_json, "--output", target_json]
        if max_gap is not None:
            cmd += ["--max-gap", str(max_gap)]
        if min_duration is not None:
            cmd += ["--min-duration", str(min_duration)]
        if max_duration is not None:
            cmd += ["--max-duration", str(max_duration)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=300
        )

        if result.returncode != 0:
            return None, f"❌ Merge failed:\n{result.stderr}"

        if os.path.exists(target_json) and os.path.getsize(target_json) > 0:
            return target_json, f"✅ Segments merged!\nJSON: {target_json}"
        else:
            return None, "⚠️ Merged JSON not updated."

    except subprocess.TimeoutExpired:
        return None, "⏱️ Timeout: Merge took too long."
    except Exception as e:
        return None, f"💥 Error: {type(e).__name__}: {str(e)}"

def run_create_clips(audio_file_path, json_file_path):
    try:
        if not os.path.exists(audio_file_path):
            return None, f"❌ Audio not found: {audio_file_path}"
        if not os.path.exists(json_file_path):
            return None, f"❌ JSON not found: {json_file_path}"

        clips_dir = os.path.join(TEMP_DIR, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        target_audio = os.path.join(TEMP_DIR, "vocal_1_44100.wav")
        target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
        safe_copy(audio_file_path, target_audio)
        safe_copy(json_file_path, target_json)

        clips_script = os.path.join(PROJECT_DIR, "tools", "test_clips.py")
        cmd = [
            sys.executable, clips_script,
            "--audio", target_audio,
            "--json", target_json,
            "--output-dir", clips_dir
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=600
        )

        if result.returncode != 0:
            return None, f"❌ Clip creation failed:\n{result.stderr}"

        if os.path.isdir(clips_dir) and len(os.listdir(clips_dir)) > 0:
            return clips_dir, f"✅ Clips created!\nDir: {clips_dir}"
        else:
            return None, "⚠️ Clips directory empty or missing."

    except subprocess.TimeoutExpired:
        return None, "⏱️ Timeout: Clip creation took too long."
    except Exception as e:
        return None, f"💥 Error: {type(e).__name__}: {str(e)}"

def run_asr(clips_dir, json_file_path, language, model_size=None, device=None, compute_type=None,
            ref_check=False):
    try:
        # 确保文件在标准位置
        expected_clips = os.path.join(TEMP_DIR, "clips")
        expected_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")

        if os.path.abspath(clips_dir) != os.path.abspath(expected_clips):
            if os.path.exists(expected_clips):
                shutil.rmtree(expected_clips)
            shutil.copytree(clips_dir, expected_clips)

        safe_copy(json_file_path, expected_json)

        # ✅ 仅传 asr.py 支持的参数！
        asr_script = os.path.join(PROJECT_DIR, "tools", "asr.py")
        cmd = [sys.executable, asr_script]
        if language and language != "auto":
            cmd += ["--language", language]
        if model_size:
            cmd += ["--model_size", str(model_size)]
        if device:
            cmd += ["--device", str(device)]
        if compute_type:
            cmd += ["--compute_type", str(compute_type)]
        if ref_check:
            # 参考切片体检：纯诊断，只打印体检表，不写 JSON、不产文件、不改合成参数
            cmd += ["--ref-check"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=600
        )

        if result.returncode != 0:
            return None, f"❌ ASR failed:\n{result.stderr}"

        if os.path.exists(expected_json) and os.path.getsize(expected_json) > 0:
            return expected_json, f"✅ ASR completed!\nJSON: {expected_json}"
        else:
            return None, "⚠️ ASR ran but JSON unchanged/empty."

    except subprocess.TimeoutExpired:
        return None, "⏱️ Timeout: ASR took too long."
    except Exception as e:
        return None, f"💥 Error: {type(e).__name__}: {str(e)}"

# ==================== Annotation UI Control ====================
def _free_annotate_port():
    """杀掉占用标注端口的进程（tracked 进程若已崩，端口可能被残留进程占着）"""
    try:
        if os.name == 'nt':  # Windows 没有 lsof，用 netstat + taskkill
            subprocess.run(
                f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{ANNOTATE_PORT}') do taskkill /F /PID %a",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:  # Unix-like
            subprocess.run(
                f"lsof -ti :{ANNOTATE_PORT} | xargs kill -9",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception:
        pass  # 没有占用进程时忽略


def is_annotate_running():
    """标注 WebUI 是否真的在跑（判断开/关状态只看进程，不看状态栏文字）"""
    return bool(ANNOTATE_PROCESS) and ANNOTATE_PROCESS.poll() is None


def start_annotate_ui(json_path: str, enabled: bool):
    """Start/Stop the annotation WebUI."""
    global ANNOTATE_PROCESS

    if not enabled:
        # Stop：先优雅终止，再把端口彻底让出来
        if ANNOTATE_PROCESS and ANNOTATE_PROCESS.poll() is None:
            try:
                ANNOTATE_PROCESS.terminate()
                ANNOTATE_PROCESS.wait(timeout=5)
            except Exception:
                try:
                    ANNOTATE_PROCESS.kill()
                except Exception:
                    pass
        _free_annotate_port()
        ANNOTATE_PROCESS = None
        return "⏹️ 已停止（端口已释放）", ""

    # Start
    if not json_path or not os.path.exists(json_path):
        return "❌ 请先完成 ASR 并生成 JSON 文件", ""

    target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
    safe_copy(json_path, target_json)

    # 启动前清掉端口占用
    _free_annotate_port()

    # Additional wait to ensure port is released
    import time
    time.sleep(1)

    annotate_script = os.path.join(PROJECT_DIR, "tools", "annotate.py")
    cmd = [sys.executable, annotate_script, "--load_json", target_json, "--port", str(ANNOTATE_PORT)]

    try:
        ANNOTATE_PROCESS = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=None,  # 改为None以便查看输出
            stderr=None,  # 改为None以便查看错误
        )

        import time
        time.sleep(3)
        if ANNOTATE_PROCESS.poll() is not None:
            ANNOTATE_PROCESS = None
            return "❌ 标注 WebUI 启动失败（请检查端口是否被占用 / 控制台报错）", ""

        url = f"http://localhost:{ANNOTATE_PORT}"

        # 自动打开浏览器
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            print(f"无法自动打开浏览器: {e}")

        # 状态文字与 get_annotate_status() 保持一致，避免前端状态与真实状态不符
        return f"✅ 运行中\n🔗 {url}", url

    except Exception as e:
        ANNOTATE_PROCESS = None
        return f"💥 启动异常: {type(e).__name__}: {str(e)}", ""

def get_annotate_status():
    """Get current annotation UI status."""
    if is_annotate_running():
        url = f"http://localhost:{ANNOTATE_PORT}"
        return f"✅ 运行中\n🔗 {url}", url
    else:
        return "⏹️ 未运行", ""

# ==================== Pipeline Wrappers ====================
def run_pipeline(video_file, output_dir):
    if video_file is None:
        # 第三个输出不动（保留用户当前填的路径），避免误清空
        return None, gr.update(), "⚠️ Please upload a video file."
    audio_path, message = process_video(video_file.name, output_dir)
    # 把真实音频路径回填到"降噪"输入框：抽完音频即可直接点降噪，
    # 不用手动改路径——手改极易用了上一部片子的旧 output_audio.wav
    path_update = audio_path if audio_path else gr.update()
    return audio_path, path_update, message

def run_denoise_pipeline(audio_file_path, model_name=None, agg=None, fp16=False, engine="roformer"):
    return denoise_audio(audio_file_path, model_name, agg, fp16, engine)

def run_asr_pipeline(vocal_16k_path, vocal_44k_path, json_file_path, language,
                     cluster_threshold=None, min_cluster_size=None,
                     max_gap=None, min_duration=None, max_duration=None,
                     model_size=None, device=None, compute_type=None, ref_check=True):
    # UI 里的 "auto" 表示"不传，用脚本自己的默认值"
    if model_size in (None, "", "auto"):
        model_size = None
    if device in (None, "", "auto"):
        device = None
    if compute_type in (None, "", "auto"):
        compute_type = None

    # 初始化状态日志
    status_log = []
    
    # 步骤1: 说话人分离
    status_log.append(
        f"🗣️ 开始说话人分离...（聚类阈值={cluster_threshold}，最小簇={min_cluster_size}）"
    )
    json_path, msg1 = run_speaker_diarization(
        vocal_16k_path, json_file_path,
        cluster_threshold=cluster_threshold,
        min_cluster_size=min_cluster_size,
    )
    if json_path:
        status_log.append("✅ 说话人分离完成！")
    else:
        status_log.append("❌ 说话人分离失败！")
        return None, None, "\n".join(status_log) + "\n" + msg1

    # 步骤2: 合并说话人片段
    status_log.append(
        f"🔗 开始合并相邻说话人片段...（间隔≤{max_gap}s 合并，<{min_duration}s 丢弃，上限 {max_duration}s）"
    )
    merged_json, msg2 = run_merge_speaker_segments(
        json_path, max_gap=max_gap, min_duration=min_duration, max_duration=max_duration
    )
    if merged_json:
        status_log.append("✅ 相邻说话人片段合并完成！")
    else:
        status_log.append("❌ 合并相邻说话人片段失败！")
        return None, None, "\n".join(status_log) + "\n" + msg1 + "\n" + msg2

    # 步骤3: 创建音频片段
    status_log.append("✂️ 开始创建音频片段...")
    clips_dir, msg3 = run_create_clips(vocal_44k_path, merged_json)
    if clips_dir:
        status_log.append("✅ 音频片段创建完成！")
    else:
        status_log.append("❌ 音频片段创建失败！")
        return None, None, "\n".join(status_log) + "\n" + msg1 + "\n" + msg2 + "\n" + msg3

    # 步骤4: 运行ASR
    status_log.append(f"📝 开始语音识别...（模型={model_size}，设备={device}）")
    if ref_check:
        status_log.append("🔎 ASR 后顺带打印「参考切片体检表」（只诊断，不影响合成）")
    final_json, msg4 = run_asr(clips_dir, merged_json, language,
                               model_size=model_size, device=device, compute_type=compute_type,
                               ref_check=ref_check)
    if final_json:
        status_log.append("✅ 语音识别完成！")
    else:
        status_log.append("❌ 语音识别失败！")
        return None, None, "\n".join(status_log) + "\n" + msg1 + "\n" + msg2 + "\n" + msg3 + "\n" + msg4

    status_log.append("🎉 ASR全流程完成！")
    return final_json, clips_dir, "\n".join(status_log) + "\n" + msg1 + "\n" + msg2 + "\n" + msg3 + "\n" + msg4

# ==================== Translation Functions ====================
def load_diarization_data(file_path: str):
    """加载说话人分割数据"""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading diarization data: {e}")
        return []

def save_translated_data(data: list, file_path: str):
    """保存翻译后的数据"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving translated data: {e}")

def translate_segments(json_file_path, raw_language, target_language, api_key):
    """翻译所有片段"""
    try:
        if not json_file_path or not os.path.exists(json_file_path):
            return None, "❌ Input JSON not found."
        
        # 加载数据
        segments = load_diarization_data(json_file_path)
        if not segments:
            return None, "❌ No segments found in JSON file."
        
        # 导入翻译模块
        sys.path.append(os.path.join(PROJECT_DIR, "tools"))
        from translate import translate_segments
        
        # 翻译数据
        translated_segments = translate_segments(segments, api_key, target_language, raw_language)
        
        # 保存翻译后的数据
        save_translated_data(translated_segments, json_file_path)
        
        return json_file_path, f"✅ Translation completed!\nJSON: {json_file_path}"
    except Exception as e:
        return None, f"💥 Translation error: {type(e).__name__}: {str(e)}"

# ==================== TTS & Merge Functions ====================
def run_batch_tts_func(json_file_path, lang, emo_mode, duration_factor,
                       spk_ref_mode="segment", emo_alpha=1.0,
                       max_mel_tokens=1815, sort_by_speaker=True, emo_vector=""):
    """运行批量TTS生成（生成器：逐行推送日志，长任务可实时看进度）"""
    if not json_file_path or not os.path.exists(json_file_path):
        yield "❌ Input JSON not found."
        return

    try:
        # 确保目标文件在标准位置
        target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
        safe_copy(json_file_path, target_json)

        # 运行 batch_tts.py，把 2.5 的选项全部通过环境变量传进去
        tts_script = os.path.join(PROJECT_DIR, "tools", "batch_tts.py")
        cmd = [sys.executable, tts_script]
        env = os.environ.copy()
        # 关键：子进程 stdout 是"管道"，Python 默认走**块缓冲**（攒满 8KB 才吐），
        # 所以 TTS 加载权重那半分钟的提示全卡在缓冲区里，界面看起来像"空转"。
        # 加这个环境变量让子进程逐行输出（实测差别就是"看不到进度"和"看得到"）。
        # 注意：**不要**在这里设 PYTHONIOENCODING —— 本 Popen 是按本地编码(GBK)解码的，
        # 子进程改成 UTF-8 会让中文全变乱码。
        env["PYTHONUNBUFFERED"] = "1"
        env.update({
            "TTS_LANG": str(lang).upper(),
            "TTS_DURATION_FACTOR": str(duration_factor),
            "TTS_EMO_MODE": emo_mode,
            "SPK_REF_MODE": spk_ref_mode,
            "TTS_EMO_ALPHA": str(emo_alpha),
            "TTS_MAX_MEL_TOKENS": str(int(max_mel_tokens)),
            "TTS_SORT_BY_SPEAKER": "true" if sort_by_speaker else "false",
        })
        if emo_vector and str(emo_vector).strip():
            env["TTS_EMO_VECTOR"] = str(emo_vector).strip()

        output_lines = []

        def tail():
            return "".join(output_lines[-100:])

        yield (
            f"🚀 启动批量 TTS\n"
            f"  语言={lang} | 情绪={emo_mode}(α={emo_alpha}) | 整片语速x{duration_factor}\n"
            f"  音色参考={spk_ref_mode} | 单段长度上限={int(max_mel_tokens)}\n"
            f"  首次运行需加载 ~7GB 权重，请耐心等待..."
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            cwd=PROJECT_DIR
        )

        # 实时读取输出并逐行推送
        for line in process.stdout:
            output_lines.append(line)
            yield tail()

        process.wait()

        if process.returncode != 0:
            yield f"❌ Batch TTS failed:\n{tail()}"
        else:
            yield f"✅ Batch TTS completed successfully!\n{tail()}"

    except Exception as e:
        yield f"💥 Error: {type(e).__name__}: {str(e)}"


def run_merge_tts_video_func(enable_subtitles, burn_subtitles, output_format,
                             align_tts=True, align_max_rate=1.25, align_min_dev=0.05,
                             align_trim_overflow=False, tts_fade_ms=15, bg_pad_ms=200):
    """运行视频合并（按段对齐 / 原声外扩参数经环境变量传给脚本）"""
    try:
        merge_script = os.path.join(PROJECT_DIR, "tools", "merge_tts_video_improved.py")
        cmd = [sys.executable, merge_script]

        if enable_subtitles:
            cmd.append("--enable-subtitles")
        if burn_subtitles:
            cmd.append("--burn-subtitles")
        cmd.extend(["--output-format", output_format])

        # 对齐 / 原声外扩参数通过环境变量传给脚本
        env = os.environ.copy()
        env.update({
            "ALIGN_TTS": "true" if align_tts else "false",
            "ALIGN_MAX_RATE": str(align_max_rate),
            "ALIGN_MIN_DEV": str(align_min_dev),
            "ALIGN_TRIM_OVERFLOW": "true" if align_trim_overflow else "false",
            "TTS_FADE_MS": str(int(tts_fade_ms)),
            "BG_PAD_MS": str(int(bg_pad_ms)),
        })

        prefix = (
            f"🎬 合并参数：字幕={enable_subtitles}(硬烧={burn_subtitles}) 格式={output_format}\n"
            f"   按段对齐={align_tts}(最大倍率 {align_max_rate}, 偏差阈值 {align_min_dev}, "
            f"裁剪超长={align_trim_overflow}, 淡入淡出 {int(tts_fade_ms)}ms, 原声外扩 {int(bg_pad_ms)}ms)\n"
        )

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=1200, env=env
        )

        if result.returncode != 0:
            return f"❌ Merge TTS & Video failed:\n{prefix}\n{result.stderr}"

        output_video_path = os.path.join(RESULTS_DIR, f"output_improved.{output_format}")
        if os.path.exists(output_video_path):
            return (f"{prefix}✅ Merge TTS & Video completed successfully!\n"
                    f"Output video: {output_video_path}\n{result.stdout}"), output_video_path
        else:
            return f"{prefix}⚠️ Merge completed but output video not found.\n{result.stdout}", None

    except subprocess.TimeoutExpired:
        return "⏱️ Timeout: Merge TTS & Video took too long.", None
    except Exception as e:
        return f"💥 Error: {type(e).__name__}: {str(e)}", None
# ==================== Gradio UI ====================
def load_json_content(json_file):
    """读取 JSON 供预览面板显示"""
    if json_file and os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"JSON 加载失败: {str(e)}"}
    return {"info": "请选择 JSON 文件"}


with gr.Blocks(title="Movie-trans 视频处理全流程") as demo:
    gr.Markdown("# 🎬 Movie-trans 视频处理全流程")
    gr.Markdown(
        "抽音频 → 降噪分离 → 说话人分离 / ASR → 标注校对 → 翻译 → 批量 TTS → 回填合片。\n\n"
        "> **⚙️ 折叠区 = 高级参数**，都已按最优值预设，不确定就别动。\n"
        "> **TTS 必须单进程线性跑**：同时开两个进程不会报 OOM，而是静默降质（听起来像严重失真的噪声）。"
    )

    # ==================== ① 抽音频 & 降噪 ====================
    with gr.Tab("① 抽音频 & 降噪"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📤 从视频抽音频")
                    video_input = gr.File(label="上传视频", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                    output_dir = gr.Textbox(label="音频输出目录", value=TEMP_DIR)
                    process_btn = gr.Button("🚀 抽取音频", variant="primary")

                with gr.Group():
                    gr.Markdown("### 🔇 降噪 & 人声分离")
                    audio_input_path = gr.Textbox(
                        label="待处理音频", value=os.path.join(TEMP_DIR, "output_audio.wav"),
                        info="抽完音频会自动回填，直接点降噪即可"
                    )
                    denoise_btn = gr.Button("🔊 开始降噪", variant="primary")

                    with gr.Accordion("⚙️ 高级：降噪 / 人声分离参数", open=False):
                        denoise_engine = gr.Dropdown(
                            label="分离引擎", choices=["roformer", "uvr5"], value="roformer",
                            info="roformer=BS-Roformer（默认，A/B 实测效果最好）/ "
                                 "uvr5=VR 架构 HP2（更快、体积小，但 BGM 响的段落残留更多）"
                        )
                        denoise_model = gr.Textbox(
                            label="模型名", value="model_bs_roformer_ep_317_sdr_12.9755",
                            info="roformer 引擎填 .ckpt 模型名（如 model_bs_roformer_ep_317_sdr_12.9755）；"
                                 "uvr5 引擎填 .pth 模型名（如 HP2_all_vocals）。都不含扩展名"
                        )
                        denoise_agg = gr.Slider(
                            label="人声提取激进程度", minimum=0, maximum=20, value=10, step=1,
                            info="仅 uvr5 引擎有效；越高越激进地切掉伴奏，可能削到人声"
                        )
                        denoise_fp16 = gr.Checkbox(
                            label="半精度推理", value=True,
                            info="省显存。roformer 引擎本来就默认半精度，勾不勾都一样"
                        )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📁 结果")
                    audio_output = gr.Audio(label="抽取的音频")
                    with gr.Row():
                        vocal_16k_output = gr.Audio(label="🎤 人声 16kHz")
                        vocal_44k_output = gr.Audio(label="🎤 人声 44.1kHz")
                        bg_output = gr.Audio(label="🎧 背景音")
                    status_output = gr.Textbox(label="📝 状态", lines=12, elem_classes=["log-box"])

    # ==================== ② ASR & 切段 ====================
    with gr.Tab("② ASR & 切段"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🧠 输入")
                    asr_vocal_16k = gr.Textbox(
                        label="16kHz 人声（说话人分离用）",
                        value=os.path.join(TEMP_DIR, "vocal_1_16000.wav")
                    )
                    asr_vocal_44k = gr.Textbox(
                        label="44.1kHz 人声（切片段用）",
                        value=os.path.join(TEMP_DIR, "vocal_1_44100.wav")
                    )
                    asr_json_file = gr.Textbox(
                        label="说话人 JSON（输出位置）",
                        value=os.path.join(RESULTS_DIR, "speaker_diarization.json")
                    )
                    asr_language = gr.Dropdown(
                        label="语言", choices=["auto", "zh", "en", "ja", "ko"], value="auto",
                        info="zh 走 FunASR Paraformer；其余走 faster-whisper"
                    )
                    run_asr_btn = gr.Button("🎯 运行 ASR 全流程", variant="primary")

                with gr.Accordion("⚙️ 高级：说话人分离 / 切段粒度 / ASR", open=False):
                    with gr.Row():
                        asr_cluster_threshold = gr.Slider(
                            label="聚类阈值（越高说话人越少）",
                            minimum=0.40, maximum=0.95, value=0.72, step=0.01
                        )
                        asr_min_cluster_size = gr.Number(
                            label="最小簇大小（越小越易分出短插话）", value=15, precision=0
                        )
                    gr.Markdown(
                        "<small>**切段粒度**决定时间轴粗细，直接影响后面每段 TTS 的长短："
                        "间隔≤阈值合并、短于下限丢弃、合并后单段不超过上限。</small>"
                    )
                    with gr.Row():
                        asr_max_gap = gr.Slider(label="可合并间隔 (s)", minimum=0.0, maximum=1.5, value=0.3, step=0.05)
                        asr_min_duration = gr.Slider(label="丢弃短于 (s)", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                        asr_max_duration = gr.Slider(label="单段上限 (s)", minimum=2.0, maximum=30.0, value=10.0, step=0.5)
                    with gr.Row():
                        asr_model_size = gr.Dropdown(
                            label="Whisper 模型", choices=["large-v3", "large-v2", "medium", "small", "base"],
                            value="large-v3", info="仅非中文时生效"
                        )
                        asr_device = gr.Dropdown(label="设备", choices=["auto", "cuda", "cpu"], value="auto")
                        asr_compute_type = gr.Dropdown(
                            label="精度", choices=["auto", "float16", "int8", "float32"], value="auto"
                        )
                    asr_ref_check = gr.Checkbox(
                        label="ASR 后打印「参考切片体检」表（只诊断）", value=True,
                        info="纯只读：按「停顿降幅/语音占比/时长」标出哪些段的原片切片混了 BGM 残留，"
                             "不写 JSON、不产出文件、不影响合成。单独跑也可：python tools/asr.py --only-ref-check"
                    )

                with gr.Group():
                    gr.Markdown("### 📤 输出")
                    asr_json_output = gr.File(label="📄 转录 JSON")
                    asr_clips_dir = gr.Textbox(label="🎞️ 音频片段目录")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📋 状态")
                    asr_json_viewer = gr.JSON(label="🔍 结果预览")
                    asr_status_output = gr.Textbox(label="日志", lines=16, elem_classes=["log-box"])

                asr_json_output.change(fn=load_json_content, inputs=asr_json_output, outputs=asr_json_viewer)

    # ==================== ③ 翻译 & 标注 ====================
    with gr.Tab("③ 翻译 & 标注"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🌍 翻译")
                    trans_json_file = gr.Textbox(
                        label="说话人 JSON", value=os.path.join(RESULTS_DIR, "speaker_diarization.json")
                    )
                    with gr.Row():
                        trans_raw_language = gr.Dropdown(label="原始语言", choices=["zh", "en", "ja"], value="en")
                        trans_target_language = gr.Dropdown(label="目标语言", choices=["zh", "en", "ja"], value="zh")
                    trans_api_key = gr.Textbox(
                        label="DeepSeek API Key", value=DEEPSEEK_API_KEY, type="password",
                        info="留空则用 .env 里的 DEEPSEEK_API_KEY"
                    )
                    run_translate_btn = gr.Button("🔄 运行翻译", variant="primary")
                    gr.Markdown("<small>译文写入 `result_text`，原文保留在 `raw_text`。</small>")

                with gr.Group():
                    gr.Markdown("### 🏷️ 音频标注 WebUI（人工校对）")
                    with gr.Row():
                        annotate_status = gr.Textbox(label="状态", value="⏹️ 未运行", interactive=False, lines=2)
                        annotate_url = gr.Textbox(label="访问链接", interactive=False, lines=2)
                    with gr.Row():
                        start_annotate_btn = gr.Button("🚀 启动 / 停止标注", variant="primary")
                        refresh_status_btn = gr.Button("🔁 刷新状态")
                    gr.Markdown("<small>标注页可直接改原文/译文、对照试听 TTS、切分与合并片段。</small>")

                with gr.Group():
                    gr.Markdown("### 📤 输出")
                    trans_json_output = gr.File(label="📄 翻译 JSON")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📋 状态")
                    trans_json_viewer = gr.JSON(label="🔍 结果预览")
                    trans_status_output = gr.Textbox(label="日志", lines=16, elem_classes=["log-box"])

                trans_json_output.change(fn=load_json_content, inputs=trans_json_output, outputs=trans_json_viewer)

    # ==================== ④ TTS & 合片 ====================
    with gr.Tab("④ TTS & 合片"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🗣️ 批量 TTS（IndexTTS-2.5）")
                    gr.Markdown(
                        "<small>同声配音：每段译文用**它自己那段原片切片**同时作音色与情绪参考，"
                        "逐段跟随原声。</small>"
                    )
                    tts_json_file = gr.Textbox(
                        label="说话人 JSON", value=os.path.join(RESULTS_DIR, "speaker_diarization.json")
                    )
                    with gr.Row():
                        tts_lang = gr.Dropdown(label="合成语言", choices=["ZH", "EN", "JA", "ES", "AR"], value="ZH")
                        tts_emo_mode = gr.Dropdown(
                            label="情绪来源", choices=["ref", "text", "vector", "none"], value="ref",
                            info="ref=跟随原片表演（配音推荐）/ text=按台词推断 / vector=固定向量 / none"
                        )
                    tts_duration = gr.Slider(
                        label="整片语速因子（<1 变快，>1 变慢）",
                        minimum=0.5, maximum=2.0, value=1.0, step=0.05
                    )
                    run_batch_tts_btn = gr.Button("🎵 运行批量 TTS", variant="primary")

                    with gr.Accordion("⚙️ 高级：TTS 参数", open=False):
                        tts_spk_ref_mode = gr.Radio(
                            label="音色参考策略", choices=["segment", "fixed"], value="segment",
                            info="segment=每段用自身原片切片（同声配音正确解）；fixed=每说话人固定一段（更快但音色情绪会漂）"
                        )
                        tts_emo_alpha = gr.Slider(
                            label="情绪强度 α", minimum=0.0, maximum=1.0, value=1.0, step=0.05
                        )
                        tts_max_mel_tokens = gr.Slider(
                            label="单段长度上限 (mel tokens)", minimum=500, maximum=1815, value=1815, step=5,
                            info="2.5 上限 1815；调小会静默截断较长台词"
                        )
                        tts_sort_by_speaker = gr.Checkbox(
                            label="按说话人分组处理（仅 fixed 模式有提速收益）", value=True
                        )
                        tts_emo_vector = gr.Textbox(
                            label="情绪向量（仅 vector 模式，8 维逗号分隔）", value="0,0,0,0,0,0,0,1"
                        )

                with gr.Group():
                    gr.Markdown("### 🎞️ 与视频合并")
                    with gr.Row():
                        merge_enable_subtitles = gr.Checkbox(label="启用双语字幕", value=True)
                        merge_burn_subtitles = gr.Checkbox(label="硬烧录进画面（较慢）", value=True)
                        merge_output_format = gr.Radio(label="输出格式", choices=["mp4", "mkv"], value="mp4")
                    run_merge_tts_btn = gr.Button("🎬 运行合并", variant="primary")

                    with gr.Accordion("⚙️ 高级：按段时长对齐", open=False):
                        merge_align_tts = gr.Checkbox(
                            label="启用按段对齐（atempo 保音高，拉伸到原段时长）", value=True
                        )
                        with gr.Row():
                            merge_align_max_rate = gr.Slider(
                                label="最大伸缩倍率", minimum=1.0, maximum=2.0, value=1.25, step=0.05,
                                info="1.25 = 语速最多变 ±25%，防止为对齐把语速拉变形"
                            )
                            merge_align_min_dev = gr.Slider(
                                label="偏差阈值（低于此值不动）", minimum=0.0, maximum=0.3, value=0.05, step=0.01
                            )
                        merge_align_trim = gr.Checkbox(
                            label="仍超长时裁到段长（否则只在日志告警）", value=False
                        )
                        merge_fade_ms = gr.Slider(
                            label="每段淡入淡出 (ms)", minimum=0, maximum=200, value=15, step=5
                        )
                        merge_bg_pad = gr.Slider(
                            label="原声替换外扩 (ms)", minimum=0, maximum=500, value=200, step=10,
                            info="把原声换成伴奏时向前后多扩一点，盖掉段外的原声句末气声；相邻段按间距一半保护"
                        )

                with gr.Group():
                    gr.Markdown("### 📂 输出文件")
                    tts_output_video = gr.File(label="📥 最终视频", file_count="single")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📋 状态")
                    tts_status_output = gr.Textbox(label="🎵 TTS 日志", lines=16, elem_classes=["log-box"])
                    merge_status_output = gr.Textbox(label="🎬 合并日志", lines=16, elem_classes=["log-box"])

                run_batch_tts_btn.click(
                    fn=run_batch_tts_func,
                    inputs=[
                        tts_json_file, tts_lang, tts_emo_mode, tts_duration,
                        tts_spk_ref_mode, tts_emo_alpha, tts_max_mel_tokens,
                        tts_sort_by_speaker, tts_emo_vector,
                    ],
                    outputs=[tts_status_output]
                )

                run_merge_tts_btn.click(
                    fn=run_merge_tts_video_func,
                    inputs=[
                        merge_enable_subtitles, merge_burn_subtitles, merge_output_format,
                        merge_align_tts, merge_align_max_rate, merge_align_min_dev,
                        merge_align_trim, merge_fade_ms, merge_bg_pad,
                    ],
                    outputs=[merge_status_output, tts_output_video]
                )

    # ==================== Event bindings ====================
    process_btn.click(
        fn=run_pipeline,
        inputs=[video_input, output_dir],
        outputs=[audio_output, audio_input_path, status_output]
    )
    denoise_btn.click(
        fn=run_denoise_pipeline,
        inputs=[audio_input_path, denoise_model, denoise_agg, denoise_fp16, denoise_engine],
        outputs=[vocal_16k_output, vocal_44k_output, bg_output, status_output]
    )
    run_asr_btn.click(
        fn=run_asr_pipeline,
        inputs=[
            asr_vocal_16k, asr_vocal_44k, asr_json_file, asr_language,
            asr_cluster_threshold, asr_min_cluster_size,
            asr_max_gap, asr_min_duration, asr_max_duration,
            asr_model_size, asr_device, asr_compute_type, asr_ref_check,
        ],
        outputs=[asr_json_output, asr_clips_dir, asr_status_output]
    )

    # 翻译功能
    run_translate_btn.click(
        fn=translate_segments,
        inputs=[trans_json_file, trans_raw_language, trans_target_language, trans_api_key],
        outputs=[trans_json_output, trans_status_output]
    )

    # 标注控制
    def toggle_annotate(json_path):
        # 只看真实进程状态：之前靠 "运行中" 字符串判断，而启动返回的是 "已启动"，
        # 永远匹配不上 → 按钮只能开、关不掉
        if is_annotate_running():
            return start_annotate_ui(json_path, False)
        else:
            return start_annotate_ui(json_path, True)

    start_annotate_btn.click(
        fn=toggle_annotate,
        inputs=[trans_json_file],
        outputs=[annotate_status, annotate_url]
    )
    refresh_status_btn.click(
        fn=get_annotate_status,
        outputs=[annotate_status, annotate_url]
    )

if __name__ == "__main__":
    print(f"📁 PROJECT_DIR = {PROJECT_DIR}")
    print(f"📁 TEMP_DIR    = {TEMP_DIR}")
    print(f"📁 RESULTS_DIR = {RESULTS_DIR}")
    print(f"🌐 主 WebUI: http://localhost:7861")
    print(f"🔧 标注 WebUI 端口: {ANNOTATE_PORT}（需手动启动）")
    demo.launch(server_name="localhost", server_port=7861)
