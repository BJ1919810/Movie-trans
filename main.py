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

def denoise_audio(input_audio_path):
    try:
        if not input_audio_path or not os.path.exists(input_audio_path):
            return None, None, None, "❌ Input audio file not found."

        os.makedirs(TEMP_DIR, exist_ok=True)

        target_input = os.path.join(TEMP_DIR, "output_audio.wav")
        safe_copy(input_audio_path, target_input)

        denoise_script = os.path.join(PROJECT_DIR, "tools", "denoise.py")
        cmd = [sys.executable, denoise_script, "--input", target_input, "--output-dir", TEMP_DIR]
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

def run_speaker_diarization(audio_file_path, expected_json=None):
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

def run_merge_speaker_segments(json_file_path):
    try:
        if not json_file_path or not os.path.exists(json_file_path):
            return None, "❌ Input JSON not found."

        target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
        safe_copy(json_file_path, target_json)

        merge_script = os.path.join(PROJECT_DIR, "tools", "merge_speaker_segments.py")
        cmd = [sys.executable, merge_script, "--input", target_json, "--output", target_json]
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

def run_asr(clips_dir, json_file_path, language):
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
def start_annotate_ui(json_path: str, enabled: bool):
    """Start/Stop the annotation WebUI."""
    global ANNOTATE_PROCESS

    if not enabled:
        # Stop
        if ANNOTATE_PROCESS and ANNOTATE_PROCESS.poll() is None:
            try:
                ANNOTATE_PROCESS.terminate()
                ANNOTATE_PROCESS.wait(timeout=5)
            except:
                ANNOTATE_PROCESS.kill()
        return "⏹️ 标注 WebUI 已停止", ""

    # Start
    if not json_path or not os.path.exists(json_path):
        return "❌ 请先完成 ASR 并生成 JSON 文件", ""

    target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
    safe_copy(json_path, target_json)

    # Kill any process occupying ANNOTATE_PORT before starting
    try:
        # Windows doesn't have lsof, use netstat and taskkill instead
        if os.name == 'nt':  # Windows
            subprocess.run(f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{ANNOTATE_PORT}') do taskkill /F /PID %a", 
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:  # Unix-like systems
            subprocess.run(f"lsof -ti :{ANNOTATE_PORT} | xargs kill -9", 
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass  # Ignore errors if no process is found

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
            return "❌ 标注 WebUI 启动失败（请检查 annotate.py 权限）", ""

        url = f"http://localhost:{ANNOTATE_PORT}"
        
        # 自动打开浏览器
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            print(f"无法自动打开浏览器: {e}")
        
        return f"✅ 标注 WebUI 已启动！\n请访问: {url}", url

    except Exception as e:
        return f"💥 启动异常: {type(e).__name__}: {str(e)}", ""

def get_annotate_status():
    """Get current annotation UI status."""
    global ANNOTATE_PROCESS
    if ANNOTATE_PROCESS and ANNOTATE_PROCESS.poll() is None:
        url = f"http://localhost:{ANNOTATE_PORT}"
        return f"✅ 运行中\n🔗 {url}", url
    else:
        return "⏹️ 未运行", ""

# ==================== Pipeline Wrappers ====================
def run_pipeline(video_file, output_dir):
    if video_file is None:
        return None, "⚠️ Please upload a video file."
    return process_video(video_file.name, output_dir)

def run_denoise_pipeline(audio_file_path):
    return denoise_audio(audio_file_path)

def run_asr_pipeline(vocal_16k_path, vocal_44k_path, json_file_path, language):
    # 初始化状态日志
    status_log = []
    
    # 步骤1: 说话人分离
    status_log.append("🗣️ 开始说话人分离...")
    json_path, msg1 = run_speaker_diarization(vocal_16k_path, json_file_path)
    if json_path:
        status_log.append("✅ 说话人分离完成！")
    else:
        status_log.append("❌ 说话人分离失败！")
        return None, None, "\n".join(status_log) + "\n" + msg1

    # 步骤2: 合并说话人片段
    status_log.append("🔗 开始合并相邻说话人片段...")
    merged_json, msg2 = run_merge_speaker_segments(json_path)
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
    status_log.append("📝 开始语音识别...")
    final_json, msg4 = run_asr(clips_dir, merged_json, language)
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

# ==================== Gradio UI ====================
with gr.Blocks(title="Movie-trans Video Processing") as demo:
    gr.Markdown("# 🎬 Movie-trans 视频处理全流程")

    with gr.Tab("Extra & Denoise audio"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📤 视频处理")
                video_input = gr.File(label="上传视频", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                output_dir = gr.Textbox(label="输出目录", value=TEMP_DIR)
                process_btn = gr.Button("🚀 处理视频", variant="primary")

                gr.Markdown("## 🔇 降噪分离")
                audio_input_path = gr.Textbox(label="音频路径", value=os.path.join(TEMP_DIR, "output_audio.wav"))
                denoise_btn = gr.Button("🔊 降噪", variant="primary")

            with gr.Column():
                gr.Markdown("## 📁 结果")
                audio_output = gr.Audio(label="提取音频")
                with gr.Row():
                    vocal_16k_output = gr.Audio(label="🎤 人声 (16kHz)")
                    vocal_44k_output = gr.Audio(label="🎤 人声 (44.1kHz)")
                    bg_output = gr.Audio(label="🎧 背景音")
                status_output = gr.Textbox(label="📝 状态", lines=8)

    with gr.Tab("ASR"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🧠 ASR 流程")
                asr_vocal_16k = gr.Textbox(label="16kHz 人声", value=os.path.join(TEMP_DIR, "vocal_1_16000.wav"))
                asr_vocal_44k = gr.Textbox(label="44.1kHz 人声", value=os.path.join(TEMP_DIR, "vocal_1_44100.wav"))
                asr_json_file = gr.Textbox(label="说话人 JSON", value=os.path.join(RESULTS_DIR, "speaker_diarization.json"))
                asr_language = gr.Dropdown(label="语言", choices=["zh", "en", "ja", "ko", "auto"], value="auto")
                run_asr_btn = gr.Button("🎯 运行 ASR", variant="primary")

                gr.Markdown("## 📤 输出")
                asr_json_output = gr.File(label="📄 转录 JSON")
                asr_clips_dir = gr.Textbox(label="🎞️ 音频片段目录")

            with gr.Column():
                gr.Markdown("## 📋 ASR 状态")
                asr_json_viewer = gr.JSON(label="🔍 结果预览")
                asr_status_output = gr.Textbox(label="日志", lines=10)

                def load_json_content(json_file):
                    if json_file and os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                return json.load(f)
                        except Exception as e:
                            return {"error": f"JSON 加载失败: {str(e)}"}
                    return {"info": "请选择 JSON 文件"}

                asr_json_output.change(fn=load_json_content, inputs=asr_json_output, outputs=asr_json_viewer)

    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🌍 翻译流程")
                trans_json_file = gr.Textbox(label="说话人 JSON", value=os.path.join(RESULTS_DIR, "speaker_diarization.json"))
                trans_raw_language = gr.Dropdown(label="原始语言", choices=["zh", "en", "ja"], value="en")
                trans_target_language = gr.Dropdown(label="目标语言", choices=["zh", "en", "ja"], value="zh")
                trans_api_key = gr.Textbox(label="DeepSeek API Key", value=DEEPSEEK_API_KEY, type="password")
                run_translate_btn = gr.Button("🔄 运行翻译", variant="primary")
                
                # 标注 WebUI 控制
                gr.Markdown("## 🏷️ 音频标注 WebUI（人工校对）")
                with gr.Row():
                    annotate_status = gr.Textbox(label="状态", value="⏹️ 未运行", interactive=False, lines=2)
                    annotate_url = gr.Textbox(label="访问链接", interactive=False, lines=2)
                with gr.Row():
                    start_annotate_btn = gr.Button("🚀 启动/停止标注", variant="primary")
                    refresh_status_btn = gr.Button("🔁 刷新状态")

                gr.Markdown("## 📤 输出")
                trans_json_output = gr.File(label="📄 翻译 JSON")

            with gr.Column():
                gr.Markdown("## 📋 翻译状态")
                trans_json_viewer = gr.JSON(label="🔍 结果预览")
                trans_status_output = gr.Textbox(label="日志", lines=10)

                def load_json_content(json_file):
                    if json_file and os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                return json.load(f)
                        except Exception as e:
                            return {"error": f"JSON 加载失败: {str(e)}"}
                    return {"info": "请选择 JSON 文件"}

                trans_json_output.change(fn=load_json_content, inputs=trans_json_output, outputs=trans_json_viewer)

    # TTS & Merge 功能
    # ==================== TTS & Merge Functions ====================
    def run_batch_tts_func(json_file_path, lang, version, emo_mode, duration_factor):
        """运行批量TTS生成（生成器：逐行推送日志，长任务可实时看进度）"""
        import queue
        import threading

        if not json_file_path or not os.path.exists(json_file_path):
            yield "❌ Input JSON not found."
            return

        try:
            # 确保目标文件在标准位置
            target_json = os.path.join(RESULTS_DIR, "speaker_diarization.json")
            safe_copy(json_file_path, target_json)

            # 运行 batch_tts.py，把 2.5 的新选项通过环境变量传进去
            tts_script = os.path.join(PROJECT_DIR, "tools", "batch_tts.py")
            cmd = [sys.executable, tts_script]
            env = os.environ.copy()
            env.update({
                "TTS_LANG": lang,
                "TTS_DURATION_FACTOR": str(duration_factor),
                "TTS_EMO_MODE": emo_mode,
                "INDEXTTS_VERSION": version,
            })

            output_lines = []

            def tail():
                return "".join(output_lines[-100:])

            yield f"🚀 启动批量 TTS (lang={lang}, 版本={version}, 情绪={emo_mode}, 语速x{duration_factor})...\n首次运行需加载 ~7GB 权重，请耐心等待..."

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

    def run_merge_tts_video_func(enable_subtitles, burn_subtitles, output_format):
        """运行视频合并"""
        try:
            # 运行merge_tts_video_improved.py脚本
            merge_script = os.path.join(PROJECT_DIR, "tools", "merge_tts_video_improved.py")
            cmd = [sys.executable, merge_script]
            
            # 添加参数
            if enable_subtitles:
                cmd.append("--enable-subtitles")
            if burn_subtitles:
                cmd.append("--burn-subtitles")
            cmd.extend(["--output-format", output_format])
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=1200  # 20分钟超时
            )
            
            if result.returncode != 0:
                return f"❌ Merge TTS & Video failed:\n{result.stderr}"
            
            output_video_path = os.path.join(RESULTS_DIR, f"output_improved.{output_format}")
            if os.path.exists(output_video_path):
                return f"✅ Merge TTS & Video completed successfully!\nOutput video: {output_video_path}\n{result.stdout}", output_video_path
            else:
                return f"⚠️ Merge completed but output video not found.\n{result.stdout}", None
            
        except subprocess.TimeoutExpired:
            return "⏱️ Timeout: Merge TTS & Video took too long.", None
        except Exception as e:
            return f"💥 Error: {type(e).__name__}: {str(e)}", None

    with gr.Tab("TTS ＆ Merge"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🗣️ 批量TTS生成")
                tts_json_file = gr.Textbox(label="说话人 JSON", value=os.path.join(RESULTS_DIR, "speaker_diarization.json"), lines=2)

                # IndexTTS-2.5 新选项（通过环境变量传给 batch_tts.py）
                with gr.Row():
                    tts_lang = gr.Dropdown(label="合成语言", choices=["ZH", "EN", "JA", "ES", "AR"], value="ZH")
                    tts_version = gr.Radio(label="推理器版本", choices=["2.5", "2"], value="2.5")
                with gr.Row():
                    tts_emo_mode = gr.Radio(
                        label="情绪来源", choices=["ref", "text"], value="ref",
                        info="ref=跟随原片演员表演（推荐配音）；text=Qwen按台词字面推断（适合有声书）"
                    )
                    tts_duration = gr.Slider(
                        label="语速因子（<1 变快，>1 变慢）",
                        minimum=0.5, maximum=2.0, value=1.0, step=0.05
                    )

                run_batch_tts_btn = gr.Button("🎵 运行批量TTS", variant="primary")
                
                gr.Markdown("## 🎞️ 视频与人声合并")
                merge_enable_subtitles = gr.Checkbox(label="启用双语字幕", value=True)
                merge_burn_subtitles = gr.Checkbox(label="硬烧录字幕到视频帧（处理速度会比较慢）", value=True)
                merge_output_format = gr.Radio(label="输出格式", choices=["mp4", "mkv"], value="mp4")
                run_merge_tts_btn = gr.Button("🎬 运行视频合并", variant="primary")
                
                gr.Markdown("## 📂 输出文件")
                tts_output_video = gr.File(label="📥 最终视频文件", file_count="single")
                
                def update_output_file():
                    # 根据实际输出格式返回正确的文件路径
                    output_video_path_mp4 = os.path.join(RESULTS_DIR, "output_improved.mp4")
                    output_video_path_mkv = os.path.join(RESULTS_DIR, "output_improved.mkv")
                    if os.path.exists(output_video_path_mp4):
                        return output_video_path_mp4
                    elif os.path.exists(output_video_path_mkv):
                        return output_video_path_mkv
                    return None
                
            with gr.Column():
                gr.Markdown("## 📋 TTS & Merge 状态")
                tts_status_output = gr.Textbox(label="🎵 TTS 日志", lines=10)
                merge_status_output = gr.Textbox(label="🎬 合并日志", lines=10)
                
                # TTS & Merge Event Bindings
                run_batch_tts_btn.click(
                    fn=run_batch_tts_func,
                    inputs=[tts_json_file, tts_lang, tts_version, tts_emo_mode, tts_duration],
                    outputs=[tts_status_output]
                )

                run_merge_tts_btn.click(
                    fn=run_merge_tts_video_func,
                    inputs=[merge_enable_subtitles, merge_burn_subtitles, merge_output_format],
                    outputs=[merge_status_output, tts_output_video]
                )

    # Event bindings
    process_btn.click(
        fn=run_pipeline,
        inputs=[video_input, output_dir],
        outputs=[audio_output, status_output]
    )
    denoise_btn.click(
        fn=run_denoise_pipeline,
        inputs=[audio_input_path],
        outputs=[vocal_16k_output, vocal_44k_output, bg_output, status_output]
    )
    run_asr_btn.click(
        fn=run_asr_pipeline,
        inputs=[asr_vocal_16k, asr_vocal_44k, asr_json_file, asr_language],
        outputs=[asr_json_output, asr_clips_dir, asr_status_output]
    )
    
    # 翻译功能
    run_translate_btn.click(
        fn=translate_segments,
        inputs=[trans_json_file, trans_raw_language, trans_target_language, trans_api_key],
        outputs=[trans_json_output, trans_status_output]
    )
    
    # 标注控制
    def toggle_annotate(json_path, current_status):
        # 根据当前状态决定是启动还是停止
        if "运行中" in current_status:
            # 当前正在运行，需要停止
            return start_annotate_ui(json_path, False)
        else:
            # 当前未运行，需要启动
            return start_annotate_ui(json_path, True)
    
    start_annotate_btn.click(
        fn=toggle_annotate,
        inputs=[trans_json_file, annotate_status],
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
