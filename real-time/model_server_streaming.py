#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import traceback
import librosa
import soundfile as sf
import torch
from flask import Flask, request, jsonify, Response, stream_with_context
from pathlib import Path
import logging
import io
import base64
import numpy as np
import tempfile

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
localdir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
sys.path.append(localdir_path)

# 导入必要的模块
try:
    from uvr5.vr import AudioPre
except ImportError:
    logger.warning("无法导入uvr5模块")
    AudioPre = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    logger.warning("无法导入faster_whisper模块")
    WhisperModel = None

try:
    from funasr import AutoModel
except ImportError:
    logger.warning("无法导入funasr模块")
    AutoModel = None

import requests

# 配置参数
TEMP_DIR = os.path.join(project_path, "temp")
RESULTS_DIR = os.path.join(project_path, "results")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# UVR5配置
weight_uvr5_root = os.path.join(project_path, "uvr5", "uvr5_weights")
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = False
model_name = "HP2_all_vocals"  # hp2模型用于保留人声
agg = 10  # 人声提取激进程度
format0 = "wav"  # 导出文件格式

# ASR模型路径配置
asr_models_path = os.path.join(project_path, "asr", "models")
whisper_model_path = os.path.join(asr_models_path, "models--Systran--faster-whisper-large-v3")
funasr_asr_model_path = os.path.join(asr_models_path, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
funasr_vad_model_path = os.path.join(asr_models_path, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
funasr_punc_model_path = os.path.join(asr_models_path, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

# TTS模型路径配置
tts_checkpoint_path = os.path.join(project_path, "checkpoints")
default_ref_audio_path = os.path.join(project_path, "default_ref_voice.wav")  # 默认参考音频

# DeepSeek API配置（请替换为您的API密钥）
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 全局变量用于存储预加载的模型
preloaded_models = {}

# 初始化Flask应用
app = Flask(__name__)

def check_model_files(model_path, required_files=None):
    """
    检查模型文件是否存在且完整
    """
    if not os.path.exists(model_path):
        logger.warning(f"模型路径不存在: {model_path}")
        return False
    
    if required_files:
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                logger.warning(f"模型文件缺失: {os.path.join(model_path, file)}")
                return False
    
    # 检查是否有模型文件
    model_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.bin', '.pth', '.safetensors'))]
    if not model_files:
        logger.warning(f"模型路径中没有找到模型文件: {model_path}")
        return False
        
    logger.info(f"模型文件检查通过: {model_path}")
    return True

def preload_models():
    """
    预加载所有必需的模型
    """
    global preloaded_models
    
    try:
        logger.info("开始预加载模型...")
        
        # 1. 预加载UVR5模型
        logger.info("预加载UVR5模型...")
        uvr5_model_path = os.path.join(weight_uvr5_root, model_name + ".pth")
        if os.path.exists(uvr5_model_path) and AudioPre:
            preloaded_models['uvr5'] = AudioPre(
                agg=int(agg),
                model_path=uvr5_model_path,
                device=device,
                is_half=is_half,
            )
            logger.info("UVR5模型预加载完成")
        else:
            logger.error(f"UVR5模型文件不存在: {uvr5_model_path}")
            preloaded_models['uvr5'] = None
        
        # 2. 预加载FunASR模型（中文）
        logger.info("预加载FunASR中文模型...")
        if (AutoModel and
            os.path.exists(funasr_asr_model_path) and 
            os.path.exists(funasr_vad_model_path) and 
            os.path.exists(funasr_punc_model_path) and
            check_model_files(funasr_asr_model_path, ['model.pt']) and
            check_model_files(funasr_vad_model_path, ['model.pt']) and
            check_model_files(funasr_punc_model_path, ['model.pt'])):
            
            logger.info("使用本地模型路径加载FunASR...")
            preloaded_models['funasr_zh'] = AutoModel(
                model=funasr_asr_model_path,
                vad_model=funasr_vad_model_path,
                punc_model=funasr_punc_model_path,
            )
            logger.info("FunASR中文模型预加载完成")
        else:
            logger.warning("本地FunASR模型路径不存在或不完整，跳过加载")
            preloaded_models['funasr_zh'] = None
            
        # 3. 预加载Faster-Whisper模型（多语言）
        logger.info("预加载Faster-Whisper多语言模型...")
        # 查找whisper模型的实际路径
        whisper_actual_path = None
        if os.path.exists(whisper_model_path):
            # 检查是否有快照目录
            snapshots_path = os.path.join(whisper_model_path, "snapshots")
            if os.path.exists(snapshots_path):
                # 获取最新的快照目录
                snapshots = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
                if snapshots:
                    whisper_actual_path = os.path.join(snapshots_path, snapshots[0])
                    logger.info(f"找到Whisper模型快照路径: {whisper_actual_path}")
                else:
                    logger.warning("Whisper模型快照目录为空")
            else:
                # 直接使用模型目录
                whisper_actual_path = whisper_model_path
                logger.info(f"使用Whisper模型目录: {whisper_actual_path}")
        
        # 检查模型文件是否存在
        if WhisperModel and whisper_actual_path and check_model_files(whisper_actual_path):
            try:
                logger.info("使用本地Whisper模型路径加载...")
                preloaded_models['whisper'] = WhisperModel(
                    whisper_actual_path, 
                    device=device, 
                    compute_type="float16" if is_half and device=="cuda" else "float32"
                )
                logger.info("Faster-Whisper多语言模型预加载完成")
            except Exception as e:
                logger.error(f"加载Whisper模型失败: {e}")
                preloaded_models['whisper'] = None
        else:
            logger.warning("本地Whisper模型路径不存在或不完整，跳过加载")
            preloaded_models['whisper'] = None
            
        # 4. 初始化IndexTTS模型
        logger.info("初始化IndexTTS模型...")
        sys.path.append(os.path.join(project_path, 'index-tts'))
        
        # 检查TTS模型路径
        tts_config_path = os.path.join(project_path, "index-tts", "checkpoints", "config.yaml")
        tts_model_dir = os.path.join(project_path, "index-tts", "checkpoints")
        
        if os.path.exists(tts_config_path) and os.path.exists(tts_model_dir):
            try:
                from indextts.infer_v2 import IndexTTS2
                preloaded_models['tts'] = IndexTTS2(
                    cfg_path=tts_config_path,
                    model_dir=tts_model_dir,
                    use_fp16=True
                )
                logger.info("IndexTTS模型初始化完成(已启用DeepSpeed优化)")
            except Exception as e:
                logger.error(f"加载IndexTTS模型失败: {e}")
                preloaded_models['tts'] = None
        else:
            logger.warning("TTS模型文件不存在，跳过TTS模型加载")
            preloaded_models['tts'] = None
            
        # 5. 检查默认参考音频
        if not os.path.exists(default_ref_audio_path):
            logger.warning(f"默认参考音频不存在: {default_ref_audio_path}")
            # 尝试创建一个空的参考音频
            try:
                sr = 16000
                y = np.zeros(sr)  # 1秒静音
                sf.write(default_ref_audio_path, y, sr, subtype='PCM_16')
                logger.info("已创建默认参考音频(静音)")
            except Exception as e:
                logger.error(f"创建默认参考音频失败: {e}")
            
        logger.info("所有模型预加载完成！")
        return True
        
    except Exception as e:
        logger.error(f"模型预加载失败: {e}")
        traceback.print_exc()
        return False

def denoise_audio_hp2(input_audio_path):
    """
    使用HP2模型对音频进行降噪，只保留人声
    流程调整：
    1. 保留原始双声道输入
    2. UVR5处理双声道音频
    3. 从UVR5输出中提取右声道
    4. 生成16kHz单声道处理音频
    5. 生成16kHz单声道参考音频（完全移除44k逻辑）
    
    返回: (16kHz单声道处理音频, 16kHz单声道参考音频)
    """
    try:
        logger.info(f"🔊 使用HP2模型处理音频: {input_audio_path}")
        
        # 检查UVR5模型是否已加载
        if preloaded_models['uvr5'] is None:
            logger.error("UVR5模型未加载，无法进行降噪处理")
            return None, None
        
        # 1️⃣ 保留原始双声道输入（不再提前处理）
        logger.info("🎯 保留原始双声道输入，直接送入UVR5处理...")
        
        # 2️⃣ 使用预加载的模型处理原始音频
        pre_fun = preloaded_models['uvr5']
        output_vocal_path = TEMP_DIR
        output_ins_path = TEMP_DIR
        
        logger.info("🔄 UVR5模型处理中（保留原始双声道）...")
        pre_fun._path_audio_(input_audio_path, output_ins_path, output_vocal_path, format0)
        logger.info("✅ UVR5处理完成")
            
        # 3️⃣ 获取UVR5输出的人声文件
        vocal_filename = f"vocal_{os.path.basename(input_audio_path)}_{agg}.{format0}"
        vocal_file = os.path.join(output_vocal_path, vocal_filename)
        
        if not os.path.exists(vocal_file):
            logger.error(f"❌ 未找到UVR5处理后的人声文件: {vocal_file}")
            return None, None
        
        logger.info(f"🔍 找到UVR5输出文件: {vocal_file}")
        
        # 4️⃣ 从UVR5输出中提取右声道
        logger.info("🎤 从UVR5输出中提取右声道...")
        y, sr = librosa.load(vocal_file, sr=None, mono=False)
        
        # 添加音频数据有效性检查
        if not np.all(np.isfinite(y)):
            logger.warning("⚠️ 检测到音频数据中存在非有限值，正在进行清理...")
            y = np.nan_to_num(y, nan=0.0, posinf=0.95, neginf=-0.95)
        
        if y.ndim > 1:
            logger.info(f"🎧 检测到立体声音频（{sr}Hz），提取右声道...")
            right_channel = y[1]  # 右声道是索引1
        else:
            logger.info("_mono 检测到单声道音频，直接使用...")
            right_channel = y
            
        # 再次检查提取后的音频数据
        if not np.all(np.isfinite(right_channel)):
            logger.warning("⚠️ 右声道音频数据中存在非有限值，正在进行清理...")
            right_channel = np.nan_to_num(right_channel, nan=0.0, posinf=0.95, neginf=-0.95)
        
        # 5️⃣ 生成16kHz单声道处理音频（用于ASR）
        logger.info("🔄 重采样到16000Hz（用于ASR）...")
        vocal_processed_16k = librosa.resample(right_channel, orig_sr=sr, target_sr=16000)
        
        # 检查重采样后的音频数据
        if not np.all(np.isfinite(vocal_processed_16k)):
            logger.warning("⚠️ 重采样后的ASR音频数据中存在非有限值，正在进行清理...")
            vocal_processed_16k = np.nan_to_num(vocal_processed_16k, nan=0.0, posinf=0.95, neginf=-0.95)
            
        vocal_16k_path = os.path.join(output_vocal_path, "vocal_processed_16k.wav")
        sf.write(vocal_16k_path, vocal_processed_16k, 16000, subtype='PCM_16')
        
        # 6️⃣ 生成16kHz单声道参考音频（用于TTS，保持音色）
        logger.info("🔄 生成16kHz参考音频（用于TTS音色保持）...")
        ref_audio_16k = librosa.resample(right_channel, orig_sr=sr, target_sr=16000)
        
        # 检查重采样后的参考音频数据
        if not np.all(np.isfinite(ref_audio_16k)):
            logger.warning("⚠️ 重采样后的TTS参考音频数据中存在非有限值，正在进行清理...")
            ref_audio_16k = np.nan_to_num(ref_audio_16k, nan=0.0, posinf=0.95, neginf=-0.95)
        
        # 🛡️ 优化TTS参考音频：增强语音质量
        ref_audio_16k = np.clip(ref_audio_16k, -0.95, 0.95)  # 限制幅度
        ref_audio_16k_path = os.path.join(output_vocal_path, "vocal_ref_16k.wav")
        sf.write(ref_audio_16k_path, ref_audio_16k, 16000, subtype='PCM_24')
        
        logger.info(f"✅ 音频处理完成！")
        logger.info(f"   📊 ASR用音频: {vocal_16k_path}")
        logger.info(f"   🎵 TTS用参考: {ref_audio_16k_path}")
        
        return vocal_16k_path, ref_audio_16k_path  # ✅ 两个都是16kHz单声道
            
    except Exception as e:
        logger.error(f"💔 HP2降噪处理错误: {e}")
        traceback.print_exc()
        return None, None

def asr_recognition(audio_file, language=None):
    """
    根据语言选择合适的ASR模型进行语音识别
    """
    try:
        logger.info(f"🗣️ 开始ASR识别，语言: {language}")
        
        # 如果指定了中文或自动检测为中文，使用FunASR
        if language == "zh" or (language == "auto" and preloaded_models['funasr_zh']):
            logger.info("使用FunASR进行中文识别...")
            return asr_with_funasr(audio_file)
        # 对于其他语言，使用Faster-Whisper
        elif preloaded_models['whisper']:
            logger.info("使用Faster-Whisper进行多语言识别...")
            return asr_with_whisper(audio_file, language)
        # Fallback到FunASR
        elif preloaded_models['funasr_zh']:
            logger.info("Fallback到FunASR进行识别...")
            return asr_with_funasr(audio_file)
        else:
            logger.error("没有可用的ASR模型")
            return ""
            
    except Exception as e:
        logger.error(f"ASR识别过程错误: {e}")
        traceback.print_exc()
        return ""

def asr_with_funasr(audio_file):
    """
    使用预加载的FunASR进行中文语音识别
    """
    try:
        logger.info("使用FunASR进行中文ASR...")
        model = preloaded_models['funasr_zh']
        result = model.generate(input=audio_file)
        return result[0]["text"] if result else ""
    except Exception as e:
        logger.error(f"FunASR识别错误: {e}")
        traceback.print_exc()
        return ""

def asr_with_whisper(audio_file, language=None):
    """
    使用预加载的Faster-Whisper进行多语言语音识别
    """
    try:
        logger.info("使用Faster-Whisper进行多语言ASR...")
        model = preloaded_models['whisper']
        
        # 加载音频文件
        y, sr = librosa.load(audio_file, sr=None)
        
        # 如果采样率不是16000，需要重新采样
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # 使用Whisper进行转录
        segments, info = model.transcribe(
            audio_file, 
            beam_size=5,
            language=language if language != "auto" else None,
            condition_on_previous_text=True
        )
        
        # 合并所有片段
        text = "".join([segment.text for segment in segments])
        logger.info(f"Whisper识别结果: {text}")
        return text
    except Exception as e:
        logger.error(f"Faster-Whisper识别错误: {e}")
        traceback.print_exc()
        return ""

def translate_text(text, target_language="zh", source_language="en"):
    """
    翻译文本到目标语言
    """
    try:
        logger.info(f"🌐 翻译文本从 {source_language} 到 {target_language}")
        
        # 构造语言映射
        language_map = {
            "zh": "中文",
            "en": "英文",
            "ja": "日文"
        }
        display_target = language_map.get(target_language, "中文")
        display_source = language_map.get(source_language, "英文")
        
        # 构造提示词
        if source_language == "zh" and target_language == "en":
            system_prompt = f"你是一个专业的翻译人员，请将以下{display_source}的对话内容直接翻译成{display_target}，保持对话的自然流畅性。"
        elif source_language == "en" and target_language == "zh":
            system_prompt = f"你是一个专业的翻译人员，请将以下{display_source}的对话内容翻译成{display_target}，保持对话的自然流畅性。"
        else:
            system_prompt = f"你是一个专业的翻译人员，请将以下{display_source}的对话内容翻译成{display_target}，保持对话的自然流畅性。"
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"请翻译以下文本：\n\n{text}"
                }
            ],
            "stream": False
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        translated_text = result['choices'][0]['message']['content'].strip()
        logger.info(f"✅ 翻译完成: {translated_text}")
        return translated_text
    except Exception as e:
        logger.error(f"翻译错误: {e}")
        traceback.print_exc()
        return text  # 返回原文本作为fallback

def tts_synthesis_streaming(text, reference_audio=None, output_path=None):
    """
    使用预加载的TTS合成语音并返回音频流
    """
    try:
        logger.info(f"🎵 使用TTS合成语音流: {text}")
        
        # 检查TTS模型是否已加载
        if preloaded_models['tts'] is None:
            logger.warning("TTS模型未加载，跳过语音合成")
            return None
        
        # 确保有有效的参考音频
        if not reference_audio or not os.path.exists(reference_audio):
            if os.path.exists(default_ref_audio_path):
                logger.warning(f"⚠️ 未提供有效参考音频，使用默认参考音频: {default_ref_audio_path}")
                reference_audio = default_ref_audio_path
            else:
                logger.error("❌ 没有可用的参考音频")
                return None
        
        logger.info(f"🎯 使用参考音频: {os.path.basename(reference_audio)}")
        
        # 使用预加载的TTS模型
        tts = preloaded_models['tts']
        
        # 使用流式返回方式
        result = tts.infer(
            text=text,
            spk_audio_prompt=reference_audio,  # ✅ 现在使用16kHz参考音频
            output_path=output_path,
            #stream_return=True,
            #verbose=True
        )
        
        logger.info("✅ TTS流式合成启动成功")
        return result  # 返回生成器对象
    except Exception as e:
        logger.error(f"💔 TTS流式合成错误: {e}")
        traceback.print_exc()
        
        # 尝试不使用参考音频的fallback
        try:
            logger.info("🔄 尝试不使用参考音频...")
            tts = preloaded_models['tts']
            result = tts.infer(
                text=text,
                output_path=output_path,
                #stream_return=True,
                #verbose=True
            )
            logger.info("✅ TTS回退模式合成成功")
            return result
        except Exception as fallback_e:
            logger.error(f"💔 TTS回退模式也失败: {fallback_e}")
            return None

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    loaded_models = [k for k, v in preloaded_models.items() if v is not None]
    return jsonify({
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "loaded_models": loaded_models,
        "device": device,
        "default_ref_audio": os.path.exists(default_ref_audio_path)
    })

@app.route('/infer_wav', methods=['POST'])
def infer_wav():
    """
    简化版：返回完整的WAV文件
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "缺少JSON数据"}), 400
        
        # 提取音频数据
        audio_data_base64 = data.get("audio_data")
        if not audio_data_base64:
            return jsonify({"error": "缺少audio_data参数"}), 400
        
        
        # 保存临时文件
        audio_bytes = base64.b64decode(audio_data_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_bytes = np.nan_to_num(audio_bytes)
            f.write(audio_bytes)
            temp_audio_path = f.name
        
        try:
            # 1. 降噪处理
            vocal_16k_path, ref_audio_16k_path = denoise_audio_hp2(temp_audio_path)
            if not vocal_16k_path or not ref_audio_16k_path:
                return jsonify({"error": "降噪处理失败"}), 500
            
            # 2. ASR识别
            asr_language = data.get("asr_language", None)
            asr_result = asr_recognition(vocal_16k_path, asr_language)
            if not asr_result:
                return jsonify({"error": "ASR识别失败"}), 500
            
            # 3. 翻译
            target_language = data.get("target_language", "zh")
            source_language = "ja" if asr_language == "ja" else "en"  # 简化
            translated_text = translate_text(asr_result, target_language, source_language)
            
            # 4. TTS合成
            tts_output_path = os.path.join(RESULTS_DIR, "tts_output", f"tts_result_{int(time.time()*1000)}.wav")
            os.makedirs(os.path.dirname(tts_output_path), exist_ok=True)
            
            # 同步TTS合成
            if preloaded_models['tts']:
                tts = preloaded_models['tts']
                tts.infer(
                    text=translated_text,
                    spk_audio_prompt=ref_audio_16k_path,
                    output_path=tts_output_path
                )
            else:
                return jsonify({"error": "TTS模型未加载"}), 500
            
            # 5. 读取WAV文件
            with open(tts_output_path, "rb") as f:
                wav_bytes = f.read()
            
            # 6. 转换为Base64
            wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            return jsonify({
                "success": True,
                "text_original": asr_result,
                "text_translated": translated_text,
                "wav_data": wav_base64,
                "wav_size": len(wav_bytes)
            })
            
        finally:
            # 清理临时文件
            for path in [temp_audio_path, vocal_16k_path, ref_audio_16k_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
    
    except Exception as e:
        logger.error(f"处理失败: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 预加载模型
    if not preload_models():
        logger.error("模型预加载失败，退出程序")
        sys.exit(1)
    
    # 启动Flask服务
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 启动模型服务，监听端口: {port}")
    logger.info(f"✅ 服务准备就绪！访问 http://localhost:{port}/health 检查状态")
    app.run(host="0.0.0.0", port=port, debug=False)
