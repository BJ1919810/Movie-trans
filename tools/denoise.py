import os
import sys

import traceback

import librosa
import numpy as np
import soundfile as sf
import torch

# 添加uvr5到系统路径
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_path = os.path.dirname(script_dir)
print(project_path)
uvr5_path = os.path.join(project_path, "uvr5")
sys.path.append(uvr5_path)
sys.path.append(project_path)

# 现在可以导入AudioPre了
from uvr5.vr import AudioPre


# 配置参数
weight_uvr5_root = os.path.join(project_path, "uvr5", "uvr5_weights")
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = False  # 根据你的GPU情况调整
model_name = "HP2_all_vocals"
input_audio_path = os.path.join(project_path, "temp", "output_audio.wav")
output_vocal_path = os.path.join(project_path, "temp")
output_ins_path = os.path.join(project_path, "temp")
agg = 10  # 人声提取激进程度
format0 = "wav"  # 导出文件格式

def denoise_audio():
    try:
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        # 初始化模型
        pre_fun = AudioPre(
            agg=int(agg),
            model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
            device=device,
            is_half=is_half,
        )
        
        print(f"Processing audio: {input_audio_path}")
        
        # 直接处理音频，不重新格式化
        print("Processing audio without reformatting...")
        pre_fun._path_audio_(input_audio_path, output_ins_path, output_vocal_path, format0)
        print("Direct processing completed")
            
        print("Audio processing completed successfully!")
        print(f"Output files saved to: {project_path}")
        
        # 生成16000Hz单声道人声文件
        # 获取输出人声文件路径
        vocal_filename = f"vocal_{os.path.basename(input_audio_path)}_{agg}.{format0}"
        vocal_file = os.path.join(output_vocal_path, vocal_filename)
        
        if os.path.exists(vocal_file):
            print("Resampling vocal file to 16000Hz mono...")
            y, sr = librosa.load(vocal_file, sr=None, mono=True)
            y_resampled_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            vocal_16k_path = os.path.join(output_vocal_path, "vocal_1_16000.wav")
            sf.write(vocal_16k_path, y_resampled_16k, 16000, subtype='PCM_16')
            print(f"Saved 16000Hz mono vocal file: {vocal_16k_path}")
            
            # 保存原始采样率的人声文件（不进行重采样）
            vocal_original_path = os.path.join(output_vocal_path, "vocal_1_44100.wav")
            import shutil
            shutil.copy2(vocal_file, vocal_original_path)
            print(f"Saved original sample rate vocal file: {vocal_original_path}")
        else:
            print(f"Warning: Vocal file not found: {vocal_file}")
            
        # 处理背景音文件，保持原始采样率
        instrumental_filename = f"instrument_{os.path.basename(input_audio_path)}_{agg}.{format0}"
        instrumental_file = os.path.join(output_ins_path, instrumental_filename)
        
        if os.path.exists(instrumental_file):
            # 保存原始采样率的背景音文件（不进行重采样）
            bg_original_path = os.path.join(output_ins_path, "bg_1_44100.wav")
            import shutil
            shutil.copy2(instrumental_file, bg_original_path)
            print(f"Saved original sample rate instrumental file: {bg_original_path}")
        else:
            print(f"Warning: Instrumental file not found: {instrumental_file}")
        
        # 清理特定的临时文件
        try:
            print("Cleaning up temporary files...")
            temp_files = [
                os.path.join(output_vocal_path, f"vocal_{os.path.basename(input_audio_path)}_{agg}.{format0}"),
                os.path.join(output_ins_path, f"instrument_{os.path.basename(input_audio_path)}_{agg}.{format0}")
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Removed temporary file: {temp_file}")
                else:
                    print(f"Temporary file not found (skipping): {temp_file}")
        except Exception as e:
            print(f"Warning: Error occurred while cleaning up temporary files: {e}")
        
        # 清理资源
        del pre_fun.model
        del pre_fun
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    denoise_audio()