#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index-TTS 批量语音合成脚本

注意：此版本已修改为优先使用本地模型文件，避免重复网络下载并支持离线使用。

本地模型文件应保存在 index-tts/checkpoints 目录下的相应子目录中。
如果本地模型文件缺失或损坏，请运行 download_models.py 脚本下载所有必需的模型文件到本地。
"""

import os
import sys
import glob
import json

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目根目录到Python路径
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'index-tts'))

from indextts.infer_v2 import IndexTTS2

# 初始化模型，启用多种性能优化选项
# 强制使用CUDA，不回退到CPU模式
tts = IndexTTS2(
    cfg_path=os.path.join(project_root, "index-tts", "checkpoints", "config.yaml"),
    model_dir=os.path.join(project_root, "index-tts", "checkpoints"),
    use_fp16=True,          # 使用半精度浮点数（float16）以减少显存占用并提高推理速度
    use_cuda_kernel=True,   # 使用自定义的CUDA内核来加速BigVGAN的推理
    use_deepspeed=False,    # 暂时不启用DeepSpeed，因为它可能在某些系统上导致性能下降
    use_accel=False,         # 启用加速引擎来优化GPT模型的推理
    use_torch_compile=False, # 使用torch.compile来进一步优化模型执行
    device="cuda:0"         # 强制使用CUDA设备
)
print("Model initialized with CUDA kernel support.")

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
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        if len(parts) >= 4:
            time_part = parts[3].replace(".wav", "")
            ref_audio_files[f"{speaker}_{time_part}"] = file_path

# 为每个片段生成语音
for i, segment in enumerate(segments):
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
    
    ref_audio_path = ref_audio_files[time_key]
    
    # 构建输出文件名
    output_filename = f"result_{speaker[-2:]}_{start:.2f}-{end:.2f}.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Generating speech: {output_filename}")
    print(f"Text: {text}")
    print(f"Reference audio: {ref_audio_path}")
    
    # 执行推理
    tts.infer(
        text=text,
        spk_audio_prompt=ref_audio_path,
        output_path=output_path,
        verbose=True,
        # 性能优化参数
        max_mel_tokens=1000,  # 控制最大mel token数量
        do_sample=True,
        top_p=0.8,
        top_k=15,
        temperature=0.8,
        num_beams=3,
        repetition_penalty=10.0,
        max_text_tokens_per_segment=120
    )
    
    print(f"Speech saved to: {output_path}\n")

print("All speech segments generated successfully!")