#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
from pathlib import Path

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（tools的父目录）
project_root = os.path.dirname(script_dir)

# 添加项目根目录到Python路径
sys.path.append(project_root)

# 配置路径（使用相对于项目根目录的路径）
AUDIO_FILE = os.path.join(project_root, "temp", "vocal_1_44100.wav")
JSON_FILE = os.path.join(project_root, "results", "speaker_diarization.json")
OUTPUT_DIR = os.path.join(project_root, "temp", "clips")

def create_clips():
    # 清空输出目录（如果已存在）
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 读取说话人分离结果
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    print(f"Found {len(segments)} segments")
    
    # 按说话人分类存储
    speaker_counts = {}
    
    for i, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']
        speaker = segment['speaker']
        duration = segment['duration']
        
        # 计算说话人的片段数量
        if speaker not in speaker_counts:
            speaker_counts[speaker] = 0
        speaker_counts[speaker] += 1
        
        # 创建说话人子目录
        speaker_dir = os.path.join(OUTPUT_DIR, speaker)
        Path(speaker_dir).mkdir(parents=True, exist_ok=True)
        
        # 输出文件名 (改为wav格式) 格式: clip_{说话人ID}_{序号}_{开始时间}-{结束时间}.wav
        # 提取说话人ID，例如 SPEAKER_01 -> s1
        speaker_id = speaker.split('_')[1].lower().replace('speaker', 's')
        output_file = os.path.join(speaker_dir, f"clip_{speaker_id}_{speaker_counts[speaker]:03d}_{start:.2f}-{end:.2f}.wav")
        
        # 使用ffmpeg从音频文件中切割片段
        cmd = [
            'ffmpeg',
            '-i', AUDIO_FILE,
            '-ss', str(start),
            '-to', str(end),
            '-c:a', 'pcm_s16le',  # WAV格式需要的编码
            '-ar', '44100',       # 保持采样率一致
            '-ac', '1',           # 单声道
            '-y',                 # 覆盖已存在文件
            output_file
        ]
        
        print(f"Processing segment {i+1}/{len(segments)}: {output_file}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  Successfully created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"  Error: Cannot create {output_file}")
            print(f"  Error details: {e}")
    
    print("\nProcessing completed!")
    print("Number of segments per speaker:")
    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count} segments")

if __name__ == "__main__":
    create_clips()