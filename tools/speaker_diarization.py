#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（tools的父目录）
project_root = os.path.dirname(script_dir)

# 设置HF访问令牌为环境变量
#HF_TOKEN = "YOUR_HF_TOKEN"
#os.environ["HF_TOKEN"] = HF_TOKEN

# 设置HuggingFace Hub为离线模式
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import gc
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import torch
import torchaudio
from pathlib import Path
from pyannote.audio.pipelines.utils.hook import ProgressHook
from huggingface_hub import hf_hub_download
import warnings

# 修复PyTorch 2.6+ weights_only问题的补丁
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=True, **pickle_load_args):
    """Patch torch.load to use weights_only=False for compatibility with pyannote.audio"""
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **pickle_load_args)

# 应用补丁
torch.load = patched_torch_load

'''
pip install pyannote.audio==3.1.1
pip install numpy==1.26.4
'''

# 默认音频文件路径（使用相对于项目根目录的路径）
audio_file = os.path.join(project_root, "temp", "vocal_1_16000.wav")

def check_and_download_model(repo_id, filename, local_path):
    """检查模型文件是否存在，如果不存在则从Hugging Face下载"""
    if not os.path.exists(local_path):
        print(f"Model file {local_path} does not exist, downloading from Hugging Face...")
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 从Hugging Face下载模型文件
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(local_path),
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") != "YOUR_HF_TOKEN" else None
            )
            
            # 重命名文件以匹配预期路径（如果需要）
            if downloaded_path != local_path:
                os.rename(downloaded_path, local_path)
                
            print(f"Model file downloaded to: {local_path}")
            return True
        except Exception as e:
            print(f"Failed to download model file: {e}")
            return False
    else:
        print(f"Model file already exists: {local_path}")
        return True

def run_speaker_diarization(audio_file, output_file=None):
    try:
        print("Initializing pyannote speaker diarization pipeline...")
        
        # 检查输入音频文件是否存在且不为空
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Input audio file not found: {audio_file}")
        
        if os.path.getsize(audio_file) == 0:
            raise ValueError(f"Input audio file is empty: {audio_file}")
        
        # 构建本地模型检查点路径
        segmentation_model_path = os.path.join(project_root, "asr", "models", "models--pyannote--segmentation-3.0", "snapshots", "e66f3d3b9eb0873085418a7b813d3b369bf160bb", "pytorch_model.bin")
        embedding_model_path = os.path.join(project_root, "asr", "models", "models--pyannote--wespeaker-voxceleb-resnet34-LM", "snapshots", "837717ddb9ff5507820346191109dc79c958d614", "pytorch_model.bin")
        
        # 检查并下载缺失的模型文件
        print("Checking model files...")
        seg_model_downloaded = check_and_download_model(
            "pyannote/segmentation-3.0",
            "pytorch_model.bin",
            segmentation_model_path
        )
        
        emb_model_downloaded = check_and_download_model(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            "pytorch_model.bin",
            embedding_model_path
        )
        
        # 如果任何一个模型下载失败，则抛出异常
        if not seg_model_downloaded or not emb_model_downloaded:
            raise RuntimeError("模型文件下载失败，请检查网络连接或HF_TOKEN设置")
        
        # 重新设置为离线模式以避免后续不必要的网络请求
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        # 直接使用本地模型路径初始化说话人分离管道
        # 注意：这里我们绕过标准的from_pretrained方法，直接使用本地模型
        pipeline = SpeakerDiarization(
            segmentation=segmentation_model_path,
            embedding=embedding_model_path
        )
        
        # 实例化管道参数
        '''
        'single': 0,    # 单连接 → 链式簇
        'complete': 1,  # 全连接 → 紧凑簇
        'average': 2,   # 平均连接 → 平衡型 ******
        'centroid': 3,  # 质心连接  ****
        'median': 4,    # 中位数连接 *****
        'ward': 5,      # Ward方差最小化 → 对球形簇极佳
        'weighted': 6   # 加权平均
        '''
        # 使用默认参数配置
        params = {
            "segmentation": {
                "min_duration_off": 0
            },
            "clustering": {
                "method": "average",
                "threshold": 0.72,  # 值越高，speaker数量越小
                "min_cluster_size": 15  # 小于该值的小cluster会被归为附近的大cluster里
            }
        }
        
        print("Parameters configuration:")
        print(params)
        
        # 实例化管道
        pipeline.instantiate(params)
        
        # 自动选择设备（GPU优先，如果没有则使用CPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"Using device: {device}")
        
        print(f"Processing audio file: {audio_file}")
        
        # 加载整个音频文件
        print("Loading entire audio file...")
        waveform, sample_rate = torchaudio.load(audio_file)
        print(f"Loaded audio file: {waveform.shape[1]/sample_rate:.2f} seconds")
        
        # 强制垃圾回收
        gc.collect()
        
        # 创建临时音频对象
        audio_data = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # 运行说话人分离，使用ProgressHook监控进度
        print("Running speaker diarization (this may take a while)...")
        with ProgressHook() as hook:
            diarization = pipeline(audio_data, hook=hook)
        
        # 强制垃圾回收
        gc.collect()
        
        # 准备JSON格式的结果
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
                "duration": round(turn.end - turn.start, 2)
            })
        
        # 确定输出文件路径
        if output_file:
            json_file = output_file
            # 确保输出目录存在
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
        else:
            # 使用默认路径
            results_dir = os.path.join(project_root, "results")
            os.makedirs(results_dir, exist_ok=True)
            json_file = os.path.join(results_dir, "speaker_diarization.json")
        
        # 将结果保存为JSON格式
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Diarization results saved to: {json_file}")
        
        # 同时保存RTTM格式（为了兼容性）
        rttm_file = audio_file.replace(".wav", ".rttm")
        with open(rttm_file, "w") as rttm:
            diarization.write_rttm(rttm)
        
        print(f"RTTM results saved to: {rttm_file}")
        
        # 打印结果
        print("\nSpeaker diarization results:")
        for result in results:
            print(f"start={result['start']}s stop={result['end']}s speaker={result['speaker']}")
            
        return json_file
        
    except Exception as e:
        print(f"Error occurred during speaker diarization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Run speaker diarization on an audio file")
    parser.add_argument("--audio", type=str, help="Path to the input audio file")
    parser.add_argument("--output", type=str, help="Path to the output JSON file")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用命令行参数或默认值
    audio_file = args.audio if args.audio else os.path.join(project_root, "temp", "vocal_1_16000.wav")
    output_file = args.output if args.output else os.path.join(project_root, "results", "speaker_diarization.json")
    
    json_file = run_speaker_diarization(audio_file, output_file)
    if json_file:
        print(f"\nProcess completed successfully. Results saved to {json_file}")
    else:
        print("\nProcess failed.")