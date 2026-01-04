#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本

此脚本用于下载项目所需的所有模型文件，包括：
1. 克隆index-tts仓库
2. 下载ASR相关模型到asr/models目录
3. 下载HuggingFace缓存模型到checkpoints/hf_cache目录
4. 下载Index-TTS相关模型到index-tts/checkpoints目录
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

# 设置HF_TOKEN环境变量
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"  # 请替换为实际的HF_TOKEN

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"项目根目录: {PROJECT_ROOT}")

def clone_index_tts_repo():
    """克隆index-tts仓库到指定目录"""
    repo_url = "https://github.com/index-tts/index-tts"
    target_dir = os.path.join(PROJECT_ROOT, "index-tts")
    
    print("开始克隆index-tts仓库...")
    
    # 检查目录是否已存在且包含内容
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"目录 {target_dir} 已存在且非空，跳过克隆")
        return True
    elif os.path.exists(target_dir):
        # 目录存在但为空，删除它以便重新克隆
        print(f"目录 {target_dir} 存在但为空，删除后重新克隆")
        os.rmdir(target_dir)
    
    try:
        # 克隆仓库
        result = subprocess.run([
            "git", "clone", repo_url, target_dir
        ], check=True, capture_output=True, text=True)
        print("index-tts仓库克隆成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"克隆index-tts仓库失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def download_asr_models():
    """下载ASR相关模型到asr/models目录"""
    asr_models_dir = os.path.join(PROJECT_ROOT, "asr", "models")
    os.makedirs(asr_models_dir, exist_ok=True)
    
    print("开始下载ASR模型...")
    
    # ASR模型列表
    asr_models = {
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch": {
            "repo_id": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "revision": "v1.0.0"
        },
        "speech_fsmn_vad_zh-cn-16k-common-pytorch": {
            "repo_id": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "revision": "v1.0.0"
        },
        "punc_ct-transformer_zh-cn-common-vocab272727-pytorch": {
            "repo_id": "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "revision": "v1.0.0"
        },
        # 注意：模型目录名前已添加"models--"前缀
        "models--pyannote--segmentation-3.0": {
            "repo_id": "pyannote/segmentation-3.0",
            "revision": "main"
        },
        "models--pyannote--wespeaker-voxceleb-resnet34-LM": {
            "repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "revision": "main"
        }
    }
    
    success_count = 0
    for model_name, model_info in asr_models.items():
        print(f"正在下载 {model_name}...")
        try:
            # 检查模型是否已存在
            model_path = os.path.join(asr_models_dir, model_name)
            if os.path.exists(model_path):
                print(f"  模型 {model_name} 已存在，跳过下载")
                success_count += 1
                continue
                
            # 下载模型
            snapshot_download(
                repo_id=model_info["repo_id"],
                revision=model_info["revision"],
                cache_dir=asr_models_dir,
                local_files_only=False,
                token=os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") != "YOUR_HF_TOKEN" else None
            )
            
            # 重命名下载的目录
            downloaded_dir = os.path.join(asr_models_dir, model_info["repo_id"].replace("/", "--"))
            if os.path.exists(downloaded_dir) and not os.path.exists(model_path):
                os.rename(downloaded_dir, model_path)
                
            print(f"  ✓ {model_name} 下载完成")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {model_name} 下载失败: {e}")
    
    print(f"ASR模型下载完成 ({success_count}/{len(asr_models)})")
    return success_count == len(asr_models)

def download_hf_cache_models():
    """下载HuggingFace缓存模型到checkpoints/hf_cache目录"""
    hf_cache_dir = os.path.join(PROJECT_ROOT, "checkpoints", "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    print("开始下载HF缓存模型...")
    
    # 设置HF缓存目录
    os.environ['HF_HUB_CACHE'] = hf_cache_dir
    
    # HF缓存模型列表
    hf_models = {
        "w2v-bert-2.0": {
            "repo_id": "facebook/w2v-bert-2.0",
            "filename": None  # 下载整个仓库
        },
        "campplus": {
            "repo_id": "funasr/campplus",
            "filename": None  # 下载整个仓库
        },
        "bigvgan_v2_22khz_80band_256x": {
            "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
            "filename": None  # 下载整个仓库
        },
        "MaskGCT": {
            "repo_id": "amphion/MaskGCT",
            "filename": None  # 下载整个仓库
        }
    }
    
    success_count = 0
    for model_name, model_info in hf_models.items():
        print(f"正在下载 {model_name}...")
        try:
            if model_info["filename"]:
                # 下载单个文件
                # 正确构建HuggingFace缓存路径
                repo_dir = os.path.join(hf_cache_dir, "models--" + model_info["repo_id"].replace("/", "--"))
                blobs_dir = os.path.join(repo_dir, "blobs")
                
                # 检查文件是否已存在（通过检查blobs目录）
                if os.path.exists(blobs_dir) and os.listdir(blobs_dir):
                    print(f"  文件 {model_name}/{model_info['filename']} 已存在，跳过下载")
                    success_count += 1
                    continue
                
                # 确保目录存在
                os.makedirs(blobs_dir, exist_ok=True)
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=model_info["filename"],
                    cache_dir=hf_cache_dir,
                    token=os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") != "YOUR_HF_TOKEN" else None
                )
            else:
                # 下载整个仓库
                # 正确构建HuggingFace缓存路径
                model_path = os.path.join(hf_cache_dir, "models--" + model_info["repo_id"].replace("/", "--"))
                snapshots_path = os.path.join(model_path, "snapshots")
                
                # 检查模型是否已存在（通过检查snapshots目录）
                if os.path.exists(snapshots_path) and os.listdir(snapshots_path):
                    print(f"  模型 {model_name} 已存在，跳过下载")
                    success_count += 1
                    continue
                
                snapshot_download(
                    repo_id=model_info["repo_id"],
                    cache_dir=hf_cache_dir,
                    local_files_only=False,
                    token=os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") != "YOUR_HF_TOKEN" else None
                )
                
            print(f"  ✓ {model_name} 下载完成")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {model_name} 下载失败: {e}")
    
    print(f"HF缓存模型下载完成 ({success_count}/{len(hf_models)})")
    return success_count == len(hf_models)

def download_faster_whisper_model():
    """下载Faster-Whisper大型模型到asr/models目录"""
    asr_models_dir = os.path.join(PROJECT_ROOT, "asr", "models")
    os.makedirs(asr_models_dir, exist_ok=True)
    
    print("开始下载Faster-Whisper大型模型...")
    
    # 模型配置
    model_name = "models--Systran--faster-whisper-large-v3"
    repo_id = "Systran/faster-whisper-large-v3"
    model_path = os.path.join(asr_models_dir, model_name)
    
    # 检查模型是否已存在
    if os.path.exists(model_path):
        print(f"  模型 {model_name} 已存在，跳过下载")
        return True
    
    # 定义需要下载的文件
    files = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
        "preprocessor_config.json",
        "vocabulary.json"
    ]
    
    success = False
    for attempt in range(2):
        try:
            print(f"  正在下载 {model_name} (尝试 {attempt + 1}/2)...")
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=files,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            print(f"  ✓ {model_name} 下载完成")
            success = True
            break
        except Exception as e:
            print(f"  ✗ {model_name} 下载失败 (尝试 {attempt + 1}/2): {e}")
            if attempt < 1:
                print("  等待2秒后重试...")
                time.sleep(2)
    
    if not success:
        print(f"  ✗ {model_name} 下载最终失败")
        return False
    
    return True


def download_index_tts_models():
    """下载Index-TTS相关模型到index-tts/checkpoints目录"""
    index_tts_checkpoints_dir = os.path.join(PROJECT_ROOT, "index-tts", "checkpoints")
    os.makedirs(index_tts_checkpoints_dir, exist_ok=True)
    
    print("开始下载Index-TTS模型...")
    
    # IndexTTS-2仓库中的文件列表
    index_tts_files = [
        ".gitattributes",
        "README.md",
        "bpe.model",
        "config.yaml",
        "feat1.pt",
        "feat2.pt",
        "gpt.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt"
    ]
    
    # 必需的核心模型文件（除了.gitattributes与README.md）
    essential_files = [f for f in index_tts_files if f not in [".gitattributes", "README.md"]]
    
    success_count = 0
    
    # 检查核心模型文件是否存在
    missing_essential_files = []
    for filename in essential_files:
        local_path = os.path.join(index_tts_checkpoints_dir, filename)
        if not os.path.exists(local_path):
            missing_essential_files.append(filename)
    
    # 如果qwen0.6bemo4-merge目录不存在或为空，也认为是缺失
    qwen_model_dir = os.path.join(index_tts_checkpoints_dir, "qwen0.6bemo4-merge")
    qwen_model_exists = os.path.exists(qwen_model_dir) and os.path.isdir(qwen_model_dir) and os.listdir(qwen_model_dir)
    
    # 如果所有必需文件都存在且qwen模型也存在，则跳过下载
    if not missing_essential_files and qwen_model_exists:
        print("  所有Index-TTS模型文件已存在，跳过下载")
        success_count = len(index_tts_files) + 1  # +1 是因为 qwen0.6bemo4-merge
        print(f"Index-TTS模型下载完成 ({success_count}/{len(index_tts_files)+1})")
        return True
    
    # 否则下载缺失的文件和qwen模型
    print("检测到部分模型文件缺失，开始下载...")
    
    # 检查并下载 qwen0.6bemo4-merge 文件夹
    if not qwen_model_exists:
        # 如果目录存在但为空，则删除它
        if os.path.exists(qwen_model_dir):
            shutil.rmtree(qwen_model_dir)
        
        # 下载 qwen0.6bemo4-merge 模型
        print("正在下载 qwen0.6bemo4-merge...")
        try:
            snapshot_download(
                repo_id="IndexTeam/IndexTTS-2",
                allow_patterns="qwen0.6bemo4-merge/*",
                local_dir=index_tts_checkpoints_dir,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") != "YOUR_HF_TOKEN" else None
            )
            print("  ✓ qwen0.6bemo4-merge 下载完成")
            success_count += 1
        except Exception as e:
            print(f"  ✗ qwen0.6bemo4-merge 下载失败: {e}")
    else:
        print("  模型 qwen0.6bemo4-merge 已存在，跳过下载")
        success_count += 1
    
    # 检查并下载其他必需文件
    for filename in index_tts_files:
        # .gitattributes 和 README.md 总是重新下载以确保最新
        if filename not in [".gitattributes", "README.md"] and filename not in missing_essential_files:
            print(f"  文件 {filename} 已存在，跳过下载")
            success_count += 1
            continue
            
        local_path = os.path.join(index_tts_checkpoints_dir, filename)
        print(f"正在下载 {filename}...")
        try:
            hf_hub_download(
                repo_id="IndexTeam/IndexTTS-2",
                filename=filename,
                local_dir=index_tts_checkpoints_dir,
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") != "YOUR_HF_TOKEN" else None
            )
            print(f"  ✓ {filename} 下载完成")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {filename} 下载失败: {e}")
    
    print(f"Index-TTS模型下载完成 ({success_count}/{len(index_tts_files)+1})")  # +1 是因为 qwen0.6bemo4-merge
    return success_count == len(index_tts_files) + 1

def main():
    """主函数"""
    print("=" * 60)
    print("开始下载所有模型文件...")
    print("=" * 60)
    
    # 1. 克隆index-tts仓库
    if not clone_index_tts_repo():
        print("克隆index-tts仓库失败，退出程序")
        return False
    
    # 2. 下载ASR模型
    if not download_asr_models():
        print("ASR模型下载未完全成功")
    
    # 3. 下载HF缓存模型
    if not download_hf_cache_models():
        print("HF缓存模型下载未完全成功")
    
    # 4. 下载Faster-Whisper大型模型
    if not download_faster_whisper_model():
        print("Faster-Whisper大型模型下载未完全成功")
    
    # 5. 下载Index-TTS模型
    if not download_index_tts_models():
        print("Index-TTS模型下载未完全成功")
    
    print("=" * 60)
    print("所有模型下载任务已完成!")
    print("请检查上面的输出确认是否有下载失败的模型。")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断了程序执行")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)