#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本

此脚本用于下载项目所需的所有模型文件，包括：
1. 克隆index-tts仓库
2. 下载ASR相关模型到asr/models目录
3. 下载 IndexTTS 辅助模型到 index-tts/checkpoints/hf_cache（**唯一权威位置**）
4. 下载Index-TTS相关模型到index-tts/checkpoints目录

⚠️ 关于"模型放哪里"的约定（2026-09-14 规范，改动前务必读完）
   ------------------------------------------------------------------
   * IndexTTS-2.5 主权重 + 全部辅助模型，**只在** ``index-tts/checkpoints/`` 这一处：
       index-tts/checkpoints/                 主权重（gpt.pth / s2mel.pth / codec.pth / config.yaml …）
       index-tts/checkpoints/hf_cache/        辅助模型的扁平布局
         ├── w2v-bert-2.0/                    facebook/w2v-bert-2.0
         ├── bigvgan/                         nvidia/bigvgan_v2_22khz_80band_256x
         ├── campplus_cn_common.bin           funasr/campplus
         └── semantic_codec_model.safetensors amphion/MaskGCT
   * 这个位置由 ``indextts.utils.model_download.ensure_models_available(model_dir)`` 决定，
     是**运行时真正读取的地方**。本脚本直接调用它，不再自己拼路径。
   * ASR/说话人分离的模型在 ``asr/models/``，与上面互不相干。
   * **不要**再往 ``<项目根>/checkpoints/hf_cache/`` 下载东西 —— 那是历史遗留的第二套目录，
     会导致同一批模型存两份（实测重复约 2.5 GB）。
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"项目根目录: {PROJECT_ROOT}")

# HF_TOKEN：从系统环境变量或项目根 .env 读取（见 .env.example）
# pyannote 模型需要授权，请到 https://huggingface.co/settings/tokens 创建只读 token
sys.path.insert(0, PROJECT_ROOT)
try:
    from env_config import get_api_key
    _hf_token = get_api_key("HF_TOKEN")
except Exception:
    _hf_token = os.environ.get("HF_TOKEN", "")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
else:
    print("!! 未配置 HF_TOKEN，pyannote 模型下载可能失败（请在 .env 中填写）")

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

def download_aux_models():
    """下载 IndexTTS 的辅助模型（w2v-bert-2.0 / MaskGCT codec / CAMPPlus / BigVGAN）

    ⚠️ 权威位置只有一个：``index-tts/checkpoints/hf_cache/``
    （＝运行时 ``ensure_models_available(model_dir)`` 用的 ``{model_dir}/hf_cache``）。

    这里**直接复用 index-tts 自己的解析器**，而不是自己拼路径 + 自己 snapshot_download
    —— 这样"下载脚本"与"运行时"永远用同一套查找规则，不会出现
    「下到 A、运行时找 B → 又下一遍」的情况：

        1. ``{model_dir}/hf_cache/`` 已有的扁平布局  → 复用
        2. 历史本地位置（``{model_dir}/w2v-bert-2.0``、``{model_dir}/nvidia/...``）→ 复用
        3. 旧版 HuggingFace 缓存布局（``models--owner--name/snapshots/<hash>/``）→ 迁移复用
        4. 都没有才下载（ModelScope / hf-mirror / HuggingFace 自动选路）

    旧版本这个函数把模型下到 ``<项目根>/checkpoints/hf_cache/``，
    和运行时用的 ``index-tts/checkpoints/hf_cache/`` 是两套目录 → 同一批模型存两份
    （实测重复约 2.5 GB）。**不要再改回去。**
    """
    model_dir = os.path.join(PROJECT_ROOT, "index-tts", "checkpoints")
    if not os.path.isdir(model_dir):
        print(f"  ✗ 找不到 {model_dir}，请先执行： python Download_indextts25.py")
        return False

    # HF 直连不通时走镜像（可用环境变量覆盖；国内实测 huggingface.co 超时）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print(f"辅助模型目录（权威位置）: {os.path.join(model_dir, 'hf_cache')}")
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "index-tts"))
        from indextts.utils.model_download import ensure_models_available
        paths = ensure_models_available(model_dir)
    except Exception as e:
        print(f"  ✗ 辅助模型准备失败: {e}")
        return False

    print("  ✓ 辅助模型就位：")
    for key, path in paths.items():
        print(f"      {key:16s} {path}")
    return True

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
    
    # 3. 下载 IndexTTS 辅助模型（权威位置：index-tts/checkpoints/hf_cache）
    if not download_aux_models():
        print("辅助模型下载未完全成功")
    
    # 4. 下载Faster-Whisper大型模型
    if not download_faster_whisper_model():
        print("Faster-Whisper大型模型下载未完全成功")
    
    # 5. IndexTTS-2.5 权重：不在本脚本内处理，必须用专用脚本（ModelScope 源）
    print("=" * 60)
    print("提示：IndexTTS-2.5 权重请单独执行： python Download_indextts25.py")
    print("=" * 60)
    
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