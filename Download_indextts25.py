#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
IndexTTS-2.5 模型下载脚本（ModelScope 源，国内速度快）

从 https://www.modelscope.cn/models/IndexTeam/IndexTTS-2.5 下载权重到
``index-tts/checkpoints``，并提供完整性校验。

用法：
    python Download_indextts25.py                # 下载缺失的文件
    python Download_indextts25.py --force        # 强制重新下载全部
    python Download_indextts25.py --dir D:\xxx   # 指定输出目录

说明：
- 已存在且非空的文件默认跳过，中断后可重跑续传。
- 首次升级会备份旧的 config.yaml 为 config.v2.yaml.bak（回滚用）。
- 辅助模型（w2v-bert-2.0 / MaskGCT codec / CAMPPlus / BigVGAN）不在此仓库里，
  脚本末尾会检查它们是否已存在于本地，缺失时给出下载提示。
"""

import argparse
import os
import shutil
import sys
import time

MODEL_ID = "IndexTeam/IndexTTS-2.5"

# 远程文件字节大小（ModelScope 清单），用于判断本地文件是不是 2.5 版本
# —— 升级时 gpt.pth / s2mel.pth 等同名旧权重会被正确识别并重新下载
EXPECTED_SIZES = {
    "config.yaml": 2860,
    "configuration.json": 25,
    "feat1.pt": 57170,
    "feat2.pt": 374866,
    "codec.pth": 607290935,
    "gpt.pth": 3259599833,
    "s2mel.pth": 414908601,
    "multilingual_zh_ja_yue_char_del.tiktoken": 907395,
    "wav2vec2bert_stats.pt": 9343,
    "qwen0.6bemo4-merge/model.safetensors": 1192135096,
    "qwen0.6bemo4-merge/tokenizer.json": 11422654,
}

# 单文件（相对于仓库根）
SINGLE_FILES = [
    "config.yaml",
    "configuration.json",
    "feat1.pt",
    "feat2.pt",
    "codec.pth",
    "gpt.pth",
    "s2mel.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
]

# qwen 情感模型（目录）
QWEN_DIR = "qwen0.6bemo4-merge"
QWEN_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "Modelfile",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

# 辅助模型（本地已有则跳过）
AUX_CHECKS = {
    "w2v-bert-2.0": ["w2v-bert-2.0", os.path.join("hf_cache", "w2v-bert-2.0")],
    "MaskGCT semantic codec": [
        os.path.join("semantic_codec", "model.safetensors"),
        os.path.join("hf_cache", "semantic_codec_model.safetensors"),
    ],
    "CAMPPlus": [
        "campplus_cn_common.bin",
        os.path.join("hf_cache", "campplus_cn_common.bin"),
    ],
    "BigVGAN": [
        os.path.join("nvidia", "bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt"),
        os.path.join("hf_cache", "bigvgan", "bigvgan_generator.pt"),
    ],
}


def _human(size):
    if size is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024


def _ensure_modelscope():
    try:
        from modelscope.hub.file_download import model_file_download  # noqa: F401
        return True
    except ImportError:
        print("!! 未安装 modelscope，请先执行： pip install -U modelscope")
        return False


def _download_one(model_id, remote_path, local_path, retries=3):
    """下载单个文件到 local_path（原子化：先下到 .part 再改名）。"""
    from modelscope.hub.file_download import model_file_download

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".part"
    for attempt in range(1, retries + 1):
        try:
            got = model_file_download(
                model_id=model_id,
                file_path=remote_path,
                local_dir=os.path.dirname(local_path),
            )
            # modelscope 可能直接落到 local_dir，也可能返回缓存路径
            if got and os.path.isfile(got) and os.path.abspath(got) != os.path.abspath(local_path):
                shutil.copy2(got, tmp_path)
                os.replace(tmp_path, local_path)
            elif os.path.isfile(tmp_path):
                os.replace(tmp_path, local_path)
            if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"   ! 第 {attempt}/{retries} 次下载失败：{e}")
            if attempt < retries:
                time.sleep(2)
    if os.path.isfile(tmp_path):
        os.remove(tmp_path)
    return False


def _need_download(local_path, force, remote_path=None):
    """判断是否需要下载：不存在 / 为空 / 大小与 2.5 清单不符（说明是旧版本权重）。"""
    if force:
        return True, "强制"
    if not os.path.isfile(local_path):
        return True, "缺失"
    size = os.path.getsize(local_path)
    if size == 0:
        return True, "空文件"
    expect = EXPECTED_SIZES.get(remote_path)
    if expect is not None and size != expect:
        return True, f"版本不符({_human(size)} != {_human(expect)})"
    return False, ""


def main():
    parser = argparse.ArgumentParser(description="下载 IndexTTS-2.5 模型（ModelScope）")
    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index-tts", "checkpoints"),
        help="模型输出目录，默认 <项目根>/index-tts/checkpoints",
    )
    parser.add_argument("--force", action="store_true", help="强制重新下载（覆盖已存在文件）")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"模型目录: {out_dir}")
    print(f"模型来源: ModelScope {MODEL_ID}\n")

    if not _ensure_modelscope():
        return 1

    # 升级前备份旧 config.yaml
    config_path = os.path.join(out_dir, "config.yaml")
    if os.path.isfile(config_path) and not os.path.isfile(os.path.join(out_dir, "config.v2.yaml.bak")):
        try:
            with open(config_path, encoding="utf-8") as f:
                if "version: 2.5" not in f.read():
                    shutil.copy2(config_path, os.path.join(out_dir, "config.v2.yaml.bak"))
                    print(">> 已备份旧配置 -> config.v2.yaml.bak")
        except Exception:  # noqa: BLE001
            pass

    total, ok = 0, 0
    # 1) 主模型文件
    for name in SINGLE_FILES:
        local_path = os.path.join(out_dir, name)
        total += 1
        need, why = _need_download(local_path, args.force, name)
        if not need:
            print(f"[skip] {name} ({_human(os.path.getsize(local_path))})")
            ok += 1
            continue
        print(f"[down] {name} ({why}) ...")
        t0 = time.time()
        if _download_one(MODEL_ID, name, local_path):
            print(f"   ✓ {name} ({_human(os.path.getsize(local_path))}, {time.time() - t0:.1f}s)")
            ok += 1
        else:
            print(f"   ✗ {name} 下载失败")

    # 2) qwen 情感模型
    qwen_ok = True
    for name in QWEN_FILES:
        remote = f"{QWEN_DIR}/{name}"
        local_path = os.path.join(out_dir, QWEN_DIR, name)
        need, why = _need_download(local_path, args.force, remote)
        if not need:
            continue
        print(f"[down] {remote} ({why}) ...")
        if not _download_one(MODEL_ID, remote, local_path):
            print(f"   ✗ {remote} 下载失败")
            qwen_ok = False
    total += 1
    if qwen_ok:
        print(f"[ ok ] {QWEN_DIR}/")
        ok += 1

    # 3) 辅助模型检查
    print("\n--- 辅助模型检查 ---")
    missing_aux = []
    for label, candidates in AUX_CHECKS.items():
        found = any(
            os.path.exists(os.path.join(out_dir, c)) for c in candidates
        )
        print(f"  {'✓' if found else '✗'} {label}")
        if not found:
            missing_aux.append(label)

    print(f"\n完成：{ok}/{total}")
    if ok < total:
        print("!! 有文件未下载成功，重跑本脚本即可续传。")
    if missing_aux:
        print("!! 缺失辅助模型：" + "、".join(missing_aux))
        print("   它们会在首次推理时自动下载到 checkpoints/hf_cache/，")
        print("   或手动用 modelscope 下载 facebook/w2v-bert-2.0、amphion/MaskGCT、")
        print("   funasr/campplus、nvidia/bigvgan_v2_22khz_80band_256x。")
    else:
        print("辅助模型齐全，可离线推理。")
    return 0 if ok == total else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
