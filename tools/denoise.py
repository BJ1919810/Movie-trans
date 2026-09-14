"""人声/伴奏分离（双引擎，输出命名保持一致，下游无需改动）

引擎（--engine）
  roformer  → uvr5/bsroformer.py 的 BS-Roformer（~610MB），**默认**。
              主人 A/B 试听结论：单走这个效果最好 —— 高频残留最少、
              齿音与气息保留完整，不损失"湿润感"。
  uvr5      → uvr5/vr.py 的 VR 架构 HP2_all_vocals（~60MB）。
              速度快、体积小，但 BGM 响的段落残留明显更多。备选。

roformer 引擎的注意点
  * 权重放 uvr5/uvr5_weights/，文件名必须含 "bs_roformer" 或 "mel_band_roformer"
    才能被自动识别；同名 .yaml 会被自动加载（没有则用内置默认配置）。
  * 必须在 CUDA 上跑：uvr5/bsroformer.py 内部无条件用 torch.amp.autocast("cuda")。
  * 默认半精度（--fp32 可关）：8GB 卡上更稳。
  * `--agg` 只对 uvr5 引擎有意义，roformer 会忽略它。

输出（两种引擎完全一致）
  <out>/vocal_1_44100.wav   人声，原始 44.1k 立体声（TTS 参考切片从这里切）
  <out>/vocal_1_16000.wav   人声，16k 单声道（喂 pyannote / ASR）
  <out>/bg_1_44100.wav      伴奏

不要在这后面加"去噪 / 去回响 / 清高频"后处理（2026-09-14 定论）
  主人明确：加了声音会**变干**。原理：denoise / de-reverb 类模型是**全带重建**，
  会连人声的泛音尾巴与房间残响一起重画，湿润感必然丢失。
  也试过"只削超出人声包络的宽带残留"（纯频谱阈值）：HF(3-10k)/语音核 这个比值
  **分不开"鸟鸣"和"正常齿音"**（正常段 80 分位 +8.7dB vs 残留段 +10.1dB，几乎重合），
  结果把全片高频一起削了 8~11dB —— 同样变闷。已删除，别再往回加。
"""
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

# 修复 PyTorch 2.6+ weights_only 问题（与 speaker_diarization.py 同样的补丁）：
# 权重是本地文件，这里允许反序列化完整对象。
_original_torch_load = torch.load


def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=True, **kw):
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module,
                                weights_only=False, **kw)


torch.load = _patched_torch_load

# 现在可以导入AudioPre了
from uvr5.vr import AudioPre


# 配置参数
weight_uvr5_root = os.path.join(project_path, "uvr5", "uvr5_weights")
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True  # cuda 上默认半精度（roformer 需要，8GB 卡更稳）
model_name = "model_bs_roformer_ep_317_sdr_12.9755"
input_audio_path = os.path.join(project_path, "temp", "output_audio.wav")
output_vocal_path = os.path.join(project_path, "temp")
output_ins_path = os.path.join(project_path, "temp")
agg = 10  # 人声提取激进程度（仅 uvr5 引擎）
format0 = "wav"  # 导出文件格式

# 默认引擎：roformer（主人 A/B 试听：BS-Roformer 单走效果最好）
engine = "roformer"

# 默认模型名（按引擎区分；--model-name 未显式给出时用它）
DEFAULT_MODEL = {
    "roformer": "model_bs_roformer_ep_317_sdr_12.9755",
    "uvr5": "HP2_all_vocals",
}


def _separate(pre_fun):
    """跑一次分离（两种引擎的 _path_audio_ 签名一致：input, others_root, vocal_root, format）"""
    pre_fun._path_audio_(input_audio_path, output_ins_path, output_vocal_path, format0)


def _resolve_produced(pre_fun):
    """返回 (人声文件, 伴奏文件) 的实际路径（两种引擎命名不同，在这里抹平）"""
    base = os.path.splitext(os.path.basename(input_audio_path))[0]
    if engine == "roformer":
        cfg = pre_fun.config
        target = cfg["training"]["target_instrument"]
        others = [i for i in cfg["training"]["instruments"] if i != target]
        vocal = os.path.join(output_vocal_path, f"{base}_{target}.{format0}")
        inst = os.path.join(output_ins_path, f"{base}_{others[0]}.{format0}")
    else:
        vocal = os.path.join(output_vocal_path, f"vocal_{base}_{agg}.{format0}")
        inst = os.path.join(output_ins_path, f"instrument_{base}_{agg}.{format0}")
    return vocal, inst


def denoise_audio():
    temp_files = []
    try:
        print(f"Engine: {engine}")
        print(f"Loading model: {model_name}")
        print(f"Device: {device}  half={is_half}")

        if engine == "roformer":
            if device != "cuda":
                print("!! roformer 引擎需要 CUDA（uvr5/bsroformer.py 内部写死了 autocast('cuda')）")
                return
            from uvr5.bsroformer import Roformer_Loader

            ckpt = os.path.join(weight_uvr5_root, model_name + ".ckpt")
            cfg_path = os.path.join(weight_uvr5_root, model_name + ".yaml")
            if not os.path.isfile(ckpt):
                print(f"!! 找不到权重: {ckpt}")
                print("   （可从 hf-mirror / UVR 生态下载同名 .ckpt 与 .yaml 放进 uvr5/uvr5_weights/）")
                print(f"   或改用旧引擎：--engine uvr5 --model-name {DEFAULT_MODEL['uvr5']}")
                sys.exit(2)
            if not os.path.isfile(cfg_path):
                print(f"   注意：没有配置文件 {cfg_path}，将使用脚本内置默认配置")
            pre_fun = Roformer_Loader(
                model_path=ckpt,
                config_path=cfg_path,
                device=device,
                is_half=is_half,
            )
        else:
            pre_fun = AudioPre(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )

        print(f"Processing audio: {input_audio_path}")
        _separate(pre_fun)
        print("Direct processing completed")

        produced_vocal, produced_inst = _resolve_produced(pre_fun)
        temp_files = [produced_vocal, produced_inst]

        # 生成16000Hz单声道人声文件（喂 pyannote / ASR）
        if os.path.exists(produced_vocal):
            print("Resampling vocal file to 16000Hz mono...")
            y, sr = librosa.load(produced_vocal, sr=None, mono=True)
            y_resampled_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            vocal_16k_path = os.path.join(output_vocal_path, "vocal_1_16000.wav")
            sf.write(vocal_16k_path, y_resampled_16k, 16000, subtype='PCM_16')
            print(f"Saved 16000Hz mono vocal file: {vocal_16k_path}")

            # 保存原始采样率的人声文件（不进行重采样）
            vocal_original_path = os.path.join(output_vocal_path, "vocal_1_44100.wav")
            import shutil
            shutil.copy2(produced_vocal, vocal_original_path)
            print(f"Saved original sample rate vocal file: {vocal_original_path}")
        else:
            print(f"Warning: Vocal file not found: {produced_vocal}")

        # 处理背景音文件，保持原始采样率
        if os.path.exists(produced_inst):
            bg_original_path = os.path.join(output_ins_path, "bg_1_44100.wav")
            import shutil
            shutil.copy2(produced_inst, bg_original_path)
            print(f"Saved original sample rate instrumental file: {bg_original_path}")
        else:
            print(f"Warning: Instrumental file not found: {produced_inst}")

        # 清理特定的临时文件
        try:
            print("Cleaning up temporary files...")
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Removed temporary file: {temp_file}")
                else:
                    print(f"Temporary file not found (skipping): {temp_file}")
        except Exception as e:
            print(f"Warning: Error occurred while cleaning up temporary files: {e}")

        # 清理资源
        try:
            del pre_fun.model
        except Exception:
            pass
        del pre_fun
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Audio processing completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="人声/伴奏分离（uvr5 VR 架构 / roformer）")
    parser.add_argument("--input", default=input_audio_path,
                        help="输入音频路径（默认 <项目根>/temp/output_audio.wav）")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录（默认 <项目根>/temp）")
    parser.add_argument("--engine", choices=["roformer", "uvr5"], default="roformer",
                        help="分离引擎：roformer（BS-Roformer，默认，效果最好）/ uvr5（VR HP2，快但残留多）")
    parser.add_argument("--model-name", default=None,
                        help=f"模型名。roformer 默认 {DEFAULT_MODEL['roformer']}；"
                             f"uvr5 默认 {DEFAULT_MODEL['uvr5']}（不含扩展名）")
    parser.add_argument("--agg", type=int, default=agg,
                        help="人声提取激进程度（默认 10，仅 uvr5 引擎有效）")
    parser.add_argument("--fp16", action="store_true",
                        help="半精度推理（roformer 引擎默认已开启）")
    parser.add_argument("--fp32", action="store_true",
                        help="强制单精度（关掉 roformer 的默认半精度）")
    args = parser.parse_args()

    input_audio_path = os.path.abspath(args.input)
    if not os.path.isfile(input_audio_path):
        print(f"Error: input audio not found: {input_audio_path}")
        sys.exit(1)

    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(project_path, "temp")
    os.makedirs(out_dir, exist_ok=True)
    output_vocal_path = out_dir
    output_ins_path = out_dir
    engine = args.engine
    model_name = args.model_name or DEFAULT_MODEL[engine]
    agg = args.agg
    # roformer 在 8GB 卡上默认半精度；要单精度用 --fp32
    is_half = (not args.fp32) and (args.fp16 or engine == "roformer")

    print(f"input={input_audio_path}")
    print(f"output_dir={out_dir} engine={engine} model={model_name} agg={agg} half={is_half}")

    denoise_audio()
