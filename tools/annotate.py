#!/usr/bin/env python3
# annotate.py —— 音频标注 WebUI（适配 Movie-trans 输出格式）
"""
人工校对界面：逐段看/听片段，改原文与译文，必要时切分或合并。

功能
  * 双栏编辑：原文 raw_text 与 译文 result_text 都能改（原文错会连累译文）
  * 对照试听：若该段已跑过 batch_tts，自动挂上对应的 TTS 结果一起听
  * 每段显示 说话人 / 起止时间 / 时长，顶部显示页码进度
  * 提交、翻页都会自动写回 JSON（wav_path / tts_path 这类内部字段不落盘）

注意：并发跑 TTS 会静默降质，标注页开着的时候不要去跑 batch_tts。
"""
import sys
import os
import json
import glob
import re
import argparse
import copy

# ---------------------------------------------------------------------------
# Windows 编码兜底：stdout 被重定向 / 被管道接走时，Python 会退回本地编码
# （中文系统 = GBK），脚本里打印的 emoji（🎧 ✅ 🔄 …）会抛 UnicodeEncodeError。
# 只放宽错误处理、**不改编码**，编不出的字符降级成 '?'，中文不受影响。
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:   # noqa: BLE001
        pass
import uuid
import librosa
import numpy as np
import soundfile

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda: None
except:
    pass

import gradio as gr

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TTS 结果目录（命名规则必须和 tools/merge_tts_video_improved.py 一致）
TTS_OUTPUT_DIR = os.path.join(project_root, "results", "tts_output")

# 写回 JSON 时排除的内部字段
INTERNAL_KEYS = ("wav_path", "tts_path")

# 切片文件名形如 clip_00_005_17.90-26.65.wav
CLIP_NAME_RE = re.compile(r"^clip_\d+_\d+_([\d.]+)-([\d.]+)\.wav$", re.IGNORECASE)

# 全局状态
g_data_json = []
g_index = 0
# 界面固定构建这么多槽位；「每页条数」只控制显示前几条（其余整组隐藏），
# 这样返回值个数恒定，Gradio 不会因为输出数量变化报错。
MAX_ROWS = 10
g_batch = 5          # 默认每页显示条数（<= MAX_ROWS）
g_max_json_index = -1
g_key_raw = "raw_text"
g_key_result = "result_text"
g_key_path = "wav_path"
g_load_file = ""
g_last_save = "尚未保存"


# ---------------------------------------------------------------------------
# 数据装载 / 保存
# ---------------------------------------------------------------------------
def _build_tts_path(item):
    """按 merge 脚本的命名规则找该段的 TTS 结果；没有则返回 None"""
    speaker = str(item.get("speaker", ""))
    parts = speaker.split("_")
    if len(parts) < 2:
        return None
    start = item.get("start")
    end = item.get("end")
    if start is None or end is None:
        return None
    filename = f"result_{parts[1]}_{start:.2f}-{end:.2f}.wav"
    path = os.path.join(TTS_OUTPUT_DIR, filename)
    return path if os.path.exists(path) else None


def _tts_path(item):
    """带缓存的 TTS 路径查询（缓存键为 tts_path，写回 JSON 时会被排除）"""
    if "tts_path" not in item:
        item["tts_path"] = _build_tts_path(item)
    return item["tts_path"]


def _resolve_wav_path(item, index):
    """推断该段对应的原片切片路径。

    切片文件名用的是 `{start:.2f}-{end:.2f}`（两位小数），而 JSON 里可能是
    `17.9` 这种一位小数——所以不能只靠字符串拼接，必须按数值容差匹配。
    """
    for key in ("wav", "audio", "path", "file"):
        if item.get(key):
            return item[key]

    speaker = str(item.get("speaker", "UNKNOWN"))
    start = float(item.get("start", 0) or 0)
    end = float(item.get("end", 0) or 0)
    clips_dir = os.path.join(project_root, "temp", "clips", speaker)

    # 1) 先按两位小数拼（覆盖绝大多数情况）
    matches = sorted(glob.glob(os.path.join(clips_dir, f"clip_*_{start:.2f}-{end:.2f}.wav")))
    if matches:
        return matches[0]

    # 2) 数值容差匹配（忽略小数位数差异）
    for candidate in sorted(glob.glob(os.path.join(clips_dir, "clip_*.wav"))):
        m = CLIP_NAME_RE.match(os.path.basename(candidate))
        if not m:
            continue
        if abs(float(m.group(1)) - start) <= 0.01 and abs(float(m.group(2)) - end) <= 0.01:
            return candidate

    # 3) 兜底：按序号猜一个路径（可能不存在，UI 会显示为空）
    spk_num = speaker.split("_")[1] if len(speaker.split("_")) > 1 else "00"
    return os.path.join(clips_dir, f"clip_{spk_num}_{str(index + 1).zfill(3)}_{start}-{end}.wav")


def _caption(index, item, has_tts):
    """每段的标题行：序号 · 说话人 · 起止 · 时长"""
    start = float(item.get("start", 0) or 0)
    end = float(item.get("end", 0) or 0)
    dur = max(0.0, end - start)
    speaker = item.get("speaker", "UNKNOWN")
    badge = "　🔊" if has_tts else ""
    return f"**#{index + 1}**　`{speaker}`　{start:.2f} → {end:.2f}s　**{dur:.2f}s**{badge}"


def reload_data(index, batch):
    """取当前页数据，整理成 UI 需要的字段"""
    datas = g_data_json[index: index + batch]
    output = []
    for offset, d in enumerate(datas):
        tts = _tts_path(d)
        output.append({
            "index": index + offset,
            "raw": d.get(g_key_raw, "") or "",
            "result": d.get(g_key_result, "") or "",
            "path": d.get(g_key_path, ""),
            "tts": tts,
            "caption": _caption(index + offset, d, bool(tts)),
        })
    return output


def _progress_text():
    total = len(g_data_json)
    if total == 0:
        return "⚠️ 没有数据（请先跑完 ASR）"
    first = g_index + 1
    last = min(g_index + g_batch, total)
    page_tts = sum(1 for d in g_data_json[g_index: g_index + g_batch] if _tts_path(d))
    return (
        f"**进度：第 {first}–{last} 段 / 共 {total} 段**　｜　"
        f"本页已合成 TTS：{page_tts}/{last - first + 1}　｜　💾 {g_last_save}"
    )


def _row_outputs(datas):
    """生成一页所有行的输出（组可见性, caption, 原声, TTS, 原文, 译文, 勾选）

    最后一页不满时，空位整组隐藏——不留一排空框。
    """
    outputs = []
    for i in range(MAX_ROWS):
        if i < len(datas):
            d = datas[i]
            outputs += [
                gr.update(visible=True),
                d["caption"],
                d["path"] if (d["path"] and os.path.exists(d["path"])) else None,
                d["tts"],
                d["raw"],
                d["result"],
                False,
            ]
        else:
            outputs += [
                gr.update(visible=False),
                "", None, None, "", "", False,
            ]
    return outputs


def _page_outputs(index, batch):
    """统一的翻页/刷新返回值：[index, progress] + 各行输出"""
    datas = reload_data(index, batch)
    return [index, _progress_text()] + _row_outputs(datas)


def _save_file():
    global g_last_save
    if not g_load_file:
        return
    try:
        # 写回时剔除内部字段（wav_path / tts_path 由页面自己推断）
        data_to_save = [
            {k: v for k, v in item.items() if k not in INTERNAL_KEYS}
            for item in g_data_json
        ]
        with open(g_load_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        import datetime
        g_last_save = f"已保存 {datetime.datetime.now().strftime('%H:%M:%S')}"
    except Exception as e:
        g_last_save = f"保存失败：{e}"
        print(f"Save error: {e}")


def _load_file():
    global g_data_json, g_max_json_index, g_key_raw, g_key_result
    if not g_load_file or not os.path.exists(g_load_file):
        g_data_json = []
        g_max_json_index = -1
        return
    try:
        with open(g_load_file, "r", encoding="utf-8") as f:
            g_data_json = json.load(f)

        if isinstance(g_data_json, dict) and "segments" in g_data_json:
            g_data_json = g_data_json["segments"]

        if g_data_json:
            first = g_data_json[0]
            # 文本字段：优先译文，同时兼容只有原文的老数据
            g_key_result = "result_text" if "result_text" in first else ""
            g_key_raw = "raw_text" if "raw_text" in first else ""
            if not g_key_result and not g_key_raw:
                g_key_result, g_key_raw = "result_text", "raw_text"
                print("Warning: JSON 里没有 raw_text/result_text 字段，将按这两个名字新建")

            # 音频路径：显式字段优先，否则按时间戳去 temp/clips 里找
            for key in ("wav", "audio", "path", "file", "wav_path"):
                if key in first and first[key]:
                    g_key_path = key
                    break
            else:
                g_key_path = "wav_path"
                missing = 0
                for i, item in enumerate(g_data_json):
                    item["wav_path"] = _resolve_wav_path(item, i)
                    if not os.path.exists(item["wav_path"]):
                        missing += 1
                if missing:
                    print(f"Warning: {missing}/{len(g_data_json)} 段找不到对应原声切片")

        g_max_json_index = len(g_data_json) - 1
        print(f"Loaded {len(g_data_json)} segments from {g_load_file}")
    except Exception as e:
        print(f"Load error: {e}")


def set_global(load_json):
    global g_load_file
    g_load_file = load_json
    _load_file()


# ---------------------------------------------------------------------------
# 交互逻辑
# ---------------------------------------------------------------------------
def b_change_index(index, batch):
    global g_index, g_batch
    g_index, g_batch = int(index), min(int(batch), MAX_ROWS)
    g_index = min(max(0, g_index), max(0, g_max_json_index))
    return _page_outputs(g_index, g_batch)


def b_next_index(index, batch):
    _save_file()
    new_index = min(int(index) + int(batch), max(0, g_max_json_index))
    return b_change_index(new_index, batch)


def b_previous_index(index, batch):
    _save_file()
    return b_change_index(max(0, int(index) - int(batch)), batch)


def _checked_indices(checkbox_list):
    """勾选且**在当前页可见范围内**的段下标。

    注意：界面固定构建 MAX_ROWS 个槽位，隐藏槽位的值不会被用户看到，
    必须排除掉，否则"反选/提交/删除"会误伤页外的段。
    """
    return [
        i for i, c in enumerate(checkbox_list)
        if c and i < g_batch and g_index + i < len(g_data_json)
    ]


def b_submit_change(*text_list):
    """提交本页文本：前半是原文，后半是译文（只处理当前页可见的行）"""
    global g_index
    half = len(text_list) // 2
    for i in range(min(g_batch, half)):
        idx = g_index + i
        if idx >= len(g_data_json):
            break
        raw = (text_list[i] or "").strip()
        result = (text_list[half + i] or "").strip()
        if g_key_raw and g_data_json[idx].get(g_key_raw, "") != raw:
            g_data_json[idx][g_key_raw] = raw
        if g_key_result and g_data_json[idx].get(g_key_result, "") != result:
            g_data_json[idx][g_key_result] = result
    _save_file()
    return _page_outputs(g_index, g_batch)


def b_refresh_tts(*_args):
    """重新扫描 TTS 结果（跑完 batch_tts 后不用重启标注页）"""
    for item in g_data_json:
        item.pop("tts_path", None)
    return _page_outputs(g_index, g_batch)


def b_delete_audio(*checkbox_list):
    """删除勾选的段（只从 JSON 移除，不动磁盘上的 wav）"""
    global g_max_json_index
    _save_file()
    to_delete = _checked_indices(checkbox_list)
    for idx in reversed(to_delete):
        g_data_json.pop(idx)
    g_max_json_index = len(g_data_json) - 1
    new_index = min(g_index, max(0, g_max_json_index))
    _save_file()
    return b_change_index(new_index, g_batch)


def b_invert_selection(*checkbox_list):
    """反选：只作用于当前页可见的行，隐藏槽位一律置 False"""
    return [
        (not x if isinstance(x, bool) else True) if i < g_batch else False
        for i, x in enumerate(checkbox_list)
    ]


def get_next_path(filename):
    base_dir = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(100):
        new_path = os.path.join(base_dir, f"{base_name}_{str(i).zfill(2)}.wav")
        if not os.path.exists(new_path):
            return new_path
    return os.path.join(base_dir, f"{uuid.uuid4()}.wav")


def b_audio_split(audio_breakpoint, *checkbox_list):
    """把勾选的那一段按切分点切成两段"""
    global g_max_json_index
    _save_file()
    checked = _checked_indices(checkbox_list)
    if len(checked) == 1:
        idx = g_index + checked[0]
        item = g_data_json[idx]
        path = item.get(g_key_path, "")
        if not path or not os.path.exists(path):
            return _page_outputs(g_index, g_batch)
        data, sr = librosa.load(path, sr=None, mono=True)
        split_frame = int(float(audio_breakpoint) * sr)
        if 0 < split_frame < len(data):
            new_path = get_next_path(path)
            soundfile.write(new_path, data[split_frame:], sr)
            soundfile.write(path, data[:split_frame], sr)
            new_item = copy.deepcopy(item)
            new_item[g_key_path] = new_path
            new_item.pop("tts_path", None)
            item.pop("tts_path", None)
            g_data_json.insert(idx + 1, new_item)
            _save_file()
    g_max_json_index = len(g_data_json) - 1
    return _page_outputs(g_index, g_batch)


def b_merge_audio(interval_s, *checkbox_list):
    """把勾选的多段合并成一段（间隔处补静音）"""
    global g_max_json_index
    _save_file()
    indices = sorted(g_index + i for i in _checked_indices(checkbox_list))
    if len(indices) > 1:
        base_item = g_data_json[indices[0]]
        audios, texts, sr_ref = [], [], None
        for idx in indices:
            item = g_data_json[idx]
            path = item.get(g_key_path, "")
            if not path or not os.path.exists(path):
                continue
            data, sr = librosa.load(path, sr=sr_ref, mono=True)
            sr_ref = sr
            audios.append(data)
            texts.append(item.get(g_key_result, "") or item.get(g_key_raw, ""))
            if idx != indices[0]:
                os.remove(path)  # 合并后多余的切片删掉
        if audios and sr_ref:
            merged = []
            for i, audio in enumerate(audios):
                if i > 0:
                    merged.append(np.zeros(int(sr_ref * float(interval_s))))
                merged.append(audio)
            soundfile.write(base_item[g_key_path], np.concatenate(merged), sr_ref)
            base_item[g_key_result] = "".join(texts)
            base_item.pop("tts_path", None)
            for idx in reversed(indices[1:]):
                g_data_json.pop(idx)
            _save_file()
    g_max_json_index = len(g_data_json) - 1
    return _page_outputs(g_index, g_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_json", default="", help="Path to speaker_diarization.json")
    parser.add_argument("--port", type=int, default=9871, help="WebUI port")
    args = parser.parse_args()

    if not args.load_json:
        print("Error: --load_json is required")
        sys.exit(1)

    set_global(args.load_json)

    custom_css = """
    .center-content {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        width: 100%;
    }
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 20px;
    }
    """

    with gr.Blocks(title="🎬 音频标注 WebUI", css=custom_css) as demo:
        # 创建居中容器
        with gr.Column(elem_classes=["center-content"]):
            with gr.Column(elem_classes=["main-container"]):
                gr.Markdown("## 🏷️ 音频标注 · 原文 & 译文校对")
                progress_md = gr.Markdown(_progress_text())

                gr.Markdown(
                    "<small>提交 / 翻页会自动写回 JSON ｜ 原声 = TTS 的参考与时长基准，"
                    "TTS 对照需要该段已经跑过批量合成</small>"
                )

                with gr.Row():
                    btn_prev = gr.Button("⏮️ 上一页")
                    btn_next = gr.Button("⏭️ 下一页")
                    btn_submit = gr.Button("💾 提交本页", variant="primary")
                    btn_refresh_tts = gr.Button("🔄 刷新 TTS 对照")

                with gr.Row():
                    btn_merge = gr.Button("🔗 合并选中")
                    btn_split = gr.Button("✂️ 切分选中")
                    btn_delete = gr.Button("🗑️ 删除选中")
                    btn_invert = gr.Button("🔄 反选")

                with gr.Row():
                    page_size = gr.Dropdown(
                        label="每页条数", choices=[3, 5, 10], value=5, scale=1,
                        info="嫌挤就调小，每条占的高度不变"
                    )
                    max_index = max(0, g_max_json_index)
                    index_slider = gr.Slider(
                        0, max_index, value=g_index, step=1,
                        label="起始索引（拖动直接跳页）", scale=4
                    )

                with gr.Row():
                    split_sec = gr.Slider(
                        0, 60, value=1.0, step=0.1, label="切分点 (秒，相对该段起点)", scale=1
                    )
                    merge_interval = gr.Slider(
                        0, 2, value=0.3, step=0.01, label="合并时的间隔静音 (秒)", scale=1
                    )

                # 动态生成批次控件
                groups, captions, audios_orig, audios_tts = [], [], [], []
                textboxes_raw, textboxes_result, checkboxes = [], [], []

                gr.Markdown(
                    "<small>每段一张卡片：上排两个播放器（**原声** / **TTS 对照**），"
                    "下排两个输入框（**原文 raw_text** / **译文 result_text**），"
                    "右上角勾选后可用于 合并 / 切分 / 删除</small>"
                )

                # 每条一张卡片，给足宽度（原声/TTS 各占半宽，原文/译文各占半宽）
                for i in range(MAX_ROWS):
                    with gr.Group() as grp:
                        with gr.Row():
                            with gr.Column(scale=10):
                                cap = gr.Markdown("")
                            with gr.Column(scale=0, min_width=48):
                                chk = gr.Checkbox(show_label=False, container=False)
                        with gr.Row():
                            aud_o = gr.Audio(
                                label="🎧 原声", interactive=False,
                                show_download_button=False, scale=1
                            )
                            aud_t = gr.Audio(
                                label="🔊 TTS 对照", interactive=False,
                                show_download_button=False, scale=1
                            )
                        with gr.Row():
                            txt_raw = gr.Textbox(
                                show_label=False, lines=3, scale=1,
                                placeholder="原文 raw_text"
                            )
                            txt_result = gr.Textbox(
                                show_label=False, lines=3, scale=1,
                                placeholder="译文 result_text"
                            )
                        groups.append(grp)
                        captions.append(cap)
                        audios_orig.append(aud_o)
                        audios_tts.append(aud_t)
                        textboxes_raw.append(txt_raw)
                        textboxes_result.append(txt_result)
                        checkboxes.append(chk)

        # 输出顺序：[index, progress] + 每行(组, caption, 原声, TTS, 原文, 译文, 勾选)
        row_outputs = []
        for i in range(MAX_ROWS):
            row_outputs += [
                groups[i], captions[i], audios_orig[i], audios_tts[i],
                textboxes_raw[i], textboxes_result[i], checkboxes[i],
            ]
        page_outputs = [index_slider, progress_md] + row_outputs

        # 翻页（每页条数由下拉框决定，不再固定 10）
        btn_prev.click(b_previous_index, [index_slider, page_size], page_outputs)
        btn_next.click(b_next_index, [index_slider, page_size], page_outputs)
        index_slider.release(b_change_index, [index_slider, page_size], page_outputs)
        page_size.change(
            lambda size, idx: b_change_index(idx, size), [page_size, index_slider], page_outputs
        )

        # 提交（原文 body + 译文 body 一起传）
        btn_submit.click(b_submit_change, textboxes_raw + textboxes_result, page_outputs)

        # TTS 对照重扫
        btn_refresh_tts.click(b_refresh_tts, [], page_outputs)

        # 合并 / 切分 / 删除
        btn_merge.click(b_merge_audio, [merge_interval] + checkboxes, page_outputs)
        btn_split.click(b_audio_split, [split_sec] + checkboxes, page_outputs)
        btn_delete.click(b_delete_audio, checkboxes, page_outputs)

        # 反选
        btn_invert.click(b_invert_selection, checkboxes, checkboxes)

        # 初始化加载
        demo.load(b_change_index, [index_slider, page_size], page_outputs)

    print(f"Starting annotation WebUI: http://localhost:{args.port}")
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=args.port,
        inbrowser=False,
        quiet=True
    )
