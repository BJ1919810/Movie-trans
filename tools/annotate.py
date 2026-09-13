#!/usr/bin/env python3
# annotate.py —— 音频标注 WebUI（适配 Movie-trans 输出格式）
import sys
import os
import json
import argparse
import copy
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

# 全局状态
g_data_json = []
g_index = 0
g_batch = 10
g_max_json_index = -1
g_json_key_text = "text"
g_json_key_path = "wav_path"
g_load_file = ""
g_load_format = "json"

def reload_data(index, batch):
    datas = g_data_json[index : index + batch]
    output = []
    for d in datas:
        text_val = d.get(g_json_key_text, "")
        path_val = d.get(g_json_key_path, "")
        output.append({"text": text_val, "path": path_val})
    return output

def b_change_index(index, batch):
    global g_index, g_batch
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)
    outputs = []
    # Textboxes
    for i, d in enumerate(datas):
        outputs.append(gr.update(label=f"Text {i + index}", value=d["text"]))
    for _ in range(g_batch - len(datas)):
        outputs.append(gr.update(label="Text", value=""))
    # Audios
    for d in datas:
        outputs.append(d["path"] if os.path.exists(d["path"]) else None)
    for _ in range(g_batch - len(datas)):
        outputs.append(None)
    # Checkboxes
    for _ in datas:
        outputs.append(False)
    for _ in range(g_batch - len(datas)):
        outputs.append(False)
    # 修复：确保index不超过最大索引值
    max_index = max(0, g_max_json_index)
    index = min(index, max_index)
    return [index] + outputs

def b_next_index(index, batch):
    _save_file()
    # 修复：确保g_max_json_index至少为0
    max_index = max(0, g_max_json_index)
    if index + batch <= max_index:
        new_index = index + batch
    else:
        new_index = index
    # 修复：确保new_index不超过最大索引值
    new_index = min(new_index, max_index)
    return [new_index] + b_change_index(new_index, batch)[1:]

def b_previous_index(index, batch):
    _save_file()
    new_index = max(0, index - batch)
    # 修复：确保g_max_json_index至少为0
    max_index = max(0, g_max_json_index)
    new_index = min(new_index, max_index)
    return [new_index] + b_change_index(new_index, batch)[1:]

def b_submit_change(*text_list):
    global g_data_json
    for i, new_text in enumerate(text_list):
        idx = g_index + i
        if idx < len(g_data_json):
            new_text = new_text.strip()
            if g_data_json[idx].get(g_json_key_text, "") != new_text:
                g_data_json[idx][g_json_key_text] = new_text
    _save_file()
    return [g_index] + b_change_index(g_index, g_batch)[1:]

def b_delete_audio(*checkbox_list):
    global g_data_json, g_max_json_index
    _save_file()
    to_delete = []
    for i, checked in enumerate(checkbox_list):
        if checked and g_index + i < len(g_data_json):
            to_delete.append(g_index + i)
    for idx in reversed(to_delete):
        g_data_json.pop(idx)
    g_max_json_index = len(g_data_json) - 1
    # 修复：确保g_max_json_index至少为0
    max_index = max(0, g_max_json_index)
    new_index = min(g_index, max_index) if max_index >= 0 else 0
    _save_file()
    return [new_index] + b_change_index(new_index, g_batch)[1:]

def b_invert_selection(*checkbox_list):
    return [not x if isinstance(x, bool) else True for x in checkbox_list]

def get_next_path(filename):
    base_dir = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(100):
        new_path = os.path.join(base_dir, f"{base_name}_{str(i).zfill(2)}.wav")
        if not os.path.exists(new_path):
            return new_path
    return os.path.join(base_dir, f"{uuid.uuid4()}.wav")

def b_audio_split(audio_breakpoint, *checkbox_list):
    global g_data_json
    _save_file()
    checked_indices = [i for i, c in enumerate(checkbox_list) if c and g_index + i < len(g_data_json)]
    if len(checked_indices) == 1:
        idx = g_index + checked_indices[0]
        item = g_data_json[idx]
        path = item.get(g_json_key_path, "")
        if not path or not os.path.exists(path):
            return [g_index] + b_change_index(g_index, g_batch)[1:]
        data, sr = librosa.load(path, sr=None, mono=True)
        split_frame = int(audio_breakpoint * sr)
        if 0 < split_frame < len(data):
            first = data[:split_frame]
            second = data[split_frame:]
            new_path = get_next_path(path)
            soundfile.write(new_path, second, sr)
            soundfile.write(path, first, sr)
            new_item = copy.deepcopy(item)
            new_item[g_json_key_path] = new_path
            g_data_json.insert(idx + 1, new_item)
            _save_file()
    g_max_json_index = len(g_data_json) - 1
    return [g_index] + b_change_index(g_index, g_batch)[1:]

def b_merge_audio(interval_s, *checkbox_list):
    global g_data_json
    _save_file()
    indices = [g_index + i for i, c in enumerate(checkbox_list) if c and g_index + i < len(g_data_json)]
    if len(indices) > 1:
        indices.sort()
        base_item = g_data_json[indices[0]]
        audios = []
        texts = []
        sr_ref = None
        for idx in indices:
            item = g_data_json[idx]
            path = item.get(g_json_key_path, "")
            if not path or not os.path.exists(path):
                continue
            data, sr = librosa.load(path, sr=sr_ref, mono=True)
            sr_ref = sr
            audios.append(data)
            texts.append(item.get(g_json_key_text, ""))
            if idx != indices[0]:
                os.remove(path)  # 删除被合并的音频
        # 插入静音
        merged = []
        for i, audio in enumerate(audios):
            if i > 0:
                silence = np.zeros(int(sr_ref * interval_s))
                merged.append(silence)
            merged.append(audio)
        soundfile.write(base_item[g_json_key_path], np.concatenate(merged), sr_ref)
        base_item[g_json_key_text] = "".join(texts)
        # 删除多余条目
        for idx in reversed(indices[1:]):
            g_data_json.pop(idx)
        _save_file()
    g_max_json_index = len(g_data_json) - 1
    return [g_index] + b_change_index(g_index, g_batch)[1:]

def _save_file():
    if not g_load_file:
        return
    try:
        # 创建一个不包含wav_path属性的数据副本
        data_to_save = []
        for item in g_data_json:
            # 创建不包含wav_path的新字典
            saved_item = {k: v for k, v in item.items() if k != "wav_path"}
            data_to_save.append(saved_item)
        
        with open(g_load_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Save error: {e}")

def _load_file():
    global g_data_json, g_max_json_index
    if not g_load_file or not os.path.exists(g_load_file):
        g_data_json = []
        g_max_json_index = -1
        return
    try:
        with open(g_load_file, "r", encoding="utf-8") as f:
            g_data_json = json.load(f)
        # 自动探测 key（兼容 Movie-trans JSON）
        if g_data_json:
            first = g_data_json[0]
            # 音频路径候选
            path_keys = ["wav", "audio", "path", "wav_path", "file"]
            for k in path_keys:
                if k in first:
                    global g_json_key_path
                    g_json_key_path = k
                    break
            # 文本候选 - 优先获取 result_text，如果没有则获取 raw_text
            if "result_text" in first:
                global g_json_key_text
                g_json_key_text = "result_text"
            elif "raw_text" in first:
                g_json_key_text = "raw_text"
            else:
                g_json_key_text = ''
            
            # 如果没有找到音频路径字段，则动态构造路径
            if g_json_key_path not in first:
                # 为每个条目添加wav_path字段
                for i, item in enumerate(g_data_json):
                    speaker = item.get("speaker", "UNKNOWN")
                    start_time = item.get("start", 0)
                    end_time = item.get("end", 0)
                    # 构造符合temp/clips目录结构的路径（基于时间戳而不是索引）
                    wav_filename = f"clip_{speaker.split('_')[1].zfill(2)}_{str(i+1).zfill(3)}_{start_time}-{end_time}.wav"
                    wav_path = os.path.join(project_root, "temp", "clips", speaker, wav_filename)
                    item["wav_path"] = wav_path
                g_json_key_path = "wav_path"
                
                # 修正文件路径以匹配实际的文件命名（基于时间戳）
                import glob
                for item in g_data_json:
                    speaker = item.get("speaker", "UNKNOWN")
                    start_time = item.get("start", 0)
                    end_time = item.get("end", 0)
                    # 查找匹配的文件（基于时间戳）
                    pattern = os.path.join(project_root, "temp", "clips", speaker, f"clip_*_{start_time}-{end_time}.wav")
                    matches = glob.glob(pattern)
                    if matches:
                        item["wav_path"] = matches[0]
                    else:
                        # 如果没有精确匹配，尝试近似匹配
                        pattern = os.path.join(project_root, "temp", "clips", speaker, f"clip_*_{start_time}-*.wav")
                        matches = glob.glob(pattern)
                        if matches:
                            item["wav_path"] = matches[0]
                
        g_max_json_index = len(g_data_json) - 1
    except Exception as e:
        print(f"Load error: {e}")

def set_global(load_json):
    global g_load_file
    g_load_file = load_json
    _load_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_json", default="", help="Path to speaker_diarization.json")
    parser.add_argument("--port", type=int, default=9871, help="WebUI port")
    args = parser.parse_args()

    if not args.load_json:
        print("Error: --load_json is required")
        sys.exit(1)

    set_global(args.load_json)

    # 定义自定义CSS样式
    custom_css = """
    .large-checkbox input[type="checkbox"] {
        transform: scale(2);
        margin: 10px;
    }
    .large-checkbox label {
        font-size: 1.2em;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        width: 100%;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    """
    
    with gr.Blocks(title="🎬 Audio Annotation WebUI", css=custom_css) as demo:
        # 创建居中容器
        with gr.Column(elem_classes=["center-content"]):
            with gr.Column(elem_classes=["main-container"]):
                gr.Markdown("## 🏷️ 音频标注工具（支持切分/合并/删除）")
                gr.Markdown("✅ 提交文本 → 保存到文件 | 🔄 刷新页面会丢失未提交修改！")

                with gr.Row():
                    btn_prev = gr.Button("⏮️ 上一页")
                    btn_next = gr.Button("⏭️ 下一页")
                    btn_submit = gr.Button("💾 提交文本", variant="primary")
                    btn_merge = gr.Button("🔗 合并选中")
                    btn_split = gr.Button("✂️ 切分选中")
                    btn_delete = gr.Button("🗑️ 删除选中")
                    btn_invert = gr.Button("🔄 反选")

                with gr.Row():
                    # 修复：确保g_max_json_index至少为0，避免Slider范围错误
                    max_index = max(0, g_max_json_index)
                    index_slider = gr.Slider(0, max_index, value=g_index, step=1, label="起始索引", scale=3)
                    split_sec = gr.Slider(0, 120, value=1.0, step=0.1, label="切分点 (秒)", scale=2)
                    merge_interval = gr.Slider(0, 2, value=0.3, step=0.01, label="合并间隔 (秒)", scale=2)

                # 动态生成批次控件
                textboxes = []
                audios = []
                checkboxes = []
                with gr.Row():
                    with gr.Column():
                        for i in range(g_batch):
                            with gr.Row():
                                txt = gr.Textbox(label=f"Text {i}", scale=4, lines=6)  # 增加文本框高度（原来是2行，现在是4行）
                                aud = gr.Audio(label=f"Audio {i}", scale=4)
                                chk = gr.Checkbox(label="✓", scale=1, elem_classes=["large-checkbox"])  # 添加CSS类来增大复选框
                                textboxes.append(txt)
                                audios.append(aud)
                                checkboxes.append(chk)

        # 绑定事件
        def make_update_fn():
            inputs = [index_slider] + textboxes + checkboxes
            outputs = [index_slider] + textboxes + audios + checkboxes
            return inputs, outputs

        # 上一页/下一页
        btn_prev.click(b_previous_index, [index_slider, gr.State(g_batch)], [index_slider] + textboxes + audios + checkboxes)
        btn_next.click(b_next_index, [index_slider, gr.State(g_batch)], [index_slider] + textboxes + audios + checkboxes)

        # 提交
        btn_submit.click(b_submit_change, textboxes, [index_slider] + textboxes + audios + checkboxes)

        # 合并/切分/删除
        btn_merge.click(b_merge_audio, [merge_interval] + checkboxes, [index_slider] + textboxes + audios + checkboxes)
        btn_split.click(b_audio_split, [split_sec] + checkboxes, [index_slider] + textboxes + audios + checkboxes)
        btn_delete.click(b_delete_audio, checkboxes, [index_slider] + textboxes + audios + checkboxes)

        # 反选
        btn_invert.click(b_invert_selection, checkboxes, checkboxes)

        # 初始化加载
        demo.load(b_change_index, [index_slider, gr.State(g_batch)], [index_slider] + textboxes + audios + checkboxes)

    print(f"Starting annotation WebUI: http://localhost:{args.port}")
    demo.launch(
        server_name="0.0.0.0",  # 改为0.0.0.0以允许外部访问
        server_port=args.port,
        inbrowser=False,
        quiet=True
    )