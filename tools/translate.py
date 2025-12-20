import json
import requests
import os
import re
import sys
import argparse
from typing import List, Dict

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目根目录到Python路径
sys.path.append(project_root)

# DeepSeek API配置
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def load_diarization_data(file_path: str) -> List[Dict]:
    """加载说话人分割数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_translated_data(data: List[Dict], file_path: str):
    """保存翻译后的数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def prepare_translation_context(segments: List[Dict]) -> str:
    """准备翻译上下文"""
    context = "请保持说话人标识，并只输出翻译后的对话内容，确保标点符号的正确使用：\n\n"
    for i, segment in enumerate(segments):
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment.get('raw_text', '')
        context += f"{i+1}. {speaker}: {text}\n"
    return context

def translate_with_deepseek(text: str, api_key: str, language: str = "zh", raw_language: str = "zh") -> str:
    """使用DeepSeek API进行翻译"""
    # 构造语言映射
    language_map = {
        "zh": "中文",
        "en": "英文",
        "ja": "日文"
    }
    display_language = language_map.get(language, "英文")
    display_raw_language = language_map.get(raw_language, "中文")
    
    # 根据不同的语言组合构造提示词
    if raw_language == "zh" and language == "en":
        # 中文到英文：直接翻译
        system_prompt = f"你是一个专业的翻译人员，请将以下{display_raw_language}的对话内容直接翻译成{display_language}，保持对话的自然流畅性。请保持说话人标识，并只输出翻译后的对话内容，保证标点符号的正确使用，每行以序号开头，例如：1. SPEAKER_01: Hello."
    elif raw_language == "ja" and language == "en":
        # 日文到英文：直接翻译（原文为英文的地方保留原文）
        system_prompt = f"你是一个专业的翻译人员，请将以下{display_raw_language}的对话内容直接翻译成{display_language}，如果原文中有英文内容请保留原文。请保持说话人标识，并只输出翻译后的对话内容，保证标点符号的正确使用，每行以序号开头，例如：1. SPEAKER_01: Hello."
    elif raw_language == "en" and language == "zh":
        # 英文到中文：特殊名词、组织简称、专业术语保留英文形式
        system_prompt = f"你是一个专业的翻译人员，请将以下{display_raw_language}的对话内容翻译成{display_language}，保持对话的自然流畅性，并考虑上下文进行情景化翻译。请保持说话人标识，并只输出翻译后的对话内容，保证标点符号的正确使用，每行以序号开头，例如：1. SPEAKER_01: 你好。"
    elif raw_language == "ja" and language == "zh":
        # 日文到中文：若有英文短句则保留（比如"Nice idea！"）
        system_prompt = f"你是一个专业的翻译人员，请将以下{display_raw_language}的对话内容翻译成{display_language}，保持对话的自然流畅性，并考虑上下文进行情景化翻译，如果原文中有英文短句（如\"Nice idea!\"），请保留原文。请保持说话人标识，并只输出翻译后的对话内容，保证标点符号的正确使用，每行以序号开头，例如：1. SPEAKER_01: 你好。"
    else:
        # 默认提示词
        system_prompt = f"你是一个专业的翻译人员，请将以下对话内容翻译成{display_language}，保持对话的自然流畅性，并考虑上下文进行情景化翻译。请保持说话人标识，并只输出翻译后的对话内容，保证标点符号的正确使用，每行以序号开头，例如：1. SPEAKER_01: 你好。"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"请翻译以下对话：\n\n{text}"
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def parse_translated_text(translated_text: str, segments: List[Dict], language: str = "zh") -> List[Dict]:
    """解析翻译后的文本并匹配到原始片段"""
    # 创建新的数据结构
    translated_segments = []
    
    # 按行分割翻译结果
    lines = translated_text.split('\n')
    
    # 创建一个字典来映射序号到翻译内容
    translation_map = {}
    for line in lines:
        # 匹配序号和内容，例如 "1. SPEAKER_01: 你好"
        match = re.match(r'^(\d+)\.\s*(SPEAKER_\d+):\s*(.*)$', line.strip())
        if match:
            index = int(match.group(1)) - 1  # 转换为0基索引
            speaker = match.group(2)
            translation = match.group(3)
            translation_map[index] = {
                'speaker': speaker,
                'translation': translation
            }
    
    # 将翻译结果匹配到原始片段
    for i, segment in enumerate(segments):
        new_segment = segment.copy()
        # 如果找到了对应的翻译
        if i in translation_map:
            new_segment['result_text'] = translation_map[i]['translation'].replace(' ', '，') if language == "zh" else translation_map[i]['translation']
        else:
            # 否则保留原文
            new_segment['result_text'] = segment.get('raw_text', '')
        translated_segments.append(new_segment)
    
    return translated_segments

def translate_segments(segments: List[Dict], api_key: str, language: str = "zh", raw_language: str = "zh") -> List[Dict]:
    """翻译所有片段"""
    # 准备上下文
    context = prepare_translation_context(segments)
    print("Sending the following content for translation:")
    print(context)
    
    # 发送翻译请求
    translated_text = translate_with_deepseek(context, api_key, language, raw_language)
    
    if not translated_text:
        print("Translation failed")
        # 如果翻译失败，返回原始片段并添加空的result_text字段
        for segment in segments:
            if 'result_text' not in segment:
                segment['result_text'] = segment.get('text', '')  # 使用原始文本作为默认值
        return segments
    
    print("Translation result:")
    print(translated_text)
    
    # 解析翻译结果
    translated_segments = parse_translated_text(translated_text, segments, language)
    
    return translated_segments

def safe_copy(src, dst):
    """Copy src to dst only if they are different files."""
    import shutil
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs != dst_abs:
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        shutil.copy2(src_abs, dst_abs)
    return dst_abs

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="翻译ASR结果")
    parser.add_argument("--rawlg", type=str, default="ja", choices=["zh", "en", "ja"], 
                        help="原始语言 (zh=中文, en=英文, ja=日文)")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"], 
                        help="目标翻译语言 (zh=中文, en=英文)")
    args = parser.parse_args()
    
    # 使用项目根目录构建输入输出文件路径
    input_file = os.path.join(project_root, "results", "speaker_diarization.json")
    output_file = os.path.join(project_root, "results", "speaker_diarization.json")
    
    # 加载数据
    print("Loading speaker segmentation data...")
    segments = load_diarization_data(input_file)
    print(f"Loaded {len(segments)} segments")
    
    # 翻译数据
    print("Performing translation...")
    translated_segments = translate_segments(segments, DEEPSEEK_API_KEY, args.language, args.rawlg)
    
    # 保存翻译后的数据
    print("Saving translation results...")
    save_translated_data(translated_segments, output_file)
    print(f"Translation results saved to: {output_file}")

# 导出函数供其他模块使用
__all__ = ['load_diarization_data', 'save_translated_data', 'translate_segments']

if __name__ == "__main__":
    main()
