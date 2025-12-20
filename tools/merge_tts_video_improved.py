import os
import sys
import json
import argparse
from pydub import AudioSegment

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目根目录到Python路径
sys.path.append(project_root)

def get_existing_audio():
    """获取已存在的音频文件"""
    existing_audio_path = os.path.join(project_root, "temp", "output_audio.wav")
    
    if os.path.exists(existing_audio_path):
        print(f"Using existing audio file: {existing_audio_path}")
        return existing_audio_path
    else:
        # 如果不存在，则从原始视频中提取音频
        print("Existing audio file not found, extracting audio from original video...")
        video_path = os.path.join(project_root, "1.mp4")
        extracted_audio_path = os.path.join(project_root, "temp", "output_audio.wav")
        
        print("Extracting audio from original video...")
        cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {extracted_audio_path} -y"
        os.system(cmd)
        print(f"Audio extracted to: {extracted_audio_path}")
        
        return extracted_audio_path

def insert_background_audio(original_audio_path, bg_audio_path, segments):
    """根据时间戳将背景音频插入到指定位置"""
    print("Loading original audio...")
    original_audio = AudioSegment.from_wav(original_audio_path)
    
    print("Loading background audio...")
    bg_audio = AudioSegment.from_wav(bg_audio_path)
    
    print("Inserting background audio into specified positions...")
    # 分别保存原始音频和背景音频，便于后续独立处理
    audio_with_bg = original_audio
    bg_segments_positions = []
    
    # 按时间倒序处理，避免位置偏移
    sorted_segments = sorted(segments, key=lambda x: x['start'], reverse=True)
    
    for segment in sorted_segments:
        start_time = segment['start']
        end_time = segment['end']
        
        # 计算插入位置（毫秒）
        start_position = int(start_time * 1000)
        end_position = int(end_time * 1000)
        
        # 从背景音频中提取对应片段
        bg_segment = bg_audio[start_position:end_position]
        
        # 直接将背景音频片段插入到指定位置
        before_insert = audio_with_bg[:start_position]
        after_insert = audio_with_bg[end_position:]
        audio_with_bg = before_insert + bg_segment + after_insert
        
        # 保存背景音频片段的位置信息，便于后续处理
        bg_segments_positions.append({
            'start': start_position,
            'end': end_position,
            'segment': bg_segment
        })
        
        print(f"Background audio inserted at {start_time}s - {end_time}s position")
    
    return audio_with_bg, bg_segments_positions

def overlay_tts_audio(final_audio, tts_output_dir, segments):
    """将TTS音频叠加到对应位置，同时保持背景音乐音量"""
    print("Overlaying TTS audio to corresponding positions...")
    
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        text = segment['result_text']
        
        # 构造TTS音频文件名
        speaker_num = speaker.split('_')[1]
        start_str = f"{start_time:.2f}"
        end_str = f"{end_time:.2f}"
        
        tts_filename = f"result_{speaker_num}_{start_str}-{end_str}.wav"
        tts_filepath = os.path.join(tts_output_dir, tts_filename)
        
        # 检查TTS音频文件是否存在
        if os.path.exists(tts_filepath):
            print(f"Processing: {tts_filename}")
            print(f"Text: {text}")
            
            # 加载TTS音频
            tts_audio = AudioSegment.from_wav(tts_filepath)
            
            # 自动归一化TTS音频至-3 dBFS
            tts_audio = tts_audio.normalize(-3.0)
            
            # 计算叠加位置（毫秒）
            overlay_position = int(start_time * 1000)
            overlay_end_position = int(end_time * 1000)
            
            # 将TTS音频叠加到最终音频的指定位置（直接叠加，不处理原音频）
            final_audio = final_audio.overlay(tts_audio, position=overlay_position)
            
            print(f"TTS audio overlaid at {start_time} seconds position")
        else:
            print(f"Warning: Cannot find TTS audio file {tts_filepath}")
    
    return final_audio

def generate_ass_subtitle(segments, output_ass_path):
    """生成ASS字幕文件，包含原始文本和翻译文本"""
    print("Generating ASS subtitle file...")
    
    # 获取视频分辨率
    video_width, video_height = 852, 480  # 根据ffprobe结果设置
    
    # ASS文件头部信息
    ass_header = f"""[Script Info]
Title: Bilingual Subtitles
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ResultText,Arial,27,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1
Style: RawText,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1
Style: Default,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            raw_text = segment.get('raw_text', '')
            result_text = segment.get('result_text', '')
            
            # 将秒数转换为ASS时间格式 (HH:MM:SS.cc)
            def seconds_to_ass_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                centiseconds = int((seconds * 100) % 100)
                return f"{hours:01d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
            
            start_ass_time = seconds_to_ass_time(start_time)
            end_ass_time = seconds_to_ass_time(end_time)
            
            # 双行字幕：上面是翻译文本(result_text)，下面是原始文本(raw_text)
            # 使用不同样式分别设置字体大小
            # 处理换行符，确保文本可以正确换行显示
            raw_text_escaped = raw_text.replace('\n', '\\N').replace('\r', '')
            result_text_escaped = result_text.replace('\n', '\\N').replace('\r', '')
            subtitle_text = f"{{\\rResultText}}{result_text_escaped}\\N{{\\rRawText}}{raw_text_escaped}"
            
            f.write(f"Dialogue: 0,{start_ass_time},{end_ass_time},Default,,0,0,0,,{subtitle_text}\n")
    
    print(f"ASS subtitle file saved to: {output_ass_path}")

def merge_audio_with_video(video_path, final_audio_path, output_video_path, subtitle_path=None, burn_subtitles=False):
    """将处理后的音频与视频合并，使用loudnorm进行响度归一化，并根据需要封装软字幕或硬烧录字幕"""
    print("Merging processed audio with video...")
    
    # 获取输出文件扩展名以确定格式
    output_extension = os.path.splitext(output_video_path)[1].lower()
    
    if subtitle_path and os.path.exists(subtitle_path):
        if burn_subtitles:
            # 硬烧录字幕到视频帧中
            print("Burning subtitles into video frames...")
            # 使用FFmpeg的ass滤镜将字幕渲染到视频上
            # 对Windows路径进行转义处理，避免FFmpeg解析错误
            escaped_subtitle_path = subtitle_path.replace('\\', '\\\\').replace(':', '\\:')
            
            # 如果输出是MP4格式，先生成MKV再转码为MP4
            if output_extension == ".mp4":
                # 创建临时MKV文件路径
                temp_mkv_path = output_video_path.replace(".mp4", "_temp.mkv")
                # Add -map 0:v -map 1:a to ensure audio/video sources are explicit
                cmd1 = f'ffmpeg -i "{video_path}" -i "{final_audio_path}" -map 0:v -map 1:a -vf "ass=\'{escaped_subtitle_path}\'" -c:v libx264 -preset medium -crf 23 -c:a aac -af loudnorm=I=-16:TP=-1.5:LRA=11 "{temp_mkv_path}" -y'
                os.system(cmd1)
                
                # 将MKV转码为MP4，直接使用处理后的音频文件
                print("Converting MKV to MP4 format...")
                # Add -map 0:v -map 0:a for rigor (although MKV usually has only one audio/video stream)
                cmd2 = f'ffmpeg -i "{temp_mkv_path}" -map 0:v -map 0:a -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 192k "{output_video_path}" -y'
                os.system(cmd2)
                
                # 删除临时MKV文件
                if os.path.exists(temp_mkv_path):
                    os.remove(temp_mkv_path)
                    
                print(f"MP4 video with burned-in subtitles saved to: {output_video_path}")
            else:
                # 直接生成MKV格式
                # Also add -map 0:v -map 1:a
                cmd = f'ffmpeg -i "{video_path}" -i "{final_audio_path}" -map 0:v -map 1:a -vf "ass=\'{escaped_subtitle_path}\'" -c:v libx264 -preset medium -crf 23 -c:a aac -af loudnorm=I=-16:TP=-1.5:LRA=11 "{output_video_path}" -y'
                os.system(cmd)
                print(f"MKV video with burned-in subtitles saved to: {output_video_path}")
        else:
            # 软字幕封装
            if output_extension == ".mkv":
                print("Embedding soft subtitles into MKV video...")
                # MKV格式支持完整的ASS字幕，无需转换编码
                cmd = f"ffmpeg -i {video_path} -i {final_audio_path} -i {subtitle_path} -c:v copy -c:a aac -af loudnorm=I=-16:TP=-1.5:LRA=11 -map 0:v:0 -map 1:a:0 -map 2:s:0 -c:s copy {output_video_path} -y"
                os.system(cmd)
                print(f"MKV video with soft subtitles saved to: {output_video_path}")
            else:
                print("Embedding soft subtitles into MP4 video...")
                # MP4格式使用mov_text编码封装字幕
                cmd = f"ffmpeg -i {video_path} -i {final_audio_path} -i {subtitle_path} -c:v copy -c:a aac -af loudnorm=I=-16:TP=-1.5:LRA=11 -map 0:v:0 -map 1:a:0 -map 2:s:0 -c:s mov_text {output_video_path} -y"
                os.system(cmd)
                print(f"MP4 video with soft subtitles saved to: {output_video_path}")
    else:
        # 不使用字幕，直接合并音频和视频
        cmd = f"ffmpeg -i {video_path} -i {final_audio_path} -c:v copy -c:a aac -af loudnorm=I=-16:TP=-1.5:LRA=11 -map 0:v:0 -map 1:a:0 {output_video_path} -y"
        os.system(cmd)
        if output_extension == ".mkv":
            print(f"MKV video saved to: {output_video_path}")
        else:
            print(f"MP4 video saved to: {output_video_path}")

def load_subtitle_data(json_path):
    """加载字幕数据"""
    if not os.path.exists(json_path):
        print(f"Warning: Subtitle data file does not exist: {json_path}")
        return None
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 检查数据格式
    if isinstance(data, dict) and 'segments' in data:
        segments = data['segments']
    elif isinstance(data, list):
        segments = data
    else:
        print("Warning: Subtitle data format is incorrect")
        return None
        
    # 验证每个片段是否包含所需字段
    for i, segment in enumerate(segments):
        if 'start' not in segment or 'end' not in segment:
            print(f"Warning: Segment {i} is missing start or end field")
            return None
            
    return segments

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Merge TTS audio with video')
    parser.add_argument('--enable-subtitles', action='store_true', help='Enable bilingual subtitles (raw_text and result_text)')
    parser.add_argument('--burn-subtitles', action='store_true', help='Burn subtitles directly into the video frames (hardsub)')
    parser.add_argument('--output-format', choices=['mp4', 'mkv'], default='mkv', help='Output video format (default: mkv)')
    args = parser.parse_args()
    
    # 定义路径（使用项目根目录）
    bg_audio_path = os.path.join(project_root, "temp", "bg_1_44100.wav")
    tts_output_dir = os.path.join(project_root, "results", "tts_output")
    diarization_file = os.path.join(project_root, "results", "speaker_diarization.json")
    video_path = os.path.join(project_root, "1.mp4")
    
    # 根据参数决定输出格式
    output_extension = "." + args.output_format
    output_video_path = os.path.join(project_root, "results", f"output_improved{output_extension}")
    final_audio_path = os.path.join(project_root, "temp", "final_audio_improved.wav")
    
    # 初始化字幕路径
    subtitle_path = None
    
    # 如果启用了字幕功能，则加载字幕数据
    if args.enable_subtitles:
        segments_data = load_subtitle_data(diarization_file)
        
        if segments_data:
            # 生成ASS字幕文件，统一命名为subtitle.ass，并产出到results目录中
            subtitle_path = os.path.join(project_root, "results", "subtitle.ass")
            generate_ass_subtitle(segments_data, subtitle_path)
        else:
            print("Warning: Unable to load subtitle data, will continue processing without subtitles")
    
    # 读取说话人分割信息
    print("Reading speaker diarization information...")
    if not os.path.exists(diarization_file):
        print(f"Error: Speaker diarization file does not exist: {diarization_file}")
        return
        
    with open(diarization_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # 1. 获取原始音频文件
    original_audio_path = get_existing_audio()
    
    # 2. 将背景音频插入到指定位置
    audio_with_bg, bg_segments_positions = insert_background_audio(original_audio_path, bg_audio_path, segments)
    
    # 3. 将TTS音频叠加到对应位置
    final_audio = overlay_tts_audio(audio_with_bg, tts_output_dir, segments)
    
    # 对最终音频进行响度归一化处理
    print("Performing loudness normalization on final audio...")
    # 先保存为临时文件
    temp_audio_path = os.path.join(project_root, "temp", "temp_final_audio.wav")
    final_audio.export(temp_audio_path, format="wav")
    
    # 使用ffmpeg进行loudnorm处理
    normalized_audio_path = final_audio_path
    cmd = f"ffmpeg -i {temp_audio_path} -af loudnorm=I=-16:TP=-1.5:LRA=11 {normalized_audio_path} -y"
    os.system(cmd)
    
    # 删除临时文件
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    print(f"Loudness normalized audio saved to: {normalized_audio_path}")
    
    # 5. 将处理后的音频与视频合并
    merge_audio_with_video(video_path, final_audio_path, output_video_path, subtitle_path, args.burn_subtitles)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()