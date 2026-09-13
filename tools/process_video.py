#!/usr/bin/env python3
"""
Script to extract audio from a video file and save it as a WAV file with specific parameters:
- Sample rate: 48000 Hz
- Channels: Mono (1 channel)

This version uses ffmpeg directly through subprocess for more reliable operation.
"""


'''
ffmpeg -i /data/aigc/bj/Movie-trans/temp/output_audio.wav -t 60 -c copy /data/aigc/bj/Movie-trans/temp/1.wav
'''
import os
import subprocess
import sys


def extract_audio_from_video(video_path, output_audio_path, sample_rate=44100):
    """
    Extract audio from a video file and save it as a WAV file with specified sample rate and mono channel.
    
    Args:
        video_path (str): Path to the input video file
        output_audio_path (str): Path to save the output WAV file
        sample_rate (int): Desired sample rate for the output audio (default: 44100)
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Build ffmpeg command
    # -i: input file
    # -ac 2: set audio channels to 2 (stereo)
    # -ar 44100: set audio sample rate to 44100 Hz
    # -acodec pcm_s16le: set audio codec to PCM 16-bit little endian (WAV)
    # -y: overwrite output file without asking
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ac', '2',           # Stereo
        '-ar', str(sample_rate),  # Sample rate
        '-acodec', 'pcm_s16le',   # WAV codec
        '-y',                 # Overwrite output file
        output_audio_path
    ]
    
    try:
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
        print(f"Audio extracted successfully!")
        print(f"Input video: {video_path}")
        print(f"Output audio: {output_audio_path}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: Stereo")
        
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except Exception as e:
        raise RuntimeError(f"Error processing video: {e}")


def main():
    import argparse

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="用 ffmpeg 从视频中抽取音频（44.1kHz 立体声 WAV）")
    parser.add_argument("video", nargs="?", default="./1.mp4",
                        help="输入视频路径（默认 <项目根>/1.mp4）")
    parser.add_argument("--output-dir", default=None,
                        help="音频输出目录（默认 <项目根>/temp）")
    parser.add_argument("--sample-rate", type=int, default=44100, help="输出采样率（默认 44100）")
    args = parser.parse_args()

    # 解析视频路径（相对路径按项目根解析，保持历史行为）
    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(project_root, video_path)

    output_dir = args.output_dir or os.path.join(project_root, "temp")
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "output_audio.wav")

    try:
        # Process the video file
        extract_audio_from_video(video_path, audio_path, sample_rate=args.sample_rate)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()