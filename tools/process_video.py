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
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_root = os.path.dirname(script_dir)
    
    # Define output path (relative to project root)
    audio_file = "./temp/output_audio.wav"
    audio_path = os.path.join(project_root, audio_file)
    
    # Check if video file path is provided as command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        # If relative path, make it absolute
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.getcwd(), video_path)
    else:
        # Default to ./1.mp4 if no argument provided (relative to project root)
        video_file = "./1.mp4"
        video_path = os.path.join(project_root, video_file)
    
    try:
        # Process the video file
        extract_audio_from_video(video_path, audio_path)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()