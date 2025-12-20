#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json

MAX_DURATION = 10 # 最大合并后片段允许持续时长（秒）

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（tools的父目录）
project_root = os.path.dirname(script_dir)

# 添加项目根目录到Python路径
sys.path.append(project_root)

def merge_adjacent_segments(segments, max_gap_duration=0.3):
    """
    合并属于同一说话人的相邻片段
    
    Args:
        segments: 包含片段信息的列表，每个片段包含start、end、speaker字段
        max_gap_duration: 允许合并的最大间隔时间（秒），默认0.4秒
    
    Returns:
        合并后的片段列表
    """
    if not segments:
        return segments
    
    # 按开始时间排序
    segments.sort(key=lambda x: x['start'])
    
    merged_segments = []
    current_segment = segments[0].copy()
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        # 检查是否为同一说话人
        if current_segment['speaker'] == next_segment['speaker']:
            # 检查间隔时间是否小于等于最大允许间隔
            gap = next_segment['start'] - current_segment['end']
            if gap <= max_gap_duration:
                duration = next_segment['end'] - current_segment['start']
                if duration <= MAX_DURATION:
                    # 合并片段：更新结束时间和持续时间
                    current_segment['end'] = next_segment['end']
                    current_segment['duration'] = round(duration, 2)
                continue
        
        # 不能合并，保存当前片段，开始新的片段
        merged_segments.append(current_segment)
        current_segment = next_segment.copy()
    
    # 添加最后一个片段
    merged_segments.append(current_segment)
    
    return merged_segments

def filter_short_segments(segments, min_duration=0.3):
    """
    过滤掉持续时间小于指定阈值的片段
    
    Args:
        segments: 包含片段信息的列表
        min_duration: 最小持续时间（秒），默认0.3秒
    
    Returns:
        过滤后的片段列表
    """
    filtered_segments = []
    removed_count = 0
    
    for segment in segments:
        if segment['duration'] >= min_duration:
            filtered_segments.append(segment)
        else:
            removed_count += 1
            print(f"  Discard too short segment: {segment['start']:.2f}-{segment['end']:.2f}s ({segment['duration']:.2f}s) - {segment['speaker']}")
    
    print(f"Discarded {removed_count} too short segments")
    return filtered_segments

def merge_speaker_segments(json_file_path, output_file_path=None, max_gap_duration=0.3, min_duration=0.3):
    """
    读取JSON文件，合并相邻的同说话人片段，过滤过短片段，并保存结果
    
    Args:
        json_file_path: 输入的JSON文件路径
        output_file_path: 输出的JSON文件路径，默认为覆盖原文件
        max_gap_duration: 允许合并的最大间隔时间（秒），默认1秒
        min_duration: 最小持续时间（秒），默认0.3秒
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    print(f"Original segments count: {len(segments)}")
    
    # 显示合并前的一些统计信息
    speaker_stats = {}
    for segment in segments:
        speaker = segment['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = 0
        speaker_stats[speaker] += 1
    
    print("Segments count by speaker before merging:")
    for speaker, count in speaker_stats.items():
        print(f"  {speaker}: {count}")
    
    # 合并相邻片段
    merged_segments = merge_adjacent_segments(segments, max_gap_duration)
    
    print(f"Segments count after first merge: {len(merged_segments)}")
    
    # 过滤过短的片段
    filtered_segments = filter_short_segments(merged_segments, min_duration)
    
    print(f"Segments count after filtering: {len(filtered_segments)}")
    
    # 显示合并后的统计信息
    speaker_stats_merged = {}
    for segment in filtered_segments:
        speaker = segment['speaker']
        if speaker not in speaker_stats_merged:
            speaker_stats_merged[speaker] = 0
        speaker_stats_merged[speaker] += 1
    
    print("Segments count by speaker after filtering:")
    for speaker, count in speaker_stats_merged.items():
        print(f"  {speaker}: {count}")
    
    # 计算合并了多少片段
    merged_count = len(segments) - len(merged_segments)
    print(f"Merged {merged_count} segments")
    
    # 计算过滤了多少片段
    filtered_count = len(merged_segments) - len(filtered_segments)
    print(f"Filtered out {filtered_count} too short segments")
    
    # 确定输出文件路径
    if output_file_path is None:
        output_file_path = json_file_path
    
    # 保存过滤后的结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_segments, f, ensure_ascii=False, indent=2)
    
    print(f"Final result saved to: {output_file_path}")
    
    return filtered_segments

def main():
    # JSON文件路径（使用相对于项目根目录的路径）
    json_file = os.path.join(project_root, "results", "speaker_diarization.json")
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} does not exist")
        return
    
    print("Starting to merge adjacent segments from the same speaker...")
    print("=" * 50)
    
    # 合并片段
    merged_segments = merge_speaker_segments(json_file, max_gap_duration=0.3, min_duration=0.3)
    
    print("=" * 50)
    print("Merge completed!")
    
    # 显示一些合并后的片段示例
    print("\nFirst 10 segments after merging:")
    for i, segment in enumerate(merged_segments[:10]):
        print(f"  [{i+1}] {segment['start']:.2f}-{segment['end']:.2f}s ({segment['duration']:.2f}s) - {segment['speaker']}")
    
    if len(merged_segments) > 10:
        print(f"  ... and {len(merged_segments) - 10} more segments")

if __name__ == "__main__":
    main()