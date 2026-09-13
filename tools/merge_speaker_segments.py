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

def merge_adjacent_segments(segments, max_gap_duration=0.3, max_duration=MAX_DURATION):
    """
    合并属于同一说话人的相邻片段

    Args:
        segments: 包含片段信息的列表，每个片段包含start、end、speaker字段
        max_gap_duration: 允许合并的最大间隔时间（秒），默认0.3秒
        max_duration: 合并后片段允许的最大时长（秒），超过则不合并

    Returns:
        合并后的片段列表

    注意（历史 bug）：合并会让当前段"吞掉"下一段。若合并后超过 max_duration，
    必须**放弃本次合并并把当前段落盘、以新段开始**——早期版本在这里 `continue`
    导致当前段既没合并、也没写入结果，台词被静默丢弃。
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
                if duration <= max_duration:
                    # 合并片段：更新结束时间和持续时间
                    current_segment['end'] = next_segment['end']
                    current_segment['duration'] = round(duration, 2)
                    continue  # 仅在真正合并后跳过落盘
                # 合并会超长 -> 不合并，落到下面正常落盘并开新段（勿改成 continue！）

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

def merge_speaker_segments(json_file_path, output_file_path=None, max_gap_duration=0.3, min_duration=0.3, max_duration=MAX_DURATION):
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
    merged_segments = merge_adjacent_segments(segments, max_gap_duration, max_duration)
    
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
    import argparse

    parser = argparse.ArgumentParser(description="合并同一说话人的相邻片段并过滤过短片段")
    parser.add_argument("--input", type=str, default=os.path.join(project_root, "results", "speaker_diarization.json"),
                        help="输入 JSON（默认 results/speaker_diarization.json）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON（默认覆盖 --input）")
    parser.add_argument("--max-gap", type=float, default=0.3, help="允许合并的最大间隔秒数（默认 0.3）")
    parser.add_argument("--min-duration", type=float, default=0.3, help="短于该时长的片段被丢弃（默认 0.3）")
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION, help=f"合并后允许的最大时长（默认 {MAX_DURATION}）")
    args = parser.parse_args()

    json_file = args.input
    output_file = args.output or json_file

    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} does not exist")
        sys.exit(1)

    print("Starting to merge adjacent segments from the same speaker...")
    print(f"input={json_file} output={output_file} max_gap={args.max_gap} min_duration={args.min_duration} max_duration={args.max_duration}")
    print("=" * 50)

    # 合并片段
    merged_segments = merge_speaker_segments(
        json_file,
        output_file_path=output_file,
        max_gap_duration=args.max_gap,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

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