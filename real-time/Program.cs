// 🐾 清雨出品｜.NET 10 + NAudio 2.2.1 防死循环完美版 ✅
// ✅ 修复所有编译错误，特别是WasapiLoopbackCapture.Recording问题
// ✅ 使用状态变量替代已弃用的Recording属性
// ✅ 完整的空引用检查
// ✅ 播放TTS时暂停捕获，彻底解决死循环

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace RealtimeTranslator
{
    internal class Program
    {
        private static readonly CancellationTokenSource _cts = new();
        private static readonly HttpClient _httpClient = new();

        public static async Task Main(string[] args)
        {
            Console.WriteLine("🐾 清雨的智能语音助手启动中... (Ctrl+C 退出)");
            Console.WriteLine("🎯 防死循环版：播放时暂停捕获");
            Console.WriteLine("🔴 初始状态: 等待检测语音");
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                _cts.Cancel();
                Console.WriteLine("\n👋 正在安全退出...");
            };

            try
            {
                var service = new AudioStreamingService(_httpClient, _cts.Token);
                await service.RunAsync().WaitAsync(_cts.Token);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("🛑 程序已取消");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"💥 严重错误: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
            }
            finally
            {
                _httpClient.Dispose();
                _cts.Dispose();
                Console.WriteLine("✅ 资源已清理");
            }
        }
    }

    internal class AudioStreamingService : IDisposable
    {
        private const float SpeechStartThreshold = 0.025f;    // 语音开始阈值
        private const float SilenceThreshold = 0.015f;        // 静音阈值
        private const int SilenceDurationSeconds = 1;         // 静音持续1秒截断
        private const int TargetSampleRate = 16000;           // 录制采样率
        private const int MinRecordingDurationMs = 200;       // 最小录制时长

        // 调试设置
        private bool _debugMode = true;
        private DateTime _lastDebugTime = DateTime.MinValue;
        private DateTime _recordingStartTime;

        private readonly HttpClient _httpClient;
        private readonly CancellationToken _cancellationToken;
        private readonly string _serverUrl = "http://localhost:5000";

        // 播放
        private WaveOutEvent? _waveOut;

        // 录制
        private WasapiLoopbackCapture? _capture;
        private WaveFileWriter? _currentWriter;
        private string? _currentFilePath;
        private RecorderState _state = RecorderState.WaitingForSpeech;
        private int _silenceFrameCount;
        private int _maxSilenceFrames;
        private long _totalFramesRecorded;
        private readonly object _stateLock = new();

        // 音频格式信息
        private WaveFormat? _sourceFormat;

        // 🔥 防死循环关键：播放状态标志 + 捕获状态标志
        private bool _isPlayingTts = false;
        private bool _isCapturing = false;  // 🔥 新增：跟踪捕获状态
        private readonly object _playbackLock = new();

        private enum RecorderState
        {
            WaitingForSpeech,
            Recording,
            Processing
        }

        public AudioStreamingService(HttpClient httpClient, CancellationToken cancellationToken)
        {
            _httpClient = httpClient;
            _cancellationToken = cancellationToken;
        }

        public async Task RunAsync()
        {
            try
            {
                await StartLoopbackCaptureAsync();
            }
            finally
            {
                Dispose();
            }
        }

        private async Task StartLoopbackCaptureAsync()
        {
            var enumerator = new MMDeviceEnumerator();
            var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            Console.WriteLine($"🔊 设备: {device.FriendlyName}");

            _capture = new WasapiLoopbackCapture(device);
            _sourceFormat = _capture.WaveFormat;
            
            Console.WriteLine($"🔍 源音频格式详情:");
            Console.WriteLine($"   采样率: {_sourceFormat.SampleRate}Hz");
            Console.WriteLine($"   声道数: {_sourceFormat.Channels}");
            Console.WriteLine($"   位深度: {_sourceFormat.BitsPerSample}位");

            _capture.DataAvailable += OnDataAvailable;
            _capture.RecordingStopped += (s, e) => 
            {
                if (e.Exception != null)
                {
                    Console.WriteLine($"❌ 录制停止异常: {e.Exception.Message}");
                }
                // 更新捕获状态
                lock (_playbackLock)
                {
                    _isCapturing = false;
                }
            };

            _maxSilenceFrames = SilenceDurationSeconds * TargetSampleRate;
            Directory.CreateDirectory("recordings");
            Directory.CreateDirectory("downloaded_audio");
            
            Console.WriteLine("🔴 等待检测语音...");
            Console.WriteLine($"🎯 语音阈值: {SpeechStartThreshold:F4} | 静音阈值: {SilenceThreshold:F4}");
            Console.WriteLine($"⏱️  静音截断时长: {SilenceDurationSeconds}秒 | 最小录制时长: {MinRecordingDurationMs}毫秒");
            
            // 🔥 启动捕获并更新状态
            _capture.StartRecording();
            lock (_playbackLock)
            {
                _isCapturing = true;
            }
            Console.WriteLine("🎧 音频捕获已启动");

            await Task.Delay(-1, _cancellationToken);
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            if (e.BytesRecorded == 0 || _cancellationToken.IsCancellationRequested) return;

            // 🔥 关键检查：如果正在播放TTS，跳过处理
            lock (_playbackLock)
            {
                if (_isPlayingTts)
                {
                    if (_debugMode)
                    {
                        Console.WriteLine("🔇 暂停捕获：正在播放TTS结果");
                    }
                    return;
                }
            }

            try
            {
                ProcessAudioBuffer(e.Buffer, e.BytesRecorded);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  处理异常: {ex.Message}");
                if (_debugMode)
                {
                    Console.WriteLine($"   详细信息: {ex.StackTrace}");
                }
            }
        }

        private byte[]? ResampleAudioBuffer(byte[] buffer, int bytesRecorded)
        {
            if (_sourceFormat == null) return null;

            try
            {
                var sourceProvider = new RawSourceWaveStream(buffer, 0, bytesRecorded, _sourceFormat);
                
                IWaveProvider monoWaveProvider;
                if (_sourceFormat.Channels == 1)
                {
                    monoWaveProvider = sourceProvider;
                }
                else if (_sourceFormat.Channels == 2)
                {
                    var toSample = new WaveToSampleProvider(sourceProvider);
                    var toMono = new StereoToMonoSampleProvider(toSample) { LeftVolume = 0.5f, RightVolume = 0.5f };
                    monoWaveProvider = new SampleToWaveProvider(toMono);
                }
                else
                {
                    Console.WriteLine($"⚠️  不支持的声道数: {_sourceFormat.Channels}");
                    return null;
                }

                var targetWaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(TargetSampleRate, 1);
                
                var resampler = new MediaFoundationResampler(monoWaveProvider, targetWaveFormat)
                {
                    ResamplerQuality = 60
                };

                int sourceDurationMs = (int)((bytesRecorded * 1000L) / _sourceFormat.AverageBytesPerSecond);
                int targetBufferSize = (int)(TargetSampleRate * (sourceDurationMs / 1000.0f) * 4);
                
                var readBuffer = new byte[Math.Max(4096, targetBufferSize)];
                var result = new List<byte>();
                int totalRead = 0;

                while (totalRead < readBuffer.Length)
                {
                    int read = resampler.Read(readBuffer, 0, readBuffer.Length);
                    if (read <= 0) break;
                    
                    for (int i = 0; i < read; i += 4)
                    {
                        if (i + 3 >= read) break;
                        
                        float sample = BitConverter.ToSingle(readBuffer, i);
                        sample = Math.Clamp(sample, -0.8f, 0.8f);
                        byte[] sampleBytes = BitConverter.GetBytes(sample);
                        result.AddRange(sampleBytes);
                    }
                    
                    totalRead += read;
                }

                if (result.Count == 0) return null;
                return result.ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"🔧 重采样失败: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
                return null;
            }
        }

        private float CalculateRmsForFloat32(ReadOnlySpan<float> samples)
        {
            if (samples.Length == 0) return 0;
            
            double sum = 0;
            int count = 0;
            
            foreach (var s in samples)
            {
                var clamped = Math.Clamp(s, -1.0f, 1.0f);
                var squared = clamped * clamped;
                
                if (!float.IsFinite(squared))
                {
                    continue;
                }
                
                sum += squared;
                count++;
            }
            
            if (count == 0) return 0;
            return (float)Math.Sqrt(sum / count);
        }

        private void StartNewSegment()
        {
            var ts = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            _currentFilePath = Path.Combine("recordings", $"seg_{ts}.wav");
            var fmt = WaveFormat.CreateIeeeFloatWaveFormat(TargetSampleRate, 1);
            
            try
            {
                _currentWriter = new WaveFileWriter(_currentFilePath, fmt);
                _state = RecorderState.Recording;
                _silenceFrameCount = 0;
                _totalFramesRecorded = 0;
                
                if (_debugMode)
                {
                    Console.WriteLine($"🎯 新文件: {Path.GetFileName(_currentFilePath)}");
                    Console.WriteLine($"📊 目标格式: {fmt.SampleRate}Hz, {fmt.Channels}声道");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 创建文件失败: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
                _state = RecorderState.WaitingForSpeech;
            }
        }

        private void WriteAudioBytes(byte[] bytes)
        {
            if (_currentWriter != null && bytes.Length > 0)
            {
                try
                {
                    _currentWriter.Write(bytes, 0, bytes.Length);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"📝 写入失败: {ex.Message}");
                    Console.WriteLine($"   详细信息: {ex.StackTrace}");
                    _state = RecorderState.WaitingForSpeech;
                }
            }
        }

        private async Task StopCurrentSegmentAndProcessAsync()
        {
            string? path;
            WaveFileWriter? writer;

            lock (_stateLock)
            {
                if (_state != RecorderState.Processing) return;
                path = _currentFilePath;
                writer = _currentWriter;
                _currentFilePath = null;
                _currentWriter = null;
            }

            writer?.Dispose();
            
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                Console.WriteLine("⚠️  无有效音频文件");
                _state = RecorderState.WaitingForSpeech;
                return;
            }

            var fileInfo = new FileInfo(path);
            Console.WriteLine($"📦 准备发送: {Path.GetFileName(path)}");
            Console.WriteLine($"📊 大小: {fileInfo.Length / 1024:F1}KB | 采样率: {TargetSampleRate}Hz");

            try
            {
                await SendToServerAndPlayAsync(path);
                Console.WriteLine("✅ 处理完成");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 处理失败: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
            }
            finally
            {
                try { File.Delete(path!); } catch { }
                lock (_stateLock)
                {
                    _state = RecorderState.WaitingForSpeech;
                    Console.WriteLine("🔴 等待下一段语音...");
                }
            }
        }

        private async Task SendToServerAndPlayAsync(string audioPath)
        {
            Console.WriteLine($"🚀 发送到: {_serverUrl}/infer_wav");
            
            try
            {
                // 1. 读取音频文件
                byte[] audioBytes = await File.ReadAllBytesAsync(audioPath, _cancellationToken);
                string audioBase64 = Convert.ToBase64String(audioBytes);
                
                // 2. 构造请求
                var requestData = new 
                { 
                    audio_data = audioBase64,
                    filename = Path.GetFileName(audioPath),
                    target_language = "zh",
                    asr_language = "ja"      // 源语言改为日语
                };
                
                var json = JsonSerializer.Serialize(requestData);
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");

                // 3. 发送请求
                using var res = await _httpClient.PostAsync($"{_serverUrl}/infer_wav", content, _cancellationToken);
                
                if (!res.IsSuccessStatusCode)
                {
                    string errorContent = await res.Content.ReadAsStringAsync(_cancellationToken);
                    Console.WriteLine($"❌ 服务器错误响应: {errorContent}");
                    throw new Exception($"服务器错误: {res.StatusCode}");
                }

                // 4. 读取WAV文件内容
                string responseContent = await res.Content.ReadAsStringAsync(_cancellationToken);
                
                // 5. 解析JSON响应
                using var jsonDoc = JsonDocument.Parse(responseContent);
                var root = jsonDoc.RootElement;
                
                if (root.TryGetProperty("wav_data", out var wavDataElem))
                {
                    // 6. 解码WAV文件
                    string wavBase64 = wavDataElem.GetString() ?? "";
                    byte[] wavBytes = Convert.FromBase64String(wavBase64);
                    
                    // 7. 保存到临时文件
                    string tempWavPath = Path.Combine("downloaded_audio", $"result_{DateTime.Now:yyyyMMdd_HHmmss_fff}.wav");
                    await File.WriteAllBytesAsync(tempWavPath, wavBytes, _cancellationToken);
                    
                    Console.WriteLine($"✅ WAV文件已下载: {tempWavPath} ({wavBytes.Length / 1024:F1}KB)");
                    
                    // 8. 播放WAV文件（带防死循环）
                    await PlayWavFileWithPauseCaptureAsync(tempWavPath);
                    
                    Console.WriteLine("🔊 播放完成，文件已保存到 downloaded_audio 文件夹");
                }
                else if (root.TryGetProperty("error", out var errorElem))
                {
                    throw new Exception($"服务器错误: {errorElem.GetString()}");
                }
                else
                {
                    throw new Exception("无效的服务器响应");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 处理失败: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
                throw;
            }
        }

        private async Task PlayWavFileWithPauseCaptureAsync(string wavFilePath)
        {
            try
            {
                Console.WriteLine($"🎵 正在播放: {Path.GetFileName(wavFilePath)}");
                
                // 🔥 关键1：暂停音频捕获
                lock (_playbackLock)
                {
                    _isPlayingTts = true;
                    
                    // 🔥 修复：使用_isCapturing状态变量替代Recording属性
                    if (_isCapturing && _capture != null)
                    {
                        Console.WriteLine("⏸️  暂停音频捕获...");
                        _capture.StopRecording();
                        _isCapturing = false;  // 更新状态
                    }
                }
                
                // 2. 释放之前的播放器
                if (_waveOut != null)
                {
                    _waveOut.Stop();
                    _waveOut.Dispose();
                }
                _waveOut = null;
                
                // 3. 创建新的播放器
                using var audioFile = new AudioFileReader(wavFilePath);
                _waveOut = new WaveOutEvent { DesiredLatency = 100 };
                _waveOut.Init(audioFile);
                
                // 4. 开始播放
                _waveOut.Play();
                
                // 5. 等待播放完成
                while (_waveOut.PlaybackState == PlaybackState.Playing && !_cancellationToken.IsCancellationRequested)
                {
                    await Task.Delay(100, _cancellationToken);
                }
                
                Console.WriteLine("✅ 播放完成");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 播放失败: {ex.Message}");
                Console.WriteLine($"   详细信息: {ex.StackTrace}");
            }
            finally
            {
                // 🔥 关键2：恢复音频捕获
                lock (_playbackLock)
                {
                    _isPlayingTts = false;
                    
                    // 🔥 修复：检查捕获对象是否存在且未在捕获
                    if (_capture != null && !_isCapturing)
                    {
                        Console.WriteLine("▶️  恢复音频捕获...");
                        try
                        {
                            _capture.StartRecording();
                            _isCapturing = true;  // 更新状态
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"⚠️  恢复捕获失败: {ex.Message}");
                            // 尝试重新初始化捕获
                            try
                            {
                                var enumerator = new MMDeviceEnumerator();
                                var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
                                _capture = new WasapiLoopbackCapture(device);
                                _capture.DataAvailable += OnDataAvailable;
                                _capture.RecordingStopped += (s, e) => 
                                {
                                    if (e.Exception != null)
                                    {
                                        Console.WriteLine($"❌ 录制停止异常: {e.Exception.Message}");
                                    }
                                    lock (_playbackLock)
                                    {
                                        _isCapturing = false;
                                    }
                                };
                                _capture.StartRecording();
                                _isCapturing = true;
                                Console.WriteLine("🔄 音频捕获已重新初始化");
                            }
                            catch (Exception initEx)
                            {
                                Console.WriteLine($"❌ 重新初始化捕获失败: {initEx.Message}");
                            }
                        }
                    }
                }
            }
        }

        private void ProcessAudioBuffer(byte[] buffer, int bytesRecorded)
        {
            if (_sourceFormat == null) return;

            try
            {
                var audioBytes = ResampleAudioBuffer(buffer, bytesRecorded);
                if (audioBytes == null || audioBytes.Length == 0) return;

                // 🔑 关键修复：使用安全的方式处理内存
                float[] floatArray = new float[audioBytes.Length / 4];
                Buffer.BlockCopy(audioBytes, 0, floatArray, 0, audioBytes.Length);
                
                var floatSpan = floatArray.AsSpan();
                var rms = CalculateRmsForFloat32(floatSpan);

                if (_debugMode && (DateTime.Now - _lastDebugTime).TotalMilliseconds > 500)
                {
                    Console.WriteLine($"📊 RMS: {rms:F6} | 状态: {_state} | 数据长度: {audioBytes.Length}字节");
                    _lastDebugTime = DateTime.Now;
                }

                lock (_stateLock)
                {
                    switch (_state)
                    {
                        case RecorderState.WaitingForSpeech:
                            if (rms > SpeechStartThreshold)
                            {
                                Console.WriteLine($"\n🎤 检测到语音! RMS: {rms:F6} > {SpeechStartThreshold:F4}");
                                Console.WriteLine("🟢 开始录制...");
                                StartNewSegment();
                                WriteAudioBytes(audioBytes);
                                _recordingStartTime = DateTime.Now;
                                _totalFramesRecorded = floatSpan.Length;
                            }
                            break;

                        case RecorderState.Recording:
                            _totalFramesRecorded += floatSpan.Length;
                            WriteAudioBytes(audioBytes);

                            if (rms <= SilenceThreshold)
                            {
                                _silenceFrameCount += floatSpan.Length;
                                var silenceSeconds = (float)_silenceFrameCount / TargetSampleRate;
                                
                                if (_silenceFrameCount > TargetSampleRate * 0.5 && _debugMode)
                                {
                                    Console.WriteLine($"🤫 静音: {silenceSeconds:F1}s/{SilenceDurationSeconds}s (RMS: {rms:F6})");
                                }
                                
                                var recordingDurationMs = (DateTime.Now - _recordingStartTime).TotalMilliseconds;
                                if (_silenceFrameCount >= _maxSilenceFrames && recordingDurationMs >= MinRecordingDurationMs)
                                {
                                    var durationSeconds = recordingDurationMs / 1000.0;
                                    Console.WriteLine($"🛑 静音超时，结束录制");
                                    Console.WriteLine($"📏 时长: {durationSeconds:F2}s | 帧数: {_totalFramesRecorded}");
                                    _state = RecorderState.Processing;
                                    _ = StopCurrentSegmentAndProcessAsync();
                                }
                            }
                            else
                            {
                                if (_silenceFrameCount > 0 && _debugMode)
                                {
                                    Console.WriteLine($"🔊 语音恢复! RMS: {rms:F6}");
                                }
                                _silenceFrameCount = 0;
                            }
                            break;
                        case RecorderState.Processing:
                            break;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"🔄 音频处理失败: {ex.Message}");
                if (_debugMode)
                {
                    Console.WriteLine($"   详细信息: {ex.StackTrace}");
                }
            }
        }

        public void Dispose()
        {
            try
            {
                lock (_playbackLock)
                {
                    if (_isCapturing && _capture != null)
                    {
                        _capture.StopRecording();
                        _isCapturing = false;
                    }
                }
                
                _capture?.Dispose();
                _currentWriter?.Dispose();
                
                if (_waveOut != null)
                {
                    _waveOut.Stop();
                    _waveOut.Dispose();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"🧹 清理异常: {ex.Message}");
            }
        }
    }
}