import pathlib
import os
import typing
import bentoml
import warnings
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGE_CODE = "en"

with bentoml.importing():
    import whisperx
    import torch

@bentoml.service(
    traffic={"timeout": 300},  # 增加超时时间处理长音频
    resources={"gpu": 1, "memory": "12Gi"},  # 增加内存
    image=bentoml.images.PythonImage(python_version="3.11")
    .system_packages("ffmpeg")
    .system_packages("libsndfile1-dev")
    # 兼容性修复：使用稳定版本组合
    .python_packages("torch==2.1.2")
    .python_packages("torchaudio==2.1.2")
    .python_packages("transformers==4.35.2")
    .python_packages("faster-whisper==0.10.1")
    .python_packages("whisperx==3.1.1")
    .python_packages("pyannote.audio==3.1.1")
    .python_packages("pyannote.core==5.0.0")
    .python_packages("librosa==0.10.1")
    .python_packages("soundfile==0.12.1")
    .python_packages("bentoml>=1.2.0"),
)
class WhisperX:
    """
    Enhanced WhisperX service based on the original implementation
    Source: https://github.com/m-bain/whisperX
    
    Improvements:
    - Fixed CUDA/PyTorch/pyannote compatibility issues
    - Enhanced error handling and recovery
    - Better segments processing and formatting
    - Intelligent device detection
    - Graceful fallback strategies
    """
    
    def __init__(self):
        """初始化 WhisperX 服务"""
        # 兼容性设置
        self._setup_compatibility()
        
        # 智能设备检测
        self._detect_device()
        
        # 模型加载
        self._load_models()
        
        logger.info("🎉 WhisperX 服务初始化完成")
    
    def _setup_compatibility(self):
        """设置兼容性环境"""
        # 忽略版本兼容性警告
        warnings.filterwarnings("ignore", message=".*pyannote.*")
        warnings.filterwarnings("ignore", message=".*torch.*")
        warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
        
        # PyTorch 兼容性设置
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.empty_cache()
        
        logger.info("✅ 兼容性环境设置完成")
    
    def _detect_device(self):
        """智能设备检测和配置"""
        if torch.cuda.is_available():
            self.device = "cuda"
            compute_type = "int8"  # 更稳定的精度
            self.batch_size = 8    # 保守的批大小
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 使用 CUDA: {gpu_name}")
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            compute_type = "int8"
            self.batch_size = 4
            logger.info("🍎 使用 Apple MPS")
            
        else:
            self.device = "cpu"
            compute_type = "int8"
            self.batch_size = 4  # 原版是16，但为了兼容性降低
            logger.info("💻 使用 CPU")
        
        self.compute_type = compute_type
        logger.info(f"📊 设备配置: {self.device}, 精度: {compute_type}, 批大小: {self.batch_size}")
    
    def _load_models(self):
        """加载模型（保持原版逻辑，增加容错）"""
        try:
            # 主模型加载 - 与原版逻辑一致
            logger.info("🔄 加载 Whisper 模型...")
            self.model = whisperx.load_model(
                "large-v2", 
                self.device, 
                compute_type=self.compute_type, 
                language=LANGUAGE_CODE
            )
            logger.info("✅ Whisper 模型加载成功")
            
        except Exception as e:
            logger.warning(f"⚠️ large-v2 模型加载失败: {e}")
            logger.info("🔄 尝试加载 medium 模型...")
            try:
                self.model = whisperx.load_model(
                    "medium", 
                    self.device, 
                    compute_type=self.compute_type, 
                    language=LANGUAGE_CODE
                )
                logger.info("✅ medium 模型加载成功")
            except Exception as e2:
                logger.warning(f"⚠️ medium 模型也失败: {e2}")
                logger.info("🔄 使用 base 模型作为最后备选...")
                self.model = whisperx.load_model(
                    "base", 
                    self.device, 
                    compute_type=self.compute_type, 
                    language=LANGUAGE_CODE
                )
                logger.info("✅ base 模型加载成功")
        
        # 对齐模型加载 - 与原版逻辑一致，增加容错
        try:
            logger.info("🔄 加载对齐模型...")
            self.model_a, self.metadata = whisperx.load_align_model(
                language_code=LANGUAGE_CODE, 
                device=self.device
            )
            self.alignment_enabled = True
            logger.info("✅ 对齐模型加载成功")
            
        except Exception as e:
            logger.warning(f"⚠️ 对齐模型加载失败: {e}")
            logger.info("📝 继续使用基础转录功能")
            self.model_a = None
            self.metadata = None
            self.alignment_enabled = False
    
    def _process_segments(self, segments):
        """增强的 segments 处理"""
        if not segments:
            return []
        
        processed_segments = []
        for i, segment in enumerate(segments):
            # 基础信息（保持原版格式）
            processed_segment = {
                "id": i,
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
            }
            
            # 如果有词级对齐信息，添加到 segment 中
            if "words" in segment:
                processed_segment["words"] = segment["words"]
                
                # 计算额外统计信息
                words = segment["words"]
                if words:
                    processed_segment["word_count"] = len(words)
                    processed_segment["confidence"] = sum(
                        word.get("score", 0) for word in words if "score" in word
                    ) / len(words) if any("score" in word for word in words) else None
            
            # 计算段落时长
            duration = processed_segment["end"] - processed_segment["start"]
            processed_segment["duration"] = round(duration, 2)
            
            # 估算语速（每分钟单词数）
            word_count = len(processed_segment["text"].split())
            if duration > 0:
                wpm = (word_count / duration) * 60
                processed_segment["words_per_minute"] = round(wpm, 1)
            
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _generate_summary_stats(self, result, processing_time):
        """生成转录统计摘要"""
        segments = result.get("segments", [])
        
        if not segments:
            return {
                "total_segments": 0,
                "total_duration": 0,
                "total_words": 0,
                "processing_time": processing_time
            }
        
        # 计算总时长
        total_duration = max(seg.get("end", 0) for seg in segments) if segments else 0
        
        # 计算总单词数
        total_words = sum(len(seg.get("text", "").split()) for seg in segments)
        
        # 计算平均置信度（如果有的话）
        confidences = []
        for seg in segments:
            if "words" in seg:
                for word in seg["words"]:
                    if "score" in word:
                        confidences.append(word["score"])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # 计算处理速度比率
        speed_ratio = total_duration / processing_time if processing_time > 0 else 0
        
        return {
            "total_segments": len(segments),
            "total_duration": round(total_duration, 2),
            "total_words": total_words,
            "average_confidence": round(avg_confidence, 3) if avg_confidence else None,
            "processing_time": round(processing_time, 2),
            "speed_ratio": round(speed_ratio, 1),  # 例如: 10x 表示处理速度是实时播放的10倍
            "alignment_enabled": self.alignment_enabled
        }
    
    @bentoml.api
    def transcribe(self, audio_file: pathlib.Path) -> typing.Dict[str, typing.Any]:
        """
        转录音频文件 - 保持原版接口，增强功能
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            转录结果字典，包含 segments 和统计信息
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎵 开始转录: {audio_file}")
            
            # 加载音频 - 与原版一致
            audio = whisperx.load_audio(audio_file)
            
            # 转录 - 与原版一致
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            
            # 对齐处理 - 与原版逻辑一致，增加容错
            if self.alignment_enabled and self.model_a is not None:
                try:
                    result = whisperx.align(
                        result["segments"], 
                        self.model_a, 
                        self.metadata, 
                        audio, 
                        self.device, 
                        return_char_alignments=False
                    )
                    logger.info("✅ 词级对齐完成")
                except Exception as e:
                    logger.warning(f"⚠️ 对齐失败，使用基础转录: {e}")
                    # 继续使用原始结果
            
            # 处理时间
            processing_time = time.time() - start_time
            
            # 增强的 segments 处理
            if "segments" in result:
                result["segments"] = self._process_segments(result["segments"])
            
            # 添加统计信息（新增功能）
            result["stats"] = self._generate_summary_stats(result, processing_time)
            
            # 添加服务信息（新增功能）
            result["service_info"] = {
                "device": self.device,
                "compute_type": self.compute_type,
                "batch_size": self.batch_size,
                "alignment_enabled": self.alignment_enabled
            }
            
            logger.info(f"✅ 转录完成，耗时: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 转录失败: {e}")
            
            # 返回错误信息（保持接口一致性）
            return {
                "segments": [],
                "text": "",
                "language": LANGUAGE_CODE,
                "error": str(e),
                "stats": {
                    "processing_time": processing_time,
                    "status": "failed"
                }
            }
    
    @bentoml.api
    def health_check(self) -> typing.Dict[str, typing.Any]:
        """健康检查端点（新增）"""
        return {
            "status": "healthy",
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "alignment_enabled": self.alignment_enabled,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": (
                torch.backends.mps.is_available() 
                if hasattr(torch.backends, 'mps') else False
            )
        }
