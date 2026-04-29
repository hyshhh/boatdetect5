"""
Pipeline 模块 — 船弦号识别视频处理流水线

核心组件：
  - ShipDetector: YOLO 船只检测与跟踪
  - AgentResult: 识别结果结构
  - TrackManager: 跟踪状态管理（track ID → 弦号绑定）
  - ShipPipeline: 主流水线编排（级联/并发双模式）
  - FPSMeter: 推理速度 FPS 统计
  - InputSource: 视频/相机/视频流输入
  - DemoRenderer: 可视化渲染
  - ScreenshotSaver: 截图保存
  - HullNumberLocator: PaddleOCR 弦号定位
"""

from pipeline.detector import ShipDetector  # noqa: F401
from agent import AgentResult  # noqa: F401
from pipeline.tracker import TrackManager  # noqa: F401
from pipeline.pipeline import ShipPipeline  # noqa: F401
from pipeline.fps import FPSMeter, LatencyMeter  # noqa: F401
from pipeline.video_input import InputSource  # noqa: F401
from pipeline.demo import DemoRenderer  # noqa: F401
from pipeline.output import ScreenshotSaver  # noqa: F401
from pipeline.hull_number_locator import HullNumberLocator  # noqa: F401

__all__ = [
    "ShipDetector",
    "AgentResult",
    "TrackManager",
    "ShipPipeline",
    "FPSMeter",
    "LatencyMeter",
    "InputSource",
    "DemoRenderer",
    "ScreenshotSaver",
    "HullNumberLocator",
]
