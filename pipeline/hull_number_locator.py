"""
HullNumberLocator — 基于 PaddleOCR 的弦号定位模块

在 YOLO crop 中定位弦号文字区域，返回原帧坐标系下的虚线框位置。
支持坐标转换：crop 坐标 → 原帧坐标。

版本兼容：PaddleOCR 2.5+ / 2.6+ / 2.7+（PP-OCRv4）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """一个检测到的文字区域。"""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 原帧坐标
    text: str                         # 识别到的文字内容
    confidence: float                 # 置信度


class HullNumberLocator:
    """
    弦号定位器 — 使用 PaddleOCR 在 crop 中定位弦号文字区域。

    坐标转换流程：
      1. PaddleOCR 输出 crop 内的四点坐标
      2. 转换为 crop 内的矩形 bbox
      3. 通过 inverse_crop_info 映射回原帧坐标

    inverse_crop_info 格式：
      {
          "crop_origin": (cx1, cy1),  # crop 在原帧中的左上角坐标
          "scale_x": float,           # resize 缩放比 X（原尺寸/crop 尺寸）
          "scale_y": float,           # resize 缩放比 Y
      }
    """

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        rec_batch_num: int = 1,
    ):
        """
        Args:
            use_gpu: 是否使用 GPU（需要 paddlepaddle-gpu）。
            lang: OCR 语言，"en" 适合英文/数字弦号。
            det_db_thresh: 文字检测阈值（越低越敏感）。
            det_db_box_thresh: 文字检测框阈值。
            rec_batch_num: 识别批处理大小。
        """
        self._use_gpu = use_gpu
        self._lang = lang
        self._det_db_thresh = det_db_thresh
        self._det_db_box_thresh = det_db_box_thresh
        self._rec_batch_num = rec_batch_num
        self._ocr = None
        self._init_error: str | None = None

        self._lazy_init()

    def _lazy_init(self) -> None:
        """延迟初始化 PaddleOCR（首次调用时加载模型）。"""
        if self._ocr is not None:
            return

        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                det_db_thresh=self._det_db_thresh,
                det_db_box_thresh=self._det_db_box_thresh,
                rec_batch_num=self._rec_batch_num,
                show_log=False,
                # 版本兼容：PaddleOCR 2.5+ 使用 PP-OCRv4 默认模型
                # 不指定 model_dir，使用内置默认模型
            )
            logger.info(
                "PaddleOCR 初始化成功 (gpu=%s, lang=%s)",
                self._use_gpu, self._lang,
            )
        except ImportError:
            self._init_error = (
                "PaddleOCR 未安装。请安装: "
                "pip install paddlepaddle-gpu==2.5.2 paddleocr==2.7.0.3"
            )
            logger.warning(self._init_error)
        except Exception as e:
            self._init_error = f"PaddleOCR 初始化失败: {e}"
            logger.warning(self._init_error)

    @property
    def available(self) -> bool:
        """PaddleOCR 是否可用。"""
        return self._ocr is not None

    @property
    def init_error(self) -> str | None:
        """初始化错误信息。"""
        return self._init_error

    def locate(
        self,
        crop: np.ndarray,
        inverse_crop_info: dict | None = None,
    ) -> list[TextRegion]:
        """
        在 crop 图像中定位文字区域。

        Args:
            crop: YOLO 裁剪的船只图像 (BGR)。
            inverse_crop_info: 坐标逆变换信息，用于将 crop 坐标映射回原帧坐标。
                格式见类文档。None 则返回 crop 内坐标。

        Returns:
            检测到的文字区域列表（按置信度降序）。
        """
        if not self.available:
            return []

        if crop is None or crop.size == 0:
            return []

        try:
            results = self._ocr.ocr(crop, cls=True)
        except Exception as e:
            logger.debug("PaddleOCR 推理异常: %s", e)
            return []

        if not results or not results[0]:
            return []

        regions: list[TextRegion] = []

        for line in results[0]:
            if not line or len(line) < 2:
                continue

            # line[0] = 四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # line[1] = (text, confidence)
            points = line[0]
            text_info = line[1]

            if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                continue

            text = str(text_info[0]).strip()
            confidence = float(text_info[1])

            if not text:
                continue

            # 将四点坐标转为矩形 bbox（crop 内坐标）
            pts = np.array(points, dtype=np.float32)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))

            # 坐标转换到原帧坐标系
            if inverse_crop_info:
                x_min, y_min, x_max, y_max = self._transform_to_frame(
                    x_min, y_min, x_max, y_max, inverse_crop_info,
                )

            regions.append(TextRegion(
                bbox=(x_min, y_min, x_max, y_max),
                text=text,
                confidence=confidence,
            ))

        # 按置信度降序排列
        regions.sort(key=lambda r: r.confidence, reverse=True)

        return regions

    @staticmethod
    def _transform_to_frame(
        x1: int, y1: int, x2: int, y2: int,
        info: dict,
    ) -> tuple[int, int, int, int]:
        """
        将 crop 内坐标转换为原帧坐标。

        坐标转换公式（考虑了 crop 时的 padding 和 resize 缩放）：
          frame_x = crop_x * scale_x + crop_origin_x
          frame_y = crop_y * scale_y + crop_origin_y

        Args:
            x1, y1, x2, y2: crop 内的矩形坐标。
            info: inverse_crop_info 字典。

        Returns:
            原帧坐标 (fx1, fy1, fx2, fy2)。
        """
        cx, cy = info.get("crop_origin", (0, 0))
        sx = info.get("scale_x", 1.0)
        sy = info.get("scale_y", 1.0)

        fx1 = int(x1 * sx + cx)
        fy1 = int(y1 * sy + cy)
        fx2 = int(x2 * sx + cx)
        fy2 = int(y2 * sy + cy)

        return fx1, fy1, fx2, fy2


def build_inverse_crop_info(
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
    pad: int = 20,
    target_min: int = 256,
    target_max: int = 512,
) -> dict:
    """
    根据 YOLO 检测框和 crop 参数，构建坐标逆变换信息。

    该函数复刻 detector.py 中 crop 的生成逻辑，计算从 crop 坐标
    映射回原帧坐标所需的参数。

    Args:
        x1, y1, x2, y2: YOLO 检测框在原帧中的坐标。
        frame_w, frame_h: 原帧尺寸。
        pad: crop 时的 padding 像素数（与 detector.py 一致）。
        target_min: crop resize 的最小目标尺寸（与 detector.py 一致）。
        target_max: crop resize 的最大目标尺寸（与 detector.py 一致）。

    Returns:
        inverse_crop_info 字典。
    """
    # 1. 计算带 padding 的 crop 区域在原帧中的位置
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(frame_w, x2 + pad)
    cy2 = min(frame_h, y2 + pad)

    # 2. 计算原始 crop 尺寸
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    if crop_w <= 0 or crop_h <= 0:
        return {"crop_origin": (cx1, cy1), "scale_x": 1.0, "scale_y": 1.0}

    # 3. 计算 resize 后的尺寸（与 detector.py 逻辑一致）
    max_dim = max(crop_w, crop_h)
    if max_dim < target_min:
        scale = target_min / max_dim
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
    elif max_dim > target_max:
        scale = target_max / max_dim
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
    else:
        new_w = crop_w
        new_h = crop_h

    # 4. 计算缩放比（原 crop 尺寸 / resize 后尺寸）
    scale_x = crop_w / max(new_w, 1)
    scale_y = crop_h / max(new_h, 1)

    return {
        "crop_origin": (cx1, cy1),
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
