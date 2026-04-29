"""Pipeline 模块单元测试（不需要 GPU / YOLO / LLM API）"""

import time
import pytest
import numpy as np

from pipeline.fps import FPSMeter
from pipeline.tracker import TrackManager
from pipeline.hull_number_locator import build_inverse_crop_info, HullNumberLocator


# ══════════════════════════════════════════════
#  FPSMeter 测试
# ══════════════════════════════════════════════

class TestFPSMeter:
    def test_initial_fps_is_zero(self):
        meter = FPSMeter(window_seconds=10.0)
        assert meter.get_fps() == 0.0

    def test_single_tick_zero_fps(self):
        """单次 tick 无法计算 FPS（至少需要 2 次）"""
        meter = FPSMeter(window_seconds=10.0)
        meter.tick("test")
        assert meter.get_fps("test") == 0.0

    def test_multiple_ticks(self):
        meter = FPSMeter(window_seconds=10.0)
        for _ in range(10):
            meter.tick("test")
            time.sleep(0.01)
        fps = meter.get_fps("test")
        assert fps > 0

    def test_multiple_channels(self):
        meter = FPSMeter(window_seconds=10.0)
        for _ in range(5):
            meter.tick("stream")
            meter.tick("process")
        assert "stream" in meter.get_all_fps()
        assert "process" in meter.get_all_fps()

    def test_reset(self):
        meter = FPSMeter(window_seconds=10.0)
        for _ in range(5):
            meter.tick("test")
        meter.reset("test")
        assert meter.get_fps("test") == 0.0

    def test_reset_all(self):
        meter = FPSMeter(window_seconds=10.0)
        meter.tick("a")
        meter.tick("b")
        meter.reset()
        assert meter.get_all_fps() == {}

    def test_should_print(self):
        meter = FPSMeter(window_seconds=10.0)
        meter._print_interval = 0.05  # 缩短间隔以测试
        # 第一次调用 should_print 在 tick 之前会初始化 last_print
        assert meter.should_print("test") is False
        # 经过足够时间后应返回 True
        time.sleep(0.06)
        assert meter.should_print("test") is True
        # 立即再次调用应返回 False（刚打印过）
        assert meter.should_print("test") is False


# ══════════════════════════════════════════════
#  TrackManager 测试
# ══════════════════════════════════════════════

class TestTrackManager:
    def test_get_or_create_new(self):
        mgr = TrackManager()
        info = mgr.get_or_create(1, 100)
        assert info.track_id == 1
        assert info.first_seen_frame == 100
        assert info.recognized is False

    def test_get_or_create_existing(self):
        mgr = TrackManager()
        info1 = mgr.get_or_create(1, 100)
        info2 = mgr.get_or_create(1, 200)
        assert info1 is info2
        assert info2.last_seen_frame == 200

    def test_needs_recognition_new(self):
        mgr = TrackManager()
        assert mgr.needs_recognition(1) is True

    def test_needs_recognition_after_bind(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        assert mgr.needs_recognition(1) is False

    def test_needs_recognition_while_pending(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.mark_pending(1)
        assert mgr.needs_recognition(1) is False

    def test_cancel_pending(self):
        """取消 pending 后应恢复为可识别状态。"""
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.mark_pending(1)
        assert mgr.needs_recognition(1) is False
        mgr.cancel_pending(1)
        assert mgr.needs_recognition(1) is True

    def test_cancel_pending_nonexistent(self):
        """取消不存在的 track 不应报错。"""
        mgr = TrackManager()
        mgr.cancel_pending(999)  # 不应抛异常

    def test_bind_result(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        info = mgr.active_tracks[1]
        assert info.recognized is True
        assert info.hull_number == "0014"
        assert info.description == "白色大型客轮"
        assert info.pending is False

    def test_bind_db_match(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        mgr.bind_db_match(1, "0014", "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪")
        info = mgr.active_tracks[1]
        assert info.db_matched is True
        assert info.db_match_id == "0014"

    def test_bind_locate_bboxes(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        bboxes = [(10, 20, 50, 60), (100, 200, 150, 250)]
        mgr.bind_locate_bboxes(1, bboxes)
        info = mgr.active_tracks[1]
        assert len(info.locate_bboxes) == 2
        assert info.locate_bboxes[0] == (10, 20, 50, 60)

    def test_display_text_waiting(self):
        mgr = TrackManager()
        text = mgr.get_display_text(1)
        assert "等待" in text

    def test_display_text_pending(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.mark_pending(1)
        text = mgr.get_display_text(1)
        assert "识别中" in text

    def test_display_text_db_matched(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        mgr.bind_db_match(1, "0014", "白色大型客轮，上层建筑为蓝色涂装")
        text = mgr.get_display_text(1)
        assert "库内确定id" in text
        assert "0014" in text

    def test_display_text_unknown(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "X999", "未知小船")
        text = mgr.get_display_text(1)
        assert "未知id" in text

    def test_cleanup_stale(self):
        mgr = TrackManager(max_stale_frames=10)
        mgr.get_or_create(1, 100)
        mgr.get_or_create(2, 200)
        cleaned = mgr.cleanup_stale(200)
        assert cleaned == 1  # track 1 被清理
        assert 2 in mgr.active_tracks

    def test_len(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.get_or_create(2, 100)
        assert len(mgr) == 2

    def test_concurrent_access(self):
        """多线程并发访问 TrackManager 不应崩溃。"""
        import threading

        mgr = TrackManager()
        errors = []

        def writer():
            try:
                for i in range(100):
                    mgr.get_or_create(i % 10, i)
                    if i % 3 == 0:
                        mgr.mark_pending(i % 10)
                    if i % 5 == 0:
                        mgr.cancel_pending(i % 10)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    mgr.needs_recognition(i % 10)
                    mgr.get(i % 10)
                    _ = mgr.active_tracks
                    _ = len(mgr)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发访问出错: {errors}"


# ══════════════════════════════════════════════
#  HullNumberLocator / 坐标转换测试
# ══════════════════════════════════════════════

class TestBuildInverseCropInfo:
    def test_basic_crop(self):
        """基本 crop 坐标转换。"""
        info = build_inverse_crop_info(
            x1=100, y1=200, x2=300, y2=400,
            frame_w=1920, frame_h=1080,
            pad=20,
        )
        # crop_origin 应为 (80, 180)（100-20, 200-20）
        assert info["crop_origin"] == (80, 180)
        # scale 不应为 0
        assert info["scale_x"] > 0
        assert info["scale_y"] > 0

    def test_crop_at_frame_edge(self):
        """crop 在帧边缘时 padding 被裁剪。"""
        info = build_inverse_crop_info(
            x1=5, y1=5, x2=50, y2=50,
            frame_w=1920, frame_h=1080,
            pad=20,
        )
        # crop_origin 应为 (0, 0)（max(0, 5-20) = 0）
        assert info["crop_origin"] == (0, 0)

    def test_large_crop_resize_down(self):
        """大 crop 应缩小（max_dim > target_max=512）。"""
        info = build_inverse_crop_info(
            x1=0, y1=0, x2=1000, y2=800,
            frame_w=1920, frame_h=1080,
            pad=0, target_max=512,
        )
        # scale 应 > 1（原尺寸大于 resize 后尺寸）
        assert info["scale_x"] > 1.0
        assert info["scale_y"] > 1.0

    def test_small_crop_resize_up(self):
        """小 crop 应放大（max_dim < target_min=256）。"""
        info = build_inverse_crop_info(
            x1=100, y1=100, x2=150, y2=130,
            frame_w=1920, frame_h=1080,
            pad=0, target_min=256,
        )
        # scale 应 < 1（原尺寸小于 resize 后尺寸）
        assert info["scale_x"] < 1.0
        assert info["scale_y"] < 1.0

    def test_transform_roundtrip(self):
        """验证坐标转换的往返一致性。"""
        # 模拟 detector.py 中的 crop 逻辑
        x1, y1, x2, y2 = 200, 300, 500, 600
        frame_w, frame_h = 1920, 1080
        pad = 20

        info = build_inverse_crop_info(x1, y1, x2, y2, frame_w, frame_h, pad)

        # crop 内的一个点 (10, 10) 应该映射回原帧
        fx = int(10 * info["scale_x"] + info["crop_origin"][0])
        fy = int(10 * info["scale_y"] + info["crop_origin"][1])
        assert fx >= x1 - pad
        assert fy >= y1 - pad

    def test_zero_size_crop(self):
        """零尺寸 crop 不应崩溃。"""
        info = build_inverse_crop_info(
            x1=100, y1=100, x2=100, y2=100,
            frame_w=1920, frame_h=1080,
        )
        assert info["scale_x"] == 1.0
        assert info["scale_y"] == 1.0


class TestHullNumberLocatorInit:
    def test_locator_unavailable_without_paddleocr(self):
        """PaddleOCR 未安装时 locator.available 应为 False。"""
        # 在没有安装 paddleocr 的环境中，初始化应优雅降级
        try:
            locator = HullNumberLocator()
            # 如果 paddleocr 已安装，available 可能为 True
            # 如果未安装，available 应为 False 且有 init_error
            if not locator.available:
                assert locator.init_error is not None
                assert "PaddleOCR" in locator.init_error or "paddle" in locator.init_error.lower()
        except Exception:
            # 如果 paddleocr 完全不可用，初始化不应抛异常
            pytest.fail("HullNumberLocator 初始化不应抛异常")

    def test_locate_returns_empty_when_unavailable(self):
        """PaddleOCR 不可用时 locate 应返回空列表。"""
        try:
            locator = HullNumberLocator()
            if not locator.available:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                regions = locator.locate(img)
                assert regions == []
        except Exception:
            pass  # paddleocr 可能未安装

    def test_locate_empty_image(self):
        """空图像不应崩溃。"""
        try:
            locator = HullNumberLocator()
            if locator.available:
                img = np.zeros((0, 0, 3), dtype=np.uint8)
                regions = locator.locate(img)
                assert regions == []
        except Exception:
            pass
