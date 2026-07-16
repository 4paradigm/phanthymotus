import sys
import types
import unittest
from unittest import mock


class FakeImage:
    def __init__(self, height, width):
        self.shape = (height, width, 3)

    def __getitem__(self, key):
        y_slice, x_slice = key
        return FakeImage(y_slice.stop - y_slice.start, x_slice.stop - x_slice.start)


class MarkedImage(FakeImage):
    def __init__(
        self,
        height,
        width,
        marker,
        origin_x=0,
        origin_y=0,
    ):
        super().__init__(height, width)
        self.marker = marker
        self.origin_x = origin_x
        self.origin_y = origin_y

    def __getitem__(self, key):
        y_slice, x_slice = key
        return MarkedImage(
            height=y_slice.stop - y_slice.start,
            width=x_slice.stop - x_slice.start,
            marker=self.marker,
            origin_x=self.origin_x + x_slice.start,
            origin_y=self.origin_y + y_slice.start,
        )

    def contains_marker(self):
        marker_x, marker_y = self.marker
        height, width = self.shape[:2]
        return (
            self.origin_x <= marker_x < self.origin_x + width
            and self.origin_y <= marker_y < self.origin_y + height
        )


class OCRTiledStrategyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plugins.ocr_tiled_strategy import (
            AdaptiveTiledOCRStrategy,
            LargeImageStrategyConfig,
        )

        cls.strategy_type = AdaptiveTiledOCRStrategy
        cls.config_type = LargeImageStrategyConfig

    def setUp(self):
        self.cv2 = types.SimpleNamespace(
            IMREAD_COLOR=1,
            IMREAD_REDUCED_COLOR_2=2,
            IMREAD_REDUCED_COLOR_4=4,
            IMREAD_REDUCED_COLOR_8=8,
        )

    @staticmethod
    def _decoded_image(width=2600, height=2000):
        return types.SimpleNamespace(
            image=FakeImage(height, width),
            source_size=(width, height),
            factor=1,
        )

    @staticmethod
    def _jpeg(width, height):
        return (
            b"\xff\xd8\xff\xc0\x00\x11\x08"
            + height.to_bytes(2, "big")
            + width.to_bytes(2, "big")
            + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
        )

    def test_rejects_overlap_not_smaller_than_tile(self):
        with self.assertRaisesRegex(ValueError, "overlap must be smaller"):
            self.config_type.from_mapping(
                {"tile_size": 1280, "overlap": 1280}
            )

    def test_strategy_is_opt_in_when_config_section_is_absent(self):
        config = self.config_type.from_mapping(None)

        self.assertFalse(config.enabled)

    def test_decode_plan_preserves_detail_without_crossing_hard_limit(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        cases = {
            4000: (1, 3200),
            6000: (2, 3000),
            7000: (2, 3200),
            9000: (4, 2250),
        }

        for source_side, expected in cases.items():
            with self.subTest(source_side=source_side):
                plan = strategy._plan_jpeg_decode(
                    self.cv2, (source_side, source_side)
                )
                self.assertEqual((plan.factor, max(plan.target_size)), expected)

    def test_decode_plan_never_exceeds_hard_limit(self):
        strategy = self.strategy_type(
            {"decode_side": 4096, "decode_hard_limit": 4096},
            global_max_side=1600,
        )

        plan = strategy._plan_jpeg_decode(self.cv2, (9000, 4500))

        self.assertEqual(plan.factor, 4)
        self.assertLessEqual(max(plan.target_size), 4096)

    def test_rejects_tile_larger_than_decode_hard_limit(self):
        with self.assertRaisesRegex(ValueError, "tile_size must not exceed"):
            self.config_type.from_mapping(
                {"tile_size": 5000, "decode_hard_limit": 4096}
            )

    def test_decode_plan_rejects_image_too_large_for_reduced_decode(self):
        strategy = self.strategy_type({}, global_max_side=1600)

        with self.assertRaisesRegex(ValueError, "JPEG dimensions exceed"):
            strategy._plan_jpeg_decode(self.cv2, (40000, 30000))

    def test_tile_grid_anchors_right_and_bottom_edges(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 20},
            global_max_side=1600,
        )

        tiles = strategy._select_tiles((2500, 2100))

        self.assertIn((1220, 820, 2500, 2100), tiles)
        self.assertEqual(len(tiles), len(set(tiles)))

    def test_tile_limit_is_deterministic_and_covers_corners(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 6},
            global_max_side=1600,
        )

        first = strategy._select_tiles((4000, 3000))
        second = strategy._select_tiles((4000, 3000))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertIn((0, 0, 1280, 1280), first)
        self.assertIn((2720, 1720, 4000, 3000), first)

    def test_offsets_then_scales_tile_box_to_source_pixels(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        decoded_items = strategy._offset_items(
            [
                {
                    "text": "small",
                    "bbox": [10, 20, 110, 60],
                    "score": 0.8,
                }
            ],
            offset_x=1000,
            offset_y=500,
        )

        source_items = strategy._scale_items(
            decoded_items,
            scale_x=2.0,
            scale_y=2.0,
            bounds=(4000, 3000),
        )

        self.assertEqual(source_items[0]["bbox"], [2020, 1040, 2220, 1120])

    def test_scale_clips_boxes_to_source_bounds_without_mutating_input(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        item = {
            "text": "edge",
            "bbox": [-5, -4, 2100, 1600],
            "score": 0.7,
        }

        scaled = strategy._scale_items(
            [item], scale_x=2.0, scale_y=2.0, bounds=(4000, 3000)
        )

        self.assertEqual(scaled[0]["bbox"], [0, 0, 4000, 3000])
        self.assertEqual(item["bbox"], [-5, -4, 2100, 1600])

    def test_dedup_keeps_higher_confidence_duplicate(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        candidates = [
            strategy._candidate(
                {
                    "text": "Old Navy",
                    "bbox": [10, 10, 110, 40],
                    "score": 0.7,
                },
                from_tile=False,
            ),
            strategy._candidate(
                {
                    "text": "old   navy",
                    "bbox": [12, 11, 112, 41],
                    "score": 0.9,
                },
                from_tile=True,
            ),
        ]

        self.assertEqual(strategy._deduplicate(candidates), [candidates[1].item])

    def test_dedup_prefers_tile_when_scores_tie(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        global_item = strategy._candidate(
            {"text": "TEST", "bbox": [0, 0, 100, 30], "score": 0.8},
            from_tile=False,
        )
        tile_item = strategy._candidate(
            {"text": "test", "bbox": [1, 1, 101, 31], "score": 0.8},
            from_tile=True,
        )

        self.assertEqual(
            strategy._deduplicate([global_item, tile_item]),
            [tile_item.item],
        )

    def test_dedup_retains_overlapping_different_text(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        candidates = [
            strategy._candidate(
                {"text": "price", "bbox": [0, 0, 100, 30], "score": 0.9},
                from_tile=False,
            ),
            strategy._candidate(
                {"text": "total", "bbox": [2, 1, 102, 31], "score": 0.8},
                from_tile=True,
            ),
        ]

        self.assertEqual(len(strategy._deduplicate(candidates)), 2)

    def test_results_are_sorted_top_to_bottom_then_left_to_right(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        candidates = [
            strategy._candidate(
                {"text": "right", "bbox": [100, 10, 150, 30], "score": 0.9},
                from_tile=True,
            ),
            strategy._candidate(
                {"text": "bottom", "bbox": [0, 50, 50, 70], "score": 0.9},
                from_tile=True,
            ),
            strategy._candidate(
                {"text": "left", "bbox": [10, 10, 60, 30], "score": 0.9},
                from_tile=False,
            ),
        ]

        self.assertEqual(
            [item["text"] for item in strategy._deduplicate(candidates)],
            ["left", "right", "bottom"],
        )

    def test_large_image_runs_global_then_bounded_sequential_tiles(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 6},
            global_max_side=1600,
        )
        strategy._decode_image = mock.Mock(
            return_value=self._decoded_image(width=4000, height=3000)
        )
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1200, width=1600)
        )
        calls = []

        def infer(image):
            calls.append(image.shape[:2])
            return []

        strategy.recognize(b"jpeg", infer)

        self.assertEqual(calls[0], (1200, 1600))
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(max(shape) <= 1280 for shape in calls[1:]))

    def test_partial_tile_failure_returns_successful_passes(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 2},
            global_max_side=1600,
        )
        strategy._decode_image = mock.Mock(return_value=self._decoded_image())
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1231, width=1600)
        )
        calls = 0

        def infer(_image):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("tile failed")
            if calls == 3:
                return [
                    {
                        "text": "kept",
                        "bbox": [10, 20, 110, 60],
                        "score": 0.8,
                    }
                ]
            return []

        with self.assertLogs("plugins.ocr_tiled_strategy", level="WARNING"):
            result = strategy.recognize(b"jpeg", infer)

        self.assertEqual([item["text"] for item in result], ["kept"])
        self.assertEqual(calls, 3)

    def test_global_failure_still_attempts_tiles(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 1},
            global_max_side=1600,
        )
        strategy._decode_image = mock.Mock(return_value=self._decoded_image())
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1231, width=1600)
        )
        infer = mock.Mock(
            side_effect=[
                RuntimeError("global failed"),
                [
                    {
                        "text": "tile",
                        "bbox": [5, 5, 50, 30],
                        "score": 0.9,
                    }
                ],
            ]
        )

        with self.assertLogs("plugins.ocr_tiled_strategy", level="WARNING"):
            result = strategy.recognize(b"jpeg", infer)

        self.assertEqual(result[0]["text"], "tile")
        self.assertEqual(infer.call_count, 2)

    def test_total_inference_failure_raises_first_error(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 1},
            global_max_side=1600,
        )
        strategy._decode_image = mock.Mock(return_value=self._decoded_image())
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1231, width=1600)
        )

        with self.assertLogs("plugins.ocr_tiled_strategy", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "global failed"):
                strategy.recognize(
                    b"jpeg", mock.Mock(side_effect=ValueError("global failed"))
                )

    def test_empty_success_is_not_reported_as_failure(self):
        strategy = self.strategy_type(
            {"tile_size": 1280, "overlap": 192, "max_tiles": 1},
            global_max_side=1600,
        )
        strategy._decode_image = mock.Mock(return_value=self._decoded_image())
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1231, width=1600)
        )

        self.assertEqual(strategy.recognize(b"jpeg", lambda _image: []), [])

    def test_small_non_jpeg_uses_one_single_pass(self):
        strategy = self.strategy_type({"enabled": True}, global_max_side=1600)
        strategy._decode_image = mock.Mock(
            return_value=self._decoded_image(width=800, height=600)
        )
        infer = mock.Mock(
            return_value=[
                {
                    "text": "small",
                    "bbox": [10, 20, 100, 50],
                    "score": 0.9,
                }
            ]
        )

        result = strategy.recognize(b"png", infer)

        self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])
        infer.assert_called_once()

    def test_non_jpeg_below_trigger_keeps_existing_global_resize(self):
        strategy = self.strategy_type({"enabled": True}, global_max_side=1600)
        strategy._decode_image = mock.Mock(
            return_value=self._decoded_image(width=2000, height=1500)
        )
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1200, width=1600)
        )
        seen_shapes = []

        def infer(image):
            seen_shapes.append(image.shape[:2])
            return [
                {
                    "text": "scaled",
                    "bbox": [80, 40, 160, 80],
                    "score": 0.9,
                }
            ]

        result = strategy.recognize(b"png", infer)

        self.assertEqual(seen_shapes, [(1200, 1600)])
        self.assertEqual(result[0]["bbox"], [100, 50, 200, 100])

    def test_decode_4000_jpeg_keeps_3200_pixels_for_strategy(self):
        strategy = self.strategy_type({}, global_max_side=1600)
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.INTER_AREA = 3
        cv2_module.imdecode = mock.Mock(return_value=FakeImage(3000, 4000))
        cv2_module.resize = mock.Mock(return_value=FakeImage(2400, 3200))
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            decoded = strategy._decode_image(self._jpeg(4000, 3000))

        cv2_module.imdecode.assert_called_once_with("encoded", 1)
        cv2_module.resize.assert_called_once_with(
            mock.ANY, (3200, 2400), interpolation=3
        )
        self.assertEqual(decoded.image.shape[:2], (2400, 3200))
        self.assertEqual(decoded.source_size, (4000, 3000))

    def test_tiled_pipeline_recovers_small_text_without_duplicate_large_text(self):
        strategy = self.strategy_type(
            {
                "enabled": True,
                "tile_size": 1280,
                "overlap": 192,
                "max_tiles": 6,
            },
            global_max_side=1600,
        )
        source = MarkedImage(
            height=3000,
            width=4000,
            marker=(3500, 2500),
        )
        strategy._decode_image = mock.Mock(
            return_value=types.SimpleNamespace(
                image=source,
                source_size=(4000, 3000),
                factor=1,
            )
        )
        strategy._resize_longest = mock.Mock(
            return_value=FakeImage(height=1200, width=1600)
        )

        def infer(image):
            if not isinstance(image, MarkedImage):
                return [
                    {
                        "text": "large text",
                        "bbox": [100, 100, 400, 200],
                        "score": 0.9,
                    }
                ]
            if image.contains_marker():
                local_x = 3500 - image.origin_x
                local_y = 2500 - image.origin_y
                return [
                    {
                        "text": "small text",
                        "bbox": [
                            local_x,
                            local_y,
                            local_x + 100,
                            local_y + 40,
                        ],
                        "score": 0.85,
                    }
                ]
            return []

        result = strategy.recognize(b"jpeg", infer)

        self.assertEqual(
            [item["text"] for item in result],
            ["large text", "small text"],
        )
        self.assertEqual(
            sum(item["text"] == "large text" for item in result),
            1,
        )
        self.assertTrue(
            all(0 <= item["bbox"][0] < item["bbox"][2] <= 4000 for item in result)
        )
        self.assertTrue(
            all(0 <= item["bbox"][1] < item["bbox"][3] <= 3000 for item in result)
        )
        self.assertTrue(
            all(
                isinstance(coordinate, int)
                for item in result
                for coordinate in item["bbox"]
            )
        )


if __name__ == "__main__":
    unittest.main()
