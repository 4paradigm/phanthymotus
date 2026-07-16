import types
import unittest


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


if __name__ == "__main__":
    unittest.main()
