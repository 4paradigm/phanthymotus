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


if __name__ == "__main__":
    unittest.main()
