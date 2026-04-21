import unittest

from models.internVL import InternVL2_5_8B
from models.qwen import Qwen2_5_VL


class ParserContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Bypass heavyweight model initialization; parser methods do not depend on weights.
        self.internvl = InternVL2_5_8B.__new__(InternVL2_5_8B)
        self.qwen = Qwen2_5_VL.__new__(Qwen2_5_VL)

    def test_internvl_parses_historical_integer_box_format(self):
        text = "<ref>person</ref><box>[[100, 200, 500, 700]]</box>"
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=1000, img_height=1000)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [100.0, 200.0, 400.0, 500.0])

    def test_internvl_fallback_accepts_decimal_boxes(self):
        text = "<box>[[100.5, 200.25, 500.75, 700.5]]</box>"
        boxes, parser_fallback_used = self.internvl._parse_boxes(text, img_width=1000, img_height=1000)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 100.5)
        self.assertAlmostEqual(boxes[0][1], 200.25)

    def test_qwen_parses_current_corner_format(self):
        text = "<|box_start|>(28, 56), (280, 420)<|box_end|>"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=280, img_height=420)
        self.assertFalse(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [28.0, 56.0, 252.0, 364.0])

    def test_qwen_fallback_accepts_decimal_corner_format(self):
        text = "<|box_start|>([28.5, 56.25]), ([280.5, 420.75])<|box_end|>"
        boxes, parser_fallback_used = self.qwen._parse_boxes(text, img_width=280, img_height=420)
        self.assertTrue(parser_fallback_used)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 28.5)
        self.assertAlmostEqual(boxes[0][1], 56.25)


if __name__ == "__main__":
    unittest.main()
