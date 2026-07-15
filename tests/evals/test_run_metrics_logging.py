import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from main import _estimate_tokens_from_payload, _estimate_tokens_from_text


class RunMetricsLoggingHelpersTest(unittest.TestCase):
    def test_estimate_tokens_from_text_is_positive_for_nonempty_input(self) -> None:
        estimated = _estimate_tokens_from_text("python machine learning")
        self.assertGreater(estimated, 0)

    def test_estimate_tokens_from_text_scales_with_input_size(self) -> None:
        small = _estimate_tokens_from_text("small")
        large = _estimate_tokens_from_text("small " * 100)
        self.assertGreater(large, small)

    def test_estimate_tokens_from_payload_handles_nested_data(self) -> None:
        payload = {
            "status": "pass",
            "skills": ["python", "machine learning", "git"],
            "details": {"count": 3, "notes": ["ok"]},
        }
        estimated = _estimate_tokens_from_payload(payload)
        self.assertGreater(estimated, 0)


if __name__ == "__main__":
    unittest.main()