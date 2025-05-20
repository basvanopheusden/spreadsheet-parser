import sys
import types
import unittest

# Provide a minimal openai stub if the real package is unavailable
if "openai" not in sys.modules:

    def make_client(*args, **kwargs):
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )
        )

    openai_stub = types.SimpleNamespace(
        ChatCompletion=types.SimpleNamespace(create=lambda **kwargs: None),
        OpenAI=make_client,
    )
    sys.modules["openai"] = openai_stub

from spreadsheet_parser.analysis import _cb_rank_value


class TestCBRankValue(unittest.TestCase):
    def test_handles_numeric_input(self):
        self.assertEqual(_cb_rank_value(123), 123)
        self.assertEqual(_cb_rank_value(456.0), 456)

    def test_handles_strings(self):
        self.assertEqual(_cb_rank_value("1,234"), 1234)
        self.assertIsNone(_cb_rank_value(None))


if __name__ == "__main__":
    unittest.main()
