import unittest

from devtools.server_soak import validate_ready


class ServerSoakTest(unittest.TestCase):
    def test_ready_idle_status(self) -> None:
        validate_ready(
            {
                "state": "ready",
                "last_error": "",
                "requests": {"active": 0},
            }
        )

    def test_rejects_error_or_active_request(self) -> None:
        for status in (
            {"state": "ready", "last_error": "bad", "requests": {"active": 0}},
            {"state": "ready", "last_error": "", "requests": {"active": 1}},
            {"state": "loading", "last_error": "", "requests": {"active": 0}},
        ):
            with self.assertRaises(RuntimeError):
                validate_ready(status)


if __name__ == "__main__":
    unittest.main()
