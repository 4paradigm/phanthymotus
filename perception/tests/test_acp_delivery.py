"""utils.acp must reach agent-core, which serves HTTPS only on 15678.

The default was plain http://127.0.0.1:15678: agent-core dropped the
connection, the failure was only logged, and every music completion was lost.
"""

import ssl
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import acp  # noqa: E402


class AcpDeliveryTest(unittest.TestCase):
    def test_posts_over_https_to_loopback_without_checking_the_self_signed_cert(self):
        with mock.patch.object(acp.urllib.request, "urlopen") as urlopen:
            acp._post("a1", "completed", {})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://localhost:15678/api/acp/complete")
        context = urlopen.call_args.kwargs["context"]
        self.assertFalse(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_NONE)


if __name__ == "__main__":
    unittest.main()
