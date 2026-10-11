# Does a non-ASCII URL reach the network seam, and what does the real opener do with it?
import http.client, socket
from unittest import mock
from ai_arch_toolkit.toolkit.tools import _http
from ai_arch_toolkit.toolkit.tools._web import http_get

# Block real sockets: make connect fail after the request line is built.
class NoConnect(http.client.HTTPConnection):
    def connect(self):  # never reach the network
        raise OSError("blocked in test")
orig_send = http.client.HTTPConnection.send
def fake_send(self, data):  # the request line is encoded before sending
    raise OSError("blocked in test")
with mock.patch.object(http.client.HTTPConnection, "connect", lambda self: None), \
     mock.patch.object(http.client.HTTPConnection, "send", fake_send):
    for url in ["http://example.invalid/café", "http://example.invalid/ok"]:
        try:
            http_get(url)
        except Exception as e:
            print(url, "->", type(e).__name__, "|", str(e)[:90])
