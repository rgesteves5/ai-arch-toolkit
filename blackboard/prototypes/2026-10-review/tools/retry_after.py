import email.message, io, urllib.error
from unittest import mock
from ai_arch_toolkit.toolkit.tools import _http
from ai_arch_toolkit.toolkit.tools._news import hacker_news

def err(status, retry_after):
    h = email.message.Message(); h["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x/", status, "Too Many", h, io.BytesIO(b"{}"))

# 1) a 429 with Retry-After: inf
with mock.patch.object(_http, "_open", side_effect=err(429, "inf")):
    try:
        hacker_news.__wrapped__() if hasattr(hacker_news, "__wrapped__") else hacker_news()
    except Exception as e:
        print("first call:", type(e).__name__, str(e)[:100])
print("resting:", _http._THROTTLE.resting("hacker-news.firebaseio.com"))
# 2) the next call, no network needed
with mock.patch.object(_http, "_open", side_effect=AssertionError("no request expected")):
    try:
        hacker_news()
    except Exception as e:
        print("second call:", type(e).__name__, str(e)[:100])
