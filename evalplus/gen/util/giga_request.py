import signal
import time

import requests
from giga import GigaChat


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(client: GigaChat, *args, **kwargs) -> dict:
    key = "temperature"
    if key in kwargs and kwargs[key] == 0:
        kwargs[key] = 1
        kwargs["top_p"] = 0
    else:
        kwargs["top_p"] = 0.95
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = client.chat(*args, **kwargs)
            signal.alarm(0)
        except requests.HTTPError as e:
            print("requests.HTTPError. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(5)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret
