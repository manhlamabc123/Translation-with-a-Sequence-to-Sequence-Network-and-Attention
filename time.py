import time
import math

def as_minutes(s):
    minute = math.floor(s / 60)
    second = minute * 60
    return '%dm %ds' % (minute, second)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s 
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))