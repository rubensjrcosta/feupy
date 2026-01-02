"""Core marker utilities."""

import random

def repeat_items(items, n, shuffle=False):
    result = (items * (n // len(items))) + items[: n % len(items)]
    if shuffle:
        random.shuffle(result)
    return result
