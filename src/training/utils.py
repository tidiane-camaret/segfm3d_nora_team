from contextlib import contextmanager
import signal
import threading
import torch
import numpy as np
import hashlib

@contextmanager
def defer_keysignal():
    signum = signal.SIGINT
    # Based on https://stackoverflow.com/a/71330357/1319998

    original_handler = None
    defer_handle_args = None

    def defer_handle(*args):
        nonlocal defer_handle_args
        defer_handle_args = args

    # Do nothing if
    # - we don't have a registered handler in Python to defer
    # - or the handler is not callable, so either SIG_DFL where the system
    #   takes some default action, or SIG_IGN to ignore the signal
    # - or we're not in the main thread that doesn't get signals anyway
    original_handler = signal.getsignal(signum)
    if (
        original_handler is None
        or not callable(original_handler)
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    try:
        signal.signal(signum, defer_handle)
        yield
    finally:
        signal.signal(signum, original_handler)
        if defer_handle_args is not None:
            original_handler(*defer_handle_args)


@contextmanager
def defer_keysignal_with_grad():
    # Wrap the existing defer_keysignal context manager
    with defer_keysignal():
        # Turn on PyTorch gradient mode
        torch.set_grad_enabled(True)
        try:
            yield
        finally:
            # Ensure gradient mode is turned off at the end
            torch.set_grad_enabled(False)


def compute_binary_dsc(pred, mask):
    intersect = np.sum(pred * mask)
    return 2 * intersect / (np.sum(pred) + np.sum(mask))



def fast_id_to_deterministic_float(id):
    hash_digest = hashlib.sha256(str(id).encode("utf-8")).digest()
    seed = int.from_bytes(hash_digest[:4], "big", signed=False)
    float_val = float(seed) / 4294967296.0
    return float_val


fast_ids_to_deterministic_floats = np.vectorize(fast_id_to_deterministic_float)
