"""Stop an embedded-model probe when the core that owns it disappears."""
import os
import time


def watch_parent(parent):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, parent)  # SYNCHRONIZE only.
        if not handle: os._exit(130)
        try:
            # A handle identifies the original process even if its PID is reused.
            result = kernel.WaitForSingleObject(handle, 0xffffffff)
            os._exit(130 if result == 0 else 1)
        finally: kernel.CloseHandle(handle)
    else:
        while True:
            if os.getppid() != parent: os._exit(130)
            time.sleep(1)
