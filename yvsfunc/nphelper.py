# Helper functions for numpy
from typing import Union
import ctypes
import vapoursynth as vs
core = vs.core
import numpy as np

from .misc import y_error_msg

__all__ = [
    'np_get_dtype',
    'np_read_plane',
    'np_write_plane',
    'make_eye',
]

def np_get_dtype(frame: Union[vs.VideoFrame, vs.VideoNode]) -> np.DTypeLike:
    func_name = 'np_get_dtype'
    if frame.format.sample_type == vs.INTEGER:
        if frame.format.bytes_per_sample == 1:
            return np.uint8
        elif frame.format.bytes_per_sample == 2:
            return np.uint16
        elif frame.format.bytes_per_sample == 4: # Supported by v4 API
            return np.uint32
        else:
            y_error_msg(func_name, 'input format not supported')
    else:
        if frame.format.bytes_per_sample == 2:
            return np.float16
        elif frame.format.bytes_per_sample == 4:
            return np.float32
        else:
            y_error_msg(func_name, 'input format not supported')


def np_read_plane(frame: vs.VideoFrame, plane: int, keep_stride: bool = False) -> np.ndarray:
    '''
    Cast the view of plane as `numpy.ndarray` using `get_read_ptr`.
    '''
    dtype = np_get_dtype(frame)
    pt_cast = ctypes.cast(frame.get_read_ptr(plane), ctypes.POINTER(ctypes.c_uint8 * frame.get_stride(plane) * frame.height))
    arr = np.ctypeslib.as_array(pt_cast, shape=()).view(dtype)
    return arr if keep_stride else arr[:, :frame.width]


def np_write_plane(frame: vs.VideoFrame, plane: int, keep_stride: bool = False) -> np.ndarray:
    '''
    Cast the view of plane as `numpy.ndarray` using `get_write_ptr`.
    '''
    dtype = np_get_dtype(frame)
    pt_cast = ctypes.cast(frame.get_write_ptr(plane), ctypes.POINTER(ctypes.c_uint8 * frame.get_stride(plane) * frame.height))
    arr = np.ctypeslib.as_array(pt_cast, shape=()).view(dtype)
    return arr if keep_stride else arr[:, :frame.width]


def make_eye(n: int) -> vs.VideoNode:
    '''
    A square clip filled with identity matrix
    '''
    clip = core.std.BlankClip(format=vs.GRAYS, width=n, height=n, length=1)
    def _make_eye(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        dst_f = fout.get_write_array(0)
        for x in range(n):
            dst_f[x, x] = 1.0
        return fout
    return core.std.ModifyFrame(clip, clips=clip, selector=_make_eye)
