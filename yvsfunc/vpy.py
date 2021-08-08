from typing import Union
import vapoursynth as vs
import inspect
import html
import runpy

from .misc import y_error_msg

__all__ = [
    'get_param',
    'read_vpy',
]

def u2a(str_unicode: str) -> bytes:
    '''
    Encode UTF-8 string to ASCII bytes
    '''
    return str_unicode.encode('ascii', 'xmlcharrefreplace')


def a2u(bytes_ascii: bytes) -> str:
    '''
    Decode ASCII bytes to UTF-8 string
    '''
    return html.unescape(bytes_ascii.decode('utf8', 'xmlcharrefreplace'))


def get_param(key: str, decode: bool = True) -> Union[str, bytes]:
    '''
    Get parameter by name from VPY

    Note: this should only be called directly in a VPY file
    '''
    is_in_vspipe = True
    frame = inspect.currentframe().f_back
    try:
        # Detect reserved variable to see if it's in vspipe
        key_u2a = a2u(frame.f_globals['YVSFUNC_WITH_U2A'])
        if key_u2a == '1':
            is_in_vspipe = False
    except (AttributeError, KeyError):
        pass
    key_b = frame.f_globals[key]
    try:
        if isinstance(key_b, bytes) and decode:
            return key_b.decode('utf-8') if is_in_vspipe else a2u(key_b)
        else:
            return key_b
    except KeyError:
        print('get_param: key not found. Make sure you are calling directly from the vpy script.')
        raise


def read_vpy(script: str, params: dict = dict(), output_idx: int = 0) -> vs.VideoNode:
    '''
    Read VPY from script

    A reserved flag YVSFUNC_WITH_U2A is set to distinguish between read_vpy and vspipe
    so that the VPY file will be compatible with both environments.
    '''
    params_ascii = dict(
        YVSFUNC_WITH_U2A = u2a('1')
    )
    if 'YVSFUNC_WITH_U2A' in params.keys():
        y_error_msg('read_vpy', 'YVSFUNC_WITH_U2A is a reserved key for read_vpy. Remove it from the input dict.')
    for i, k in params.items():
        params_ascii[i] = u2a(str(k))
    runpy.run_path(script, params_ascii, '__vapoursynth__')
    return vs.get_output(output_idx)
