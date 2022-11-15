from typing import Any, Dict
import vapoursynth as vs
core = vs.core
from vsutil import get_subsampling, get_y

from .misc import y_error_msg
from .planes import join_uv
from .resample import aa2x, get_nnedi3, ResClip

__all__ = [
    'KNLMYUV',
]


def KNLMYUV(
    clip: vs.VideoNode,
    args_1p: Dict[Any, Any] = dict(d=0, a=4, h=0.2),
    args_2p: Dict[Any, Any] = dict(),
    chromaloc_in_s: str = 'left',
    show_ref: bool = False
) -> vs.VideoNode:
    '''
    A KNLMeansCL wrapper for YUV420 input:
    1. resize luma to half size
    2. denoise in YUV444 (effective because params are relatively large)
    3. denoise luma with re-upscaled luma as reference clip (effective because of a ref clip)
    '''
    func_name = 'KNLMYUV'
    if clip.format.color_family != vs.YUV:
        y_error_msg(func_name, 'input format must be YUV420 or YUV444')
    ss = get_subsampling(clip)
    if ss == '444':
        knlm_args_1p = dict()
        knlm_args_1p.update(args_1p)
        knlm_args_1p['channels'] = 'YUV'
        return core.knlm.KNLMeansCL(clip, **knlm_args_1p)
    elif ss != '420':
        y_error_msg(func_name, 'input format must be YUV420 or YUV444')
    else:
        y = get_y(clip)
        if chromaloc_in_s == 'left':
            sx = -0.5
            sy = 0
        elif chromaloc_in_s == 'top_left':
            sx = -0.5
            sy = -0.5
        elif chromaloc_in_s == 'center':
            sx = 0
            sy = 0
        else:
            y_error_msg(func_name, 'chroma location not supported')
        yd = core.resize.Bicubic(y, y.width / 2, y.height / 2, src_left=sx, src_top=sy, filter_param_a=-0.5, filter_param_b=0.25)
        half = join_uv(yd, clip)
        knlm_args_1p = dict()
        knlm_args_1p.update(args_1p)
        knlm_args_1p['channels'] = 'YUV'
        dn_uv = core.knlm.KNLMeansCL(half, **knlm_args_1p)
        yh_dn = ResClip(get_y(dn_uv), sx=-sx / 2, sy=-sy / 2)
        ypre = aa2x(yh_dn, nnedi3=get_nnedi3()).spline36(y.width, y.height)
        if show_ref:
            return join_uv(ypre, dn_uv)
        knlm_args_2p = dict()
        knlm_args_2p.update(args_1p)
        knlm_args_2p.update(args_2p)
        knlm_args_2p['channels'] = 'Y'
        knlm_args_2p['rclip'] = ypre
        dn_y = core.knlm.KNLMeansCL(y, **knlm_args_2p)
        return join_uv(dn_y, dn_uv)
