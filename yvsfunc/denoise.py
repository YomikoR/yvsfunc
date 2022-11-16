from __future__ import annotations
from typing import Any, Dict
import vapoursynth as vs
core = vs.core

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
    if clip.format.color_family != vs.YUV:
        raise ValueError('KNLMYUV: input format must be YUV420 or YUV444')
    if clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0:
        knlm_args_1p = dict()
        knlm_args_1p.update(args_1p)
        knlm_args_1p['channels'] = 'YUV'
        return core.knlm.KNLMeansCL(clip, **knlm_args_1p)
    elif clip.format.subsampling_w == 1 and clip.format.subsampling_h == 1:
        y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
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
            raise ValueError('KNLMYUV: the chroma location is not supported')
        yd = core.resize.Bicubic(y, y.width / 2, y.height / 2, src_left=sx, src_top=sy, filter_param_a=-0.5, filter_param_b=0.25)
        half = core.std.ShufflePlanes([yd, clip], [0, 1, 2], vs.YUV)
        knlm_args_1p = dict()
        knlm_args_1p.update(args_1p)
        knlm_args_1p['channels'] = 'YUV'
        dn_uv = core.knlm.KNLMeansCL(half, **knlm_args_1p)
        yh_dn = ResClip(core.std.ShufflePlanes(dn_uv, 0, vs.GRAY), sx=-sx / 2, sy=-sy / 2)
        ypre = aa2x(yh_dn, nnedi3=get_nnedi3()).spline36(y.width, y.height)
        if show_ref:
            return core.std.ShufflePlanes([ypre, dn_uv], [0, 1, 2], vs.YUV)
        knlm_args_2p = dict()
        knlm_args_2p.update(args_1p)
        knlm_args_2p.update(args_2p)
        knlm_args_2p['channels'] = 'Y'
        knlm_args_2p['rclip'] = ypre
        dn_y = core.knlm.KNLMeansCL(y, **knlm_args_2p)
        return core.std.ShufflePlanes([dn_y, dn_uv], [0, 1, 2], vs.YUV)
    else:
        raise ValueError('KNLMYUV: input format must be YUV420 or YUV444')
