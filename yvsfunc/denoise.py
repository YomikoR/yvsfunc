from typing import Any, Dict, List, Optional, Union
import vapoursynth as vs
core = vs.core
from vsutil import depth, get_subsampling, get_y

from .misc import y_error_msg
from .planes import join_uv
from .resample import aa2x, get_nnedi3, ResClip

__all__ = [
    'vn_denoise',
    'KNLMYUV',
]

def vn_denoise(clip: vs.VideoNode, ref: Optional[vs.VideoNode] = None, sigma: Union[float, List[float]] = 15, radius: int = 2, cuda: bool = True, **bm3d_args: Any) -> vs.VideoNode:
    '''
    Profile 'vn' of BM3D (optionally) prefiltered with QTGMC deshimmering. \\
    Output is ALWAYS in YUV444PS.
    '''
    func_name = 'vn_denoise'
    if clip.format.color_family == vs.RGB:
        y_error_msg(func_name, 'RGB input not directly supported')
    if ref is None:
        import havsfunc as haf
        ref = haf.QTGMC(clip, InputType=1, Sharpness=0)
    if clip.format.color_family == vs.GRAY:
        ref = depth(ref, 32)
        clip = depth(clip, 32)
        chroma = False
    else:
        ref = core.resize.Spline36(ref, format=vs.YUV444PS)
        clip = core.resize.Spline36(clip, format=vs.YUV444PS)
        chroma = True
    if cuda:
        bm3d_default_args = dict(block_step=6, bm_range=12, ps_num=2, ps_range=6, chroma=chroma, fast=True)
        bm3d_default_args.update(bm3d_args)
        denoise = core.bm3dcuda.BM3D(clip, ref=ref, sigma=sigma, radius=radius, **bm3d_default_args)
        if radius > 1:
            denoise = core.bm3d.VAggregate(denoise, radius=radius, sample=1)
    else:
        import mvsfunc as mvf
        denoise = mvf.BM3D(clip, ref=ref, sigma=sigma, radius2=radius, profile2='vn', **bm3d_args)
    return denoise


def KNLMYUV(clip: vs.VideoNode, args_1p: Dict[Any, Any] = dict(d=0, a=4, h=0.2), args_2p: Dict[Any, Any] = dict(), chromaloc_in_s: str = 'left', opencl: bool = False, show_ref: bool = False) -> vs.VideoNode:
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
        yd = core.resize.Spline36(y, y.width / 2, y.height / 2, src_left=sx, src_top=sy)
        half = join_uv(yd, clip)
        knlm_args_1p = dict()
        knlm_args_1p.update(args_1p)
        knlm_args_1p['channels'] = 'YUV'
        dn_uv = core.knlm.KNLMeansCL(half, **knlm_args_1p)
        yh_dn = ResClip(get_y(dn_uv), sx=-sx / 2, sy=-sy / 2)
        ypre = aa2x(yh_dn, nnedi3=get_nnedi3(opencl=opencl)).spline36(y.width, y.height)
        if show_ref:
            return join_uv(ypre, dn_uv)
        knlm_args_2p = dict()
        knlm_args_2p.update(args_1p)
        knlm_args_2p.update(args_2p)
        knlm_args_2p['channels'] = 'Y'
        knlm_args_2p['rclip'] = ypre
        dn_y = core.knlm.KNLMeansCL(y, **knlm_args_2p)
        return join_uv(dn_y, dn_uv)
