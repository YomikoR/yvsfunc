# Preview tools
from typing import Optional
import vapoursynth as vs
core = vs.core
from vsutil import depth, get_depth, get_subsampling, split

from .misc import y_error_msg

__all__ = [
    'playback',
    'show_yuv',
    'show_over_range',
]

def to_rgb(clip: vs.VideoNode, output_depth: int = 16) -> vs.VideoNode:
    '''
    Default output is 16-bit
    '''
    if clip.format.color_family == vs.RGB:
        return depth(clip, 16)
    else:
        rgbs = core.resize.Spline36(clip, format=vs.RGBS)
        return depth(rgbs, output_depth)


def playback(clip: vs.VideoNode, icc: Optional[str] = None, csp: str = '709', intent: str = 'relative') -> vs.VideoNode:
    '''
    Wrapper for iccc.Playback for video playback, like in mpv.
    '''
    return to_rgb(clip).iccc.Playback(display_icc=icc, playback_csp=csp, intent=intent)


def show_yuv(clip: vs.VideoNode) -> vs.VideoNode:
    func_name = 'show_yuv'
    if clip.format.color_family != vs.YUV:
        y_error_msg(func_name, 'input format must by YUV')
    y, u, v = split(clip)
    ss = get_subsampling(clip)
    if ss == '420':
        uv = core.std.StackHorizontal([u, v])
    elif ss == '444':
        uv = core.std.StackVertical([u, v])
    else:
        y_error_msg(func_name, 'subsampling not supported')
    if get_depth(clip) == 32:
        # There could be a range issue when previewed. Ignored.
        uv = core.std.Expr(uv, 'x 0.5 +')
    return core.std.StackVertical([y, uv])


def show_over_range(clip: vs.VideoNode, tolerance: float = 0) -> vs.VideoNode:
    '''
    Revert the pixel values that are out of limited range while allowing a given tolerance of error
    '''
    func_name = 'show_over_range'
    def _genExpr(rangemin: float, rangemax: float, tol: float=0) -> str:
        return 'x {chmin} < {rmax} x {chmax} > {rmin} x ? ?'.format(rmin=rangemin, rmax=rangemax, chmin=rangemin - tol, chmax=rangemax + tol)
    if get_depth(clip) == 32:
        if clip.format.color_family == vs.GRAY:
            return core.std.Expr(clip, _genExpr(0, 1, tolerance))
        elif clip.format.color_family == vs.YUV:
            return core.std.Expr(clip, [_genExpr(0, 1, tolerance), _genExpr(-0.5, 0.5, tolerance)])
        else:
            y_error_msg(func_name, 'color format not supported.')
    else:
        bd = clip.format.bits_per_sample - 8
        if clip.format.color_family == vs.GRAY:
            return core.std.Expr(clip, _genExpr(16 << bd, 235 << bd, tolerance))
        elif clip.format.color_family == vs.YUV:
            return core.std.Expr(clip, [_genExpr(16 << bd, 235 << bd, tolerance), _genExpr(16 << bd, 240 << bd, tolerance)])
        else:
            y_error_msg(func_name, 'color format not supported.')
