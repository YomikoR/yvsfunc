# Preview tools
from __future__ import annotations
from typing import Optional
import vapoursynth as vs
core = vs.core

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
        return core.fmtc.bitdepth(clip, bits=16)
    else:
        rgbs = core.resize.Spline36(clip, format=vs.RGBS)
        return core.fmtc.bitdepth(rgbs, bits=output_depth)


def playback(clip: vs.VideoNode, icc: Optional[str] = None, csp: str = '709', intent: str = 'relative') -> vs.VideoNode:
    '''
    Wrapper for iccc.Playback for video playback, like in mpv.
    '''
    return to_rgb(clip).iccc.Playback(display_icc=icc, playback_csp=csp, intent=intent)


def show_yuv(clip: vs.VideoNode) -> vs.VideoNode:
    if clip.format.color_family != vs.YUV:
        raise ValueError('show_yuv: input format must by YUV')
    y, u, v = core.std.SplitPlanes(clip)
    if clip.format.subsampling_w == 1 and clip.format.subsampling_h == 1:
        uv = core.std.StackHorizontal([u, v])
    elif clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0:
        uv = core.std.StackVertical([u, v])
    else:
        raise ValueError('show_yuv: subsampling not supported')
    if clip.format.bits_per_sample == 32:
        # There could be a range issue when previewed. Ignored.
        uv = core.std.Expr(uv, 'x 0.5 +')
    return core.std.StackVertical([y, uv])


def show_over_range(clip: vs.VideoNode, tolerance: float = 0) -> vs.VideoNode:
    '''
    Revert the pixel values that are out of limited range while allowing a given tolerance of error
    '''
    def _genExpr(rangemin: float, rangemax: float, tol: float=0) -> str:
        return 'x {chmin} < {rmax} x {chmax} > {rmin} x ? ?'.format(rmin=rangemin, rmax=rangemax, chmin=rangemin - tol, chmax=rangemax + tol)
    if clip.format.bits_per_sample == 32:
        if clip.format.color_family == vs.GRAY:
            return core.std.Expr(clip, _genExpr(0, 1, tolerance))
        elif clip.format.color_family == vs.YUV:
            return core.std.Expr(clip, [_genExpr(0, 1, tolerance), _genExpr(-0.5, 0.5, tolerance)])
        else:
            raise ValueError('show_over_range: color format not supported.')
    else:
        bd = clip.format.bits_per_sample - 8
        if clip.format.color_family == vs.GRAY:
            return core.std.Expr(clip, _genExpr(16 << bd, 235 << bd, tolerance))
        elif clip.format.color_family == vs.YUV:
            return core.std.Expr(clip, [_genExpr(16 << bd, 235 << bd, tolerance), _genExpr(16 << bd, 240 << bd, tolerance)])
        else:
            raise ValueError('show_over_range: color format not supported.')
