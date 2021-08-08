from typing import List, Optional
import vapoursynth as vs
core = vs.core
from vsutil import depth, get_y

__all__ = [
    'sep_fields',
    'weave',
    'apply_borders',
    'get_y32',
    'join_uv',
]

def sep_fields(clip: vs.VideoNode, tff: bool = True, progressive: bool = True) -> List[vs.VideoNode]:
    sep = core.std.SeparateFields(clip, tff=tff)
    if progressive:
        if 'API R4.' in core.version():
            sep = core.std.RemoveFrameProps(sep, '_Field')
        else:
            sep = core.std.SetFrameProp(sep, '_Field', delete=True)
    return [sep[0::2], sep[1::2]]


def weave(clip: vs.VideoNode, clip2: Optional[vs.VideoNode] = None, progressive: bool = True) -> vs.VideoNode:
    clips = clip if clip2 is None else core.std.Interleave([clip, clip2])
    wv = core.std.DoubleWeave(clips, True)[::2]
    if progressive:
        return core.std.SetFrameProp(wv, '_FieldBased', intval=0)
    else:
        return wv


def apply_borders(clip: vs.VideoNode, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0, color: Optional[int] = None):
    crop = core.std.Crop(clip, left=left, top=top, right=right, bottom=bottom)
    return core.std.AddBorders(crop, left=left, top=top, right=right, bottom=bottom, color=color)


def get_y32(clip: vs.VideoNode) -> vs.VideoNode:
    if clip.format.color_family == vs.RGB:
        return core.resize.Spline36(clip, format=vs.GRAYS, matrix_s='709')
    else:
        return depth(get_y(clip), 32)


def join_uv(clip_y: vs.VideoNode, clip_uv: vs.VideoNode) -> vs.VideoNode:
    return core.std.ShufflePlanes([clip_y, clip_uv], [0, 1, 2], vs.YUV)
