from __future__ import annotations
from typing import List, Optional
import vapoursynth as vs
core = vs.core

__all__ = [
    'sep_fields',
    'weave',
]

def sep_fields(clip: vs.VideoNode, tff: bool = True, progressive: bool = True
) -> List[vs.VideoNode]:
    sep = core.std.SeparateFields(clip, tff=tff)
    if progressive:
        sep = core.std.RemoveFrameProps(sep, '_Field')
    return [sep[0::2], sep[1::2]]


def weave(clip: vs.VideoNode, clip2: Optional[vs.VideoNode] = None, progressive: bool = True
) -> vs.VideoNode:
    clips = clip if clip2 is None else core.std.Interleave([clip, clip2])
    wv = core.std.DoubleWeave(clips, True)[::2]
    if progressive:
        return core.std.SetFrameProp(wv, '_FieldBased', intval=0)
    else:
        return wv
