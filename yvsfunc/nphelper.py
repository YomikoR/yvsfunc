# Helper functions for numpy
import vapoursynth as vs
core = vs.core

__all__ = [
    'make_eye',
]

def make_eye(dim: int) -> vs.VideoNode:
    '''
    A square clip filled with identity matrix
    '''
    clip = core.std.BlankClip(format=vs.GRAYS, width=dim, height=dim, length=1)
    def _make_eye(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        for x in range(dim):
            fout[0][x, x] = 1.0
        return fout
    return core.std.ModifyFrame(clip, clips=clip, selector=_make_eye)
