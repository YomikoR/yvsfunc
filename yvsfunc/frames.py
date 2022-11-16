from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union
import vapoursynth as vs
core = vs.core
import sys
from functools import partial
from itertools import accumulate
from fractions import Fraction

__all__ = [
    'frame_time',
    'qp_read',
    'qp_splice',
    'join_clips',
    'cut_clips',
    'tdec',
    'mdec',
    'clip_time',
    'gen_timestamps',
]

def frame_time(n24: int = 0, n30: int = 0, n60: int = 0, n120: int = 0, n18: int = 0, print_output: bool = False
) -> Union[str, int]:
    '''
    Calculate time according to frame rates. \\
    TIVTC occasionally creates 3 consecutive 18 fps frames, so included as well.
    '''
    t24 = 1001 / 24
    t30 = 1001 / 30
    t60 = 1001 / 60
    t18 = 1001 / 18
    t120 = 1001 / 120
    t = t24 * n24 + t30 * n30 + t60 * n60 + t18 * n18 + t120 * n120 # in ms
    t = round(t) # in rounded ms
    if print_output:
        # h:mm:ss:ms
        ms = t % 1000
        s = (t // 1000) % 60
        m = (t // 60000) % 60
        h = (t // 60000) // 60
        return f'{h}:{m:02}:{s:02}:{ms:03}'
    else:
        return t


def qp_read(qpfile: str, suffix: str = 'IK') -> List[int]:
    '''
    Read qpfile and return a list of frame numbers
    '''
    frame_nums = []
    with open(qpfile, 'r') as qpf:
        lines = qpf.readlines()
        for line in lines:
            line_spl = line.split(' ')
            if len(line_spl) > 1 and line_spl[1][0] in suffix:
                try:
                    fno = int(line_spl[0])
                except:
                    pass
                frame_nums.append(fno)
    return frame_nums


def qp_splice(clips: List[vs.VideoNode], qp_output: str, suffix: str = 'K'):
    '''
    Use together with mvsfunc.VFRSplice
    '''
    frame_num = 0
    with open(qp_output, 'w') as qpf:
        for clip in clips:
            qpf.write(f'{frame_num} ' + suffix + '\n')
            frame_num += len(clip)


def join_clips(clips: List[vs.VideoNode]) -> Tuple[vs.VideoNode, List[int]]:
    '''
    Input a list of n clips, \\
    output the concatenated clip and a list of starting and ending frame numbers with length (n+1). \\
    Used for saving hardware resources in scenefiltering.
    '''
    joined = core.std.Splice(clips)
    frame_nums = [0] + [len(clip) for clip in clips]
    return joined, list(accumulate(frame_nums))


def cut_clips(clip: vs.VideoNode, frame_nums: List[int]) -> List[vs.VideoNode]:
    '''
    This reverts join_clips.

    Usage:
        clip, index = join_clips([clipa, clipb]) \\
        clip = filter(clip) \\
        clipa, clipb = cut_clips(clip, index)
    '''
    return [clip[frame_nums[j]:frame_nums[j + 1]] for j in range(len(frame_nums) - 1)]


def tdec(clip: vs.VideoNode, **args: Any) -> vs.VideoNode:
    '''
    TDecimate wrapper that replaces combed frames by its successor
    '''
    tdec_args = dict(mode=1)
    tdec_args.update(args)
    iv = core.tivtc.TDecimate(clip, **tdec_args)
    iv_next = iv[1:] + iv[-1]
    def _replace_next(n: int, f: vs.VideoFrame, c: vs.VideoNode, c2: vs.VideoNode) -> vs.VideoNode:
        return c2 if f.props['_Combed'] > 0 else c
    return core.std.FrameEval(iv, partial(_replace_next, c=iv, c2=iv_next), iv)


def mdec(clip: vs.VideoNode, drop: Union[int, List[int]], cycle: int = 5, modify_duration: Optional[bool] = None
) -> vs.VideoNode:
    '''
    Manual decimation
    '''
    if isinstance(drop, int):
        drop = [drop]
    keep = list(range(cycle))
    for d in drop:
        keep.remove(d % cycle)
    return core.std.SelectEvery(clip, cycle=cycle, offsets=keep, modify_duration=modify_duration)


def clip_time(clip: vs.VideoNode, default_duration: float = 1001 / 24) -> float:
    '''
    Total time of clip in ms
    '''
    total_ms: float = 0
    for f in clip.frames(close=True):
        try:
            dur_num = f.props['_DurationNum']
            dur_den = f.props['_DurationDen']
            total_ms += 1000 * dur_num / dur_den
        except KeyError:
            total_ms += default_duration
    return total_ms


def gen_timestamps(clip: vs.VideoNode, output_file: str, fallback_fps_num: int = 30000, fallback_fps_den: int = 1001) -> None:
    '''
    Generate timestamps by reading all the frames.
    '''
    durations: List[Fraction] = []
    num_frames = len(clip)

    print('Reading frames for timestamps creation...', file=sys.stderr)
    for n, f in enumerate(clip.frames(close=True)):
        print(f'\r{n + 1}/{num_frames}', end='', file=sys.stderr)
        try:
            dur_num = f.props['_DurationNum']
            dur_den = f.props['_DurationDen']
            durations.append(Fraction(dur_num, dur_den))
        except KeyError:
            durations.append(Fraction(fallback_fps_den, fallback_fps_num))
    print('\nDone.', file=sys.stderr)

    with open(output_file, 'w') as outf:
        outf.write('# timestamp format v2\n')
        for ts in accumulate(durations):
            outf.write(f'{round(ts * 1000)}\n')
