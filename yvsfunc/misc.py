from __future__ import annotations
from typing import Callable, List, Optional, Union
import vapoursynth as vs
core = vs.core
from functools import partial

__all__ = [
    'repair',
    'bb_ref',
    'bic_blur',
    'pl_deband',
    'DescaleAA_mask',
    'non_telop_mask',
    'make_rgb_mask',
    'apply_borders',
    'neutralize_rgb_shift',
]

# https://github.com/dnjulek/jvsfunc/blob/164b02aad92d746fed2d56207868ebbe91caa38c/jvsfunc.py#L50-L66
def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: int = 1, pixel: Optional[float] = None) -> vs.VideoNode:
    if mode == 0:
        return clip
    elif mode < 0:
        return repair(repairclip, clip, -mode, pixel)
    mode_list = [1, 2, 3, 4, 11, 12, 13, 14]
    if mode not in mode_list:
        raise ValueError('repair: only modes 1-4 and 11-14 are implemented.')
    if pixel is None:
        sp = False
    else:
        pixel = abs(pixel)
        if pixel > 1.0:
            raise ValueError('repair: Shifting more than 1 pixel is not supported.')
        sp = (pixel != 1.0)
    if sp:
        p = pixel
        q = 1.0 - pixel
        pp = p * p
        qq = q * q
        pq = p * q
        plt = f'y[-1,-1] {pp} * y[-1,0] {pq} * + y[0,-1] {pq} * + y {qq} * + '
        prt = f'y[1,-1] {pp} * y[1,0] {pq} * + y[0,-1] {pq} * + y {qq} * + '
        plb = f'y[-1,1] {pp} * y[-1,0] {pq} * + y[0,1] {pq} * + y {qq} * + '
        prb = f'y[1,1] {pp} * y[1,0] {pq} * + y[0,1] {pq} * + y {qq} * + '
        pl = f'y[-1,0] {p} * y {q} * + '
        pr = f'y[1,0] {p} * y {q} * + '
        pt = f'y[0,-1] {p} * y {q} * + '
        pb = f'y[0,1] {p} * y {q} * + '
        pixels = plt + prt + plb + prb + pl + pr + pt + pb
    else:
        pixels = 'y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y[1,0] y[-1,1] y[0,1] y[1,1] '
    if mode <= 4:
        expr = f'y sort9 dup{9 - mode} max! dup{mode - 1} min! drop9 x min@ max@ clamp'
    else:
        mode = mode - 10
        expr = f'sort8 dup{8 - mode} max! dup{mode - 1} min! drop8 y min@ min ymin! y max@ max ymax! x ymin@ ymax@ clamp'

    return core.akarin.Expr([clip, repairclip], pixels + expr)


def bb_ref(clip: vs.VideoNode, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0, **bbmod_args) -> vs.VideoNode:
    from awsmfunc import bbmod
    def _bb_ref(clip: vs.VideoNode, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0, radius: int = 0, **bbmod_args) -> vs.VideoNode:
        ref = bbmod(clip, top=top, bottom=bottom, left=left, right=right, **bbmod_args)
        return core.edgefixer.Reference(clip, ref=ref, top=top, bottom=bottom, left=left, right=right, radius=radius)
    if left > 0:
        clip = _bb_ref(clip, left=left, radius=left, **bbmod_args)
    if top > 0:
        clip = _bb_ref(clip, top=top, radius=top, **bbmod_args)
    if right > 0:
        clip = _bb_ref(clip, right=right, radius=right, **bbmod_args)
    if bottom > 0:
        clip = _bb_ref(clip, bottom=bottom, radius=bottom, **bbmod_args)
    return clip


def bic_blur(clip: vs.VideoNode, b: float = 1, it: int = 1) -> vs.VideoNode:
    blur_ker = partial(core.fmtc.resample, kernel='bicubic', a1=b, a2=0, fh=-1, fv=-1)
    ret = clip
    for _ in range(it):
        ret = blur_ker(ret)
    return ret


def pl_deband(
    clip: vs.VideoNode,
    planes: List[int] = [0, 1, 2],
    it: Union[int, List[int]] = 3,
    thr: Union[int, List[int]] = 4,
    rad: Union[int, List[int]] = [20, 12],
    grain: Union[int, List[int]] = 3
) -> vs.VideoNode:
    '''
    Yet another wrapper for placebo.Deband
    '''
    def param_to_list(value: Optional[Union[int, float]], name: str) -> List[Union[int, float]]:
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f'pl_deband: input list of <{name}> is empty')
            while len(value) < 3:
                value.append(value[-1])
            return value
        else:
            return [value] * 3
    it = param_to_list(it, 'it')
    thr = param_to_list(thr, 'thr')
    rad = param_to_list(rad, 'rad')
    if clip.format.color_family in [vs.RGB, vs.GRAY] or isinstance(grain, list):
        grain = param_to_list(grain)
    else:
        # Don't add grain to chroma planes unless intended
        grain = [grain, 0, 0]
    def _deband(planeno):
        if planeno in planes:
            return core.placebo.Deband(clip.std.ShufflePlanes(planeno, vs.GRAY), planes=1, iterations=it[planeno], threshold=thr[planeno], radius=rad[planeno], grain=grain[planeno])
        else:
            return clip.std.ShufflePlanes(planeno, vs.GRAY)
    debanded = [_deband(num) for num in range(clip.format.num_planes)]
    if clip.format.num_planes == 1:
        return debanded[0]
    else:
        return core.std.ShufflePlanes(debanded, [0, 0, 0], clip.format.color_family)


def DescaleAA_mask(
    descale_diff: vs.VideoNode,
    rescaled: Optional[vs.VideoNode] = None,
    thr: Union[int, float] = 10,
    thr_lo: Optional[Union[int, float]] = None,
    thr_hi: Optional[Union[int, float]] = None,
    expand: int = 3,
    inflate: int = 3,
    pexpand: int = 0,
    pinflate: int = 0,
    show_credit: bool = False
) -> vs.VideoNode:
    '''
    The mask generated by fvsfunc.DescaleAA
    '''
    bits = descale_diff.format.bits_per_sample
    if bits == 32:
        max_val = 1.0
        thr = thr / (235 - 16)
    else:
        max_val = (1 << bits) - 1
        thr = thr * max_val // 0xFF
    diffmask = descale_diff
    for _ in range(expand):
        diffmask = core.std.Maximum(diffmask)
    for _ in range(inflate):
        diffmask = core.std.Inflate(diffmask)
    if show_credit or rescaled is None:
        return diffmask.std.Binarize(thr)
    if thr_lo is None:
        thr_lo = 4 * max_val // 0xFF
    if thr_hi is None:
        thr_hi = 24 * max_val // 0xFF
    edgemask = core.std.Prewitt(rescaled, planes=0)
    for _ in range(pexpand):
        edgemask = core.std.Maximum(edgemask)
    for _ in range(pinflate):
        edgemask = core.std.Inflate(edgemask)
    edgemask = edgemask.std.Expr("x {thrhigh} >= {maxvalue} x {thrlow} <= 0 x ? ?".format(thrhigh=thr_hi, maxvalue=max_val, thrlow=thr_lo))
    return core.std.Expr([diffmask,edgemask], 'x {thr} >= 0 y ?'.format(thr=thr)).std.Inflate().std.Deflate()


def non_telop_mask(
    src: vs.VideoNode,
    nc: vs.VideoNode,
    thr: Union[int, float] = 3,
    prefilter: Union[Callable[[vs.VideoNode], vs.VideoNode], int] = 0
) -> vs.VideoNode:
    if callable(prefilter):
        s = prefilter(src)
        n = prefilter(nc)
    elif prefilter <= 0:
        s = src
        n = nc
    elif prefilter == 1:
        return non_telop_mask(src, nc, thr=thr, prefilter=bic_blur)
    else:
        return non_telop_mask(src, nc, thr=thr, prefilter=partial(core.std.Convolution, matrix=[1] * 9))
    if isinstance(thr, list):
        if len(thr) == 0:
            raise ValueError('non_telop_mask: input list of <thr> is empty')
        while len(thr) < 3:
            thr.append(thr[-1])
        else:
            thr = [thr] * 3
    diff = core.std.Expr([s, n], 'x y - abs')
    return core.std.Binarize(diff, thr if s.format.num_planes == 3 else thr[0])


def make_rgb_mask(clip: vs.VideoNode, mask_func: Callable[[vs.VideoNode], vs.VideoNode], op: int = 1) -> vs.VideoNode:
    rgb = core.resize.Bicubic(clip, format=vs.RGB48, matrix_in_s='709')
    rgbi = core.std.Interleave(core.std.SplitPlanes(rgb))
    maski = mask_func(rgbi)
    return core.akarin.Expr([maski[0::3], maski[1::3], maski[2::3]], f'x y z sort3 dup{op} res! drop3 res@')


def apply_borders(
    clip: vs.VideoNode,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    color: Optional[int] = None
) -> vs.VideoNode:
    crop = core.std.Crop(clip, left=left, top=top, right=right, bottom=bottom)
    return core.std.AddBorders(crop, left=left, top=top, right=right, bottom=bottom, color=color)


def neutralize_rgb_shift(
    shift: list[float],
    coef: list[float] = [0.2126, 0.7152, 0.0722] # Default 709
) -> list[float]:
    tot = shift[0] * coef[0] + shift[1] * coef[1] + shift[2] * coef[2]
    return [v - tot for v in shift]
