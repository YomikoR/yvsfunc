# I don't use znedi3...
from typing import Any, Callable, Dict, List, Optional, Union
import vapoursynth as vs
from vsutil.info import get_subsampling
core = vs.core
import vsutil.types
from vsutil import depth, fallback, get_depth, join, plane, split
from functools import partial
from math import ceil, floor

from .misc import y_error_msg, repair, bic_blur
from .planes import get_y32

__all__ = [
    'ResClip',
    'descale',
    'bdescale',
    'fdescale',
    'get_nnedi3',
    'get_nnedi3cl',
    'get_eedi3',
    'get_eedi3cl',
    'nn444',
    'aa2x',
    'ee2x',
    'daa_mod',
    'aa_limit',
]

class ResClip:
    '''
    Wrapper of vs.VideoNode with cropping args for resampling. We should reduce the number of resizer calls!

    Usually, the cropping args are used to represent the effective region of the frame.
    '''
    def __init__(self, clip: vs.VideoNode, sx: float = 0, sy: float = 0, sw: Optional[float] = None, sh: Optional[float] = None):
        self.clip = clip
        self.sx = sx
        self.sy = sy
        self.sw = fallback(sw, clip.width)
        self.sh = fallback(sh, clip.height)

    def copy(self, clip: Optional[vs.VideoNode] = None, sx: Optional[float] = None, sy: Optional[float] = None, sw: Optional[float] = None, sh: Optional[float] = None) -> 'ResClip':
        '''
        Copy with substitutions
        '''
        clip = fallback(clip, self.clip)
        sx = fallback(sx, self.sx)
        sy = fallback(sy, self.sy)
        sw = fallback(sw, self.sw)
        sh = fallback(sh, self.sh)
        return ResClip(clip, sx=sx, sy=sy, sw=sw, sh=sh)

    def width(self) -> int:
        return self.clip.width

    def height(self) -> int:
        return self.clip.height

    def make_fmtc_dict(self) -> Dict[str, float]:
        return dict(
            sx = self.sx,
            sy = self.sy,
            sw = self.sw,
            sh = self.sh,
        )

    def make_resize_dict(self) -> Dict[str, float]:
        return dict(
            src_left = self.sx,
            src_top = self.sy,
            src_width = self.sw,
            src_height = self.sh,
        )

    def transpose(self):
        self.clip = core.std.Transpose(self.clip)
        self.sx, self.sy = self.sy, self.sx
        self.sw, self.sh = self.sh, self.sw

    def crop(self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return
        self.clip = core.std.Crop(self.clip, left=left, right=right, top=top, bottom=bottom)
        self.sx = self.sx - left
        self.sy = self.sy - top

    def show(self) -> vs.VideoNode:
        return core.text.Text(self.clip, f'sx={self.sx}\nsy={self.sy}\nsw={self.sw}\nsh={self.sh}')

    def bicubic(self, width: int, height: int, b: float = 0, c: float = 0.5, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height, filter_param_a=b, filter_param_b=c))
        return core.resize.Bicubic(self.clip, **args)

    def hermite(self, width: int, height: int, **resizer_args) -> vs.VideoNode:
        return self.bicubic(width, height, b=0, c=0, **resizer_args)

    def spline36(self, width: int, height: int, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height))
        return core.resize.Spline36(self.clip, **args)


### Descale wrappers
#   For two typical cases in anime upscaling:
#       upscaled with non-reflection border handling causing ringing (use bdescale)
#       upscaled and cropped to kill border ringing (use fdescale)


def descale(clip: Union[ResClip, vs.VideoNode], width: int = 1280, height: int = 720, kernel: str = 'bicubic', b: float = 0, c: float = 0.5, taps: int = 3, src_left: float = 0, src_top: float = 0, src_width: Optional[float] = None, src_height: Optional[float] = None, with_diff: bool = False) -> Union[ResClip, List[ResClip]]:
    '''
    A descale wrapper that also returns rescaling error with `with_diff=True`
    '''
    func_name = 'descale'

    if isinstance(clip, vs.VideoNode):
        clip = ResClip(clip)
    src_width = fallback(src_width, width)
    src_height = fallback(src_height, height)

    ratio_w = src_width / clip.width()
    ratio_h = src_height / clip.height()

    dst_sx = src_left + ratio_w * clip.sx
    dst_sw =            ratio_w * clip.sw
    dst_sy = src_top  + ratio_h * clip.sy
    dst_sh =            ratio_h * clip.sh

    clip_depth = get_depth(clip.clip)
    y32 = get_y32(clip.clip)
    kernel = kernel.lower()
    if kernel.startswith('de'):
        kernel = kernel[2:]

    descaler = _get_descaler(func_name, kernel, b, c, taps)
    down = descaler(y32, width, height, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height)
    if with_diff:
        scaler = _get_scaler(func_name, kernel, b, c, taps)
        up = scaler(down, y32.width, y32.height, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height)
        diff = core.std.Expr([y32, up], 'x y - abs')
        if clip_depth != 32:
            down = depth(down, clip_depth)
            diff = depth(diff, clip_depth, range_in=vsutil.types.Range.LIMITED, range=vsutil.types.Range.FULL)
        return [
            ResClip(down, dst_sx, dst_sy, dst_sw, dst_sh),
            ResClip(diff, clip.sx, clip.sy, clip.sw, clip.sh)
        ]
    else:
        return ResClip(depth(down, clip_depth), dst_sx, dst_sy, dst_sw, dst_sh)


def bdescale(clip: Union[ResClip, vs.VideoNode], width: int = 1280, height: int = 720, kernel: str = 'bicubic', b: float = 0, c: float = 0.5, taps: int = 3, src_left: float = 0, src_top: float = 0, src_width: Optional[float] = None, src_height: Optional[float] = None, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0, color: Optional[float] = None, with_diff: bool = False) -> Union[vs.VideoNode, List[vs.VideoNode]]:
    '''
    Descale with (black) borders added internally to suppress border ringing. \\
    Cropping args are adjusted accordingly by border inputs, but NOT by the effective resolution. \\
    Correspondingly, cropping args of the output ResClips are invalid. Hence just returned as `vs.VideoNode`.
    '''
    if left == 0 and top == 0 and right == 0 and bottom == 0:
        if with_diff:
            down, diff = descale(clip, width=width, height=height, kernel=kernel, b=b, c=c, taps=taps, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height, with_diff=with_diff)
            return [down.clip, diff.clip]
        else:
            down = descale(clip, width=width, height=height, kernel=kernel, b=b, c=c, taps=taps, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height, with_diff=with_diff)
            return down.clip

    src_width = fallback(src_width, width)
    src_height = fallback(src_height, height)

    if isinstance(clip, vs.VideoNode):
        y = ResClip(plane(clip, 0))
    else:
        y = ResClip(plane(clip.clip, 0))

    ratio_w = src_width / y.width()
    ratio_h = src_height / y.height()

    # The minimal number of extra pixels on the left
    extra_left = left * ratio_w - src_left
    extra_left_int = ceil(extra_left)
    if extra_left_int < 0:
        # Negative values are not allowed
        extra_left -= extra_left_int
        extra_left_int = 0
    # On the right
    extra_right = right * ratio_w + src_left + src_width - width
    extra_right_int = ceil(extra_right)
    if extra_right_int < 0:
        # We don't adjust extra_right because it's not used for cropping
        extra_right_int = 0
    # On the top
    extra_top = top * ratio_h - src_top
    extra_top_int = ceil(extra_top)
    if extra_top_int < 0:
        extra_top -= extra_top_int
        extra_top_int = 0
    # On the bottom
    extra_bottom = bottom * ratio_h + src_top + src_height - height
    extra_bottom_int = ceil(extra_bottom)
    if extra_bottom_int < 0:
        extra_bottom_int = 0

    # Descale
    descale_cropping_args = dict(
        src_left = extra_left_int - extra_left,
        src_width = extra_left + extra_right + width,
        width = extra_left_int + extra_right_int + width,
        src_top = extra_top_int - extra_top,
        src_height = extra_top + extra_bottom + height,
        height = extra_top_int + extra_bottom_int + height,
    )
    descale_border_args = dict(
        left = extra_left_int,
        right = extra_right_int,
        top = extra_top_int,
        bottom = extra_bottom_int,
    )
    src_border_args = dict(
        left = left,
        right = right,
        top = top,
        bottom = bottom,
    )
    y.clip = core.std.AddBorders(y.clip, color=color, **src_border_args)
    if with_diff:
        down, diff = descale(y, b=b, c=c, taps=taps, with_diff=True, **descale_cropping_args)
        down.crop(**descale_border_args)
        diff.crop(**src_border_args)
        return [down.clip, diff.clip]
    else:
        down = descale(y, b=b, c=c, taps=taps, with_diff=False, **descale_cropping_args)
        down.crop(**descale_border_args)
        return down.clip


def fdescale(clip: Union[ResClip, vs.VideoNode], ratio: float, base_width: Optional[int] = None, base_height: Optional[int] = None, kernel: str = 'bicubic', b: float = 0, c: float = 0.5, taps: int = 3, src_left: float = 0, src_top: float = 0, src_width: Optional[float] = None, src_height: Optional[float] = None, with_diff: bool = False) -> Union[ResClip, List[ResClip]]:
    '''
    Descale with a certain ratio, assuming
        1) effective region is the entire input (existing cropping args are discarded) \\
        2) cropped in a symmetric way

    We need at least the parities of base_width and base_height. Default values are multiples of 16 and 18, respectively. A good starting point in practice may be 1536x864.
    '''
    if ratio <= 0 or ratio >= 1:
        y_error_msg('fdescale', 'ratio must be within (0, 1)')
    if isinstance(clip, ResClip):
        clip = clip.clip
    y = ResClip(plane(clip, 0)) # NOTE cropping cleared
    src_width = y.width() * ratio
    src_height = y.height() * ratio
    base_width = fallback(base_width, ceil(src_width / 16) * 16)
    base_height = fallback(base_height, ceil(src_height / 18) * 18)
    # Strip
    width = base_width - 2 * floor((base_width - src_width) / 2)
    height = base_height - 2 * floor((base_height - src_height) / 2)
    src_left = (width - src_width) / 2
    src_top = (height - src_height) / 2
    return descale(y, width=width, height=height, kernel=kernel, b=b, c=c, taps=taps, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height, with_diff=with_diff)


### AA related wrappers


def get_nnedi3(opencl: bool = False, **nnedi3_args: Any) -> Callable[..., vs.VideoNode]:
    args = dict(nsize=4, nns=4, qual=2, pscrn=1)
    args.update(nnedi3_args)
    return partial(core.nnedi3cl.NNEDI3CL if opencl else core.nnedi3.nnedi3, **args)


def get_nnedi3cl(**nnedi3_args: Any) -> Callable[..., vs.VideoNode]:
    return get_nnedi3(opencl=True, **nnedi3_args)


def get_eedi3(opencl: bool = False, **eedi3_args: Any) -> Callable[..., vs.VideoNode]:
    args = dict(alpha=0.25, beta=0.25, gamma=40, nrad=2, mdis=20, vthresh0=12, vthresh1=24, vthresh2=4)
    args.update(eedi3_args)
    return partial(core.eedi3m.EEDI3CL if opencl else core.eedi3m.EEDI3, **args)


def get_eedi3cl(**eedi3_args) -> Callable[..., vs.VideoNode]:
    return get_eedi3(opencl=True, **eedi3_args)


def nn444(clip, opencl: bool = False, **nnedi3_args: Any) -> vs.VideoNode:
    '''
    Use nnedi3 to upscale chroma planes. Only works for MPEG2 inputs.
    '''
    func_name = 'nn444'
    if clip.format.color_family != vs.YUV:
        y_error_msg(func_name, 'format not supported')
    if get_subsampling(clip) == '420':
        nnedi3 = get_nnedi3(opencl=opencl, **nnedi3_args)
        y, u, v = split(clip)
        # Interleave to save number of filter calls
        uv = core.std.Interleave([u, v])
        uv_up1 = nnedi3(uv.std.Transpose(), field=1, dh=True)
        uv_up2 = nnedi3(uv_up1.std.Transpose(), field=0, dh=True).resize.Spline36(src_top=0.5)
        return join([y, uv_up2[0::2], uv_up2[1::2]])
    elif get_subsampling(clip) == '422':
        nnedi3 = get_nnedi3(opencl=opencl, **nnedi3_args)
        y, u, v = split(clip)
        uv = core.std.Interleave([u, v])
        uv_up = nnedi3(uv.std.Transpose(), field=1, dh=True).std.Transpose()
        return join([y, uv_up[0::2], uv_up[1::2]])
    else:
        y_error_msg(func_name, 'format not supported')


def interpolate(clip: ResClip, nnedi3: Optional[Callable[..., vs.VideoNode]] = None, eedi3: Optional[Callable[..., vs.VideoNode]] = None, field: int = 1, dh: bool = True, with_nn: bool = False) -> Union[ResClip, List[ResClip]]:
    # Using nnedi3 by default
    deint: Optional[Callable[..., vs.VideoNode]] = None
    if nnedi3 is None:
        if eedi3 is None:
            deint = get_nnedi3()
        else:
            deint = eedi3
    elif eedi3 is None:
        deint = nnedi3
    # Adjust output cropping args
    if dh:
        sh = clip.sh * 2
        sy = clip.sy * 2 + 0.5 - field
    else:
        sh = clip.sh
        sy = clip.sy

    if deint is None: # corresponding to nnedi3 + eedi3
        ip_nn = nnedi3(clip.clip, field=field, dh=dh)
        ip_ee = eedi3(clip.clip, field=field, dh=dh, vcheck=3, sclip=ip_nn)
        if with_nn:
            return [clip.copy(ip_ee, sy=sy, sh=sh), clip.copy(ip_nn, sy=sy, sh=sh)]
        else:
            return clip.copy(ip_ee, sy=sy, sh=sh)
    else:
        ip = deint(clip.clip, field=field, dh=dh)
        return clip.copy(ip, sy=sy, sh=sh)


def aa2x(clip: Union[ResClip, vs.VideoNode], nnedi3: Optional[Callable[..., vs.VideoNode]] = None, eedi3: Optional[Callable[..., vs.VideoNode]] = None) -> ResClip:
    '''
    The usual 2x filter with nnedi3 and/or eedi3.
    '''
    if isinstance(clip, vs.VideoNode):
        clip = ResClip(clip)
    up1 = interpolate(clip, nnedi3, eedi3)
    up1.transpose()
    up2 = interpolate(up1, nnedi3, eedi3)
    up2.transpose()
    return up2


def ee2x(clip: Union[ResClip, vs.VideoNode], nnedi3: Optional[Callable[..., vs.VideoNode]] = None, eedi3: Optional[Callable[..., vs.VideoNode]] = None, with_nn: bool = False) -> Union[ResClip, List[ResClip]]:
    '''
    Compared to `aa2x`, here we use the nnedi3 result for the vcheck of the second interpolation. \\
    Set `with_nn=True` to return the nnedi3 result as well.
    '''
    if nnedi3 is None or eedi3 is None:
        y_error_msg('ee2x', 'both nnedi3 and eedi3 must be available')
    if isinstance(clip, vs.VideoNode):
        clip = ResClip(clip)
    # First interpolation
    nn_up1, ee_up1 = interpolate(clip, nnedi3=nnedi3, eedi3=eedi3, with_nn=True)
    nn_up1.transpose()
    ee_up1.transpose()
    # Second interpolation
    nn_up2 = interpolate(nn_up1, field=1, dh=True, nnedi3=nnedi3)
    ee_up2 = nn_up2.copy(eedi3(ee_up1.clip, field=1, dh=True, vcheck=3, sclip=nn_up2.clip))
    ee_up2.transpose()
    # Output
    if with_nn:
        nn_up2.transpose()
        return [ee_up2, nn_up2]
    else:
        return ee_up2


def daa_mod(clip: vs.VideoNode, ref: Optional[vs.VideoNode] = None, opencl: bool = False, b: float = 1, rep1: int = 0, px1: Optional[float] = None, rep2: int = 13, px2: Optional[float] = None) -> vs.VideoNode:
    '''
    Modified from `havsfunc.daa` reducing its default strength. \\
    Sometimes it fixes residual interlacing (with some detail loss). Try `ref=TFM(PP=5)`.
    '''
    nnedi3 = get_nnedi3(opencl=opencl)
    nn = nnedi3(clip, field=3)
    nn0 = repair(nn[0::2], clip, mode=rep1, pixel=px1)
    nn1 = repair(nn[1::2], clip, mode=rep1, pixel=px1)
    dbl = core.std.Merge(nn0, nn1)
    dblD = core.std.MakeDiff(ref if ref is not None else clip, dbl)
    blrD = bic_blur(dbl, b=b)
    shrpD = core.std.MakeDiff(dbl, blrD)
    DD = repair(shrpD, dblD, mode=rep2, pixel=px2)
    return core.std.MergeDiff(dbl, DD)


def aa_limit(ref: vs.VideoNode, strong: vs.VideoNode, weak: vs.VideoNode, **lim_args: Any):
    '''
    Limiting the results of AA
    '''
    import mvsfunc as mvf
    args = dict(thr=2, elast=4)
    args.update(lim_args)
    lim = mvf.LimitFilter(strong, weak, **args)
    return core.std.Expr([strong, weak, ref, lim], 'x z - y z - xor y a ?')


### Descale helpers


def _get_descaler(func_name: str, kernel: str, b: float, c: float, taps: int) -> Callable[[Any], vs.VideoNode]:
    if kernel == 'bilinear':
        return core.descale.Debilinear
    elif kernel == 'bicubic':
        return partial(core.descale.Debicubic, b=b, c=c)
    elif kernel == 'lanczos':
        return partial(core.descale.Delanczos, taps=taps)
    elif kernel == 'spline16':
        return core.descale.Despline16
    elif kernel == 'spline36':
        return core.descale.Despline36
    elif kernel == 'spline64':
        return core.descale.Despline64
    else:
        y_error_msg(func_name, 'invalid kernel specified')


def _get_scaler(func_name: str, kernel: str, b: float, c: float, taps: int) -> Callable[[Any], vs.VideoNode]:
    if kernel == 'bilinear':
        return core.resize.Bilinear
    elif kernel == 'bicubic':
        return partial(core.resize.Bicubic, filter_param_a=b, filter_param_b=c)
    elif kernel == 'lanczos':
        return partial(core.resize.Lanczos, filter_param_a=taps)
    elif kernel == 'spline16':
        return core.resize.Spline16
    elif kernel == 'spline36':
        return core.resize.Spline36
    elif kernel == 'spline64':
        return core.resize.Spline64
    else:
        y_error_msg(func_name, 'invalid kernel specified')
