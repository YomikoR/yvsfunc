from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import vapoursynth as vs
core = vs.core
from functools import partial
from math import ceil, floor, sqrt

from .misc import repair, bic_blur

__all__ = [
    'ResClip',
    'descale',
    'bdescale',
    'fdescale',
    'get_nnedi3',
    'get_nnedi3cl',
    'get_eedi3',
    'get_eedi3cl',
    'intra_aa',
    'aa2x',
    'ee2x',
    'daa_mod',
    'aa_limit',
    'rgb2opp',
    'opp2rgb',
    'xyz2lab',
    'lab2xyz',
    'lab2labview',
    'labview2lab',
    'deltaE76',
    'deltaE94',
]

class ResClip:
    '''
    Wrapper of vs.VideoNode with cropping args for resampling. We should reduce the number of resizer calls!

    Usually, the cropping args are used to represent the effective region of the frame.
    '''
    def __init__(self,
        clip: vs.VideoNode,
        sx: float = 0,
        sy: float = 0,
        sw: Optional[float] = None,
        sh: Optional[float] = None
    ):
        self.clip = clip
        self.sx = sx
        self.sy = sy
        self.sw = clip.width if sw is None else sw
        self.sh = clip.height if sh is None else sh

    def copy(
        self,
        clip: Optional[vs.VideoNode] = None,
        sx: Optional[float] = None,
        sy: Optional[float] = None,
        sw: Optional[float] = None,
        sh: Optional[float] = None
    ) -> ResClip:
        '''
        Copy with substitutions
        '''
        clip = self.clip if clip is None else clip
        sx = self.sx if sx is None else sx
        sy = self.sy if sy is None else sy
        sw = self.sw if sw is None else sw
        sh = self.sh if sh is None else sh
        return ResClip(clip, sx=sx, sy=sy, sw=sw, sh=sh)

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

    def transpose(self, inplace: bool = True):
        if inplace:
            self.clip = core.std.Transpose(self.clip)
            self.sx, self.sy = self.sy, self.sx
            self.sw, self.sh = self.sh, self.sw
        else:
            ret = self.copy()
            ret.transpose()
            return ret

    def crop(self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0, inplace: bool = True):
        if inplace:
            if left == 0 and right == 0 and top == 0 and bottom == 0:
                return
            self.clip = self.std.Crop(left=left, right=right, top=top, bottom=bottom)
            self.sx = self.sx - left
            self.sy = self.sy - top
        else:
            ret = self.copy()
            ret.crop(left=left, right=right, top=top, bottom=bottom)
            return ret

    def show(self) -> vs.VideoNode:
        return self.text.Text(f'sx={self.sx}\nsy={self.sy}\nsw={self.sw}\nsh={self.sh}')

    def bicubic(self, width: int, height: int, b: float = 0, c: float = 0.5, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height, filter_param_a=b, filter_param_b=c))
        return self.resize.Bicubic(**args)

    def hermite(self, width: int, height: int, **resizer_args) -> vs.VideoNode:
        return self.bicubic(width, height, b=0, c=0, **resizer_args)

    def spline36(self, width: int, height: int, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height))
        return self.resize.Spline36(**args)

    def spline64(self, width: int, height: int, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height))
        return self.resize.Spline64(**args)

    def lanczos(self, width: int, height: int, taps: int = 3, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height, filter_param_a=taps))
        return self.resize.Lanczos(**args)

    def bilinear(self, width: int, height: int, **resizer_args) -> vs.VideoNode:
        args = self.make_resize_dict()
        args.update(resizer_args)
        args.update(dict(width=width, height=height))
        return self.resize.Bilinear(**args)

    def double(self, clip2x: vs.VideoNode, inplace: bool = True):
        '''
        Use it for centered doubling, e.g. Waifu2x
        '''
        if inplace:
            self.clip = clip2x
            self.sx *= 2
            self.sy *= 2
            self.sw *= 2
            self.sh *= 2
        else:
            ret = self.copy()
            ret.double(clip2x)
            return ret

    def __getattr__(self, name: str):
        try:
            return getattr(self.clip, name)
        except AttributeError:
            raise AttributeError(f'Attribute {name} is not found in ResClip.')

### Descale wrappers
#   For two typical cases in anime upscaling:
#       upscaled with non-reflection border handling causing ringing (use bdescale)
#       upscaled and cropped to kill border ringing (use fdescale)


def descale(
    clip: Union[ResClip, vs.VideoNode],
    width: int = 1280,
    height: int = 720,
    kernel: str = 'bicubic',
    b: float = 0,
    c: float = 0.5,
    taps: int = 3,
    src_left: float = 0,
    src_top: float = 0,
    src_width: Optional[float] = None,
    src_height: Optional[float] = None,
    with_diff: bool = False
) -> Union[ResClip, List[ResClip]]:
    '''
    A descale wrapper that also returns rescaling error with `with_diff=True`
    '''
    if isinstance(clip, vs.VideoNode):
        clip = ResClip(clip)
    if src_width is None:
        src_width = width
    if src_height is None:
        src_height = height

    ratio_w = src_width / clip.width
    ratio_h = src_height / clip.height

    dst_sx = src_left + ratio_w * clip.sx
    dst_sw =            ratio_w * clip.sw
    dst_sy = src_top  + ratio_h * clip.sy
    dst_sh =            ratio_h * clip.sh

    clip_depth = clip.clip.format.bits_per_sample
    y32 = clip.std.ShufflePlanes(0, vs.GRAY).fmtc.bitdepth(bits=32)
    kernel = kernel.lower()
    if kernel.startswith('de'):
        kernel = kernel[2:]

    descaler = _get_descaler(kernel, b, c, taps)
    down = descaler(y32, width, height, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height)
    if with_diff:
        scaler = _get_scaler(kernel, b, c, taps)
        up = scaler(down, y32.width, y32.height, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height)
        diff = core.std.Expr([y32, up], 'x y - abs')
        if clip_depth != 32:
            down = down.fmtc.bitdepth(bits=32)
            diff = diff.fmtc.bitdepth(bits=32, fulls=False, fulld=True)
        return [
            ResClip(down, dst_sx, dst_sy, dst_sw, dst_sh),
            ResClip(diff, clip.sx, clip.sy, clip.sw, clip.sh)
        ]
    else:
        return ResClip(down.fmtc.bitdepth(bits=clip_depth), dst_sx, dst_sy, dst_sw, dst_sh)


def bdescale(
    clip: Union[ResClip, vs.VideoNode],
    width: int = 1280,
    height: int = 720,
    kernel: str = 'bicubic',
    b: float = 0,
    c: float = 0.5,
    taps: int = 3,
    src_left: float = 0,
    src_top: float = 0,
    src_width: Optional[float] = None,
    src_height: Optional[float] = None,
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    color: Optional[float] = None,
    with_diff: bool = False
) -> Union[vs.VideoNode, List[vs.VideoNode]]:
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

    if src_width is None:
        src_width = width
    if src_height is None:
        src_height = height

    y = ResClip(clip.std.ShufflePlanes(0, vs.GRAY))

    ratio_w = src_width / y.width
    ratio_h = src_height / y.height

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
    y.clip = y.std.AddBorders(color=color, **src_border_args)
    if with_diff:
        down, diff = descale(y, kernel=kernel, b=b, c=c, taps=taps, with_diff=True, **descale_cropping_args)
        down.crop(**descale_border_args)
        diff.crop(**src_border_args)
        return [down.clip, diff.clip]
    else:
        down = descale(y, kernel=kernel, b=b, c=c, taps=taps, with_diff=False, **descale_cropping_args)
        down.crop(**descale_border_args)
        return down.clip


def fdescale(
    clip: Union[ResClip, vs.VideoNode],
    ratio: float,
    base_width: Optional[int] = None,
    base_height: Optional[int] = None,
    kernel: str = 'bicubic',
    b: float = 0,
    c: float = 0.5,
    taps: int = 3,
    src_left: float = 0,
    src_top: float = 0,
    src_width: Optional[float] = None,
    src_height: Optional[float] = None,
    with_diff: bool = False
    ) -> Union[ResClip, List[ResClip]]:
    '''
    Descale with a certain ratio, assuming
        1) effective region is the entire input (existing cropping args are discarded) \\
        2) cropped in a symmetric way

    We need at least the parities of base_width and base_height. Default values are multiples of 16 and 18, respectively. A good starting point in practice may be 1536x864.
    '''
    if ratio <= 0 or ratio >= 1:
        raise ValueError('fdescale: ratio must be within (0, 1)')
    y = ResClip(clip.std.ShufflePlanes(0, vs.GRAY)) # NOTE cropping cleared
    src_width = y.width * ratio
    src_height = y.height * ratio
    if base_width is None:
        base_width = ceil(src_width / 16) * 16
    if base_height is None:
        base_height = ceil(src_height / 18) * 18
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
    return partial(core.nnedi3cl.NNEDI3CL if opencl else core.znedi3.nnedi3, **args)


def get_nnedi3cl(**nnedi3_args: Any) -> Callable[..., vs.VideoNode]:
    return get_nnedi3(opencl=True, **nnedi3_args)


def get_eedi3(opencl: bool = False, **eedi3_args: Any) -> Callable[..., vs.VideoNode]:
    args = dict(alpha=0.25, beta=0.25, gamma=40, nrad=2, mdis=20, vthresh0=12, vthresh1=24, vthresh2=4)
    args.update(eedi3_args)
    return partial(core.eedi3m.EEDI3CL if opencl else core.eedi3m.EEDI3, **args)


def get_eedi3cl(**eedi3_args) -> Callable[..., vs.VideoNode]:
    return get_eedi3(opencl=True, **eedi3_args)


def interpolate(
    clip: Union[ResClip, vs.VideoNode],
    nnedi3: Optional[Callable[..., vs.VideoNode]] = None,
    eedi3: Optional[Callable[..., vs.VideoNode]] = None,
    field: Optional[int] = None,
    dh: bool = True,
    with_nn: bool = False
    ) -> Union[ResClip, List[ResClip]]:
    if isinstance(clip, vs.VideoNode):
        clip = ResClip(clip)
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
        if field is None:
            field = 1 if clip.sy > 0 else 0
        sh = clip.sh * 2
        sy = clip.sy * 2 + 0.5 - field
    else:
        if field is None:
            field = 1
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


def intra_aa(
    clip: vs.VideoNode,
    nnedi3: Optional[Callable[..., vs.VideoNode]] = None,
    eedi3: Optional[Callable[..., vs.VideoNode]] = None
) -> vs.VideoNode:
    clip = ResClip(clip)
    aa0 = interpolate(clip, nnedi3, eedi3, field=0, dh=False).clip
    aa1 = interpolate(clip, nnedi3, eedi3, field=1, dh=False).clip
    clip.transpose()
    aa2 = interpolate(clip, nnedi3, eedi3, field=0, dh=False).std.Transpose()
    aa3 = interpolate(clip, nnedi3, eedi3, field=1, dh=False).std.Transpose()
    return core.akarin.Expr([aa0, aa1, aa2, aa3], 'x y z a sort4 dup1 r1! dup2 r2! drop4 r1@ r2@ + 2 /')


def aa2x(
    clip: Union[ResClip, vs.VideoNode],
    nnedi3: Optional[Callable[..., vs.VideoNode]] = None,
    eedi3: Optional[Callable[..., vs.VideoNode]] = None
) -> ResClip:
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


def ee2x(
    clip: Union[ResClip, vs.VideoNode],
    nnedi3: Optional[Callable[..., vs.VideoNode]] = None,
    eedi3: Optional[Callable[..., vs.VideoNode]] = None,
    with_nn: bool = False
) -> Union[ResClip, List[ResClip]]:
    '''
    Compared to `aa2x`, here we use the nnedi3 result for the vcheck of the second interpolation. \\
    Set `with_nn=True` to return the nnedi3 result as well.
    '''
    if nnedi3 is None or eedi3 is None:
        raise ValueError('ee2x: both nnedi3 and eedi3 must be available')
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


def daa_mod(
    clip: vs.VideoNode,
    ref: Optional[vs.VideoNode] = None,
    opencl: bool = False,
    b: float = 1,
    rep1: int = 0,
    px1: Optional[float] = None,
    rep2: int = 13,
    px2: Optional[float] = None
) -> vs.VideoNode:
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


def aa_limit(ref: vs.VideoNode, strong: vs.VideoNode, weak: vs.VideoNode, **lim_args: Any
) -> vs.VideoNode:
    '''
    Limiting the results of AA
    '''
    import mvsfunc as mvf
    args = dict(thr=2, elast=4)
    args.update(lim_args)
    lim = mvf.LimitFilter(strong, weak, **args)
    return core.std.Expr([strong, weak, ref, lim], 'x z - y z - xor y a ?')


def rgb2opp(clip: vs.VideoNode, normalize: bool = False) -> vs.VideoNode:
    ''' Set normalize=True if assuming the Gaussian noise on R, G and B planes are iid, and the noise levels in channels of the output are to be the same
        Otherwise output is linearly scaled to fit into common YUV ranges
    '''
    assert clip.format.id == vs.RGBS
    if normalize:
        coef = [1/3, 1/3, 1/3, 0, 1/sqrt(6), -1/sqrt(6), 0, 0, 1/sqrt(18), 1/sqrt(18), -2/sqrt(18), 0]
    else:
        coef = [1/3, 1/3, 1/3, 0, 1/2, -1/2, 0, 0, 1/4, 1/4, -1/2, 0]
    opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.YUV, coef=coef)
    opp = core.std.SetFrameProps(opp, _Matrix=vs.MATRIX_UNSPECIFIED, BM3D_OPP=1)
    return opp


def opp2rgb(clip: vs.VideoNode, normalize: bool = False) -> vs.VideoNode:
    assert clip.format.id == vs.YUV444PS
    if normalize:
        coef = [1, sqrt(3/2), 1/sqrt(2), 0, 1, -sqrt(3/2), 1/sqrt(2), 0, 1, 0, -sqrt(2), 0]
    else:
        coef = [1, 1, 2/3, 0, 1, -1, 2/3, 0, 1, 0, -4/3, 0]
    rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.RGB, coef=coef)
    rgb = core.std.SetFrameProps(rgb, _Matrix=vs.MATRIX_RGB)
    rgb = core.std.RemoveFrameProps(rgb, 'BM3D_OPP')
    return rgb


def xyz2lab(clip: vs.VideoNode):
    assert clip.format.color_family == vs.RGB
    clip = core.fmtc.bitdepth(clip, bits=32)
    r, g, b = core.std.SplitPlanes(clip)
    r = core.std.Expr(r, 'x 0.9642119944 /')
    b = core.std.Expr(b, 'x 0.8251882845 /')
    clip = core.std.ShufflePlanes([r, g, b], [0, 0, 0], vs.RGB)
    clip = core.fmtc.transfer(clip, transs='linear', transd='lstar')
    clip = core.fmtc.matrix(clip, coef=[0, 100, 0, 0, 12500/29, -12500/29, 0, 0, 0, 5000/29, -5000/29, 0])
    return clip


def lab2xyz(clip: vs.VideoNode):
    assert clip.format.id == vs.RGBS
    clip = core.fmtc.matrix(clip, coef=[0.01, 0.00232, 0, 0, 0.01, 0, 0, 0, 0.01, 0, -0.0058, 0])
    clip = core.fmtc.transfer(clip, transs='lstar', transd='linear')
    r, g, b = core.std.SplitPlanes(clip)
    r = core.std.Expr(r, 'x 0.9642119944 *')
    b = core.std.Expr(b, 'x 0.8251882845 *')
    clip = core.std.ShufflePlanes([r, g, b], [0, 0, 0], vs.RGB)
    return clip


def lab2labview(clip: vs.VideoNode):
    '''The L*a*b* values may be reasonably viewed in YUV format
    '''
    assert clip.format.id == vs.RGBS
    clip = core.std.Expr(clip, ['x 100 /', 'x 255 /'])
    clip = core.std.ShufflePlanes(clip, [0, 1, 2], vs.YUV)
    return clip


def labview2lab(clip: vs.VideoNode):
    '''This reverts lab2labview
    '''
    assert clip.format.id == vs.YUV444PS
    clip = core.std.ShufflePlanes(clip, [0, 1, 2], vs.RGB)
    clip = core.std.Expr(clip, ['x 100 *', 'x 255 *'])
    return clip


### deltaE functions
# dE94 is already rather slow so no dE2000 anyway


def deltaE76(clip1: vs.VideoNode, clip2: vs.VideoNode):
    L1, a1, b1 = core.std.SplitPlanes(clip1) # x y z
    L2, a2, b2 = core.std.SplitPlanes(clip2) # a b c
    delE = 'x a - dup * y b - dup * + z c - dup * + sqrt'
    return core.std.Expr([L1, a1, b1, L2, a2, b2], delE)


def deltaE94(clip1: vs.VideoNode, clip2: vs.VideoNode):
    L1, a1, b1 = core.std.SplitPlanes(clip1) # x y z
    L2, a2, b2 = core.std.SplitPlanes(clip2) # a b c
    K1 = 0.045
    K2 = 0.015
    delL = 'x a -'
    dela = 'y b -'
    delb = 'z c -'
    C1 = 'y dup * z dup * + sqrt'
    C2 = 'b dup * c dup * + sqrt'
    delC = f'{C1} {C2} -'
    delH2 = f'{dela} dup * {delb} dup * + {delC} dup * -'
    SC = f'1 {K1} {C1} * +'
    SH = f'1 {K2} {C1} * +'
    delE = f'{delL} dup * {delC} {SC} / dup * + {delH2} {SH} dup * / + sqrt'
    return core.std.Expr([L1, a1, b1, L2, a2, b2], delE)


### Descale helpers


def _get_descaler(kernel: str, b: float, c: float, taps: int) -> Callable[[Any], vs.VideoNode]:
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
        raise ValueError(f'invalid kernel specified: {kernel}')


def _get_scaler(kernel: str, b: float, c: float, taps: int) -> Callable[[Any], vs.VideoNode]:
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
        raise ValueError(f'invalid kernel specified: {kernel}')
