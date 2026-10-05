"""Build SREAvatar's CUDA extensions (pytorch3d, diff_gaussian_rasterization) for this PC's GPU (Windows).

    build_cuda_ext_win.bat                          # both, arch auto-detected
    build_cuda_ext_win.bat --steps pytorch3d        # subset
    build_cuda_ext_win.bat --arch "8.9;12.0+PTX"    # explicit TORCH_CUDA_ARCH_LIST
    build_cuda_ext_win.bat --verify-only            # just check what is installed

Run it through build_cuda_ext_win.bat, which loads vcvars64 first -- nvcc needs cl.exe
on PATH. Sources are cloned once into third_modules/; wheels go to build/wheels/ and are
installed right away.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TM = ROOT / 'third_modules'
WHEELS = ROOT / 'build' / 'wheels'

# CUDA toolkit must match what torch was built against (cu128), NOT whatever nvcc is on PATH.
CUDA_HOME = Path(os.environ.get('CUDA_HOME') or
                 r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8')


# ----------------------------------------------------------------------------- helpers
def log(msg):
    print('\n[build] %s' % msg, flush=True)


def run(cmd, cwd=None, env=None):
    cmd = [str(c) for c in cmd]
    print('  $ %s' % ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def clone(url, dst, tag=None, recursive=False):
    """Clone url into dst (skip if present) and check out tag."""
    dst = Path(dst)
    if dst.exists():
        print('  exists, skip clone: %s' % dst)
    else:
        cmd = ['git', 'clone']
        if recursive:
            cmd.append('--recursive')
        cmd += [url, str(dst)]
        run(cmd)
    if tag:
        # release tags are spelled inconsistently across these repos (v0.7.8 / V0.7.8)
        for cand in (tag, tag.upper(), tag.lower(), tag.lstrip('vV')):
            if subprocess.run(['git', 'rev-parse', '-q', '--verify', '%s^{commit}' % cand],
                              cwd=str(dst), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                run(['git', 'checkout', '-q', cand], cwd=dst)
                if recursive:
                    run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=dst)
                return
        sys.exit('  ERROR: tag %s not found in %s' % (tag, dst))


def detect_arch():
    """TORCH_CUDA_ARCH_LIST covering every GPU in this box, plus a PTX fallback."""
    import torch
    caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    if not caps:
        sys.exit('  ERROR: no CUDA device visible; cannot auto-detect arch')
    archs = sorted('%d.%d' % c for c in caps)
    return ';'.join(archs[:-1] + [archs[-1] + '+PTX'])


def build_env(arch, extra=None):
    env = os.environ.copy()
    env['CUDA_HOME'] = env['CUDA_PATH'] = str(CUDA_HOME)
    # prepend so 12.8's nvcc wins over any newer toolkit already on PATH
    env['PATH'] = str(CUDA_HOME / 'bin') + os.pathsep + env.get('PATH', '')
    env['TORCH_CUDA_ARCH_LIST'] = arch
    env['DISTUTILS_USE_SDK'] = '1'          # let torch reuse the vcvars MSVC environment
    env['MAX_JOBS'] = str(min(16, os.cpu_count() or 8))
    env.pop('CUDA_VISIBLE_DEVICES', None)   # arch detection already done
    if extra:
        env.update(extra)
    return env


def pip_wheel(src_dir, env, name):
    """Build a wheel from src_dir into build/wheels/ and install it."""
    run([sys.executable, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps',
         '-w', str(WHEELS), '.'], cwd=src_dir, env=env)
    new = sorted(WHEELS.glob('%s-*.whl' % name), key=lambda w: w.stat().st_mtime)
    if not new:
        sys.exit('  ERROR: no wheel produced for %s' % name)
    wheel = new[-1]
    log('installing %s' % wheel.name)
    run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
         str(wheel)], env=env)
    return wheel


# ----------------------------------------------------------------------------- patches
def patch_pytorch3d(src):
    """pulsar/global.h gives Windows the missing `uint`/`ushort` names as preprocessor
    macros. `#define ushort unsigned short` then corrupts CUDA 12.8's own libcu++
    headers: cuda/std/__tuple_dir/vector_types.h token-pastes the name with 1..4, so
    `tuple_size<ushort1>` preprocesses to `tuple_size<unsigned short1>` and every
    pulsar .cu file dies with `error: expected a ">"`. Typedefs name the same types
    without taking part in token pasting, so rewrite the macros as typedefs."""
    f = src / 'pytorch3d' / 'csrc' / 'pulsar' / 'global.h'
    txt = f.read_text(encoding='utf-8')
    out, done = txt, []
    for name, ctype in (('uint', 'unsigned int'), ('ushort', 'unsigned short')):
        if re.search(r'typedef\s+%s\s+%s\s*;' % (ctype, name), out):
            continue
        out, n = re.subn(r'#define[ \t]+%s[ \t]+%s[ \t]*' % (name, ctype),
                         'typedef %s %s;' % (ctype, name), out)
        if n:
            done.append('%s (x%d)' % (name, n))
    if out == txt:
        print('  already patched: %s' % f.name)
        return
    print('  rewrote #define -> typedef in %s: %s' % (f.name, ', '.join(done)))
    f.write_text(out, encoding='utf-8')


# ------------------------------------------------------------------------------- steps
def step_pytorch3d(arch):
    src = TM / 'pytorch3d'
    clone('https://github.com/facebookresearch/pytorch3d.git', src, tag='v0.7.8')
    patch_pytorch3d(src)
    env = build_env(arch, {'PYTORCH3D_NO_NINJA': '0', 'FORCE_CUDA': '1'})
    return pip_wheel(src, env, 'pytorch3d')


def step_dgr(arch):
    mip = TM / 'mip-splatting'
    # --recursive is required: the rasterizer needs the bundled glm headers
    clone('https://github.com/autonomousvision/mip-splatting.git', mip, recursive=True)
    src = mip / 'submodules' / 'diff-gaussian-rasterization'
    glm = src / 'third_party' / 'glm' / 'glm'
    if not glm.exists():
        run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=mip)
    if not glm.exists():
        sys.exit('  ERROR: glm submodule missing at %s' % glm)
    env = build_env(arch)
    return pip_wheel(src, env, 'diff_gaussian_rasterization')


STEPS = [
    ('pytorch3d', step_pytorch3d),
    ('dgr', step_dgr),
]


# ------------------------------------------------------------------------------ verify
def verify():
    """Report the SASS/PTX archs baked into each installed .pyd, then run a real kernel."""
    import torch

    cuobjdump = CUDA_HOME / 'bin' / 'cuobjdump.exe'
    sp = Path(torch.__file__).parent.parent
    targets = {
        'pytorch3d': sp / 'pytorch3d' / '_C.cp310-win_amd64.pyd',
        'diff_gaussian_rasterization': sp / 'diff_gaussian_rasterization' / '_C.cp310-win_amd64.pyd',
    }
    dev = torch.cuda.get_device_capability(0)
    print('\n' + '=' * 72)
    print('GPU: %s  sm_%d%d' % (torch.cuda.get_device_name(0), dev[0], dev[1]))
    print('=' * 72)
    for name, pyd in targets.items():
        if not pyd.exists():
            print('  %-28s NOT INSTALLED' % name)
            continue
        sass = ptx = set()
        if cuobjdump.exists():
            out = subprocess.run([str(cuobjdump), '-lelf', str(pyd)],
                                 capture_output=True, text=True).stdout
            sass = set(re.findall(r'sm_(\d+)', out))
            out = subprocess.run([str(cuobjdump), '-lptx', str(pyd)],
                                 capture_output=True, text=True).stdout
            ptx = set(re.findall(r'sm_(\d+)', out))
        ok = str(dev[0] * 10 + dev[1]) in (sass | ptx)
        print('  %-28s SASS=%-18s PTX=%-12s %s' % (
            name,
            ','.join('sm_' + a for a in sorted(sass, key=int)) or '-',
            ','.join('sm_' + a for a in sorted(ptx, key=int)) or '-',
            'OK' if ok else '<-- WRONG ARCH'))

    print('\n  running real kernels:')
    checks = [
        ('pytorch3d.knn_points', _k_p3d),
        ('diff_gaussian_rasterization', _k_dgr),
    ]
    for name, fn in checks:
        try:
            r = fn()
            torch.cuda.synchronize()
            print('    PASS  %-32s %s' % (name, r))
        except Exception as e:
            print('    FAIL  %-32s %s: %s' % (name, type(e).__name__, str(e).split('\n')[0][:110]))
    print('=' * 72)


def _k_p3d():
    import torch
    from pytorch3d.ops import knn_points
    o = knn_points(torch.randn(1, 64, 3, device='cuda'), torch.randn(1, 128, 3, device='cuda'), K=4)
    return 'idx%s' % (tuple(o.idx.shape),)


def _k_dgr():
    """Render one Gaussian through a sane perspective camera (a garbage projmatrix makes
    the rasterizer allocate absurd buffers instead of reporting the real error)."""
    import math
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    H = W = 64
    fov = math.radians(60.)
    tan = math.tan(fov * .5)
    znear, zfar = .01, 100.
    view = torch.eye(4, device='cuda')
    view[2, 3] = 4.                      # camera 4 units back, row-major as CUDA glm expects
    proj = torch.zeros(4, 4, device='cuda')
    proj[0, 0] = 1. / tan
    proj[1, 1] = 1. / tan
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = proj @ view
    fields = GaussianRasterizationSettings._fields
    kw = dict(image_height=H, image_width=W, tanfovx=tan, tanfovy=tan,
              bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
              viewmatrix=view.t().contiguous(), projmatrix=full.t().contiguous(),
              sh_degree=0, campos=torch.zeros(3, device='cuda'),
              prefiltered=False, debug=False)
    defaults = {'kernel_size': .1, 'subpixel_offset': torch.zeros(H, W, 2, device='cuda'),
                'require_coord': False, 'require_depth': False, 'antialiasing': False}
    for k, v in defaults.items():
        if k in fields:
            kw[k] = v
    missing = [f for f in fields if f not in kw]
    if missing:
        return 'SKIP (unknown settings fields: %s)' % missing
    r = GaussianRasterizer(GaussianRasterizationSettings(**kw))
    N = 8
    m = torch.zeros(N, 3, device='cuda')
    out = r(means3D=m, means2D=torch.zeros_like(m, requires_grad=True), shs=None,
            colors_precomp=torch.ones(N, 3, device='cuda'),
            opacities=torch.ones(N, 1, device='cuda'),
            scales=torch.full((N, 3), .1, device='cuda'),
            rotations=torch.tensor([[1., 0, 0, 0]], device='cuda').repeat(N, 1),
            cov3D_precomp=None)
    img = out[0] if isinstance(out, tuple) else out
    return 'rendered %s sum=%.3f' % (tuple(img.shape), float(img.detach().sum()))


# -------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', nargs='+', choices=[n for n, _ in STEPS],
                    help='build only these (default: all)')
    ap.add_argument('--arch', help='TORCH_CUDA_ARCH_LIST, e.g. "8.9;12.0+PTX" (default: auto)')
    ap.add_argument('--verify-only', action='store_true', help='skip building, just report')
    args = ap.parse_args()

    if os.name != 'nt':
        sys.exit('build_cuda_ext_win.py is Windows-only.')
    if args.verify_only:
        verify()
        return

    if not (CUDA_HOME / 'bin' / 'nvcc.exe').exists():
        sys.exit('ERROR: nvcc not found under CUDA_HOME=%s' % CUDA_HOME)
    if not shutil.which('cl'):
        sys.exit('ERROR: cl.exe not on PATH. Run this through build_cuda_ext_win.bat, '
                 'which loads vcvars64 first.')

    import torch
    arch = args.arch or detect_arch()
    log('toolchain')
    print('  python  : %s' % sys.executable)
    print('  torch   : %s (built against CUDA %s)' % (torch.__version__, torch.version.cuda))
    print('  nvcc    : %s' % (CUDA_HOME / 'bin' / 'nvcc.exe'))
    print('  cl.exe  : %s' % shutil.which('cl'))
    print('  arch    : TORCH_CUDA_ARCH_LIST=%s' % arch)

    # ninja cuts these builds from ~an hour to minutes; it is a pure-python wheel
    try:
        import ninja  # noqa: F401
    except ImportError:
        log('installing ninja (build accelerator)')
        run([sys.executable, '-m', 'pip', 'install', 'ninja'])

    TM.mkdir(exist_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)

    selected = [(n, f) for n, f in STEPS if not args.steps or n in args.steps]
    built = []
    for i, (name, fn) in enumerate(selected, 1):
        log('[%d/%d] %s' % (i, len(selected), name))
        built.append(fn(arch))

    log('built wheels')
    for w in built:
        print('  %s' % w)
    verify()


if __name__ == '__main__':
    main()