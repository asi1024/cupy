from __future__ import annotations

import json
import os
import os.path
import shutil
import subprocess
import sys
from typing import Any

import setuptools
import setuptools.command.build_ext

import cupy_builder
import cupy_builder.install_build as build
from cupy_builder._context import Context
from cupy_builder._compiler import DeviceCompilerUnix, DeviceCompilerWin32


def filter_files_by_extension(
        sources: list[str],
        extension: str,
) -> tuple[list[str], list[str]]:
    sources_selected = []
    sources_others = []
    for src in sources:
        if os.path.splitext(src)[1] == extension:
            sources_selected.append(src)
        else:
            sources_others.append(src)
    return sources_selected, sources_others


def compile_device_code(
        ctx: Context,
        ext: setuptools.Extension
) -> tuple[list[str], list[str]]:
    """Compiles device code ("*.cu").

    This method invokes the device compiler (nvcc/hipcc) to build object
    files from device code, then returns the tuple of:
    - list of remaining (non-device) source files ("*.cpp")
    - list of compiled object files for device code ("*.o")
    """
    sources = [os.fspath(src) for src in ext.sources]
    sources_cu, sources_cpp = filter_files_by_extension(sources, ".cu")
    if len(sources_cu) == 0:
        # No device code used in this extension.
        return sources, []

    if sys.platform == 'win32':
        compiler = DeviceCompilerWin32(ctx)
    else:
        compiler = DeviceCompilerUnix(ctx)

    objects = []
    for src in sources_cu:
        print(f'{ext.name}: Device code: {src}')
        obj_ext = 'obj' if sys.platform == 'win32' else 'o'
        # TODO(kmaehashi): embed CUDA version in path
        obj = f'build/temp.device_objects/{src}.{obj_ext}'
        if os.path.exists(obj) and (_get_timestamp(src) < _get_timestamp(obj)):
            print(f'{ext.name}: Reusing cached object file: {obj}')
        else:
            os.makedirs(os.path.dirname(obj), exist_ok=True)
            print(f'{ext.name}: Building: {obj}')
            compiler.compile(obj, src, ext)
        objects.append(obj)

    return sources_cpp, objects


def _get_timestamp(path: str) -> float:
    stat = os.lstat(path)
    return max(stat.st_atime, stat.st_mtime, stat.st_ctime)


def dumpbin_dependents(dumpbin: str, path: str) -> list[str]:
    args = [dumpbin, '/nologo', '/dependents', path]
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE)
    except FileNotFoundError:
        print(f'*** DUMPBIN not found: {args}')
        return []
    if p.returncode != 0:
        print(f'*** DUMPBIN failed ({p.returncode}): {args}')
        return []
    sections = p.stdout.decode().split('\r\n\r\n')
    for num, section in enumerate(sections):
        if 'Image has the following dependencies:' in section:
            return [line.strip() for line in sections[num+1].splitlines()]
    print(f'*** DUMPBIN output could not be parsed: {args}')
    return []


class custom_build_ext(setuptools.command.build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def _cythonize(self, nthreads: int) -> None:
        # Defer importing Cython as it may be installed via setup_requires if
        # the user does not have Cython installed.
        import Cython.Build

        ctx = cupy_builder.get_context()
        compiler_directives = {
            'linetrace': ctx.linetrace,
            'profile': ctx.profile,
            # Embed signatures for Sphinx documentation.
            'embedsignature': True,
            # Allow not implementing reversed method
            # https://github.com/cupy/cupy/issues/5893#issuecomment-944909015
            'c_api_binop_methods': True,
            # Keep the behavior same as Cython 0.29.x.
            # https://github.com/cupy/cupy/pull/8457#issuecomment-2656568499
            'binding': False,
            'legacy_implicit_noexcept': True,
        }

        # Compile-time constants to be used in Cython code
        compile_time_env: dict[str, Any] = {}

        # Enable CUDA Python.
        # TODO: add `cuda` to `setup_requires` only when this flag is set
        use_cuda_python = ctx.use_cuda_python
        compile_time_env['CUPY_USE_CUDA_PYTHON'] = use_cuda_python
        if use_cuda_python:
            print('Using CUDA Python')

        compile_time_env['CUPY_CUFFT_STATIC'] = False
        compile_time_env['CUPY_CYTHON_VERSION'] = Cython.__version__
        if ctx.use_stub:  # on RTD
            compile_time_env['CUPY_CUDA_VERSION'] = 0
            compile_time_env['CUPY_HIP_VERSION'] = 0
        elif ctx.use_hip:  # on ROCm/HIP
            compile_time_env['CUPY_CUDA_VERSION'] = 0
            compile_time_env['CUPY_HIP_VERSION'] = build.get_hip_version()
        else:  # on CUDA
            compile_time_env['CUPY_CUDA_VERSION'] = (
                ctx.features['cuda'].get_version())
            compile_time_env['CUPY_HIP_VERSION'] = 0

        print('Compile-time constants: ' +
              json.dumps(compile_time_env, indent=4))

        if sys.platform == 'win32':
            # Disable multiprocessing on Windows (spawn)
            nthreads = 0

        Cython.Build.cythonize(
            self.extensions, verbose=True, nthreads=nthreads, language_level=3,
            compiler_directives=compiler_directives, annotate=ctx.annotate,
            compile_time_env=compile_time_env)

    def build_extensions(self) -> None:
        ctx = cupy_builder.get_context()
        num_jobs = int(os.environ.get('CUPY_NUM_BUILD_JOBS', '4'))
        if num_jobs > 1:
            self.parallel = num_jobs

        if (sys.platform == 'win32' and
                hasattr(self.compiler, 'initialize')):  # i.e., MSVCCompiler
            # Initialize to get path to the host compiler (cl.exe).
            # This also workarounds a bug in setuptools/distutils on Windows by
            # initializing the compiler before starting a thread.
            # By default, MSVCCompiler performs initialization in the
            # first compilation. However, in parallel compilation mode,
            # the init code runs in each thread and messes up the internal
            # state as the init code is not locked and is not idempotent.
            # https://github.com/pypa/setuptools/blob/v60.0.0/setuptools/_distutils/_msvccompiler.py#L322-L327
            self.compiler.initialize()
            if hasattr(self.compiler, 'cc'):
                cc = self.compiler.cc
                print(f'Detected host compiler: {cc}')
                ctx.win32_cl_exe_path = cc

        # Compile "*.pyx" files into "*.cpp" files.
        print('Cythonizing...')
        self._cythonize(num_jobs)

        # Change an extension in each source filenames from "*.pyx" to "*.cpp".
        # c.f. `Cython.Distutils.old_build_ext`
        for ext in self.extensions:
            sources_pyx, sources_others = filter_files_by_extension(
                ext.sources, '.pyx')
            sources_cpp = ['{}.cpp'.format(os.path.splitext(src)[0])
                           for src in sources_pyx]
            ext.sources = sources_cpp + sources_others
            for src in ext.sources:
                if not os.path.isfile(src):
                    raise RuntimeError(f'Fatal error: missing file: {src}')

        print('Building extensions...')
        super().build_extensions()

        if sys.platform == 'win32':
            print('Generating DLL dependency list...')

            # Find "dumpbin.exe" next to "lib.exe".
            dumpbin = os.path.join(
                os.path.dirname(self.compiler.lib), 'dumpbin.exe')

            # Save list of dependent DLLs for all extension modules.
            depends = sorted(set(sum([
                dumpbin_dependents(dumpbin, f) for f in self.get_outputs()
            ], [])))

            depends_json = os.path.join(
                self.build_lib, 'cupy', '.data', '_depends.json')
            os.makedirs(os.path.dirname(depends_json), exist_ok=True)
            with open(depends_json, 'w') as f:
                json.dump({'depends': depends}, f)

    def build_extension(self, ext: setuptools.Extension) -> None:
        ctx = cupy_builder.get_context()

        # The setuptools always uses temp dir for PEP 660 builds, which means
        # incremental compilation is not possible. Here we workaround that by
        # manually checking if the build can be skipped. See also:
        # https://github.com/pypa/setuptools/blob/v78.1.0/setuptools/command/editable_wheel.py#L333-L334
        # https://github.com/pypa/setuptools/blob/v78.1.0/setuptools/_distutils/command/build_ext.py#L532-L538
        if ctx.setup_command == 'editable_wheel' and not self.force:
            ext_build_lib = self.get_ext_fullpath(ext.name)
            ext_inplace = os.path.relpath(ext_build_lib, self.build_lib)
            if (os.path.exists(ext_inplace) and
                    max(_get_timestamp(f) for f in (ext.sources + ext.depends))
                    < _get_timestamp(ext_inplace)):
                print(f'skip building \'{ext.name}\' extension (up-to-date)')
                # Pretend as if it was just built.
                os.makedirs(os.path.dirname(ext_build_lib), exist_ok=True)
                shutil.copy2(ext_inplace, ext_build_lib)
                return

        # Compile "*.cu" files into object files.
        sources_cpp, extra_objects = compile_device_code(ctx, ext)

        # Remove device code from list of sources, and instead add compiled
        # object files to link.
        ext.sources = sources_cpp
        ext.extra_objects = extra_objects + ext.extra_objects

        # Let setuptools do the rest of the build process, i.e., compile
        # "*.cpp" files and link object files generated from "*.cu".
        super().build_extension(ext)
