import os
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


VERSION = "0.4.0"

REQUIREMENTS = ["fenics-dolfinx>0.4.0", "numpy>=1.21.0", "dolfinx_cuas>=0.4.1"]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the"
                               + "following extensions:"
                               + ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise RuntimeError("Windows not supported")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j3']

        env = os.environ.copy()
        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp, env=env)


setup(name='dolfinx_contact',

      version=VERSION,

      description='Contact custom assemblers using DOLFINx and Basix',

      author='Sarah Roggendorf',
      python_requires='>3.7.0',
      packages=['dolfinx_contact', "dolfinx_contact.one_sided", "dolfinx_contact.meshing", "dolfinx_contact.meshtie",
                "dolfinx_contact.unbiased"],
      package_data={'dolfinx_contact.wrappers': ['*.h'], 'dolfinx_contact': ["py.typed"],
                    "dolfinx_contact.one_sided": ["py.typed"], "dolfinx_contact.meshing": ["py.typed"],
                    "dolfinx_contact.meshtie": ["py.typed"],
                    "dolfinx_contact.unbiased": ["py.typed"]},
      ext_modules=[CMakeExtension('dolfinx_contact.cpp')],
      cmdclass=dict(build_ext=CMakeBuild),
      install_requires=REQUIREMENTS,
      zip_safe=False)
