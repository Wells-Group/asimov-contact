from skbuild import setup

import sys
import sysconfig

setup(name="dolfinx-contact",
      python_requires='>=3.7.0',
      version="0.3.0",
      description='DOLFINx contact kernels',
      url="https://github.com/Wells-Group/asimov-contact/",
      author='Sarah Roggendorf',
      maintainer="Jørgen S. Dokken",
      maintainer_email="dokken92@gmail.com",
      license="MIT",
      packages=['dolfinx_contact', "dolfinx_contact.one_sided"],
      cmake_args=[
          '-DPython3_EXECUTABLE=' + sys.executable,
          '-DPython3_LIBRARIES=' + sysconfig.get_config_var("LIBDEST"),
          '-DPython3_INCLUDE_DIRS=' + sysconfig.get_config_var("INCLUDEPY")],
      package_dir={"": "python"},
      cmake_install_dir="python/dolfinx_contact/")
