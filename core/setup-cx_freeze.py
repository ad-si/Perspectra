from cx_Freeze import setup, Executable

buildOptions = dict(
  packages = [],
  excludes = [],
)

import sys
base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
  Executable('tkinter-test.py', base=base)
]

setup(
  name='Perspectra',
  version = '1.0',
  description = 'Extract and perspectively correct documents in images.',
  options = dict(
    build_exe = buildOptions,
    bdist_mac = dict(
      iconfile = 'images/logo.icns',
    )
  ),
  executables = executables
)
