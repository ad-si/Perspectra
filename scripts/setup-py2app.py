import io
from setuptools import setup

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

setup(
    name='perspectra',
    version='0.2.1',
    app=['perspectra/perspectra.py'],
    url='http://github.com/ad-si/Perspectra',
    author='Adrian Sieber',
    author_email='adrian@feram.co',
    description='Extract and perspectively correct documents in images',
    long_description=read('./readme.md'),
    packages=['perspectra'],
    include_package_data=True,
    platforms='any',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: Beta',
        'Natural Language :: English',
        'Environment :: X11 Applications',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    scripts = ['cli.py'],
    setup_requires=['py2app'],
    options={'py2app': {
        'argv_emulation': True,
        'iconfile': 'images/logo.icns'
    }},
)
