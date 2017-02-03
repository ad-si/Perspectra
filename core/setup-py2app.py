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
    version='0.1.0',
    app=['perspectra/perspectra.py'],
    url='http://github.com/feramhq/perspectra',
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
        'Environment :: Desktop',
        'Intended Audience :: Prosumers',
        'Operating System :: OS Independent',
        'Topic :: Image Manipulation',
    ],
    scripts = ['cli.py'],
    setup_requires=['py2app'],
    options={'py2app': {
        'argv_emulation': True,
        'iconfile': 'images/logo.icns'
    }},
)
