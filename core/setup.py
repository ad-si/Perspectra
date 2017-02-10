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
    name='Perspectra',
    version='0.1.0',
    description='Extract and perspectively correct documents in images',
    long_description=read('./readme.md'),
    url='http://github.com/feramhq/perspectra',
    author='Adrian Sieber',
    author_email='adrian@feram.co',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Desktop',
        'Intended Audience :: Prosumers',
        'Operating System :: OS Independent',
        'Topic :: Computer Vision',
    ],
    keywords = 'document scanner perspective transformation',
    packages = ['perspectra'],
    install_requires = [
        'imageio>=2.1.0',
        'matplotlib>=2.0.0',
        'numpy>=1.12.0',
        'scikit-image>=0.12.3',
    ],
    entry_points = {
        'console_scripts': [
            'perspectra = perspectra.__main__:main',
        ],
    },
)
