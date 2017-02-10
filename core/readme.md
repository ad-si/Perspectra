# Perspectra

Extract and perspectively correct documents in images.

Check out [github:adius/awesome-scanning]
for and extensive list of alternative solutions.

[github:adius/awesome-scanning]: https://github.com/adius/awesome-scanning


## Usage

### Command Line Interface

```txt
usage: perspectra [-h] [--debug] [--gray] [--output image-path] image-path

positional arguments:
  image-path           Path to image which shall be fixed

optional arguments:
  -h, --help           show this help message and exit
  --debug              Render debugging view
  --gray               Safe image in grayscale
  --output image-path  Output path of fixed image
```


### With Docker (currently not wokring)

Run with CPython

```sh
docker run \
  --interactive \
  --tty \
  --rm \
  --name perspectra \
  --volume "$PWD":/usr/src/myapp \
  --workdir /usr/src/myapp \
  python:3 \
  python3 ./perspectra.py
```


Run with Pypy 

```sh
docker run \
  --interactive \
  --tty \
  --rm \
  --name perspectra \
  --volume "$PWD":/usr/src/myapp \
  --workdir /usr/src/myapp \
  pypy:3 \./perspectra.py
```


## Development

Create `.icns` file:

```sh
nicns --in icon-1024.png
```

Create `.app` bundle with py2app (currently not working):

```sh
python3 setup.py py2app -A
```


### With cx_Freeze

Create `.app` bundle:

```sh
python3 setup.py install
python3 setup.py bdist_mac
```


# TODO

- Calculate aspect ratio of scanned document
  and apply during perspective transformation
- Make sure besin for watershed algorithm is in no local confinement
- Add white border to sobel image or crop it by 1px in order
  to correctly handle partly cropped documents
- Check that there were at least 4 corners detected
- Dewarp pages
- Maximize contrast of image before binarizing
