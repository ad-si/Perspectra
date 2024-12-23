# Perspectra

Software and corresponding workflow to scan documents and books
with as little hardware as possible.

Check out [github:adius/awesome-scanning]
for an extensive list of alternative solutions.

[github:adius/awesome-scanning]: https://github.com/adius/awesome-scanning


Command | Input | Result
--------|-------|-------
`perspectra correct --binary=gauss-diff 01.jpeg`|![Receipt 1](images/examples/01_original.jpeg)|![Receipt 1 binarized](images/examples/01_binary.png)
`perspectra correct --binary=gauss-diff 02.jpeg`|![Receipt 2](images/examples/02_original.jpeg)|![Receipt 2 binarized](images/examples/02_binary.png)
`perspectra correct --gray 03.jpeg`|![Receipt 3](images/examples/03_original.jpeg)|![Receipt 3 grayscale](images/examples/03_gray.png)


## Installation

We recommend to use [`uv`](https://docs.astral.sh/uv/)
instead of `pip` to install the package.

```sh
uv tool install perspectra
```

To install from source:

```sh
git clone https://github.com/ad-si/Perspectra
cd Perspectra
make install
```


## Usage

### Command Line Interface

```txt
usage: perspectra [-h] [--debug] {binarize,correct,corners,renumber-pages} ...

options:
  -h, --help            show this help message and exit
  --debug               Render debugging view

subcommands:
  subcommands to handle files and correct photos

  {binarize,correct,corners,renumber-pages}
                        additional help
    binarize            Binarize image
    correct             Pespectively correct and crop photos of documents.
    corners             Returns the corners of the document in the image as
                        [top-left, top-right, bottom-right, bottom-left]
    renumber-pages      Renames the images in a directory according to their
                        page numbers. The assumed layout is `cover -> odd
                        pages -> even pages reversed`
```


## Best Practices for Taking the Photos

Your photos should ideally have following properties:

- Photos with 10 - 20 Mpx
- Contain 1 document
    - Rectangular
    - Pronounced corners
    - Only black content on white or light-colored paper
    - On dark background
    - Maximum of 30° rotation


### Camera Settings

```yaml
# Rule of thumb is the inverse of your focal length,
# but motion blur is pretty much the worst for readable documents,
# therefore use at least half of it and never less than 1/50.
shutter: 1/50 - 1/200 s

# The whole document must be sharp even if you photograph it from an angle.
# Therefore at least 8 f.
aperture: 8-12 f

# Noise is less bad than motion blur => relative high ISO
# Should be the last thing you set:
# As high as necessary as low as possible
iso: 800-6400
```

When using `Tv` (Time Value) or `Av` (Aperture Value) mode
use exposure compensation to set lightness value below 0.
You really don't want to overexpose your photos as the bright pages
are the first thing that clips.

On the other hand,
it doesn't matter if you loose background parts because they are to dark.


### Generating the Photos from a Video

A good tool for this purpose is [PySceneDetect].
It's a Python/OpenCV-based scene detection program,
using threshold/content analysis on a given video.

[PySceneDetect]: https://github.com/Breakthrough/PySceneDetect

For easy installation you can use the [docker image]

[docker image]: https://github.com/handflucht/PySceneDetect


Find good values for threshold:

```fish
docker run \
  --rm \
  --volume (pwd):/video \
  handflucht/pyscenedetect
  --input /video/page-turning.mp4 \
  --downscale-factor 2 \
  --detector content \
  --statsfile page-turning-stats.csv
```


To launch the image run:

```fish
docker run \
  --interactive \
  --tty \
  --volume=(pwd):/video \
  --entrypoint=bash \
  handflucht/pyscenedetect
```


Then run in the shell:

```bash
cd /video
scenedetect \
  --input page-turning.mp4 \
  --downscale-factor 2 \
  --detector content \
  --threshold 3 \
  --min-scene-length 80 \
  --save-images
```


TODO: The correct way to do this:
(after https://github.com/Breakthrough/PySceneDetect/issues/45 is implemented)

```fish
docker run \
  --rm \
  --volume (pwd):/video \
  handflucht/pyscenedetect \
  --input /video/page-turning.mp4 \
  --downscale-factor 2 \
  --detector content \
  --threshold 3 \
  --min-scene-length 80 \
  --save-images <TODO: path>
```

Aim for a low threshold and a long minimum scene length.
I.e. turn the page really fast and show it for a long time.
