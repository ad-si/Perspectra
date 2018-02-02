# Perspectra

Extract and perspectively correct documents in images.

Check out [github:adius/awesome-scanning]
for and extensive list of alternative solutions.

[github:adius/awesome-scanning]: https://github.com/adius/awesome-scanning


## Best Practices for Taking the Photos

TODO


## Generating the Photos from a Video

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
  --min-scene-length 50 \
  --threshold 3 \
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
  --threshold 5 \
  --save-images <TODO: path>
```


Aim for a low threshold and a long minimun scene length.
I.e. turn the page really fast and show it for a long time.
