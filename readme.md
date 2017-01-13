# Perspectra

The Perspective Transformator App.

1. Open photo of a document
1. Select corners
1. Perspectra automatically corrects the perspective and crops the image


## Usage

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


## Related

- [stackoverflow]
- [math.stackexchange]
- Zhengyou Zhang , Li-Wei He, "Whiteboard scanning and image enhancement"
  [whiteboard-scanning]
- ROBERT M. HARALICK "Determining camera parameters
  from the perspective projection of a rectangle"
  [cam-params]
- [perspective-transform]
- [document-scanner]

[stackoverflow]: http://stackoverflow.com/questions/1194352/proportions-of-a-perspective-deformed-rectangle
[whiteboard-scanning]: http://research.microsoft.com/en-us/um/people/zhang/papers/tr03-39.pdf
[cam-params]: http://portal.acm.org/citation.cfm?id=87146
[math.stackexchange]:  http://math.stackexchange.com/questions/1339924/compute-ratio-of-a-rectangle-seen-from-an-unknown-perspective
[perspective-transform]:  http://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
[document-scanner]:  http://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
