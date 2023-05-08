.PHONY: help
help: makefile
	@tail -n +4 makefile | grep ".PHONY"


pyFiles := $(shell find core/perspectra -name '*.py')


# TODO: Slow startup time on first execution
.PHONY: build-pyinstaller
build-pyinstaller: core/perspectra/__main__.py $(pyFiles)
	poetry run pyinstaller $< \
		--paths core/perspectra \
		--noconfirm \
		--name perspectra


# # TODO: Does not work yet due to import errors
# .PHONY: build-nuitka
# build-nuitka: core/perspectra/__main__.py $(pyFiles)
# 	poetry run python -m nuitka \
# 		--standalone \
# 		--output-filename=perspectra \
# 		$<
