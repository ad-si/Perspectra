.PHONY: help
help: makefile
	@tail -n +4 makefile | grep ".PHONY"


.PHONY: test
test:
	rye test


.PHONY: edit-notebooks
edit-notebooks:
	rye run marimo edit notebooks


# TODO: Re-enable this code
# pyFiles := $(shell find core/perspectra -name '*.py')
# # TODO: Slow startup time on first execution
# .PHONY: build-pyinstaller
# build-pyinstaller: core/perspectra/__main__.py $(pyFiles)
# 	pyinstaller $< \
# 		--paths core/perspectra \
# 		--noconfirm \
# 		--name perspectra


# # TODO: Does not work yet due to import errors
# .PHONY: build-nuitka
# build-nuitka: core/perspectra/__main__.py $(pyFiles)
# 	python -m nuitka \
# 		--standalone \
# 		--output-filename=perspectra \
# 		$<


images/logo.icns: images/icon-1024.png
	nicns --in $< --out $@


# TODO: Fix this
# # Create `.app` bundle with py2app
# Perspectra.app:
# 	python3 setup.py py2app -A


# TODO: Fix this
# # With cx_Freeze
# Perspectra.app:
# 	python3 setup.py install
# 	python3 setup.py bdist_mac


.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .venv
	rm -rf *.app
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	rm -rf perspectra.spec
	rm -rf src/perspectra/__pycache__
