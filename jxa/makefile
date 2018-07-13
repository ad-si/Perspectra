contentsPath = Perspectra.app/Contents
resourcesPath = $(contentsPath)/Resources
scriptsPath = $(resourcesPath)/Scripts
# tempLibsPath = $(resourcesPath)/temp-script-libraries

$(scriptsPath)/main.scpt: build/bundle.js | $(scriptsPath)
	osacompile -l JavaScript -o $@ $<

build/bundle.js: source/main.js source/config.js | build
	echo "window = this;" > $@
	./node_modules/.bin/browserify $< >> $@

build:
	- mkdir build

$(scriptsPath):
	- mkdir $@

.PHONY: clean

clean:
	rm -rf build  && \
	rm -rf $(scriptsPath)
