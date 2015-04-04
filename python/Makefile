all: build

ifndef INSTALL_DIR
INSTALL_DIR=./bin
endif

ifndef SWIG
SWIG=swig
endif

ifndef DOXYGEN
DOXYGEN=doxygen
endif

DOC_DIR=../doc/
XMM_DIR=../src
JSON_XCODE_DIR=../dependencies/libjson/ide
JSON_LIB_DIR=../dependencies/libjson/bin
JSON_SRC_DIR=../dependencies/libjson/src
XMM_SRC_FILES=$(wildcard $(XMM_DIR)/**/*.cpp $(XMM_DIR)/**/*.h)
JSON_SRC_FILES=$(wildcard $(JSON_SRC_DIR)/*.h $(JSON_SRC_DIR)/_internal/**/*.h $(JSON_SRC_DIR)/_internal/**/*.cpp)

$(JSON_LIB_DIR)/libjson.a: $(JSON_SRC_FILES)
	@echo "\n==> Compiling JSON Library\n"
	@(cd $(JSON_XCODE_DIR) && xcodebuild)
	@(cd $(JSON_XCODE_DIR) && xcodebuild install)

build/xmm_doc.i: $(XMM_SRC_FILES)
	@echo "\n==> Generating doc\n"
	@(cd ../doc/ && $(DOXYGEN))
	mkdir -p ./build/
	python src/doxy2swig.py $(DOC_DIR)xml/index.xml build/xmm_doc.i

bin/xmm/_xmm.so: $(JSON_LIB_DIR)/libjson.a build/xmm_doc.i src/xmm.i src/setup.py $(XMM_SRC_FILES)
	mkdir -p ./bin/xmm/
	@echo "\n==> Compiling XMM Library\n"
	$(SWIG) -c++ -I$(XMM_DIR)/core -I$(XMM_DIR)/models -python -outdir ./bin/xmm/ -o build/xmm_wrap.cxx src/xmm.i
	python src/setup.py build_ext --build-lib ./bin/xmm/
	cp src/__init__.py ./bin/xmm/
	cp src/util.py ./bin/xmm/
ifneq ($(INSTALL_DIR), ./bin)
	cp -r ./bin/ $(INSTALL_DIR)/
endif

build: bin/xmm/_xmm.so

clean:
	@(cd $(JSON_XCODE_DIR) && xcodebuild clean)
	rm -rf build/
	rm -rf bin/
