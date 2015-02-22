all: compile
full: compile install clean

ifndef INSTALL_DIR
INSTALL_DIR=./built-lib/xmm/
endif

ifndef SWIG
SWIG=swig
endif

ifndef DOXYGEN
DOXYGEN=doxygen
endif

DOC_DIR=../doc/
LIB_DIR=../src/models/
RTML_DIR=../src/core/
JSON_XCODE_DIR=../dependencies/libjson/ide
JSON_LIB_DIR=../dependencies/libjson/bin

SWIGOPT=#-debug-template

json:
	@echo "\n==> Compiling JSON Library\n"
	@(cd $(JSON_XCODE_DIR) && xcodebuild)
	@(cd $(JSON_XCODE_DIR) && xcodebuild install)

compile: json xmm.i setup.py $(RTML_DIR)*.cpp $(LIB_DIR)*.cpp
	@echo "\n==> Generating doc\n"
	@(cd ../doc/ && $(DOXYGEN))
	python doxy2swig.py $(DOC_DIR)xml/index.xml xmm_doc.i
	@echo "\n==> Compiling XMM Library\n"
	$(SWIG) -c++ $(SWIGOPT) -I$(LIB_DIR) -I$(RTML_DIR) -python xmm.i
	python setup.py build_ext --inplace

install:
	@echo "\n==> Installing XMM Library\n"
	mkdir -p $(INSTALL_DIR)
	mv xmm.py $(INSTALL_DIR)
	mv _xmm.so $(INSTALL_DIR)
	cp __init__.py $(INSTALL_DIR)
	cp util.py $(INSTALL_DIR)

clean:
	@echo "\n==> Cleaning files\n"
	@(cd $(JSON_XCODE_DIR) && xcodebuild clean)
	rm -f *.cxx *.so xmm.py xmm_doc.i
	rm -rf build/