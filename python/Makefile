all: compile
full: compile install clean

ifndef INSTALL_DIR
INSTALL_DIR=./built-lib/mhmm/
endif

ifndef SWIG
SWIG=swig
endif

ifndef DOXYGEN
DOXYGEN=doxygen
endif

DOC_DIR=../doc/
LIB_DIR=../src/models/
RTML_DIR=../src/base/
JSON_XCODE_DIR=../components/libjson/build/xcode
JSON_LIB_DIR=../components/libjson/built-lib

SWIGOPT=#-debug-template

json:
	@echo "\n==> Compiling JSON Library\n"
	@(cd $(JSON_XCODE_DIR) && xcodebuild -xcconfig Config_c++03.xcconfig)
	@(cd $(JSON_XCODE_DIR) && xcodebuild -xcconfig Config_c++03.xcconfig install)
	@(cd $(JSON_XCODE_DIR) && xcodebuild -xcconfig Config_c++03.xcconfig clean)

compile: json mhmm.i setup.py $(RTML_DIR)*.cpp #$(LIB_DIR)*.cpp
	@echo "\n==> Generating doc\n"
	@(cd ../doc/ && $(DOXYGEN))
	python doxy2swig.py $(DOC_DIR)xml/index.xml mhmm_doc.i
	@echo "\n==> Compiling MHMM Library\n"
	$(SWIG) -c++ $(SWIGOPT) -I$(LIB_DIR) -I$(RTML_DIR) -python mhmm.i
	python setup.py build_ext --inplace

install:
	@echo "\n==> Installing MHMM Library\n"
	mkdir -p $(INSTALL_DIR)
	mv mhmm.py $(INSTALL_DIR)
	mv _mhmm.so $(INSTALL_DIR)
	cp __init__.py $(INSTALL_DIR)

clean:
	@echo "\n==> Cleaning files\n"
	rm -f *.cxx *.so mhmm.py mhmm_doc.i
	rm -rf build/