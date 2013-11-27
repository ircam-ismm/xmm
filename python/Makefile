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
RTML_DIR = ../src/base/

SWIGOPT=#-debug-template

compile: mhmm.i setup.py $(LIB_DIR)*.h $(RTML_DIR)*.h $(RTML_DIR)*.cpp #$(LIB_DIR)*.cpp
	# @echo "\ngenerating doc\n"
	# @(cd ../doc/ && $(DOXYGEN))
	# @echo "\nCompiling Library\n"
	# python doxy2swig.py $(DOC_DIR)xml/index.xml mhmm_doc.i
	$(SWIG) -c++ $(SWIGOPT) -I$(LIB_DIR) -I$(RTML_DIR) -python mhmm.i
	python setup.py build_ext --inplace

install:
	mkdir -p $(INSTALL_DIR)
	mv mhmm.py $(INSTALL_DIR)
	mv _mhmm.so $(INSTALL_DIR)
	cp utility.py $(INSTALL_DIR)
	cp __init__.py $(INSTALL_DIR)

clean:
	@echo "\nCleaning files\n"
	rm -f *.cxx *.so mhmm.py mhmm_doc.i
	rm -rf build/
	@echo "\nDone"
