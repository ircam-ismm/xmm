all: compile install clean

INSTALL_DIR=./build/mhmm/
#INSTALL_DIR=/Users/francoise/Documents/Code/Python/__lib__/mhmm/

LIB_DIR=../src/
DOC_DIR=../doc/xml/
RTML_DIR = ../src/rtml/

SWIGOPT=#-debug-template

compile: mhmm.i setup.py $(LIB_DIR)*.h $(RTML_DIR)*.h $(RTML_DIR)*.cpp #$(LIB_DIR)*.cpp
	@echo "\nCompiling Library\n"
	python doxy2swig.py $(DOC_DIR)index.xml mhmm_doc.i
	swig -c++ $(SWIGOPT) -I$(LIB_DIR) -I$(RTML_DIR) -python mhmm.i
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
	rm -rf build/src/ build/temp*
	@echo "\nDone"
