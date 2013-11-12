# INSTALLATION

## Requirements
* Swig
* Python 2.7 + Numpy / Scipy

## Building
### Xcode:
use Xcode project in "build/xcode/"

The installation location can be changed in "config.xcconfig", as well as the path to swig.

### Command-line build:
move to this directory and type:

	make
	make install
	make clean
To install in a specific location:

	make install INSTALL_DIR=/path/to/install/location/
To specify swig location:

	make install SWIG=/path/to/swig/

# usage
Place the built python library somewhere in your python path. To add personal 
libraries located in '/Path/To/Libs' to the python path, add the following 
lines to your ".bash_profile":

	PYTHONPATH=$PYTHONPATH:/Path/To/Libs
	export PYTHONPATH
