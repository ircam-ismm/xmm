INSTALLATION

=== Requirements
- Swig
- Python 2.7 + Numpy / Scipy

=== Building
- cd to this folder and type "make"
    To change the install location, change INSTALL_DIR in the make file

=== using
Place the built python library somewhere in your python path. To add personal 
libraries located in '/Path/To/Libs' to the python path, add the following 
lines to your ".bach_profile":

PYTHONPATH=$PYTHONPATH:/Path/To/Libs
export PYTHONPATH

The library can then be imported and used in python.

