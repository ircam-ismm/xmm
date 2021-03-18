#!/bin/sh

INSTALL_DIR=bin/xmm
mkdir -p ${INSTALL_DIR}
cp _xmm.so bin/Release/xmm/xmm.py src/__init__.py ${INSTALL_DIR}