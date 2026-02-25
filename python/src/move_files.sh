#!/bin/sh

INSTALL_DIR=bin/xmm
mkdir -p ${INSTALL_DIR}
cp _xmm.so ${INSTALL_DIR}
cp bin/Release/xmm/xmm.py ${INSTALL_DIR}
cp src/__init__.py ${INSTALL_DIR}