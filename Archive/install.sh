#!/bin/bash

#if [ "$(id -u)" != "0" ]; then
#   echo "This file must be run as root." 1>&2
#   exit 1
#fi

INST_DIR=$(awk '/INSTALL/{ split($0, array , "="); print array[2]}' config.py | sed 's/"//g;')


rm -rf $INST_DIR/hiperwalk/*.py
rm -rf $INST_DIR/hiperwalk/*.nbl
rm -rf $INST_DIR/hiperwalk/hiperwalk

echo "[Hiperwalk installer] Creating $INST_DIR/hiperwalk."
mkdir -p $INST_DIR/hiperwalk

echo "[Hiperwalk installer] Copying new files."
mv *.py $INST_DIR/hiperwalk
mv *.nbl $INST_DIR/hiperwalk
cp LICENSE $INST_DIR/hiperwalk
cp README.md $INST_DIR/hiperwalk
cp -r examples $INST_DIR/hiperwalk
mv ./examples/* .
rm -rf ./examples

echo "[Hiperwalk installer] Creating dynamic link to /usr/bin/hiperwalk"
chmod +x $INST_DIR/hiperwalk/hiperwalk.py 

ln -f -s  $INST_DIR/hiperwalk/hiperwalk.py /usr/bin/hiperwalk

echo "[Hiperwalk installer] Hiperwalk installed at $INST_DIR/hiperwalk."
rm -rf install.sh
