#!/bin/bash

if [ "$(id -u)" != "0" ]; then
   echo "This file must be run as root." 1>&2
   exit 1
fi

rm -rf /usr/local/hiperwalk/*.py
rm -rf /usr/local/hiperwalk/*.nbl

echo "[Hiperwalk installer] Creating /usr/local/hiperwalk."
mkdir -p /usr/local/hiperwalk

echo "[Hiperwalk installer] Copying new files."
mv *.py /usr/local/hiperwalk
mv *.nbl /usr/local/hiperwalk
cp LICENSE /usr/local/hiperwalk
cp README.md /usr/local/hiperwalk
cp -r examples /usr/local/hiperwalk
mv ./examples/* .
rm -rf ./examples

echo "[Hiperwalk installer] Creating dynamic link to /usr/bin/hiperwalk"
chmod +x /usr/local/hiperwalk/hiperwalk.py 

ln -f -s  /usr/local/hiperwalk/hiperwalk.py /usr/bin/hiperwalk

echo "[Hiperwalk installer] Hiperwalk installed at /usr/local/hiperwalk."
rm -rf install.sh
