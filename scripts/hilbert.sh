#!/bin/bash

cd libs/Cabana
bash scripts/build.sh

cd ../..
bash scripts/build.sh

./build/examples/TestHilbert