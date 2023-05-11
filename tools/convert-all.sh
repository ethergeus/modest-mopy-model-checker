#!/bin/bash
for file in "$1"/*.{modest,jani}; do modest export-to-python $file --output ${file%.*}.py ${@:2}; done
exit 0
