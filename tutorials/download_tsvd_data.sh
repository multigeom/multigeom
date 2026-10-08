#!/usr/bin/env bash
# author: laquitainesteeve@gmail.com
# execution time: 15 secs

# Create target directory if it doesn't exist
mkdir -p data/tvsd
 
# download THINGS_normMUA.mat from GIN to data/tvsd/ (5 secs)
URL="https://gin.g-node.org/doi/TVSD/raw/master/monkeyN/THINGS_normMUA.mat"
OUTPUT="data/tvsd/monkeyN_THINGS_normMUA.mat"
curl -L "$URL" -o "$OUTPUT"

# download things_imgs.mat from GIN to data/tvsd/ (5 secs)
URL="https://gin.g-node.org/doi/TVSD/src/master/monkeyN/_logs/things_imgs.mat"
OUTPUT="data/tvsd/things_imgs.mat"
curl -L "$URL" -o "$OUTPUT"