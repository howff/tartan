#!/bin/bash

mkdir -p training
mkdir -p validation

for jpg in images/*.jpg; do
	dir=$(basename $jpg .jpg)
	file=$(basename $jpg)
	mkdir -p "training/$dir"
	mkdir -p "validation/$dir"
	ln -s  "../../$jpg"  "training/$dir/$file"
	ln -s  "../../$jpg"  "validation/$dir/$file"
	echo $dir/$jpg
done
