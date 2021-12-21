#!/bin/bash

function bcp() {
ndir=0
for dir in `ls $1`
do
    if [[ $ndir -lt $3 ]]; then
		echo "cur dir:"$dir
        ndir=$(( $ndir + 1 ))
		mkdir -p $2/$dir
		nfile=0
		for file in `ls $1/$dir`
		do
			if [[ $nfile -lt $4 ]]; then
				echo "    cur file:"$file
				nfile=$(( $nfile + 1 ))
				cp $1/$dir/$file $2/$dir/$file
			fi
		done
	fi
done
}

src="/chen/dataset/mini-imagenet/train"
dest="/chen/ahou/Swin-CBIR/database/data"
num_class=20
num_per_class=10

bcp $src $dest $num_class $num_per_class
