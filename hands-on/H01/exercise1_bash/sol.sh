#! /bin/bash

mkdir -p bandpasses_filtered

pushd bandpasses

for f in $( ls )
do
	ar=(${f//./ })
	name=${ar[0]}
	ext=${ar[1]}
	if grep -E "#?energy" $f
	then 
		type=energy
	elif grep -E "#?photon" $f
	then
		type=photon
	else
		type=unknown
	fi
	echo ../bandpasses_filtered/$name.$type.filter
	cp $f ../bandpasses_filtered/$name.$type.filter
done
popd
