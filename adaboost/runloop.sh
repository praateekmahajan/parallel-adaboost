#!/bin/bash
counter=1
while [ $counter -le 20 ]
do
	for (( k = 1; k < 33; k = k * 2 )); do
		./a.out "$k" << output.txt
	done
	counter=$[$counter+1]
done

