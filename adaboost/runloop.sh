#!/bin/bash
counter=1
while [ $counter -le 2 ]
do
	for (( k = 1; k < 33; k = k * 2 )); do
		for (( i = 1000; i < 10000001 ; i = i * 10 )); do	
			for (( j = 10; j < 10001; j = j * 10)); do
				for (( h = 5; h < 161; h = h * 2)); do
					#python ../create-dataset.py "$i" "$j"
					#./a.out "$k" >> output.txt
					./a.out "$k" "$h" "$i" "$j" >> output.txt
				done
			done
		done
	done
	counter=$[$counter+1]
done

