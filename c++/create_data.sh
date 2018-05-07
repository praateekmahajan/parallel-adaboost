#!/bin/bash
counter=1
for (( i = 1000; i < 100001 ; i = i * 10 )); do	
	for (( j = 10; j < 10001; j = j * 10)); do
		python create-dataset.py "$i" "$j" 
	done
done

