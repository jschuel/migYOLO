#!/bin/bash

directory="timestamps"
rm "$directory"/*

read -p "Enter the number images would you like to benchmark (must be a multiple of 200):" integer_input

start=$(date +%s)

screen -dmS Downsample python3 downsample.py "$integer_input"
python3 runYOLO.py

# Record end time
end=$(date +%s)

# Calculate and print execution time
execution_time=$((end - start))
echo "All scripts have finished. Total execution time: $execution_time seconds."
