#!/bin/bash

# Function to download files using wget
download_files() {
    for i in {1..5}; do
	wget "https://lobogit.unm.edu/jschueler1/migYOLO/-/raw/main/migYOLO/data_and_models${i}.zip"
    done
}

# Function to unzip the given file
unzip_directory() {
    local zip_file_path=$1
    local extract_to=$2
    unzip -q "$zip_file_path" -d "$extract_to"
}

# Function to make directories
makedirs() {
    local paths=("data/"
		 "data/benchmark/random_benchmark_image/"
                 "data/benchmark/high_occupancy_image/"
                 "data/dark/"
                 "data/raw_images/"
                 "models/")
    for path in "${paths[@]}"; do
        if [ ! -d "$path" ]; then
            mkdir -p "$path"
        fi
    done
}

# Main function
main() {
    echo "Downloading zip files"
    download_files
    echo "Done!"

    echo "Unzipping files"
    for i in {1..5}; do
        unzip_directory "data_and_models${i}.zip" ""
    done
    echo "Done!"

    echo "Making relevant directories"
    makedirs

    echo "Moving files from data_and_models1.zip"
    mv zipped_files1/benchmark/random_benchmark_image/random.MTIFF data/benchmark/random_benchmark_image/random.MTIFF
    mv zipped_files1/benchmark/high_occupancy_image/high_occupancy.MTIFF data/benchmark/high_occupancy_image/high_occupancy.MTIFF
    mv zipped_files1/dark/sample_master_dark.npy data/dark/sample_master_dark.npy
    
    echo "Moving files from data_and_models[1-5].zip"
    for i in {2..5}; do
        mv "zipped_files${i}/raw_images"/* "data/raw_images/"
    done
    
    mv zipped_files1/models/* models/
    
    echo "Cleaning up"
    for i in {1..5}; do
        rm "data_and_models${i}.zip"
	rm -rf "zipped_files${i}/"
    done

    echo "SUCCESS!"
}

# Execute main function
main
