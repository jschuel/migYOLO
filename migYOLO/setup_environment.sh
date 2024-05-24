#!/bin/bash

# Function to unzip the given file
unzip_directory() {
    local zip_file_path=$1
    local extract_to=$2
    unzip -q "$zip_file_path" -d "$extract_to"
}

# Function to make directories
makedirs() {
    local paths=("data/benchmark/random_benchmark_image/"
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
    local zip_file_path="data_and_models.zip"
    local extract_to=""

    if [ ! -f "$zip_file_path" ]; then
        echo "Script will not run unless there is a data_and_models.zip file to unzip!"
        exit 1
    fi

    echo "Unzipping data_and_models.zip"
    unzip_directory "$zip_file_path" "$extract_to"
    echo "Done!"

    echo "Making relevant directories"
    makedirs

    echo "Moving files"
    mv "zipped_files/random_benchmark_image/random.MTIFF" "data/benchmark/random_benchmark_image/random.MTIFF"
    mv "zipped_files/high_occupancy_image/high_occupancy.MTIFF" "data/benchmark/high_occupancy_image/high_occupancy.MTIFF"
    mv "zipped_files/dark/sample_master_dark.npy" "data/dark/sample_master_dark.npy"
    mv zipped_files/raw_images/* data/raw_images/
    mv zipped_files/models/* models/
    
    echo "Cleaning up"
    rm "$zip_file_path"
    rm -rf "zipped_files/"

    echo "SUCCESS!"
}

# Execute main function
main
