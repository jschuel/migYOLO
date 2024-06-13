#!/bin/bash

# Prompt the user for the ML backend name
read -p "Enter the name for your ML backend project: " backend_name

# Run the label-studio-ml init command with the provided name
label-studio-ml init "$backend_name" --script preannotate.py

