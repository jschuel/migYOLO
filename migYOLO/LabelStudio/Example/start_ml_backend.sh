#!/bin/bash                                                                                          
# Prompt the user for the ML backend name                                                            
read -p "Enter the name for the ML backend instance you would like to start: " backend_name

# Run the label-studio-ml init command with the provided name                                        
label-studio-ml start ./"$backend_name"
