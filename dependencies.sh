#!/bin/bash
#Installs all dependencies for n-body CUDA simulation

# Base development tools
sudo apt update
sudo apt install git -y
sudo apt install build-essential -y
sudo apt install cmake -y

# CUDA toolkit
sudo apt install nvidia-cuda-toolkit -y

# OpenGL dependencies
sudo apt install mesa-utils -y
sudo apt install freeglut3-dev -y
sudo apt install libgl1-mesa-dev -y
sudo apt install libglu1-mesa-dev -y

# GLFW dependencies
sudo apt install libglfw3-dev -y
sudo apt install libglew-dev -y
sudo apt install libxrandr-dev -y
sudo apt install libxinerama-dev -y
sudo apt install libxcursor-dev -y
sudo apt install libxi-dev -y

# For image processing/video capture
sudo apt install ffmpeg -y

# Ensure permissions are set properly
chmod +x compile

echo "All dependencies installed successfully!"