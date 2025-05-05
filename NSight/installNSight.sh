#!/bin/bash
# Complete NSight Systems installation with importer support
set -e

echo "Installing full NSight Systems package with all components..."

# Install basic dependencies
sudo apt-get update
sudo apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module libglu1-mesa
sudo apt-get install -y libqt5widgets5 libqt5core5a libqt5gui5 libqt5dbus5
sudo apt-get install -y sqlite3 libsqlite3-0 libpcre2-16-0

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Option 1: Install the complete host-target package (includes importer)
echo "Installing complete NSight Systems package (2025.1.3)..."
sudo apt-get install -y nsight-systems-2025.1.3 nsight-systems-target

# Option 2: If you're still having issues, try the direct download
if ! which nsys-ui > /dev/null 2>&1; then
    echo "Attempting direct download method..."
    wget https://developer.download.nvidia.com/compute/nsight-systems/2025.1.3/nsight-systems-linux-host-target-2025.1.3.deb
    sudo apt install -y ./nsight-systems-linux-host-target-*.deb
fi

# Create symbolic links if needed
if [ -d "/opt/nvidia/nsight-systems" ]; then
    latest_dir=$(find /opt/nvidia/nsight-systems -maxdepth 1 -type d -name "20*" | sort -r | head -1)
    if [ -n "$latest_dir" ]; then
        echo "Adding NSight Systems to system PATH..."
        echo "export PATH=$latest_dir/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$latest_dir/lib:$latest_dir/lib/QtCore:$latest_dir/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        
        # Apply to current session
        export PATH="$latest_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$latest_dir/lib:$latest_dir/lib/QtCore:$latest_dir/lib64:$LD_LIBRARY_PATH"
    fi
fi

echo "Installation complete. Please log out and log back in, or run 'source ~/.bashrc'"
echo "Then try running your profiling command again."