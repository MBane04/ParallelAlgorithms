# Install required libraries
sudo apt-get update
sudo apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module libglu1-mesa

# For Qt dependencies
sudo apt-get install -y libqt5widgets5 libqt5core5a libqt5gui5 libqt5dbus5

# 1. Download the standalone installer
wget https://developer.nvidia.com/nsight-systems/download

# 2. Install the package
sudo apt install ./nsight-systems-linux-public-2023.2.1.122-3259725.deb  # Adjust filename


echo "NSight Systems installation completed."