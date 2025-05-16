Python Program Installation and Execution Guide
-Ensure your Linux system has Python installed. If not, follow the steps below.

    Installing Python on Linux
    Update the package list:
        sudo apt update   # For Debian/Ubuntu
        sudo dnf update   # For Fedora
        sudo pacman -Syu  # For Arch Linux

    Install Python:
        sudo apt install python3  # For Debian/Ubuntu
        sudo dnf install python3  # For Fedora
        sudo pacman -S python     # For Arch Linux
    Verify installation:
        python3 --version

    Install Dependencies
        pip install -r requirements.txt

    Execute the Python script:
        python test1.py  
