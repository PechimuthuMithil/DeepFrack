#!/bin/bash

# Function to display a progress bar
progressbar() {
    local duration=${1}
    local total=${2}
    local progress=0
    local progress_percentage=0
    local progress_bar=''
    local empty_progress_bar=$(printf ' %.0s' $(seq 1 $total))

    while [ $progress -lt $total ]; do
        progress=$(($progress+1))
        progress_percentage=$((($progress*100/$total*100)/100))
        progress_bar+="-"
        remaining_progress_bar=${empty_progress_bar:0:$total-$progress}
        printf "\rProgress: [%s%s] %s%%" $progress_bar $remaining_progress_bar $progress_percentage
        sleep $duration
    done
    echo
}

# Function to install Python libraries
install_python_libs() {
    echo "Installing Python and necessary libraries..."
    echo

    # Check if Python is installed
    if command -v python3 &>/dev/null; then
        echo "Python already installed."
    else
        echo "Installing Python..."
        progressbar 0.03 100 &
        sudo apt-get update &>/dev/null
        sudo apt-get install python3 -y &>/dev/null || { echo "Failed to install Python. Exiting..."; exit 1; }
        echo "Python installed successfully."
    fi

    # Install required Python libraries
    echo "Installing Python libraries..."
    progressbar 0.03 100 &
    sudo -H pip3 install numpy yaml json ast matplotlib &>/dev/null || { echo "Failed to install Python libraries. Exiting..."; exit 1; }
    echo "Python libraries installed successfully."

    echo
}

# Main function
main() {
    clear
    echo " _____                      _______               _     "
    echo "(____ \                    (_______)             | |    "
    echo " _   \ \ ____ ____ ____     _____ ____ ____  ____| |  _ "
    echo "| |   | / _  ) _  )  _ \   |  ___) ___) _  |/ ___) | / )"
    echo "| |__/ ( (/ ( (/ /| | | |  | |  | |  ( ( | ( (___| |< ( "
    echo "|_____/ \____)____) ||_/   |_|  |_|   \_||_|\____)_| \_)"
    echo "                  |_|                                    "
    echo

    # Install Python and required libraries
    install_python_libs
}

# Execute the main function
main
