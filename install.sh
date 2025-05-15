#!/bin/bash

echo "╔══╣ Install: Ollama ROS (STARTING) ╠══╗"

sudo apt update

sudo apt install ros-humble-vision-msgs

pip3 install numpy==1.23.5

pip3 install soundfile

pip3 install pygame

pip3 install git+https://github.com/getuka/RubyInserter.git

pip3 install flash-attn --no-build-isolation

cd ~/colcon_ws/src/

git clone -b humble-devel https://github.com/TeamSOBITS/sobits_msgs.git

echo "╚══╣ Install: Ollama ROS (FINISHED) ╠══╝"