#!/bin/bash

echo "╔══╣ Install: Parler_tts_ros (STARTING) ╠══╗"

sudo apt update -y

sudo apt install -y ros-humble-vision-msgs

pip3 install numpy==1.23.5

pip3 install soundfile

pip3 install pygame

pip3 install git+https://github.com/getuka/RubyInserter.git

cd ~/colcon_ws/src/

git clone -b humble-devel https://github.com/TeamSOBITS/sobits_msgs.git

pip3 install parler_tts==0.2.3

pip3 uninstall -y torch torchvision torchaudio

pip3 install torch torchvision torchaudio

pip3 install transformers==4.46.1

#pip3 install flash-attn --no-build-isolation

echo "╚══╣ Install: Parler_tts_ros (FINISHED) ╠══╝"