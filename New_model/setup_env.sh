#!/bin/bash

echo "Setting up ASVspoof5 Environment..."

# Update system
apt-get update && apt-get install -y libsndfile1

# Install Python deps
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment Ready."
