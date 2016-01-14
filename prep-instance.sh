#!/bin/bash

sudo mkdir ~/ebs_disk
echo "mounting"
sudo mount /dev/xvdb ~/ebs_disk
rm -rf deep-go
git clone https://github.com/jsalvatier/deep-go.git
nohup itorch notebook --ip='*' --port=8889 --browser=none & 

