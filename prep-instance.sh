#!/bin/bash

sudo mkdir ~/ebs_disk
sudo mount /dev/xvdb ebs_disk
git clone https://github.com/jsalvatier/deep-go.git
itorch notebook --ip='*' --port=8889 --browser=none

