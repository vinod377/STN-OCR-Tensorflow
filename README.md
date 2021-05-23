# STN-OCR
Detecting and recognizing text in natural scene images.
The Algorithm consist of two stages which are **1.Text Dectection 2.Text Recognition** stages.<br /> 
The Text Dection stage uses **Resnet-Cifar** version of Deep Residual Learning for Image Recognition ("https://arxiv.org/abs/1512.03385") and Spatial Transformer Network by Max Jaderberg ("https://arxiv.org/abs/1506.02025").<br />
The Text Recognition Stage again contains Resnet-cifar version.The whole model is trained together.

## Dependencies
•Python-3.x <br />
•Tensorflow-2.3.1 <br />
•Opencv-4.x <br />
•Numpy <br />
•sklearn <br />

## Repository Structure
• main.py script creates the whole model consiting of localisation network, Grid generator and sampler and Recognition network.<br />
• stn_network.py script crates spatial transformer network, Grid genrator and bilinearsampler.<br />
• resnet_stn.py script creates detection and recognition resnet network as proposed by the author.<br />

***Note-The wok is in progress and the repo will be updated frequently.**

