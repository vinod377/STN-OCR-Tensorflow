# STN-OCR
Detecting and recognizing text in natural scene images. This is still an open problem for research community and has many usge like image-based machine translation, autonomous cars or image/video indexing. The Algorithm consist of two stages which are **1.Text Dectection 2.Text Recognition** stages. The Text Dection stage uses **Resnet-Cifar** version of Deep Residual Learning for Image Recognition ("https://arxiv.org/abs/1512.03385") and Spatial Transformer Network by Max Jaderberg ("https://arxiv.org/abs/1506.02025").The Text Detection and Recognition Stage again contains a variant Resnet-cifar version.The whole model is trained end-to-end.<br />

## Dependencies
• Python-3.x <br />
• Tensorflow-2.3.1 <br />
• Opencv-4.x <br />
• Numpy <br />
• sklearn <br />

## Repository Description
• main.py script creates the whole model consiting of localisation network, Grid generator and sampler and Recognition network.<br />
• stn_network.py script crates spatial transformer network, Grid genrator and bilinearsampler.<br />
• resnet_stn.py script creates detection and recognition resnet network as proposed by the author.<br />

## Dataset
• The Street View House Numbers (SVHN) Dataset.[http://ufldl.stanford.edu/housenumbers/]<br />
• Google FSNS dataset.[https://rrc.cvc.uab.es/?ch=6]

***Note-The wok is in progress and the repo will be updated frequently.**


### Description
- Update the runner code to take command line arguments instead of reading it from Json
- Replaced Segmentation model from 79 to 47. 

### Changes Made
- Updated the code to take CLI arguments.
- Removed dependency from Json file.
- Removed ankle and hip related code.
- Changed Segmentation model to 47.
- Updated Dockerfile.


### Related Issues
<Provide links to the related issues or feature requests.>

### Additional Notes
- The code output has dependency on mymako_osteophytes and mymako_bmd.

### Merge Request Checklists
- [ ] Code follows project coding guidelines.
- [ ] Documentation reflects the changes made.
- [x] I have already covered the unit testing.


