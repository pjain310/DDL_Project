# Clearing the Haze from Climate Models

This repository contains code to build a model to classify cloud organization patterns from satellite images.
Cloud climate feedback is one of the most challenging research areas in climate science. Traditional rule-based algorithms fail to create boundaries between different forms of clouds and thus there is a need to improve classification methods to help scientists understand how clouds will help to predict future climate change.  The code leverages mask-RCNN to improve the identification of cloud organization patterns and store them within a database called Eva.

In addition, it also contains loader functions to load our image data (pixel arrays) into EVA (https://github.com/georgia-tech-db/Eva)

## Requirements

There are packages that need to be installed to seamlessly execute the code. They can be installed easily via terminal using the following command:


```bash
pip install -r requirements.txt
```

## Modified Files

1. preprocessing.ipynb : contains code for the object which would store image-wise information about the mask, labels, in addition to utilities for resizing the image and loading it.
This file also contains code to train the model and make predictions for the test dataset. The code takes approx ~9 hrs to run. For convenience, we have added the h5 file for the model to the repository.

2. /Eva/src/loaders/loader_uadtra_cloud.py: contains our modified code to add our cloud dataset into EVA.
We will be modifying this and adding a generalised loader which can be used for images soon!

3. Preliminary Code.ipynb : exploratory data analysis and image decoding scripts included. Not used in the final analysis.


## License
[MIT](https://choosealicense.com/licenses/mit/)
