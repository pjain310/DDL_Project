# Clearing the Haze from Climate Models

This repository contains code to build a model to classify cloud organization patterns from satellite images.
Cloud climate feedback is one of the most challenging research areas in climate science. Traditional rule-based algorithms fail to create boundaries between different forms of clouds and thus there is a need to improve classification methods to help scientists understand how clouds will help to predict future climate change.  The code leverages mask-RCNN to improve the identification of cloud organization patterns and store them within a database called Eva.

In addition, it also contains loader functions to load our image data (pixel arrays) into EVA (https://github.com/georgia-tech-db/Eva) and image preprocessing utilites for database-scale operations through EVA.

## Requirements

There are packages that need to be installed to seamlessly execute the code. They can be installed easily via terminal using the following command:


```bash
pip install -r requirements.txt
```

## Project Progress

### 1. Model Architecture and Data Analysis

1. cloud-classification.ipynb : Jupyter notebook walkthorugh which contains code for the object which would store image-wise information about the mask, labels, in addition to utilities for resizing the image and loading it.
This file also contains code to train the model and make predictions for the test dataset. The code takes approx ~9 hrs to run. For convenience, we have added the h5 file for the model to the repository.

2. Preliminary Code.ipynb : exploratory data analysis and image decoding scripts included. Not used in the final analysis.

### 2. Modifications to EVA

1. /Eva/src/loaders/loader_uadtra_cloud.py: contains our modified code to add our cloud dataset into EVA. Detection of images from given location. Modify. Use: python loader_uadetra_cloud.py --data_type='<i if image and v if video>' --path='<path to folder containing train images>'  

2. /Eva/src/loaders/loader_uadtra_cloud.py: also contains generalised loader to add image files to EVA. Modification is required.

3. Eva/src/operations/: contains preprocessing filters which can be applied to databse entries in EVA (both videos and image frames). We also have an api which can be modified to use these objects outside of Eva as well.

4. Eva/src/query_parser/insert_statement.py: this has the ability to insert queries. This was created using the select_statement.py file.

5. Eva/src/udfs/maskrcnn_object_detector.py: This has the ability to detect objects based on the model which we trained using a pre-trained maskRCNN model.  It detects clouds and their types from images and classifies them into the following categories viz. 'sugar', 'gravel', 'fish' and 'flower'

Currently, the following preprocessing operations are supported:
  * Blur
  * Equalised Histogram
  * Grayscale
  * Laplacian
  * Scharr
  * Sobel

We have created a pull request for preprocessing filters.
https://github.com/georgia-tech-db/Eva/pull/43


## Team

Questions? Please feel free to reach out to us at:
- Arnav Jeurkar: ajeurkar3@gatech.edu
- Prerna Jain: prernaj@gatech.edu
- Priyank Madria: priyankmadria@gatech.edu
- Shubham Singhal: ssinghal37@gatech.edu

## License
[MIT](https://choosealicense.com/licenses/mit/)
