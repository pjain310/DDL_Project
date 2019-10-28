# Clearing the Haze from Climate Models

This repository contains code to build a model to classify cloud organization patterns from satellite images.
In addition, it also contains loader functions to load our image data (pixel arrays) into EVA (https://github.com/georgia-tech-db/Eva)

## Requirements

There are packages that need to be installed to seamlessly execute the code. They can be installed easily via terminal using the following command:


```bash
pip install -r requirements.txt
```

## Files to look out for 

The preprocessing.ipynb contains code for the object which would store image-wise information about the mask, labels, in addition to utilities for resizing the image and loading it.
This file also contains code to train the model and make predictions for the test dataset. The code takes approx ~9 hrs to run. For convenience, we have added the h5 file for the model to the repository.

The folder for EVA has been forked into our repository as well. The /Eva/src/loaders/loader_uadtra_cloud.py contains our modified code to add our cloud dataset into EVA.
We will be modifying this and adding a generalised loader which can be used for images in general soon!


## License
[MIT](https://choosealicense.com/licenses/mit/)
