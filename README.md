# MEng Micro Pasts Project

This repository holds data/scripts related to the [Micro
pasts](http://micropasts.org/) project.

## Directories

``jupyter_notebooks`` holds IPython/Jupyter notebooks.

``scripts`` hosts useful scripts.

## Installation

The Python scripts require some third-party libraries. These are listed in the
``requirements.txt`` file and can be installed via:

```console
$ pip install -r requirements.txt
```

## Running scripts - 31/10/15

In order to train and test a classifier:

First call "%run stackImages _path to image folder_"
to concatenate the image and mask set for classification. E.g:
"%run stackImages C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe1"

Next call "%run createClassifier _path to image folder_ _classifier type_"
to train the classifier on the ".npz" file created by "stackImages.py". The same
path should be used as for "stackImages" as this is where the ".npz" file is saved.
The type of classifier must also be specified, currently only LinearSVC is available.

Finally call "%run useClassifier _path to image folder_ _classifier type_"
to test the classifier on the test set and produce an average pixel error. Currently
the error metric is simply absolute difference summed over all pixel classifications.
Path should again be the same as "stackImages" as this is where the classifier will
have been saved to from the previous step in pickle form.

## Notes on Scripts

stackImages has a parameter inside the script which specifies the amount of sampling
of an image and it's mask. Currently the sampler chooses an equal number of foreground
(1 in mask) and background (0 in mask) pixels. The (N,3) and (N,) arrays are then saved
in the path folder as a ".npz".

createClassifier has parameters for choosing amount of training vs testing data (set to
1:4 as of 31/10/15). An error will be thrown if a classifier is not recognised.