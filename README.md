# ASD_classification
Classification of ASD (autism spectrum disorder) vs TD (typically developing) children based on their scanpaths of image viewing. 

This is an ICME2019 GrandChallenge "Saliency4ASD" submission, corresponding paper - "CLASSIFYING AUTISM SPECTRUM DISORDER BASED ON SCANPATHS AND SALIENCY". Please cite this work if you are using the code from this repository.


## Installation

### Additional Python packages

 `pip install numpy pandas scipy scikit-learn Pillow`

### \[Optional\] Install dlib
1. `pip install dlib`
2. Create a folder `face_detection`, download the archive http://dlib.net/files/mmod_human_face_detector.dat.bz2
   and extract the file within into `face_detection/mmod_human_face_detector.dat`

### Install SAM-ResNet (2017)

The code itself is already provided in the `saliency_attentive_model` folder. You just need to satisfy
the requirements for it to run and download the pre-trained weights:

1. Follow the instructions here: https://github.com/marcellacornia/sam#requirements.
Pay special attention to the library versions.

2. [Should not be necessary anymore, the weights are now also in this repository; these instructions are here for reproducibility] 
Download the model weights from https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_salicon2017_weights.pkl
and put those in the SAM repository folder into `weights/sam-resnet_salicon_weights.pkl`
(note the name difference between the downloaded file and its destination!)


## Usage

The script can operate on scanpath files that contain either single or multiple scanpaths, but all necessarily
recorded for the same stimulus image. To run the code on such data, use the following:

> python classifier_final.py /path/to/scanpath/file.txt /path/to/image/file.png

This will output (in separate lines) the probabilities of each of the scanpaths in the input file
belonging to the ASD class. If you desire the class label instead (NB: less fine-grained),
add `--mode class` to the command above.

The saliency prediction code may generate excessive amount of unrelated output that is difficult to
suppress. Add `--out /path/to/output/file.txt` to redirect only the relevant output to the specified
file. If you wish to accumulate the output for several scanpaths in one file, you can also pass
`--append`, which will then open the output file and append the new class labels or probabilities
to that file.

### Not re-computing the same saliency multiple times

Since in a generic case the assumption that if two images share the name (e.g. `100.png`)
they are necessarily the same image cannot be applied, the `final_classifier.py` always
re-computes saliency maps for the stimulus images.

This may cause performance issues if every single test set scanpath is stored in a separate file.
Storing the scanpaths in a single file is advised for faster computation, but either way
**the vast majority of the overhead can be avoided** by passing the `--eff` argument.
This will assume that all images with the same names are the same, and thus not
re-compute the saliency prediction for every scanpath of every image.
**Unless this assumption is satisfied**, do not pass this argument!

### Final suggested command to execute

To accumulate the output (e.g. for all scanpaths of the same class) in /tmp/probabilities.txt run:

> python classifier_final.py /path/to/scanpath/file.txt /path/to/image/file.png --out /tmp/probabilities.txt --append --eff

