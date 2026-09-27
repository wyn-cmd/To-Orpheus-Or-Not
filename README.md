# To-Orpheus-Or-Not
A machine learning program written in Python that trains off of images to learn what is Hackclub's mascot Orpheus, and what is not.

## Requirements
* OpenCV
* numpy
* tensorflow
* scikit-learn

## Usage
 In order to use the program, you will first have to train the model.

### Training
 Simply run the main.py file, it will take the images in ```data/``` and start training the model on it.
 The folder an image sits in becomes its label, so ```data/Orpheus``` and ```data/Not-Orpheus``` are the two classes. Every image is resized to 150x150 and divided by 255 before it reaches the network.
 A file called ```model.h5``` will be created. This is the model file

### Testing
 Run the test.py file, it will take images named ```test.png``` and ```test2.png```. Make sure to modify the test.py file to suit your needs.
 It will then run the model and display the predicted class.
 test.py resizes to the same 150x150 as training, so if you change the size in one script you have to change it in the other.

### Screenshots
![First Screenshot of training the model](https://github.com/wyn-cmd/To-Orpheus-Or-Not/blob/main/Screenshot-1.png?raw=true)
![Second Screenshot of testing model](https://github.com/wyn-cmd/To-Orpheus-Or-Not/blob/main/Screenshot-2.png?raw=true)
