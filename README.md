# To-Orpheus-Or-Not
A machine learning program written in Python that trains off of images to learn what is Hackclub's mascot Orpheus, and what is not.

## Requirements
* OpenCV
* tensorflow
* scikit-learn

## Usage
 In order to use the program, you will first have to train the model.

### Training
 Simply run the main.py file, it will take the images in ```data/``` and start training the model on it.
 A file called ```model.h5``` willbe created. This is the model file

### Testing
 Run the test.py file, it will take images named ```test.png``` and ```test2.png```. Make sure to modify the test.py file to suit your needs.
 It will then run the model and display the predicted class.
 
### Screenshots
![First Screenshot of training the model](https://github.com/wyn-cmd/To-Orpheus-Or-Not/blob/main/Screenshot-1.png?raw=true)
![Second Screenshot of testing model](https://github.com/wyn-cmd/To-Orpheus-Or-Not/blob/main/Screenshot-2.png?raw=true)
