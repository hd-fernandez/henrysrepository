This is my capstone project to complete my M.S. in Analytics at the University of Chicago.

The goal of this project was to enable speech recognition on a new, extremely power efficient microchip called the MAX78000. The two major
constraints of the projects were that 1.) The key word spotting must be done entire locally and 2.) It must fit the power constraints of the chip.
We could not use the current state-of-the art methods as they're too power hungry for the MAX78K to handle, so we had to develop a 2-part Neural Network
solution to complete the task within the constraints. The first part is a Fully Connected Neural Network that takes in an audio signal and estimates its
corresponding Mel-Frequency Cepstral Coefficient (MFCC), and the second part is a Convolutional Neural Network that takes the estimated MFCC and classifies
it. The dataset we trained this on is the Google Speech Commands dataset.

Also! The one of the cool features of the MAX78000 and our solution is that it's fully trainable! So you can enable any words you'd like. 
