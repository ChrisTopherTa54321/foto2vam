# Generate training data from existing faces
from Utils.Face.vam import VamFace
import argparse
import glob
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.initializers import RandomUniform
import numpy
import csv


# Work around low-memory GPU issue
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#numpy.random.seed(7)


# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#
#     def on_epoch_end(self, epoch, logs={} ):
#         print( " )

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    validationCsv = args.validationCsv
    trainingCsv = args.trainingCsv
    outputModelFile = args.outputFile
    sampleJson = args.sampleJson


    encodingFile = "{}_angle0.encoding".format( os.path.splitext(sampleJson)[0])
    if not ( os.path.exists(encodingFile)):
        raise Exception("Sample json didn't have an angle0 encoding!")
    tmpFace = VamFace(sampleJson)
    with open(encodingFile) as f:
        encoding = f.read().splitlines()

    inputSize = 2*len(encoding)
    outputSize = len(tmpFace.morphFloats)
    print( "{} : {}".format(inputSize, outputSize))
    dataSet = numpy.loadtxt(trainingCsv, delimiter=",")
    X=dataSet[:,0:inputSize]
    Y=dataSet[:,inputSize:]

    dataSet = numpy.loadtxt(validationCsv, delimiter=",")
    vX=dataSet[:,0:inputSize]
    vY=dataSet[:,inputSize:]

    print("Dataset: {}\nX: {}\nY: {}\n".format(dataSet.shape, X.shape, Y.shape))

    if os.path.exists(outputModelFile):
        print("Loading existing model")
        model = load_model(outputModelFile)
    else:
        model = generateModel( numInputs = inputSize, numOutputs=outputSize )

    print("Training...")
    while True:
        model.fit(X,Y, epochs=50, batch_size=256, verbose=0)
        scores= model.evaluate(vX,vY, verbose=0)
        print("Saving progress... {}".format(scores))
        model.save(outputModelFile)

def generateModel( numInputs, numOutputs ):
    print("Generating a model with {} inputs and {} outputs".format(numInputs, numOutputs))
    model = Sequential()
    layer1 = 10*numInputs
    layer2 = 5*numInputs
    layer3 = 3*numInputs
    print("Layer 1: {}\nLayer 2: {}\nLayer 3: {}".format(layer1, layer2, layer3))

    model.add( Dense( layer1, input_dim=numInputs, kernel_initializer='RandomUniform' ) ) #, activation='relu' ))
    model.add( LeakyReLU() )
    model.add( Dropout(.5))
    model.add( Dense( layer2, kernel_initializer='RandomUniform' ) )#, activation='relu'))
    model.add( LeakyReLU() )
    model.add( Dropout(.5))
    model.add( Dense( layer3, kernel_initializer='RandomUniform' ) )#, activation='relu'))
    model.add( LeakyReLU())
    model.add( Dense( numOutputs, activation="linear" ) )
    model.compile( loss='logcosh', optimizer='adam')
    #model.compile( loss='logcosh', optimizer='SGD')

    return model

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--trainingCsv', help="Path to training CSV", required=True)
    parser.add_argument('--validationCsv', help="Path to training containing validation JSON and encoding files", required=True)
    parser.add_argument('--outputFile', help="File to write output model to")
    parser.add_argument('--sampleJson', help="Just a json file to get the model size from")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )