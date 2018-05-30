# Generate training data from existing faces
import argparse
import os
import numpy
import collections
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.initializers import RandomUniform
from keras.optimizers import Adam

# Work around low-memory GPU issue
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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

    # First read parameters from trainingCsv and validation, ensure they match
    trainingParams = open(trainingCsv).readline()
    validationParams = open(validationCsv).readline()

    if trainingParams != validationParams:
        print("Training CSV mismatches Validation CSV! [{}] vs [{}]".format( trainingParams, validationParams ) )

    trainingParams = trainingParams.lstrip('#')
    trainingParams = trainingParams.split(',')
    configFile = trainingParams[0]
    inputSize = int(trainingParams[1])
    outputSize = int(trainingParams[2])

    print( "Using {} with {} inputs and {} outputs".format(configFile, inputSize, outputSize ))

    print( "Reading validation set...")
    dataSet = numpy.loadtxt(validationCsv, delimiter=',', comments='#')
    vX=dataSet[:,0:inputSize]
    vY=dataSet[:,inputSize:]
    print("Validation Dataset: {}\nX: {}\nY: {}\n".format(dataSet.shape, vX.shape, vY.shape))

    print( "Reading training set..." )
    dataSet = numpy.loadtxt(trainingCsv, delimiter=',', comments='#')
    X=dataSet[:,0:inputSize]
    Y=dataSet[:,inputSize:]
    print("Training Dataset: {}\nX: {}\nY: {}\n".format(dataSet.shape, X.shape, Y.shape))

    if os.path.exists(outputModelFile):
        print("Loading existing model")
        model = load_model(outputModelFile)
    else:
        model = generateModel( numInputs = inputSize, numOutputs=outputSize )

    print("Training...")
    scoreHistory = collections.deque( maxlen=5 )
    while True:
        scores= model.evaluate(vX,vY, verbose=0)
        scoreHistory.append(float(scores))
        print("Saving progress... {}  Last {}: {}".format(scores, len(scoreHistory), sum(scoreHistory)/len(scoreHistory)))
        model.save(outputModelFile)
        model.fit(X,Y, epochs=25, batch_size=16384, verbose=0, shuffle=True)

def generateModel( numInputs, numOutputs ):
    print("Generating a model with {} inputs and {} outputs".format(numInputs, numOutputs))
    model = Sequential()
    layer1 = 12*numInputs
    layer2 = 8*numInputs
    layer3 = 4*numInputs
    print("Layer 1: {}\nLayer 2: {}\nLayer 3: {}".format(layer1, layer2, layer3))

    model.add( Dense( layer1, input_dim=numInputs, kernel_initializer='RandomUniform' ) ) #, activation='relu' ))
    model.add( LeakyReLU() )
    model.add( BatchNormalization() )
    model.add( Dropout(.5))
    model.add( Dense( layer2, kernel_initializer='RandomUniform' ) )#, activation='relu'))
    model.add( LeakyReLU() )
    model.add( BatchNormalization() )
    model.add( Dropout(.5))
    #model.add( Dense( layer2, kernel_initializer='RandomUniform' ) )#, activation='relu'))
    #model.add( LeakyReLU() )
    #model.add( Dropout(.5))
    model.add( Dense( layer3, kernel_initializer='RandomUniform' ) )#, activation='relu'))
    model.add( LeakyReLU())
    model.add( Dense( numOutputs, activation="linear" ) )
    adam = Adam(lr=0.0005)
    model.compile( loss='logcosh', optimizer=adam)
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
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )