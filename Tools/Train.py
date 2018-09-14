# Generate training data from existing faces
import argparse
import os
import numpy
import collections
from keras.models import load_model, Model
from keras.initializers import RandomUniform
from keras.optimizers import Adam

from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU

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

    if os.path.exists(outputModelFile):
        print("Loading existing model")
        model = load_model(outputModelFile)
    else:
        model = generateModel( numInputs = inputSize, numOutputs=outputSize )

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

    print("Training...")
    scoreHistory = collections.deque( maxlen=5 )
    while True:
        scores= model.evaluate(vX,vY, verbose=0)
        scoreHistory.append(float(scores))
        print("Saving progress... {}  Last {}: {}".format(scores, len(scoreHistory), sum(scoreHistory)/len(scoreHistory)))
        model.save(outputModelFile)
        #model.fit(X,Y, epochs=25, batch_size=16384, verbose=0, shuffle=True)
        model.fit(X,Y, epochs=25, batch_size=256, verbose=0, shuffle=True)

def generateModel( numInputs, numOutputs ):
    print("Generating a model with {} inputs and {} outputs".format(numInputs, numOutputs))
    layer1 = 2*numInputs
    layer2 = 10*numInputs
    layer3 = 5*numInputs
    print("Layer 1: {}\nLayer 2: {}\nLayer 3: {}".format(layer1, layer2, layer3))

    input_layer = Input(shape=(numInputs,))
    
    x = Dense( layer1, activation='linear' )(input_layer)
    x = LeakyReLU()(x)
    x = Dropout(.2)(x)
    
    x = Dense( layer1, activation='linear' )(input_layer)
    x = LeakyReLU()(x)
    x = Dropout(.2)(x)
    
    x = Dense( layer1, activation='linear' )(input_layer)
    x = LeakyReLU()(x)
    x = Dropout(.2)(x)
    
    output_layer = Dense( numOutputs, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    adam = Adam(lr=0.0001)
    model.compile( optimizer=adam,
                   loss='logcosh' )

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