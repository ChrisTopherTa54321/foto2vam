# Generate training data from existing faces
from Utils.Face.vam import VamFace
import argparse
import glob
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import numpy
import csv

# Work around low-memory GPU issue
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


numpy.random.seed(7)
###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    inputPath = args.inputPath
    outputModelFile = args.outputFile

#     face = None
#     encoding = None
#     # Read in all of the files from inputpath
#     outFile = open("test.csv",'w')
#     writer = csv.writer( outFile, lineterminator='\n')
#     print("Generating test.csv")
#     for jsonFile in glob.glob(os.path.join(inputPath, '*.json')):
#         encodingFile = "{}_angle0.encoding".format( os.path.splitext(jsonFile)[0])
#         if os.path.exists(encodingFile):
#             face = VamFace(jsonFile)
#             with open(encodingFile) as f:
#                 encoding = f.read().splitlines()
#             outArray = encoding
#             outArray.extend(face.morphFloats)
#             writer.writerow(outArray)
#     exit()



    for jsonFile in glob.glob(os.path.join(inputPath, '*.json')):
        encodingFile = "{}_angle0.encoding".format( os.path.splitext(jsonFile)[0])
        if not ( os.path.exists(encodingFile)):
            continue
        tmpFace = VamFace(jsonFile)
        with open(encodingFile) as f:
            encoding = f.read().splitlines()        
        break
    inputSize = len(encoding)
    outputSize = len(tmpFace.morphFloats)
    print( "{} : {}".format(inputSize, outputSize))
    dataSet = numpy.loadtxt("test.csv", delimiter=",")
    X=dataSet[:,0:inputSize]
    Y=dataSet[:,inputSize:]
    
    print("Dataset: {}\nX: {}\nY: {}\n".format(dataSet.shape, X.shape, Y.shape))

    if os.path.exists(outputModelFile):
        print("Loading existing model")
        model = load_model(outputModelFile)
    else:
        model = generateModel( numInputs = len(encoding), numOutputs=len(tmpFace.morphFloats) )
    
    print("Training...")
    while True:
        model.fit(X,Y, epochs=100, batch_size=512, verbose=0)
        model.save(outputModelFile)
        scores= model.evaluate(X,Y, verbose=0)
        print("Saving progress... %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def generateModel( numInputs, numOutputs ):
    print("Generating a model with {} inputs and {} outputs".format(numInputs, numOutputs))
    model = Sequential()
    layer1 = 4*int(numpy.mean((numInputs,numOutputs)))
    layer2 = 2*numpy.min((numInputs,numOutputs))
    layer3 = 3*numpy.min((numInputs,numOutputs))
    print("Layer 1: {}\nLayer 2: {}\nLayer 3: {}".format(layer1, layer2, layer3))
    
    model.add( Dense( layer1, input_dim=numInputs, activation='relu' ))
    model.add( Dense( layer2, activation='tanh'))
    model.add( Dense( layer3, activation='relu'))
    model.add( Dense( numOutputs, activation='relu') )
    model.compile( loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    
    return model

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing JSON and encoding files", required=True)
    parser.add_argument('--outputFile', help="File to write output model to")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )