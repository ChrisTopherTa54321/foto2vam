# Generate training data from existing faces
from Utils.Training.config import Config
from Utils.Face.vam import VamFace
import argparse
import glob
import os
import shutil
import numpy
import datetime

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    modelFile = args.modelFile
    modelCfg = os.path.splitext(modelFile)[0] + ".json"
    config = Config.createFromFile( modelCfg )
    inputDir = args.inputDir
    recursive = args.recursive
    outputDir = args.outputDir

    # Delay heavy imports
    from keras.models import load_model

    # Work around low-memory GPU issue
    import tensorflow as tf
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)

    model = load_model(modelFile)
    modelName = os.path.splitext(os.path.basename(modelFile))[0]

    face = config.getBaseFace()
    # Read in all of the files from inputDir
    for root, subdirs, files in os.walk(inputDir):
        if len(files) > 0:
            try:
                relatedFiles = []
                for file in files:
                    relatedFiles.append( os.path.join(root, file ) )

                outRow = config.generateParams( relatedFiles )
                outShape = config.getShape()
                dataSet = numpy.array([outRow[:outShape[0]]])
                predictions = model.predict(dataSet)
                rounded = [float(round(x,5)) for x in predictions[0]]
                face.importFloatList(rounded)

                outName = root.lstrip(inputDir)
                outputFolder = os.path.join( outputDir, outName )
                try:
                    os.makedirs(outputFolder)
                except:
                    pass
                folderName = os.path.split(root)[-1]
                outputFullPath = os.path.join( outputFolder, "{}_{}.json".format(folderName, modelName))
                face.save( outputFullPath )
                print( "Generated {}".format(outputFullPath) )
            except Exception as e:
                print( "Failed to generate model from {} - {}".format(root, str(e) ) )

        if not args.recursive:
            break




###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate a VaM model from a face encoding" )
    parser.add_argument('--modelFile', help="Model to use for predictions", required=True)
    parser.add_argument('--inputDir', help="Directory containing input encodings", required=True)
    parser.add_argument("--recursive", action='store_true', default=False, help="Iterate to subdirectories of input path")
    parser.add_argument('--outputDir', help="Output VaM files directory", required=True)
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )