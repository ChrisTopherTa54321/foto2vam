# Generate training data from existing faces
from Utils.Training.config import Config
import argparse
import os
import numpy
import random

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
    multiDir = args.multiDir

    # Delay heavy imports
    from keras.models import load_model

    # Work around low-memory GPU issue
    import tensorflow as tf
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)

    model = load_model(modelFile)
    modelName = os.path.splitext(os.path.basename(modelFile))[0]
    baseName = ""

    face = config.getBaseFace()
    # Read in all of the files from inputDir
    for root, subdirs, files in os.walk(inputDir):
        for file in files:
            try:
                skipSample = random.random() < args.skipChance 
                relatedFiles = []
                if multiDir:
                    if skipSample: 
                        continue
                    if not file.endswith(".json"):
                        continue
                    baseName = os.path.splitext( file )[0]
                    for rfile in filter( lambda x: x.startswith(baseName), files ):
                        relatedFiles.append(  os.path.join(root,rfile ) )
                else:
                    if skipSample:
                        break
                    for rfile in files:
                        relatedFiles.append( os.path.join(root, rfile ) )


                outRow = config.generateParams( relatedFiles )
                outShape = config.getShape()
                dataSet = numpy.array([outRow[:outShape[0]]])
                predictions = model.predict(dataSet)
                rounded = [float(round(x,5)) for x in predictions[0]]
                face.importFloatList(rounded)

                outName = root.lstrip(inputDir)
                outName = outName.lstrip('/')
                outName = outName.lstrip('\\')
                outputFolder = os.path.join( outputDir, outName )
                try:
                    os.makedirs(outputFolder)
                except:
                    pass

                # In multiDir, the 'folder' is the baseName of the file
                if multiDir:
                    folderName = baseName
                else:
                    folderName = os.path.split(root)[-1]

                outputFullPath = os.path.join( outputFolder, "{}_{}.json".format(folderName, modelName))
                face.save( outputFullPath )
                print( "Generated {}".format(outputFullPath) )
            except Exception as e:
                print( "Failed to generate model from {} - {}".format(root, str(e) ) )

            # If not multiDir then we've already processed all of the files
            if not multiDir:
                break

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
    parser.add_argument("--multiDir", action='store_true', default=False, help="Allow multiple predictions per directory. Assume supporting files start with json files name")
    parser.add_argument("--skipChance", type=float, default=0.0, help="Chance to skip generating a model. Used for training set sampling. Defaults to 0.0")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )