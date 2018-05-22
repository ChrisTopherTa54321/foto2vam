# Generate training data from existing faces
from Utils.Face.vam import VamFace
from Utils.Face.encoded import EncodedFace
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

    # Delay heavy imports
    from keras.models import load_model

    # Work around low-memory GPU issue
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


    modelFile = args.modelFile
    inputGlob = args.inputEncoding
    outputDir = args.outputDir
    baseJson = args.baseJson
    archiveDir = args.archiveDir

    face = VamFace(baseJson)
    face.trimToAnimatable()
    model = load_model(modelFile)
    modelName = os.path.splitext(os.path.basename(modelFile))[0]
    
    if archiveDir:
        try:
            os.makedirs(archiveDir)
        except:
            pass
    

    for entry in glob.glob(inputGlob):
        try:
            encodingFile = "{}_angle0.encoding".format( os.path.splitext(entry)[0])
            encodingFile1 = "{}_angle35.encoding".format( os.path.splitext(entry)[0])
            outArray = []
            if os.path.exists(encodingFile) and os.path.exists(encodingFile1):
                encodedFace = EncodedFace.createFromFile( encodingFile )
                outArray = encodedFace.getEncodings()

                encodedFace = EncodedFace.createFromFile( encodingFile1 )
                outArray.extend( encodedFace.getEncodings())
            else:
                print("Missing encoding file {} or {}".format(encodingFile, encodingFile1))
                
            dataSet = numpy.array([outArray])
            predictions = model.predict(dataSet)
            rounded = [float(round(x,5)) for x in predictions[0]]
            face.importFloatList(rounded)
            entryName = os.path.splitext(os.path.basename(entry))[0]
            outputFile = "{}_{}.json".format(entryName, modelName)
            outputFile = os.path.join(outputDir, outputFile) 
            if args.archiveDir and os.path.exists(outputFile):
                dateString = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                backupFileName = os.path.splitext(os.path.basename(outputFile))[0]
                backupFileName = os.path.join( args.archiveDir, "{}_{}.json".format(backupFileName, dateString))
                print("Backing up {} to {}".format(outputFile,backupFileName))
                shutil.copyfile( outputFile, backupFileName)
            face.save( outputFile )
            print("Prediction saved to {}".format(outputFile))
        except Exception as e:
            print("Failed to process {}: {}".format(entry, e))
    


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate a VaM model from a face encoding" )
    parser.add_argument('--modelFile', help="Model to use for predictions", required=True)
    parser.add_argument('--inputEncoding', help="Encoding to use for input to model. Supports wildcard", required=True)
    parser.add_argument('--archiveDir', help="Backup old outputFile files to directory first")
    parser.add_argument('--baseJson', help="Base VaM file", required=True)
    parser.add_argument('--outputDir', help="Output VaM files directory", required=True)
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )