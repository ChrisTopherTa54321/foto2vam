# Quickly thrown together root script to run Tools
from Utils.Face.vam import VamFace
import argparse
import glob
import os
import fnmatch
import Tools.CreateTrainingEncodings as encodings
import Tools.MakePrediction as predictor
import shutil

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    inputPath = args.inputPath
    modelPath = args.modelPath
    outputPath = args.outputPath
    jsonPath = os.path.splitext(modelPath)[0] + ".json"

    print( "Processing images from {} and using model/json {}/{}".format(inputPath, modelPath, jsonPath))

    print( "First running CreateTrainingEncodings tool")
    params = argparse.Namespace(inputPath=inputPath, filter="*.png,*.jpg", normalizeSize=150, normalize=True, numJitters=10, numThreads=4, pydev=False, recursive=True, debugPose = False)
    encodings.main( params )

    print( "Running MakePredictions tool")
    params = argparse.Namespace(modelFile=modelPath, inputDir=inputPath, pydev=False, outputDir=outputPath, multiDir=False, skipChance=0.0, recursive=True )
    predictor.main(params)
    
    


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing images", default="Input")
    parser.add_argument('--modelPath', help="Path to model", default=os.path.join("models", "foto2vam.model") )
    parser.add_argument('--outputPath', help="Directory to store output", default="Output")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )