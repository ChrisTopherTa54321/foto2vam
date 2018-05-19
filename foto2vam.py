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
    #if args.pydev:
    #    print("Enabling debugging with pydev")
    #    import pydevd
    #    pydevd.settrace(suspend=False)

    inputPath = args.inputPath
    modelPath = args.modelPath
    outputPath = args.outputPath
    jsonPath = os.path.splitext(modelPath)[0] + ".json"

    print( "Processing images from {} and using model/json {}/{}".format(inputPath, modelPath, jsonPath))

    print( "First running CreateTrainingEncodings tool")
    params = argparse.Namespace(inputPath=inputPath, filter="*.png", normalize=True, numThreads=4, pydev=False, recursive=False)
    encodings.main( params )

    tempPath = "temp"
    print( "Moving encodings to temporary directory" )
    try:
        os.makedirs(  tempPath )
    except:
        pass
    
    try:
        os.makedirs( outputPath )
    except:
        pass

    for file in glob.glob( os.path.join( inputPath, "*.encoding" ) ):
        try:
            shutil.move( file, os.path.join( tempPath, os.path.basename(file) ) )
        except:
            print("Error moving {}".format(file))

    print( "Generating .face marker files" )
    for file in glob.glob( os.path.join( tempPath, "*angle0.encoding" ) ):
        try:
            baseName = os.path.basename(file).split("_angle0")[0]
            angle0 = baseName + "_angle0.encoding"
            angle35 = baseName + "_angle35.encoding"

            angle0 = os.path.join( tempPath, angle0 )
            angle35 = os.path.join( tempPath, angle35 )
            if os.path.exists(angle0) and os.path.exists(angle35):
                open( os.path.join( tempPath, baseName + ".face" ), "w" )
            else:
                print( "Couldn't find both angle encodings for {}".format(baseName))
        except:
            print( "Error processing {}".format(file))
            
    print( "Running MakePredictions tool")
    params = argparse.Namespace(modelFile=modelPath, inputEncoding=os.path.join(tempPath, "*.face"), baseJson=jsonPath, pydev=False, outputDir=outputPath, archiveDir=None )
    predictor.main(params)
    
    try:
        os.removedirs(tempPath)
    except:
        pass


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing images", default="Input")
    parser.add_argument('--modelPath', help="Path to model", default="models/foto2vam.model")
    parser.add_argument('--outputPath', help="Directory to store output", default="Output")
    #parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )