# Generate training data from existing faces
from Utils.Face.vam import VamFace
import argparse
import glob
import os
import numpy
import csv
import fnmatch

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    inputPath = args.inputPath
    outputName = args.outputName


    face = None
    encoding = None
    for root, subdirs, files in os.walk(inputPath):
        outCsvFile = os.path.join(root,outputName)
        outFile = None
        print( "Entering {}".format(root))
        for file in fnmatch.filter(files, '*.json'):

            # Check if we have all support encodings for this json
            basename = os.path.splitext(file)[0]
            encodingFiles = [ "{}_angle0.encoding".format( basename )
                            , "{}_angle35.encoding".format( basename ) ]

            hasAllFiles = True
            for idx,encodingFile in enumerate(encodingFiles):
                encodingFile = os.path.join(root, encodingFile)
                encodingFiles[idx] = encodingFile
                if not os.path.exists( encodingFile ):
                    hasAllFiles = False
                    break

            if not hasAllFiles:
                print("Skipping {} since it does not have all encoding files".format(file))
                continue

            # Have all encodings? Read them in
            outArray = []
            for encodingFile in encodingFiles:
                encoding = open(encodingFile).read().splitlines()
                outArray.extend(encoding)

            face = VamFace(os.path.join(root,file))
            outArray.extend(face.morphFloats)
            if outFile is None:
                print( "Creating {}".format(outCsvFile))
                outFile = open( outCsvFile, 'w' )
            writer = csv.writer( outFile, lineterminator='\n')
            writer.writerow( outArray )

        if not args.recursive:
            # Don't go any further down if not recursive!
            break

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing JSON and encoding files", required=True)
    parser.add_argument("--recursive", action='store_true', default=False, help="Iterate to subdirectories of input path")
    parser.add_argument("--outputName", help="Name of CSV file to create in each directory")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )