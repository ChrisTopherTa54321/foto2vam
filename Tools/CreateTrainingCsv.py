# Generate training data from existing faces
from Utils.Face.vam import VamFace
from Utils.Training.config import Config
import argparse
import glob
import os
import numpy
import csv
import fnmatch
import time

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
    config = Config.createFromFile( args.configFile )

    start = time.time()
    numCreated = 0
    for root, subdirs, files in os.walk(inputPath):
        outCsvFile = os.path.join(root,outputName)
        outFile = None

        print( "Entering {}".format(root))
        for file in fnmatch.filter(files, '*.json'):
            try:
                basename = os.path.splitext(file)[0]
                # Check if we have all support encodings for this json
                relatedFiles = []
                for rfile in fnmatch.filter(files, '{}*'.format(basename)):
                    relatedFiles.append( os.path.join(root, rfile ) )

                # Have all files? Convert them to CSV

                outRow = config.generateParams( relatedFiles )
                if outFile is None:
                    print( "Creating {}".format(outCsvFile))
                    outFile = open( outCsvFile, 'w' )
                    writer = csv.writer( outFile, lineterminator='\n')
                    shape = config.getShape()
                    print( "#{},{},{}".format( args.configFile, shape[0], shape[1] ), file=outFile )
                writer.writerow( outRow )
                numCreated += 1

                if numCreated % 100 == 0:
                    print( "Processed {} entries ({} s/entry)".format(numCreated, ( time.time() - start ) / numCreated ))

            except Exception as e:
                print( "Failed to generate CSV from {} - {}".format( file, str(e)))

        if not args.recursive:
            # Don't go any further down if not recursive!
            break

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Directory containing JSON and encoding files", required=True)
    parser.add_argument('--configFile', help="File with training data generation parameters", required=True)
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