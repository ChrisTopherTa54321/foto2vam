# Generate training data from existing faces
import argparse
import glob
import os
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
    outputFile = args.outputFile
    recursive = args.recursive
    fileFilter = args.filter

    print(" Creating output CSV file: {}".format(outputFile))
    # Read in all of the files from inputpath
    outFile = open(outputFile, 'w')
    #writer = csv.writer( outFile, lineterminator='\n')
    commentLine = None
    for root, subdirs, files in os.walk(inputPath):
        print("Entering directory {}".format(root))
        for file in fnmatch.filter(files, fileFilter):
            print("Reading {}".format(file))
            csvInputFile = os.path.join(root, file)
            with open( csvInputFile ) as f:
                for line in f:
                    if line.startswith("#"):
                        if commentLine is None:
                            commentLine = line
                        else:
                            if commentLine != line:
                                print("Header mismatch in {}! Got {}, expected {}".format(csvInputFile, line, commentLine ) )
                                break
                            continue # skip comment lines (except header comment)
                    outFile.write(line)

        if not recursive:
            break


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputPath', help="Root directory containing csv files to merge", required=True)
    parser.add_argument('--filter', help="Filter for files to merge", default="*.csv")
    parser.add_argument('--outputFile', help="Merged output CSV file", default="output.csv")
    parser.add_argument("--recursive", action='store_true', default=False, help="Recursively enter directories")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )