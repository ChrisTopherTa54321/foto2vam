# Copy parts of one model to another
import argparse
import os
import fnmatch
from Utils.Face.vam import VamFace

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    templateFace = VamFace( args.templateJson )
    templateFace.trimToAnimatable()
    fromFace = VamFace( args.fromJson, discardExtra = False )
    fileFilter = args.filter
    inputDir = args.toJsonDir
    outputDir = args.outputJsonDir

    for root, subdirs, files in os.walk(inputDir):
        print("Entering directory {}".format(root))
        for file in fnmatch.filter(files, fileFilter):
            try:
                toName = os.path.splitext(file)[0]
                outDir = root.lstrip(inputDir)
                outDir.lstrip('/')
                outDir.lstrip('\\')
                outDir = os.path.join( outputDir, outDir )
                outName = "{}_mergedWith_{}.json".format( os.path.splitext(file)[0], os.path.splitext(os.path.basename(args.fromJson))[0])
                toFace = VamFace( os.path.join(root, file), discardExtra = False )
                newFace = VamFace.mergeFaces( templateFace=templateFace, toFace=toFace, fromFace=fromFace)
                try:
                    os.makedirs(outDir)
                except:
                    pass
                newFace.save( os.path.join(outDir, outName ) )
            except Exception as e:
                print("Error merging {} - {}".format(file, str(e)))




###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--templateJson', help="Model with morphs to copy set as 'animatable'", required=True)
    parser.add_argument('--toJsonDir', help="Path to find models to copy morphs TO", required=True)
    parser.add_argument('--filter', help="Filter for To morphs. Defaults to *.json", default="*.json")
    parser.add_argument("--recursive", action='store_true', default=False, help="Iterate to subdirectories of toJsonDir")
    parser.add_argument('--fromJson', help="Model to copy morphs FROM", required=True)
    parser.add_argument('--outputJsonDir', help="Destination model path", required=True)

    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )