# Copy parts of one model to another
import argparse
import glob
import os
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
    toFace = VamFace( args.toJson, discardExtra = False )
    fromFace = VamFace( args.fromJson, discardExtra = False )

    newFace = VamFace.mergeFaces( templateFace=templateFace, toFace=toFace, fromFace=fromFace)
    newFace.save( args.outputJson )
    print("Merged face saved to {}".format(args.outputJson) )


###############################
# Worker function for helper processes
###############################
def worker_process_func(procId, workQueue, doneEvent, config, args):
    print("Worker {} started".format(procId))
    while not ( doneEvent.is_set() and workQueue.empty() ):
        try:
            work = workQueue.get(block=True, timeout=1)
            dirPath = work[0]
            outCsvFile = work[1]
            outFile = None
            numCreated = 0
            start = time.time()
            try:
                globPath = os.path.join( dirPath, "*.json")
                for file in glob.glob( globPath ):
                    try:
                        basename = os.path.splitext(file)[0]
                        # Check if we have all support encodings for this json
                        relatedFilesGlob = "{}*".format(basename)
                        relatedFiles = []
                        for rfile in glob.glob( relatedFilesGlob ):
                            relatedFiles.append( rfile )

                        # Have all files? Convert them to CSV
                        outRow = config.generateParams( relatedFiles )
                        if outFile is None:
                            print( "Worker {} creating {}".format(procId, outCsvFile))
                            outFile = open( outCsvFile, 'w' )
                            writer = csv.writer( outFile, lineterminator='\n')
                            shape = config.getShape()
                            print( "#{},{},{}".format( args.configFile, shape[0], shape[1] ), file=outFile )
                        writer.writerow( outRow )
                        numCreated += 1

                    except Exception as e:
                        pass
                        #print( "Failed to generate CSV from {} - {}".format( file, str(e)))
                print( "Worker {} done with {} ({} entries at {} entries/second)".format(procId, outCsvFile, numCreated, numCreated/( time.time() - start ) ) )

            except Exception as e:
                print("Worker {} failed generating {} : {}".format(procId, outCsvFile, str(e)))

        except queue.Empty:
            pass
    print("Worker {} done!".format(procId))


###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--templateJson', help="Model with morphs to copy set as 'animatable'", required=True)
    parser.add_argument('--toJson', help="Model to copy morphs TO", required=True)
    parser.add_argument('--fromJson', help="Model to copy morphs FROM", required=True)
    parser.add_argument('--outputJson', help="Destination model file", required=True)
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )