# Generate training data from existing faces

from Utils.Face.vam import VamFace
import argparse
import os
import glob
import copy
import random

###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)
    inputPath = args.inputJsonPath
    basePath = args.baseJsonPath
    outputPath = args.outputPath

    baseFace = VamFace( basePath )
    baseFace.trimToAnimatable()

    # Read in all of the files from inputpath
    inputFaces = []
    print( "Loading input faces from  {}".format(inputPath))
    for entry in glob.glob(os.path.join(inputPath, '*.json')):
        try:
            newFace = VamFace( entry )
            # Only keep the relevant morphs
            morphCnt = len(newFace.morphFloats)
            newFace.matchMorphs(baseFace)
            inputFaces.append(newFace)
        except:
            print("Error loading {}".format(entry))

    print( "Loaded {} faces".format(len(inputFaces)))
    if len(inputFaces) == 0:
        print("No starting point faces were loaded!")
        exit(-1)
    faceCnt = 0;
    print( "Generating variations")

    mutateChance = .2
    mateChance = .5
    faceVariants = [] + inputFaces

    while faceCnt < args.numFaces:
        #for face1 in faceVariants:
        face1 = faceVariants[random.randint(0,len(faceVariants)-1)]

        # Randomly take parameters from the other face
        shouldMate = random.random() < mateChance
        shouldMutate = random.random() < mutateChance

        if shouldMate or shouldMutate:
            newFace = copy.deepcopy(face1)

            if shouldMate:
                mateIdx = random.randint(0, len(faceVariants)-1)
                mate(newFace, faceVariants[mateIdx], random.randint(1, len(newFace.morphFloats)))
    
            # Randomly apply mutations to the current face
            if shouldMutate:
                mutate(newFace, random.randint(0,random.randint(1,10)) )
                
            newFace.save( os.path.join(outputPath, "face_variant_{}_{}.json".format(faceCnt, random.randint(0,99999))))
            faceVariants.append(newFace)
            faceCnt += 1
            if faceCnt % 500 == 0:
                print( "{}/{}".format(faceCnt,args.numFaces) )


def mutate(face, mutationCount):
    for i in range(mutationCount):
        face.randomize( random.randint(0, len(face.morphFloats) - 1 ) )


def mate(targetFace, otherFace, mutationCount ):
    if len(targetFace.morphFloats) != len(otherFace.morphFloats):
        raise Exception("Morph float list didn't match! {} != {}".format(len(targetFace.morphFloats), len(otherFace.morphFloats)))
    for i in range(mutationCount):
        morphIdx = random.randint(0, len(otherFace.morphFloats) - 1)
        targetFace.morphFloats[morphIdx] = otherFace.morphFloats[morphIdx]

###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser( description="Generate training data" )
    parser.add_argument('--inputJsonPath', help="Directory containing json files to start with", required=True)
    parser.add_argument('--baseJsonPath', help="JSON file with relevant morphs marked as Animatable", required=True)
    parser.add_argument('--outputPath', help="Directory to write output data to", default="output")
    parser.add_argument('--numFaces', type=int, help="Number of faces to generate before stopping", default=10000 )
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )