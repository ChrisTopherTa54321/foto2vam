import argparse
import os
import glob
import time
from Utils.VamWindow import VamWindow
from Utils.Algorithm import Algorithm
from PIL import Image

# Set DPI Awareness  (Windows 10 and 8). Makes GetWindowRect return pxiel coordinates
import ctypes
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)



###############################
# Run the program
#
def main( args ):
    if args.pydev:
        print("Enabling debugging with pydev")
        import pydevd
        pydevd.settrace(suspend=False)

    inputPath = args.inputJsonPath
    targetPath = args.targetPath
    if not os.path.isabs(targetPath):
        targetPath = os.path.join(inputPath, targetPath)

    outputPath = args.outputPath
    if not os.path.isabs(outputPath):
        outputPath = os.path.join(inputPath, outputPath)

    testJsonPath = args.testJsonPath
    if not os.path.isabs(testJsonPath):
        testJsonPath = os.path.join(inputPath, testJsonPath)


    print( "Input path: {}\nTarget Images: {}\nOutput path: {}\n\n".format( inputPath, targetPath, outputPath ) )

    # Initialize the Vam window
    window = VamWindow()

    # Locating the buttons via image comparison does not reliably work. These coordinates are the 'Window' coordinates
    # found via AutoHotKey's Window Spy, cooresponding to the Load Preset button and the location of the test file
    window.setClickLocations([(130,39), (248,178)])

    # Theses imports takes a while, so we delay loading until here
    from Utils.EncodedFace import EncodedFace
    from Utils.VamFace import VamFace
    from Utils.AlgorithmParams import Params

    # Load in our input face parameters
    protoFace = VamFace( baseFileName = "test/base.json", minFileName= "test/minimum.json", maxFileName="test/maximum.json")

    # Load in the 'target' images
    targetImages = {}
    print( "Loading target images from {}".format(targetPath))
    for entry in glob.glob(os.path.join(targetPath, '[0-9]*')):
            if os.path.isdir( entry ):
                try:
                    folderName = os.path.split(entry)[-1]
                    # test if we can cast to float
                    _ = float(folderName)
                    targetImages[folderName]=[]
                    exts = ('*.jpg', '*.png')
                    for ext in exts:
                        for image in glob.glob(os.path.join(entry,ext)):
                            targetImages[folderName].append(image)
                except:
                    pass

    targetFaces = {}
    for angle,imageList in targetImages.items():
        targetFaces[angle] = []
        for imagePath in imageList:
            try:
                print("Processing {}".format(imagePath))
                image = Image.open( imagePath )
                face = EncodedFace( image, keepImg=args.saveImages )
                targetFaces[angle].append(face)
            except:
                print("Failed to encode face from image")
        if len(targetFaces[angle]) == 0:
            print("WARNING: No images successfully loaded for angle {}".format(angle))
            del targetFaces[angle]

    # Set up the parameters for the algorithm
    params = Params( protoFace=protoFace,
                     vamWindow=window,
                     testJsonPath=os.path.join(testJsonPath, 'test.json'),
                     targetFaces = targetFaces,
                     outputPath = outputPath,
                     saveImages = args.saveImages
                      )

    # Run the algorithm!
    algo = Algorithm(params)
    algo.run( args.population )




###############################
# parse arguments
#
def parseArgs():
    parser = argparse.ArgumentParser(
         description="VaM morph from a photograph",
         formatter_class=argparse.RawDescriptionHelpFormatter,
         epilog =
'''inputJsonPath should contain base.json, maximum.json, minimum.json.

base.json should be a saved preset that has 'animatable' checked on the morphs to modify.

minimum.json and maximum.json should also have all modifiable morphs checked, but with values
at their minimum and maximum values, respectively. This is used to set the valid range for each morph.

testJsonPath is the directory where the JSON preset that the search will modify is placed. It should be
the only item in this path (so that its position on the screen does not change when listing the directory)

targetPath is the path to the folder that contains the target images. It should contain a few subdirectories
that are named the 'angle' of the image. Front shots should go into a directory named '0'. Profile images should
be in a directory something like '65' or '70'. The name of the directory is directly used as the rotation to apply
to the atom in VaM. Note: The face recognizer works very poorly on profile images. 90 is unlikely to work.

Example directory structure:
   target\\0\\front1.jpg
   target\\0\\front2.jpg
   target\\45\\front_angle.jpg
   target\\75\\profile.jpg

You can have multiple images for each angle; all will be used. Results may vary.

outputPath is where output data will be written

VaM should be loaded in desktop mode, with a Person atom with 'base.json' preset loaded. The UI should be visible,
and the face should be visible to the side of the UI. The 'Person/Control' should be selected. It is essential that
the only steps to load a preset are to click 'Load Preset' and then click 'test.json.' This is easiest if 'test.json'
is in its own directory, so its position on screen doesn't move. To ensure VaM opens this path when clicking the
'Load Preset' button you should copy base.json into this directory, rename it 'test.json' and then load the preset.
Now each time you click 'Load Preset' it should remember this folder. (Note: initial contents of this file are not used)
''')
    parser.add_argument("--saveImages", action='store_true', default=False, help="Save images of each step")
    parser.add_argument("--landmarks", action='store_true', default=False, help="Draw landmarks on images of each step")
    parser.add_argument("--saveJson", action='store_true', default=False, help="Save json of each step")
    parser.add_argument('--population', type=int, default=100, help="Population size. Smaller converges faster, larger tries a larger variety of morphs. Default: 100")
    parser.add_argument('--inputJsonPath', help="Directory containing base.json, minimum.json and maximum.json", required=True)
    parser.add_argument('--targetPath', help="Directory target image folders. Absolute, or relative to inputJsonPath", default="target")
    parser.add_argument('--outputPath', help="Directory to write output data to. Absolute, or relative to inputJsonPath", default="results")
    parser.add_argument('--testJsonPath', help="Directory where test JSON will be stored. Absolute, or relative to inputJsonPath", default="test")
    parser.add_argument("--pydev", action='store_true', default=False, help="Enable pydevd debugging")


    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    main( args )