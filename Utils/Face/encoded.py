# Class to handle faces encoded for recognition

try:
    import face_recognition_hst as face_recognition
except:
    import face_recognition
import numpy
from PIL import Image, ImageDraw
import cv2
import math
import json


class EncodedFace:
    ENCODING_TYPE = "dlib.face_recognition"
    ENCODING_VERSION = 1

    def __init__(self, image, region=None, keepImg=False, imgPadding=125, num_jitters=2, debugPose = False):
        if image is None:
            return

        nImg = numpy.array(image)

        if region is None:
            try:
                self._region = face_recognition.face_locations(nImg)[0]
            except Exception as e:
                raise Exception("Failed to find a face in the picture")

            # print("Face found at {}".format(self._region))
        else:
            self._region = _region
        top, right, bottom, left = self._region

        # Apply padding to save more of image
        top = max(0, top - imgPadding)
        left = max(0, left - imgPadding)
        bottom = min(nImg.shape[0], bottom + imgPadding)
        right = min(nImg.shape[1], right + imgPadding)

        # crop image to just the face
        self._img = nImg[top:bottom, left:right]

        # Get encodings for the face in the image
        try:
            self._encodings = face_recognition.face_encodings(self._img, num_jitters=num_jitters)[0]
            self._landmarks = face_recognition.face_landmarks(self._img)[0]
        except:
            raise Exception("Failed to find face in image")
        (_, self._angle, _) = self._estimatePose(debugPose = debugPose)

        if not keepImg:
            self._img = None

    @staticmethod
    def msgpack_encode(obj):
        if isinstance(obj, EncodedFace):
            return {'__EncodedFace__': True, 'angle': obj._angle, 'encodings': obj._encodings, 'landmarks': obj._landmarks }
        return obj
    
    @staticmethod
    def msgpack_decode(obj):
        if b'__EncodedFace__' in obj:
            decodedFace = EncodedFace(None)
            decodedFace._angle = obj[b'angle']
            decodedFace._encodings = obj[b'encodings']
            decodedFace._landmarks = obj[b'landmarks']
            obj = decodedFace
        return obj 
        


    @staticmethod
    def batchEncode( imageList, batch_size = 128, keepImage = False, debugPose = False ):
        encodings, landmarks = face_recognition.batch_face_encodings_and_landmarks( imageList, landmark_model="large", batch_size=batch_size, location_model="hog" )
        encodedList = []
        for data in zip(encodings,landmarks, imageList):
            if len(data[0]) > 0:
                encodedFace = EncodedFace(None)
                encodedFace._encodings = list(data[0][0])
                encodedFace._landmarks = data[1][0]
                encodedFace._img = data[2]
                _, encodedFace._angle, _ = encodedFace._estimatePose( debugPose = debugPose )

                if not keepImage:
                    encodedFace._img = None
            else:
                encodedFace = None
            encodedList.append(encodedFace)
        return encodedList

    @staticmethod
    def createFromFile( fileName ):
        data = open(fileName).read()
        jsonData = json.loads(data)

        if jsonData["encoding_version"] is not EncodedFace.ENCODING_VERSION:
            raise Exception("Encoding version mismatch! File was {}, reader was {}".format(jsonData["encoding_version"], EncodedFace.ENCODING_VERSION ) )

        newEncoding = EncodedFace( None )
        newEncoding._encodings = numpy.array(jsonData["encoding"])
        newEncoding._landmarks = jsonData["landmarks"]
        newEncoding._angle = jsonData["angle"]
        return newEncoding


    # Determine angle the face is facing
    def _estimatePose(self, img_size = None , landmarks = None, debugPose = False):
        if img_size is None:
            img_size = self._img.shape
        if landmarks is None:
            landmarks = self._landmarks
        # 2D image points. Adapted from https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        image_points = numpy.array([
                                    landmarks['nose_bridge'][3],  # Nose tip
                                    landmarks['chin'][8],  # Chin
                                    landmarks['left_eye'][0],  # Left eye left corner
                                    landmarks['right_eye'][3],  # Right eye right corner
                                    landmarks['top_lip'][0],  # Left mouth corner
                                    landmarks['bottom_lip'][0],  # Right mouth corner
                                ], dtype="double")

        # 3D model points
        model_points = numpy.array([
                                    (0.0, 0.0, 0.0),  # Nose tip
                                    (0.0, -330.0, -65.0),  # Chin
                                    (-225.0, 170.0, -135.0),  # Left eye left corner
                                    (225.0, 170.0, -135.0),  # Right eye right corne
                                    (-150.0, -150.0, -125.0),  # Left Mouth corner
                                    (150.0, -150.0, -125.0)  # Right mouth corner

                                ])

        # Camera internals
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = numpy.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype="double"
                                 )

        dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rMat = cv2.Rodrigues(rotation_vector)[0]
        attitude = self._rotation_matrix_to_attitude_angles(rMat)

        attitude_list = []
        for val in numpy.nditer(attitude.T):
            attitude_list.append( math.degrees(val) )

        # Display image with markings
        if debugPose:
            debugImage = self._img.copy()

            # Draw face landmarks
            for p in image_points:
                cv2.circle(debugImage, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            origin, _ = cv2.projectPoints(numpy.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            xAxis, _ = cv2.projectPoints(numpy.array([(100.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            yAxis, _ = cv2.projectPoints(numpy.array([(0.0, 100.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            zAxis, _ = cv2.projectPoints(numpy.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            origin = tuple(origin.reshape(-1,2)[0])
            xAxis = tuple(xAxis.reshape(-1,2)[0])
            yAxis = tuple(yAxis.reshape(-1,2)[0])
            zAxis = tuple(zAxis.reshape(-1,2)[0])

            origin = tuple([int(val) for val in origin])
            xAxis = tuple([int(val) for val in xAxis])
            yAxis = tuple([int(val) for val in yAxis])
            zAxis = tuple([int(val) for val in zAxis])

            cv2.line(debugImage, origin, xAxis, (255,0,0), 2 )
            cv2.line(debugImage, origin, yAxis, (0,255,0), 2 )
            cv2.line(debugImage, origin, zAxis, (0,0,255), 2 )

            for _,pointList in landmarks.items():
                for p in pointList:
                    cv2.circle(debugImage, p, 1, (0,255,0), -1)

            cv2.putText( debugImage, "Rot: {}, {}, {}".format(round(attitude_list[0],3), round(attitude_list[1],3), round(attitude_list[2],3)),
                         (10,30),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         .4,
                         (0,255,0),
                         1
                         )

            trans_list = []
            for val in numpy.nditer(translation_vector.T):
                trans_list.append(float(val))

            cv2.putText( debugImage, "Trans: {}, {}, {}".format(round(trans_list[0],3), round(trans_list[1],3), round(trans_list[2],3)),
                         (10,60),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         .4,
                         (0,255,0),
                         1
                         )


            cv2.imshow( "Debug Output", debugImage )
            cv2.waitKey(0)

        return attitude_list

    # Taken from https://stackoverflow.com/questions/44726404/camera-pose-from-solvepnp
    # Only seems to get 'yaw' correct, but that's what we want
    def _rotation_matrix_to_attitude_angles(self, R):
        cos_beta = math.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2])
        validity = cos_beta < 1e-6
        if not validity:
            alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
            beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
            gamma = math.atan2(R[2,1], R[2,2])    # roll  [x]
        else:
            alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
            beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
            gamma = 0                             # roll  [x]
        return numpy.array([alpha, beta, gamma])

    def getAngle(self):
        return self._angle

    def getImage(self):
        return self._img

    def getEncodings(self):
        return list(self._encodings)

    def getLandmarks(self):
        return self._landmarks

    def getRegion(self):
        return self._region

    def compare(self, otherFace):
        return face_recognition.face_distance([numpy.array(self._encodings)], numpy.array(otherFace._encodings)).mean()

    def getEncodingJson(self):
        return { 'angle': self._angle, 'landmarks': self._landmarks, 'encoding': self._encodings.tolist(), 'encoding_format': self.ENCODING_TYPE, 'encoding_version': self.ENCODING_VERSION }

    def saveEncodings(self, filename):
        jsonData = self.getEncodingJson()
        with open(filename, 'w') as outfile:
            json.dump(jsonData, outfile)

    def saveImage(self, filename, landmarks=True):
        if self._img is None:
            raise Exception("Image was not saved in constructor!")

        if isinstance( self._img, numpy.ndarray ):
            img = Image.fromarray( self._img )
        else:
            img = self._img
        if landmarks:
            draw = ImageDraw.Draw(img)
            for key, val in self._landmarks.items():
                draw.point(val)

        img.save(filename)
