import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition.api
import os
import PIL.Image
import dlib
import numpy as np
from PIL import ImageFile

try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

known_encodings = []

GALLERY_PATH = './grey'
TEST_PATH = './test'

def _face_encodings_(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = face_recognition.api._raw_face_landmarks(face_image, known_face_locations, model="large")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def main():
    text_in = input('Method to becnhmark:')
    METHOD = str(text_in)
    print('Chosen method is', METHOD)


    for indx in range(0,147):
        #print('Processing', indx)
        PATH = GALLERY_PATH + '/' + str(indx) + '.png'
        img = face_recognition.load_image_file(PATH)
        encodings = _face_encodings_(img, num_jitters=1)
        if len(encodings) > 1:
            print("WARNING: More than one face found in {}. Only considering the first face.".format(indx))

        if len(encodings) == 0:
            print("WARNING: No faces found in {}. Ignoring file.".format(indx))
        known_encodings.append(encodings[0])
    print('GALLERY is processed!')

    results = list()
    for i in range(1,101):
        results.append(stats(METHOD,i))

    with open(METHOD + '.txt', 'w') as f:
        for item in results:
            f.write("%s\n" % item)




def stats(method, k):
    PATH = TEST_PATH + '/' + method + '/' + str(k)
    FTA = 0 #FAILURE TO ACQUIRE
    FMR = 0 #FALSE MATCH RATE
    FNMR = 0 #FALSE NONMATCH RATE
    TP = 0 # True Positive
    for indx in range(0, len(known_encodings)):
        unknown_image = face_recognition.load_image_file(PATH + '/' +
                                                         str(indx) + '.png')

        unknown_face_encoding = _face_encodings_(unknown_image, num_jitters=1)[0]
        # results = face_recognition.compare_faces(gallery_encoding[1],
        #                                 unknown_face_encoding)
        results = face_recognition.compare_faces(known_encodings,
                                                 unknown_face_encoding, )


        if not True in results: FTA +=1
        FMR += sum(results)
        if not results[indx]:
            FNMR +=1
        if results[indx] == True:
            FMR -=1
            TP +=1



    FTA /=len(known_encodings)
    FNMR /=len(known_encodings)
    FMR /= len(known_encodings) * (len(known_encodings)-1)

    print('Finished:', k,
          '| FAILURE TO ACQUIRE', FTA,
          '| FALSE NONMATCH:',FNMR,
          '| FALSE MATCH', FMR,
          '| TP', TP)
    return [k, FTA, FNMR, FMR, TP]

if __name__ == '__main__':
    main()
