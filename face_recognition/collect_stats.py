import gzip
import shutil

import dlib
import csv
import face_recognition.api
import numpy as np
from PIL import ImageFile

import face_recognition

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
known_index = []

GALLERY_PATH = './grey'
TEST_PATH = './test'
NUM_IMG = 147


def _face_encodings_(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = face_recognition.api._raw_face_landmarks(face_image, known_face_locations, model="large")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


def main():
    text_in = input('SYS:: Method to becnhmark [pixel, eigen, same, blur, pixelated, noise]:')
    METHOD = str(text_in)
    if METHOD not in ['pixel', 'eigen', 'same', 'blur', 'pixelated', 'noise']: raise Exception('Wrong Method')
    print('SYS:: Chosen method is', METHOD)
    text_in = input('SYS:: Method to becnhmark [naive, reverse, parrot]:')
    TYPE = str(text_in)
    print('SYS:: Chosen type:', TYPE)

    if TYPE == 'naive':
        naive(METHOD)
    elif TYPE == 'reverse':
        reverse(METHOD)
    elif TYPE == 'parrot':
        parrot(METHOD)
    else:
        raise Exception('Wrong Type')

def parrot(method):
    output = list()
    if method == 'pixelated':
        range_ = [1, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
    else:
        range_ = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for k in range_:
        load_gallery('parrot', NUM_IMG, method=method, k=k)
        t = stats('parrot', method, k)
        print(t)
        output.append(t)

    path_ = TEST_PATH + '/stats/' + method + '_' + 'parrot' + '_' + 'result.csv'

    np.savetxt(path_, output, delimiter=',')



def reverse(method):
    output = list()
    if method == 'pixelated':
        range_ = [1, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
    else:
        range_ = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for k in range_:
        load_gallery('reverse', NUM_IMG, method=method, k=k)
        t = stats('reverse', method, k)
        print(t)
        output.append(t)

    path_ = TEST_PATH + '/stats/' + method + '_' + 'reverse' + '_' + 'result.csv'

    np.savetxt(path_, output, delimiter=',')


def naive(method):
    load_gallery('naive', NUM_IMG)
    output = list()
    for k in range(1, 101):
        t = stats('naive', method, k)
        print(t)
        output.append(t)

    path_ = TEST_PATH + '/stats/' + method + '_' + 'naive' + '_' + 'result.csv'

    np.savetxt(path_, output, delimiter=',')


def load_gallery(type_: 'str', num_img: 'int', method='none', k=0):
    known_encodings.clear()
    known_index.clear()
    path = ''
    if type_ == 'naive':
        path = GALLERY_PATH
    elif type_ == 'reverse':
        path = TEST_PATH + '/' + method + '/' + str(k)
        if k == 0: raise Exception('Wrong Reference')
    elif type_ == 'parrot':
        path = TEST_PATH + '/' + method + '/' + str(k)
        if k == 0: raise Exception('Wrong Reference')
    else:
        raise Exception('Wrong Type')

    for index in range(0, num_img):
        image_path = path + '/' + str(index) + '.png'
        img = face_recognition.load_image_file(image_path)
        encodings = _face_encodings_(img, num_jitters=1)
        if len(encodings) == 0:
            #print("WARNING: No faces found in {}. Ignoring file.".format(index))
            pass
        elif len(encodings) > 1:
            #print("WARNING: More than one face found in {}. Only considering the first face.".format(index))
            known_encodings.append(encodings[0])
            known_index.append(index)
        else:
            known_encodings.append(encodings[0])
            known_index.append(index)

    print('Reference Images are processed', len(known_encodings))


def stats(type_, method, k):
    if type_ == 'naive':
        PATH = TEST_PATH + '/' + method + '/' + str(k)
    elif type_ == 'reverse':
        PATH = GALLERY_PATH
    if type_ == 'parrot':
        PATH = TEST_PATH + '/' + method + '/' + str(k)

    failure_to_acquire = 0
    false_match_rate = 0
    false_non_match_rate = 0
    true_positive = 0
    genuine_score = list()
    impostor_score = list()

    if len(known_encodings) == 0:
        while len(genuine_score) < NUM_IMG:
            genuine_score.append(0.0)
        while len(impostor_score) < (NUM_IMG * (NUM_IMG - 1)):
            impostor_score.append(0.0)

        impostor_score = np.asarray(impostor_score)
        genuine_score = np.asarray(genuine_score)

        path_genuine = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + '_genuine_score.txt'
        path_impostor = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + '_impostor_score.txt'
        np.savetxt(path_genuine, genuine_score)
        np.savetxt(path_impostor, impostor_score)

        with open(path_genuine, 'rb') as f_in:
            with gzip.open(path_genuine + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        with open(path_impostor, 'rb') as f_in:
            with gzip.open(path_impostor + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return [int(k), 1.0, 0.0, 1.0, 0.0]

    for i in range(0, len(known_encodings)):
        # FACE ENCODING
        unknown_image = face_recognition.load_image_file(PATH + '/' + str(i) + '.png')
        unknown_face_encoding = _face_encodings_(unknown_image)
        if len(unknown_face_encoding) == 0:
            failure_to_acquire += 1
            false_non_match_rate += 1
        else:
            unknown_face_encoding = unknown_face_encoding[0]
            results = face_recognition.compare_faces(known_encodings, unknown_face_encoding)
            ## GENUINE AND IMPOSTOR SCORES
            distance = face_recognition.api.face_distance(known_encodings, unknown_face_encoding)
            similarity = 1 / (distance + 1)

            if not True in results: failure_to_acquire += 1

            false_match_rate += sum(results)

            index = -1
            for el in known_index:
                if el == i:
                    index = el
            # print(index, i)
            if index == -1:
                false_non_match_rate += 1
                genuine_score.append(0.0)
            elif not results[index]:
                false_non_match_rate += 1
                genuine_score.append(similarity[index])
            elif results[index]:
                genuine_score.append(similarity[index])
                false_match_rate -= 1
                true_positive += 1

            if len(distance) > 0:
                for j in range(0, len(distance)):
                    if not known_index[j] == i:
                        impostor_score.append(similarity[j])

    ## Account for unidentified faces
    while len(genuine_score) < NUM_IMG:
        genuine_score.append(0.0)
    while len(impostor_score) < (NUM_IMG * (NUM_IMG - 1)):
        impostor_score.append(0.0)

    impostor_score = np.asarray(impostor_score)
    genuine_score = np.asarray(genuine_score)

    path_genuine = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + '_genuine_score.txt'
    path_impostor = TEST_PATH + '/stats/' + method + '_' + type_ + '_' + str(k) + '_impostor_score.txt'
    np.savetxt(path_genuine, genuine_score)
    np.savetxt(path_impostor, impostor_score)

    with open(path_genuine, 'rb') as f_in:
        with gzip.open(path_genuine + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with open(path_impostor, 'rb') as f_in:
        with gzip.open(path_impostor + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not type_ == 'naive':
        failure_to_acquire += NUM_IMG - len(known_encodings)
        false_non_match_rate += NUM_IMG - len(known_encodings)

    failure_to_acquire /= NUM_IMG
    false_non_match_rate /= NUM_IMG
    false_match_rate /= NUM_IMG * (NUM_IMG - 1)


    return [int(k), failure_to_acquire, false_match_rate, false_non_match_rate, true_positive]


def stats_(method, k):
    PATH = TEST_PATH + '/' + method + '/' + str(k)
    FTA = 0  # FAILURE TO ACQUIRE
    FMR = 0  # FALSE MATCH RATE
    FNMR = 0  # FALSE NONMATCH RATE
    TP = 0  # True Positive
    for indx in range(0, len(known_encodings)):
        unknown_image = face_recognition.load_image_file(PATH + '/' +
                                                         str(indx) + '.png')

        unknown_face_encoding = _face_encodings_(unknown_image, num_jitters=1)[0]
        # results = face_recognition.compare_faces(gallery_encoding[1],
        #                                 unknown_face_encoding)
        results = face_recognition.compare_faces(known_encodings,
                                                 unknown_face_encoding, )

        if not True in results: FTA += 1
        FMR += sum(results)
        if not results[indx]:
            FNMR += 1
        if results[indx] == True:
            FMR -= 1
            TP += 1

    FTA /= NUM_IMG
    FNMR /= NUM_IMG
    FMR /= NUM_IMG * (NUM_IMG - 1)

    print('Finished:', k,
          '| FAILURE TO ACQUIRE', FTA,
          '| FALSE NONMATCH:', FNMR,
          '| FALSE MATCH', FMR,
          '| TP', TP)
    if TP == 0 and len(known_encodings)==1:
        FTA= 1
        FMR = 0



    return [k, FTA, FNMR, FMR, TP]


if __name__ == '__main__':
    main()
