import face_recognition
from face_recognition.face_recognition_cli import scan_known_people
import numpy as np

gallery_encoding = []

GALLERY_PATH = './grey'
TEST_PATH = './test'


def main():
    # text_in = input('Method to becnhmark:')
    # in_user = str(text_in)
    # print('Chosen method is', in_user)

    gallery_encoding = scan_known_people(GALLERY_PATH)


    unknown_image = face_recognition.load_image_file(GALLERY_PATH + '/' +
                                                         '0' + '.png')
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

    print()


    for indx in range(0, len(gallery_encoding[1])):
        unknown_image = face_recognition.load_image_file(TEST_PATH + '/' +
                                                         str(indx) + '.png')

        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        # results = face_recognition.compare_faces(gallery_encoding[1],
        #                                 unknown_face_encoding)
        results = face_recognition.compare_faces(gallery_encoding[1],
                                                 unknown_face_encoding,
                                                 tolerance=0.4)
        print('Name:', indx,
              ' || Identified:', True in results,
              ' || Correct:', results[indx])

if __name__ == '__main__':
    main()
