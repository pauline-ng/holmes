import cv2
import sys

def face_recognition_ex1 ():
    # Get user supplied values
    imagePath = sys.argv[1]
    imagePath = "/bigdrive/holmes/RedHeaded_League/pics/redheads/MV5BZmRlNTlmYWUtZjQxYi00NTAwLTgxZWUtNzgwYmU1NmFhM2RhXkEyXkFqcGdeQXVyMjQwMDg0Ng@@._V1_UY209_CR1,0,140,209_AL_.jpg"
    cascPath = "/bigdrive/holmes/RedHeaded_League/pics/haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


# from https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340
# pip3 install cmake face_recognition numpy opencv-python

# can download faces from https://github.com/NVlabs/ffhq-dataset (need to give acknowledgment)

import face_recognition
import cv2
import numpy as np
import os
import glob

# get current directory
cur_direc = os.getcwd()

# set path to faces pics
database_dir = os.path.join (cur_direc, 'pics/people_database/')



def read_face_database (database_dir):
    db_of_images = {}
    list_of_pic_files = [f for f in glob.glob(database_dir+'*.png')]
    for face_pic_file in list_of_pic_files:
        face_image =  face_recognition.load_image_file(face_pic_file)
        face_encoding = face_recognition.face_encodings (face_image)[0]
        name = face_pic_file.replace (database_dir, "").replace (".png", "")
        db_of_images[name] = face_encoding
        
    print ("database has been loaded with " + str(len (list_of_files)) + " faces")
    return db_of_images

# face recognition part
def face_recognition_search (face_database, encoded_face_to_search):
#    face_names = []
    #for face_encoding in faces_encodings:
    
    face_encodings = list (face_database.values())
    face_names = list (face_database.keys())
    match_results = face_recognition.compare_faces (face_encodings, encoded_face_to_search)
    
    # go through the match results and print out 
    # the names the picture matched
    for i, match_result in enumerate (match_results):
        if match_result == True:
            return face_names[i]
        
    # no matches found, return None
    return None

face_database = read_face_database (database_dir)
asst_manager_pic =  os.path.join (cur_direc, 'pics/asst_manager/john_clay_beard.png')
face_to_search = face_recognition.load_image_file(asst_manager_pic)
encoded_face_to_search = face_recognition.face_encodings (face_to_search)[0]

name_search_result = face_recognition_search (face_database, encoded_face_to_search)
print (asst_manager_pic + " matches "+ name_search_result)
