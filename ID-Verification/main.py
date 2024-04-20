# Necessary Imports
import torch
import onnxruntime as rt
import numpy as np
import pandas as pd
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Initialize faceapp
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CUDAExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# Check if the ID contains any face image or not
def detect_faces(img):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        return True
    else:
        return False

  # Extract face embeddings from photo ID
def extract_face_embeddings(image):
    # Take the image and apply to insight face
    results = faceapp.get(image)
    
    # Use for loop and extract each embedding 
    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
    return embeddings

# Match embeddings 
def compare_embeddings(embedding1, embedding2):
    # Calculate cosine similarity between the two embeddings
    similarity = pairwise.cosine_similarity(embedding1.reshape(1,-1), embedding2.reshape(1,-1)) # embedding1.reshape(1,-1)
    return similarity[0][0]

# Main Function
def main():
    image_path = "pan.png"
    img = cv2.imread(image_path)
    
    # Check if there is any face detected in the image
    face_detection = detect_faces(img)
    
    if face_detection:
        #Extract facial-embeddings
        embeddings1 = extract_face_embeddings(img)
    else:
        print("No Faces Detected !")
        
    # Capture live frames (200)
    cap = cv2.VideoCapture(0)
    face_embeddings = []
    sample = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            print('Unable to read camera')
            break
        
        # get results from insightface model
        results = faceapp.get(frame,max_num=1)
        for res in results:
            sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            
            # facial features
            embeddings = res['embedding']
            face_embeddings.append(embeddings)
            
        if sample >= 200: 
            break
        
        cv2.resizeWindow("frame", 300, 700)
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) == ord('q'): # this is trigger only when I press letter q (lowercase q) in my keyboard
#             break
            
    cap.release()
#     cv2.destroyAllWindows()

    ch = input("Match Embeddings(Y/N)? ")
    ch = ch.upper()
    if ch == "Y" or ch == "YES":
        len(face_embeddings)
        x_mean = np.asarray(face_embeddings).mean(axis=0)
        print(x_mean.shape)
        print(embeddings1.shape)
        match = compare_embeddings(embeddings1, x_mean)
        
        if match > 0.4:
            print("ID Verified with ", match*100, "% match")
        else:
            print("Face in the ID is different from person captured Live !!")



main()
