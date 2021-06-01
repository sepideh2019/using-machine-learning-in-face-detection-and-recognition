import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('H:\\ML faceDetection\\test\\alw.jpg')
faces_detected,gray_img= fr.faceDetection(test_img)
print("faces_detected:" , faces_detected)

#for (x,y,w,h) in faces_detected:
#   cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
#resized_img=cv2.resize_img(test_img,(1000,700))
#cv2.imshow("test_img", test_img )
#cv2.waitKey(1)  
#cv2.destroyAllWindows



#faces,faceID= fr.lebels_for_training_data("H:\\ML faceDetection\\Testimages")
#face_regognizer= fr.train_classifier(faces,faceID)
#face_regognizer.save("H:\\ML faceDetection\\TrainedData\\trainingData.yml")
face_regognizer=cv2.face.LBPHFaceRecognizer_create()
face_regognizer.read("H:\\ML faceDetection\\TrainedData\\trainingData.yml")


name={0:"Alwin", 1:"Mirdu",2:"Navarun",3:"Pulak",4:"Rishaw"}



#cap=cap = cv2.VideoCapture(0)
#while True:
 #   ret,





for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_regognizer.predict(roi_gray)
    print("confidence: " ,confidence)
    print("label",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):
        continue
    fr.put_text(test_img,predicted_name,x,y)

cv2.imshow("test_img", test_img )
cv2.waitKey(1)
cv2.destroyAllWindows
