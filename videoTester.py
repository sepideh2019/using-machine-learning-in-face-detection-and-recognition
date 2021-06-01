import os
import cv2
import numpy as np
import faceRecognition as fr


face_regognizer=cv2.face.LBPHFaceRecognizer_create()
face_regognizer.read("H:\\ML faceDetection\\TrainedData\\trainingData.yml")


name={0:"Alwin", 1:"Mirdu",2:"Navarun",3:"Pulak",4:"Rishaw"}


cap=cv2.VideoCapture(0)


while True:
    ret,test_img=cap.read()
    face_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in face_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)


        cv2.imshow('detected face' ,test_img)
        cv2.waitKey(10)



    for face in face_detected:
               
        (x,y,w,h)=face
        
        roi_gray=gray_img[y:y+w,x:x+h]
        label,confidence=face_regognizer.predict(roi_gray)
        print("confidence: " ,confidence)
        print("label",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if(confidence<39):
            fr.put_text(test_img,predicted_name,x,y)
                       


    cv2.imshow("test_img", test_img )
    if cv2.waitKey(10)==ord('q'):
        break

cap.release()    
cv2.destroyAllWindows
