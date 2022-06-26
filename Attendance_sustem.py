from base64 import encode
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

down_width = 400
down_height = 500
down_points = (down_width, down_height)

#BASIC KNOWLEDGE
'''
img1=face_recognition.load_image_file(r"D:\BasicImage\Anurag.jpg")
img1 = cv2.resize(img1, down_points, interpolation= cv2.INTER_LINEAR)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
testimg1=face_recognition.load_image_file(r"D:\AttendenceImage\Anurag.jpg")
testimg1 = cv2.resize(testimg1, down_points, interpolation= cv2.INTER_LINEAR)
testimg1=cv2.cvtColor(testimg1,cv2.COLOR_BGR2RGB)


faceloc=face_recognition.face_locations(img1)[0]
encodeAnurag=face_recognition.face_encodings(img1)[0]
#print(faceloc)...(133, 253, 288, 98)
cv2.rectangle(img1,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest=face_recognition.face_locations(testimg1)[0]
testencodeAnurag=face_recognition.face_encodings(testimg1)[0]
#print(faceloc)...(133, 253, 288, 98)
cv2.rectangle(testimg1,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

result=face_recognition.compare_faces([encodeAnurag],testencodeAnurag)
resultDis=face_recognition.face_distance([encodeAnurag],testencodeAnurag)
print(result,resultDis)
cv2.putText(testimg1,f"{result} and value {round(resultDis[0],2)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),1)

cv2.imshow("Anurag",img1)
cv2.imshow("Anurag_Test",testimg1)
cv2.waitKey(0)'''

path="D:/BasicImage"
images=[]
classname=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    currentImage=cv2.imread(f"{path}/{cl}")
    images.append(currentImage)
    classname.append(os.path.splitext(cl)[0])
print(classname)


def findincodings(images):
    encodeList=[]
    for img in images:
        img=cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open(r"D:\pds_project\Marked_Attendence.csv",'r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dateString=now.strftime('%y/%m/%d')
            dateString=now.strftime('%H : %M : %S')
            f.writelines(f'\n{name},{dateString},{dateString}p')

markAttendance("Anurag")

encodeListKnown=findincodings(images)
print("encoding is done")


cap=cv2.VideoCapture(0)

while True:
    _,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurrentFrame=face_recognition.face_locations(imgS)
    encodeCurrentFrame=face_recognition.face_encodings(imgS,facesCurrentFrame)

    for encodeFace,faceLoc in zip(encodeCurrentFrame,facesCurrentFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classname[matchIndex]
            #print(name)
            y1,x2,y2,x1=faceLoc
            #we multiply by 4 of each corrdinate because scaling down input image by 1/4 of web cam image
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,225,0),2)
            cv2.putText(img,name,(x1+5,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(225,225,225),2)
            markAttendance(name)


    cv2.imshow("IMAGE BY WEBCAM",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


