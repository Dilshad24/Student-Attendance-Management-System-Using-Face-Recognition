import face_recognition
import cv2
import os
import numpy as np
import pickle
import time
import datetime

Known_Faces_Dir="Known_Faces"
#Unknown_Faces_Dir="Unknown_Faces"
Tolerance=0.35
Frame_Thickness=3
Font_Thicknes=2
#Model="cnn"
Model="hog"
video=cv2.VideoCapture('test1.mp4') # could put in a filename
print("loading known faces")

known_faces=[]
known_names=[]
'''
for name in os.listdir(Known_Faces_Dir):
  for filename in os.listdir(f"{Known_Faces_Dir}/{name}"):
    try:qq
      image=face_recognition.load_image_file(f"{Known_Faces_Dir}/{name}/{filename}")
      encoding=face_recognition.face_encodings(image)[0]
      known_faces.append(encoding)
      known_names.append(name)
    except:
      pass


with open("known_faces.txt", "wb") as fp:   #Pickling
  pickle.dump(known_faces, fp)
with open("known_names.txt", "wb") as fp:   #Pickling
  pickle.dump(known_names, fp) 
'''
with open("known_faces.txt", "rb") as fp:   # Unpickling
   known_faces = pickle.load(fp)
with open("known_names.txt", "rb") as fp:   # Unpickling
   known_names = pickle.load(fp)
print(known_names)
print(len(known_names))
print("processing unknown faces")
def markAttendance(name):
  with open("Attendance.csv",'r+') as f:

    myDataList=f.readlines()
    namelist=[]
    for line in myDataList:
      entry=line.split(',')
      namelist.append(entry[0])
    if name not in namelist:
      ctime = time.asctime(time.localtime(time.time()))
      ctime = ctime.split(' ')
      cDate = ctime[4] + ' ' + ctime[1] + ' ' + ctime[0] + ' ' + ctime[2]
      ctime = ctime[3]
      f.writelines(f'\n{name},{cDate},{ctime}')



while True:
  ret,image=video.read()
  scale_percent = 90 # percent of original size
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)
  # resize image
  image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  locations=face_recognition.face_locations(image,model=Model)
  encodings=face_recognition.face_encodings(image,locations)
  #image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

  for face_encoding, face_location in zip(encodings,locations):
    results=face_recognition.compare_faces(known_faces,face_encoding,Tolerance)
    resu1t1=face_recognition.face_distance(known_faces,face_encoding)
    match=None
    if True in results:
      print(results)
      match=known_names[results.index(True)]
      print(f"Match found: {match}")

      top_left=(face_location[3],face_location[0])
      bottom_right=(face_location[1],face_location[2])
      color=[0,255,0]
      cv2.rectangle(image,top_left,bottom_right,color,Frame_Thickness)

      top_left=(face_location[3],face_location[2])
      bottom_right=(face_location[1],face_location[2]+22)
      cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
      cv2.putText(image,match,
                  (face_location[3]+10,face_location[2]+15),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  .8,(200,100,100),Font_Thicknes)
      markAttendance(match)

  cv2.imshow("Video",image)
  if cv2.waitKey(1) & 0xFF ==ord("q"):
      break