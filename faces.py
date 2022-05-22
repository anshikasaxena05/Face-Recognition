import numpy as np
import cv2
import pickle
from datetime import datetime
face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#harrcascade detects the faces in a frame
recognizer= cv2.face.LBPHFaceRecognizer_create()#the recognizer used here is open cv but we can user any other recognizer as well
recognizer.read("trainner.yml")
labels={}#"personname":1 so we need to invert it
with open("labels.pickle","rb") as f: # in file we have labels
	og_labels=pickle.load(f)
	labels={v:k for k,v in og_labels.items()}#inverting labels

name=""
cap=cv2.VideoCapture(0)
while(True):
	ret,frame=cap.read()#Reading (cam.read()) from a VideoCapture returns a tuple (return value, image).
	# With the first item you check wether the reading was successful, and if it was then you proceed to use the returned image.
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)#scalefactor is Parameter specifying how much the image size is 
	#(altered)scaledown# per detection pass at each image scale.A factor of 1.1 corresponds to an increase of 10%.Hence, increasing the scale 
	#factor	increases performance, as the number of detection passes is reduced.But too much increase might be a problem
	#minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it.minSize:Minimum possible object size
	for(x,y,w,h) in faces:
		#print(x,y,w,h)#input in form of numbers
		roi_gray=gray[y:y+h,x:x+w]#x is the horizontal initial position, w is the width, y is the vertical initial position, h is the height. 
		#roi for only the gray frame
		#here y:y+h,x:x+w are coordinates 2 diagonal coordinates (ystart,yend)   
		roi_color=frame[y:y+h,x:x+w] 
		#reccognize???## open cv has for us to train a recognizer(if we donot use open cv to train our recognizer then we can use deep learning in this place)
		#we need to have some algorithm
		id_,conf=recognizer.predict(roi_gray)#id_labels conf ->confidence
		if conf>=45:
			print(id_)
			print(labels[id_])
			name=labels[id_]
			font=cv2.FONT_HERSHEY_SIMPLEX
			name=labels[id_]
			color=(255,255,255)
			stroke=2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

		img_item="my-image.png"
		cv2.imwrite(img_item,roi_gray) 
		#draw a rectangle
		color=(147,20,255)#color of the frame in bgr not rgb
		stroke=2
		end_cord_x=x+w
		end_cord_y=y+h 
		#rectange on frame (x,y)=starting coordinates
		cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF==ord('q'):
		break
def markAttendance(name):
	with open('attendance.csv',"r+") as f:
		mydl=f.readlines()
		print(mydl)
		namel=[]
		for line in mydl:
			entry=line.split(",")
			namel.append(entry[0])
		if name not in namel:
			now=datetime.now()
			dts=now.strftime("%H:%m:%S")
			f.writelines(f'\n{name},{dts}')

#print(name)
markAttendance(name)

cap.release()
cv2.destroyAllWindows()
