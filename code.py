
import cv2 as cv
cap=cv.VideoCapture(0)
cascade_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret,frame=cap.read()
    frame=cv.cvtColor(frame,0)
    detections=cascade_classifier.detectMultiScale(frame)

    if (len(detections)>0):
        (x,y,w,h)=detections[0]
        frame=cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF ==ord('1'):
        break

cap.release()
cv.destroyAllWindows()