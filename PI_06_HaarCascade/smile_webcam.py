import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
eye_cascade = cv.CascadeClassifier('datos/haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier('datos/haarcascade_frontalface_alt.xml')     # haarcascade_frontalface_default.xml') 
smile_cascade = cv.CascadeClassifier('datos/haarcascade_smile.xml')

def gris(image):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def rostros(image):
    return face_cascade.detectMultiScale(gris(image), 1.12, 5)

def ojos(image):  
    return eye_cascade.detectMultiScale(gris(image), 1.1 , 5)

def sonrisas(image):
    return smile_cascade.detectMultiScale(gris(image), 1.3, 20)



while(True):
    # Campturamos frame por frame
    ret, frame = cap.read()

    # Desplegamos el frame
    img = frame
    faces = rostros(img)
    eyes =ojos(img)
    smiles=sonrisas(img)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        for (x_s,y_s,w_s,h_s) in eyes:
            if( (x <= x_s) and (y <= y_s) and ( x+w >= x_s+w_s) and ( y+h >= y_s+h_s)):
                cv.rectangle(img, (x_s,y_s),(x_s+w_s,y_s+h_s),(255,255,255),3)
        for (x_s,y_s,w_s,h_s) in smiles:
            if( (x <= x_s) and (y <= y_s) and ( x+w >= x_s+w_s) and ( y+h >= y_s+h_s)):
                cv.rectangle(img, (x_s,y_s),(x_s+w_s,y_s+h_s),(0,255,0),3)  
    cv.imshow('video',img)
    # Presionar q para terminar  
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#Se cierra la ventana
cap.release()
cv.destroyAllWindows()
