
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cap = cv.VideoCapture(0)

#Funciones
def hsv_convert(img):
    return cv.cvtColor(img, cv.COLOR_RGB2HSV)

#Máscara azul
def f_mask_blue(img):
    lower_blue1 = np.array([0,100,62])
    upper_blue1 = np.array([179,255,255])
    img_hsv=hsv_convert(img)
    mask_blue = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    return mask_blue


def punto_medio(imagen):
    img_cercana= imagen[180:, :]
    suma_columnas = img_cercana.sum(axis=0)
    x_pos = np.arange(len(suma_columnas))
    mid_point=int( np.dot(x_pos,suma_columnas) / np.sum( suma_columnas ) )
    return mid_point

#Funcion suma normalizada izquierda
def sum_izquierda(imagen, valor_punto_medio):
    return np.round(np.sum( imagen[:, :valor_punto_medio].sum(axis=0) )/(255*200*400),2)

#Funcion suma normalizada derecha
def sum_derecha(imagen, valor_punto_medio):
    return  np.round(np.sum( imagen[:, valor_punto_medio:].sum(axis=0) )/(255*200*400),2)

#Declaraciones de variables
P1=(135, 150)
P2=(275, 150)
P3=(100, 200)
P4=(300, 200)

ratio= 1
altura = 200*ratio
ancho = 400*ratio

pts1 = np.float32( [  P1, P2, P3, P4] )
pts2 = np.float32( [ [0,0] , [ancho,0], [0, altura] ,[ancho, altura] ] )

# Tipo de fuente
font = cv.FONT_HERSHEY_SIMPLEX

# origen de cada texto
org1 = (60, 185)
org2 = (320, 185)
org3 = (170, 100)

# Tamaño
fontScale = 0.7

# Color de la fuente
color = (150, 150, 150)

# Grosor de la linea del texto
thickness = 1

while True:

    ret, frame = cap.read()
    #cv.imshow('frame',frame)
    frame= cv.resize(frame, (400,200), interpolation=cv.INTER_NEAREST)

    ##
    img=np.copy(frame)
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_NewPerspective = cv.warpPerspective(img, matrix, (ancho,altura))
    img_bin=f_mask_blue(img_NewPerspective)
    mid_point = punto_medio(img_bin)
    cv.circle(img_bin, (mid_point,195), 5 , (100,100,100) , -1  )
    #cv.imshow('binarizada',img_bin)

    # textos
    text1 = str(sum_izquierda(img_bin, mid_point ))
    text2 = str(sum_derecha(img_bin, mid_point ))
    delta = sum_izquierda(img_bin, mid_point ) - sum_derecha(img_bin, mid_point)
    text3 = str(np.round( delta ,2  )   )

    # Usamos la función cv.putText() para agregar text
    cv.putText(img_bin, text1, org1, font, fontScale, color, thickness, cv.LINE_AA, False)
    cv.putText(img_bin, text2, org2, font, fontScale, color, thickness, cv.LINE_AA, False)
    cv.putText(img_bin, text3, org3, font, fontScale, color, thickness, cv.LINE_AA, False)

    if delta < -0.15:
        print('Girar a la derecha')
    elif delta > 0.15:
        print('Girar a la izquierda')
    else :
        print('Sin girar')

    cv.imshow('binarizada',img_bin)
     
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
