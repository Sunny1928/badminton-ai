import numpy as np
import cv2
import pandas


label = "00001"
df = pandas.read_csv(f"./part1/train/{label}/{label}_S2.csv")
cap = cv2.VideoCapture(f'./part1/train/{label}/{label}.mp4')
hitframe = 0
index = 0
scale_percent = 50
dsize = (int(1280 * scale_percent / 100), int(720 * scale_percent / 100))


while(True):
    ret, frame = cap.read()

    if hitframe == df.iloc[index]['HitFrame']-1:
        former = frame

    if hitframe == df.iloc[index]['HitFrame']:

        frame = cv2.subtract(former, frame)
        
        if index != 0:
            cv2.circle(frame, (df.iloc[index-1]['LandingX'], df.iloc[index-1]['LandingY']), 5, (0,0,255), -1)
        cv2.circle(frame, (df.iloc[index]['HitterLocationX'], df.iloc[index]['HitterLocationY']), 5, (0,255,0), -1)
        cv2.circle(frame, (df.iloc[index]['DefenderLocationX'], df.iloc[index]['DefenderLocationY']), 5, (255,0,0), -1)
        
        output = cv2.resize(frame, dsize)
        cv2.imshow('frame', output)
        cv2.waitKey(0)
        index+=1

    hitframe += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
