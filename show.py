import numpy as np
import cv2
import pandas

file_name = '%05d' % 2

df = pandas.read_csv(f"./part1/train/{file_name}/{file_name}_S2.csv")
cap = cv2.VideoCapture(f'./part1/train/{file_name}/{file_name}.mp4')
hitframe = 0
index = 0

while(True):
    ret, frame = cap.read()


    if hitframe == df.iloc[index]['HitFrame']:
        
        if index != 0:
            cv2.circle(frame, (df.iloc[index-1]['LandingX'], df.iloc[index-1]['LandingY']), 5, (0,0,255), -1)
        cv2.circle(frame, (df.iloc[index]['HitterLocationX'], df.iloc[index]['HitterLocationY']), 5, (0,255,0), -1)
        cv2.circle(frame, (df.iloc[index]['DefenderLocationX'], df.iloc[index]['DefenderLocationY']), 5, (255,0,0), -1)
        cv2.circle(frame, (df.iloc[index]['HitterLocationX'], df.iloc[index]['HitterLocationY']), 5, (0,255,0), -1)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        index+=1

    hitframe += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
