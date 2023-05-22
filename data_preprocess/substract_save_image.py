import numpy as np
import cv2
import pandas



scale_percent = 50
dsize = (int(1280 * scale_percent / 100), int(720 * scale_percent / 100))



for num in range(1,801):
    unit = '%05d' % num
    df = pandas.read_csv(f"./part1/train/{unit}/{unit}_S2.csv")
    cap = cv2.VideoCapture(f'./part1/train/{unit}/{unit}.mp4')
    hitframe = 0
    index = 0


    while(True):
        ret, frame = cap.read()

        


        if hitframe == df.iloc[index]['HitFrame']-2 or hitframe == df.iloc[index]['HitFrame']-1 or hitframe == df.iloc[index]['HitFrame'] or hitframe == df.iloc[index]['HitFrame']+1 or hitframe == df.iloc[index]['HitFrame']+2:
            output = cv2.subtract(former, frame)
            output = cv2.resize(output, dsize)
            cv2.imwrite(f"./datasets/training/0_{unit}_{hitframe}.jpg", output)
            cv2.imshow('frame', output)

        
        elif hitframe == df.iloc[index]['HitFrame']-5 or hitframe == df.iloc[index]['HitFrame']-4 or hitframe == df.iloc[index]['HitFrame']-3 or hitframe == df.iloc[index]['HitFrame']+3 or hitframe == df.iloc[index]['HitFrame']+4:
            output = cv2.subtract(former, frame)
            output = cv2.resize(output, dsize)
            cv2.imwrite(f"./datasets/training/1_{unit}_{hitframe}.jpg", output)
            # cv2.imshow('frame', output)
            
            if hitframe == df.iloc[index]['HitFrame']+4:
                index+=1

            

        former = frame
        
        hitframe += 1

        if index == len(df):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
