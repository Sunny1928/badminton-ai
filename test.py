import numpy as np
import cv2
import pandas
# import mediapipe as mp
# from ultralytics import YOLO


# mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
# mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
# mp_pose = mp.solutions.pose

df = pandas.read_csv("./part1/train/00001/00001_S2.csv")
cap = cv2.VideoCapture('./part1/train/00001/00001.mp4')
hitframe = 0
index = 0

# model = YOLO("yolov8n.pt")


# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
while(True):
    ret, frame = cap.read()


    if hitframe == df.iloc[index]['HitFrame']:
        # yolo
        # results = model(frame)
        # result = results[0]
        # bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # classes = np.array(result.boxes.cls.cpu(), dtype="int")

        # for cls, bbox in zip(classes, bboxes):
        #     (x, y, x2, y2) = bbox
        #     if cls == 0:
        #         print(bbox)
        #         cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        #         cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
                

        # mediapipe
        # img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        # results = pose.process(img2)                  # 取得姿勢偵測結果
        # # 根據姿勢偵測結果，標記身體節點和骨架
        # mp_drawing.draw_landmarks(
        #     frame,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
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
