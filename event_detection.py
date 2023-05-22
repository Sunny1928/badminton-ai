import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.optimize import curve_fit
import csv
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import find_peaks
import argparse
import pandas as pd
import court.utils as utils 
import court.court as court 
import cv2

FILE_RES = "/Users/sunny/Desktop/ai_badminton/part1/val/"

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0]
    line_s_y = line[1]
    line_e_x = line[2]
    line_e_y = line[3]
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    #斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    #截距
    b = line_s_y - k * line_s_x
    #带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis

def pred_ball(file_name):
    
    list1=[]
    frames=[]
    realx=[]
    realy=[]
    points=[]


    with open(file_name, newline='') as csvFile:
        rows = csv.reader(csvFile, delimiter=',')
        num = 0
        count=0
        for row in rows:
            list1.append(row)
        front_zeros=np.zeros(len(list1))
        for i in range(1,len(list1)):
            frames.append(int(float(list1[i][0])))
            realx.append(int(float(list1[i][2])))
            realy.append(int(float(list1[i][3])))
            if int(float(list1[i][2])) != 0:
                front_zeros[num] = count
                points.append((int(float(list1[i][2])),int(float(list1[i][3])),int(float(list1[i][0]))))
                num += 1
            else:
                count += 1

    # 羽球2D軌跡點
    points = np.array(points)
    x, y, z = points.T

    Predict_hit_points = np.zeros(len(frames))
    ang = np.zeros(len(frames))
    # from scipy.signal import find_peaks
    peaks, properties = find_peaks(y, prominence=40)

    # print(peaks)

    if(len(peaks) >= 10):
        lower = np.argmin(y[peaks[0]:peaks[1]])
        if (y[peaks[0]] - lower) < 40:
            peaks = np.delete(peaks,0)

        lower = np.argmin(y[peaks[-2]:peaks[-1]])
        if (y[peaks[-1]] - lower) < 40:
            peaks = np.delete(peaks,-1)

    # print()
    # print('Serve : ')
    start_point = 0

    for i in range(len(y)-1):
        if((y[i] - y[i+1]) / (z[i+1] - z[i]) >= 15):
            start_point = i+front_zeros[i]
            Predict_hit_points[int(start_point)] = 1
            # print(int(start_point))
            break

    # print('End : ')
    end_point = len(frames)

    # print('Predict points : ')
    plt.plot(z,y*-1,'-')
    for i in range(len(peaks)):
        # print(peaks[i]+int(front_zeros[peaks[i]]),end=',')
        if(peaks[i]+int(front_zeros[peaks[i]]) >= start_point and peaks[i]+int(front_zeros[peaks[i]]) <= end_point):
            Predict_hit_points[peaks[i]+int(front_zeros[peaks[i]])] = 1


    # 打擊的特定frame = peaks[i]+int(front_zeros[peaks[i]])
    # print()
    # print('Extra points : ')
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]+1
        upper=[]
        plt.plot(z[start:end],y[start:end]*-1,'-')
        lower = np.argmin(y[start:end]) #找到最低谷(也就是從最高點開始下墜到下一個擊球點),以此判斷扣殺或平球軌跡
        for j in range(start+lower, end+1):
            if(j-(start+lower) > 5) and (end - j > 5):
                if (y[j] - y[j-1])*3 < (y[j+1] - y[j]):
                    # print(j, end=',')
                    ang[j+int(front_zeros[j])] = 1

                point = [x[j],y[j]]
                line=[x[j-1],y[j-1],x[j+1],y[j+1]]
                # if get_point_line_distance(point,line) > 2.5:
                if angle([x[j-1],y[j-1], x[j],y[j]],[x[j],y[j], x[j+1],y[j+1]]) > 110:
                    # print(j, end=',')
                    ang[j+int(front_zeros[j])] = 1

    ang, _ = find_peaks(ang, distance=105)
    #final_predict, _  = find_peaks(Predict_hit_points, distance=10)
    for i in ang:
        Predict_hit_points[i] = 1
    Predict_hit_points, _ = find_peaks(Predict_hit_points, distance=5)
    final_predict = []
    for i in (Predict_hit_points):
        final_predict.append(i)

    # print()
    # print('Final predict : ')
    # print(list(final_predict))
    # print(data)

    return final_predict

def pred_hitter(file_name, final_predict):
    hitter = []
    balls_x = np.array(pd.read_csv(f"/Users/sunny/Desktop/aa/ball_pred/{file_name}_ball.csv")['X'])
    balls_y = np.array(pd.read_csv(f"/Users/sunny/Desktop/aa/ball_pred/{file_name}_ball.csv")['Y'])
    
    cap = cv2.VideoCapture(f'{FILE_RES}{file_name}/{file_name}.mp4')
    _, frame = cap.read()
    _, frame = cap.read()
    _, frame = cap.read()
    cap.release()
        
    _,c = court.get_court(frame)
    center = (int((c[0][0]+c[1][0])/2), int((c[1][1]+c[1][1])/2))

    for i in range(len(balls_x)):
        if balls_x[i]!=0 and balls_y[i]!=0:
            break

    
    if balls_y[i] > center[1]:
        now = "A"
    else:
        now = "B"
    
    for i in range(len(final_predict)):
        hitter.append(now)
        now = "B" if now == 'A' else 'A'
        

    return hitter

def pred_hitter_o(file_name, final_predict):

    hitter = []
    A_player = np.load(f'/Users/sunny/Desktop/aa/move_dataset/{file_name}_A_move.npy')
    B_player = np.load(f'/Users/sunny/Desktop/aa/move_dataset/{file_name}_B_move.npy')
    balls_x = np.array(pd.read_csv(f"/Users/sunny/Desktop/aa/ball_pred/{file_name}_ball.csv")['X'])
    balls_y = np.array(pd.read_csv(f"/Users/sunny/Desktop/aa/ball_pred/{file_name}_ball.csv")['Y'])

    for i in final_predict:
        # y x
        # for r in range(-1, 2):
        a = A_player[i-4][0][:2]
        b = B_player[i-4][0][:2]
        ball = (balls_y[i], balls_x[i])
        if math.dist(a, ball) >= math.dist(b, ball):
            # b_num+=1
            hitter.append("B")
        elif math.dist(a, ball) < math.dist(b, ball):
            # a_num+=1
            hitter.append("A")
            
        # a_gap = abs(A_player[i][0][0] - balls_y[i])
        # b_gap = abs(B_player[i][0][0] - balls_y[i])
        # if b_gap>=a_gap:
        #     a_num+=1
        # else:
        #     b_num+=1

        # print(i,"a", a_num,"b", b_num)

        # a_gap = abs(A_player[i][0][0] - balls_y[i])
        # b_gap = abs(B_player[i][0][0] - balls_y[i])
        # if b_gap>=a_gap:
        #     print(i, "A")
        # else:
        #     print(i, "B")

    # print(hitter)
    return hitter

def get_confusion_matrix():
    all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    confusion_matrix = np.zeros((2, 2) , dtype=np.int64)
    success_vid = 0
    for vid in range(1,801):

        ball_csv = f"./data/ball_pred_smooth/{str(vid).rjust(5,'0')}_ball.csv"
        try:
            pred = pred_ball(ball_csv)
        except:
            pred = []
        hit_labels = all_hit_labels[all_hit_labels['VideoName'] == vid]['HitFrame'].values

        success_frame = []
        for pred_frame in pred:
            pred_ac = False
            for pred_range in range(-2,3):
                if pred_frame + pred_range in hit_labels:
                    pred_ac = True
                    success_frame.append(pred_frame + pred_range)
                    break
            
            if pred_ac:
                confusion_matrix[0][0] +=1
            else:
                confusion_matrix[1][0] +=1

        if len(pred) == len(hit_labels):
            success_vid +=1

        confusion_matrix[0][1] += len(hit_labels) - len(success_frame)

    print(confusion_matrix)
    print(success_vid)
    print(np.diag(confusion_matrix)/confusion_matrix.sum(1))

def guess_hitter(ball_labels , hit_labels):
    

    ball_labels = ball_labels.drop(ball_labels[ball_labels['Visibility'] == 0].index)


    odd_labels = hit_labels[(hit_labels['ShotSeq'] % 2 == 1)]['HitFrame']
    even_labels = hit_labels[(hit_labels['ShotSeq'] % 2 == 0)]['HitFrame']
    

    frame_range = 3
    odd_hit_range = [i+j for i in odd_labels.values for j in range(-frame_range,frame_range+1)]
    even_hit_range = [i+j for i in even_labels.values for j in range(-frame_range,frame_range+1)]
    
    odd_ball_labels = ball_labels[(ball_labels['Frame'].isin(odd_hit_range))]['Y'].values
    even_ball_labels = ball_labels[(ball_labels['Frame'].isin(even_hit_range))]['Y'].values


    y_range = 0
    if((odd_ball_labels.mean() - y_range  > even_ball_labels.mean())):
        return "B"
    elif ((odd_ball_labels.mean() - y_range  < even_ball_labels.mean())):
        return "A"
    else:
        return -1

get_confusion_matrix()


# result = pd.read_csv("./sample_result.csv")

# for num in range(1,801):
#     file_name = '%05d' % num

    # real
    # realfilename = f"/Users/sunny/Desktop/aa/part1/train/{file_name}/{file_name}_S2.csv"
    # data = pd.read_csv(realfilename)
    # data_HitFrame = np.array(data['HitFrame'])
    # data_Hitter = np.array(data['Hitter'])


    # final_predict = pred_ball(file_name)
    # final_hitter = pred_hitter(file_name, final_predict)
    # print(len(final_predict))
    # print(len(data_HitFrame))
    # print(final_hitter)
    # print(data_Hitter)
    # print()
    # win = 'X'

    # for i in range(len(final_predict)):
    #     if i == len(final_predict):
    #         win = 'A'
    #     _ = pd.DataFrame([[file_name+".mp4" , i+1 , final_predict[i] , final_hitter[i]  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, win]], columns=result.columns)
    #     result = pd.concat((result , _))

# result.to_csv("./csv/0514_result.csv" , index=False)


# result = pd.read_csv("./csv/sample_result.csv")
# all_ball_labels = pd.read_csv(f"./csv/all_ball_pos_valid_V2_smooth.csv")

# for vid in range(1,170):
#     # print(vid)
#     ball_csv = f"./data/ball_val_smooth/{str(vid).rjust(5,'0')}_ball.csv"
#     try:
#         pred = pred_ball(ball_csv)
#     except:
#         pred = []
    
#     tmp = result

#     for idx , frame in enumerate(pred):
#         _ = pd.DataFrame([[str(vid).rjust(5,'0') , idx+1 , frame , "A"  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, 'X']], columns=result.columns)
#         tmp = pd.concat((tmp , _))

#     hit_labels = result[result['VideoName'] == vid]
#     ball_labels = all_ball_labels[all_ball_labels['VideoName'] == vid]
#     # print(result)
#     hitter = guess_hitter(ball_labels , tmp) 
#     if hitter == -1:
#         hitter = 'A'
#     win = 'X'
#     for idx , frame in enumerate(pred):

#         if idx + 1 == len(pred):
#             win = 'A'

#         _ = pd.DataFrame([[str(vid).rjust(5,'0')+".mp4" , idx+1 , frame , hitter  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, win]], columns=result.columns)
#         result = pd.concat((result , _))
#         hitter = "B" if hitter == 'A' else 'A'




    
# result.to_csv("./csv/0514_result.csv" , index=False)    
        


