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

# /Users/sunny/Desktop/aa/data/ball_pred_smooth/{file_name}_ball.csv

# BALL_URL = "./data/ball_pred_smooth/"
# MOVE_URL = "/Users/sunny/move_dataset/"

BALL_URL = "./data/ball_val_smooth/"
MOVE_URL = "/Users/sunny/move_dataset_val/"

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

    with open(f"{BALL_URL}{file_name}_ball.csv", newline='') as csvFile:
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
    peaks, properties = find_peaks(y, prominence=60)

    start_point = 0

    for i in range(len(y)-1):
        if((y[i] - y[i+1]) / (z[i+1] - z[i]) >= 4):
            start_point = i+front_zeros[i]
            Predict_hit_points[int(start_point)] = 1
            # print(int(start_point))
            break

    # print('End : ')
    end_point = 10000

    # print('Predict points : ')
    plt.plot(z,y*-1,'-')
    for i in range(len(peaks)):
        # print(peaks[i]+int(front_zeros[peaks[i]]),end=',')
        if(peaks[i]+int(front_zeros[peaks[i]]) >= start_point and peaks[i]+int(front_zeros[peaks[i]]) <= end_point):
            Predict_hit_points[peaks[i]+int(front_zeros[peaks[i]])] = 1


    final_predict, _  = find_peaks(Predict_hit_points, distance=10)

    return final_predict

def pred_hitter(final_predict, file_name):
    A_player = np.load(f'{MOVE_URL}{file_name}_A_move.npy')
    B_player = np.load(f'{MOVE_URL}{file_name}_B_move.npy')
    balls_x = np.array(pd.read_csv(f"{BALL_URL}{file_name}_ball.csv")['X'])
    balls_y = np.array(pd.read_csv(f"{BALL_URL}{file_name}_ball.csv")['Y'])
    hitter = []

    for i in final_predict:
        # y x
        a_num = 0
        b_num = 0

        for r in range(-3, 4):
            a = A_player[i+r][0][:2]
            b = B_player[i+r][0][:2]
            ball = (balls_y[i], balls_x[i])
            if math.dist(a, ball) >= math.dist(b, ball):
                b_num+=1
                # hitter.append("B")
            elif math.dist(a, ball) < math.dist(b, ball):
                a_num+=1
                # hitter.append("A")

        if b_num >= a_num:
            hitter.append("B")
        else:
            hitter.append("A")

        # gap_a = abs(balls_y[i]-A_player[i][0][0])
        # gap_b = abs(balls_y[i]-B_player[i][0][0])

        # if gap_a >= gap_b:
        #     # b_num+=1
        #     hitter.append("B")
        # elif gap_a < gap_b:
        #     # a_num+=1
        #     hitter.append("A")

    final_hitter = []
    for h in range(len(hitter)):
        if h == 0:
            final_hitter.append(hitter[h])
        else:
            if hitter[h]!=hitter[h-1]:
                final_hitter.append(hitter[h])
            else:
                if hitter[h-1] == 'A':
                    final_hitter.append('B')
                    final_hitter.append('A')
                elif hitter[h-1] == 'B':
                    final_hitter.append('A')
                    final_hitter.append('B')

                final_predict = np.insert(final_predict, h, int((final_predict[h-1]+final_predict[h])/2))
            
    return final_predict, final_hitter

def get_confusion_matrix():
    all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    confusion_matrix = np.zeros((2, 2) , dtype=np.int64)
    success_vid = 0
    for vid in range(1,801):

        ball_csv = f"{BALL_URL}{str(vid).rjust(5,'0')}_ball.csv"
        file_name = '%05d' % vid

        try:
            pred = pred_ball(file_name)
            pred, final_hitter = pred_hitter(pred, file_name)
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

get_confusion_matrix()

# file_name = '%05d' % 2


# realfilename = f"/Users/sunny/Desktop/aa/part1/train/{file_name}/{file_name}_S2.csv"
# data = pd.read_csv(realfilename)
# data_HitFrame = np.array(data['HitFrame'])
# data_Hitter = np.array(data['Hitter'])


# pred = pred_ball(file_name)
# pred, hitter = pred_hitter(pred, file_name)
# print(pred)
# print(data_HitFrame)

# print(hitter)
# print(data_Hitter)

# df['Hitter'] = hitter



# csv

result = pd.read_csv("./csv/sample_result.csv")
# all_ball_labels = pd.read_csv(f"./csv/all_ball_pos_val_V2_smooth.csv")

for vid in range(1,170):
    # print(vid)
    file_name = '%05d' % vid
    # ball_csv = f"./data/ball_val_smooth/{str(vid).rjust(5,'0')}_ball.csv"
    try:
        pred = pred_ball(file_name)
        pred, hitter = pred_hitter(pred, file_name)
        # print(vid, len(hitter))
    except:
        pred = []
    
    tmp = result


    # for idx , frame in enumerate(pred):
    #     _ = pd.DataFrame([[str(vid).rjust(5,'0') , idx+1 , frame , "A"  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, 'X']], columns=result.columns)
    #     tmp = pd.concat((tmp , _))

    # hit_labels = result[result['VideoName'] == vid]
    # ball_labels = all_ball_labels[all_ball_labels['VideoName'] == vid]
    # print(result)
    # hitter = guess_hitter(ball_labels , tmp) 
    # if hitter == -1:
    #     hitter = 'A'
    win = 'X'

    c = 0

    for idx , frame in enumerate(pred):

        if idx + 1 == len(pred):
            win = 'A'

        _ = pd.DataFrame([[str(vid).rjust(5,'0')+".mp4" , idx+1 , frame , hitter[c]  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, win]], columns=result.columns)
        result = pd.concat((result , _))
        # hitter = "B" if hitter == 'A' else 'A'
        c+=1

    
result.to_csv("./csv/0516_result.csv" , index=False)    





