import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from k_means import find_peaks_and_classify,find_blinks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def calculate_EAR(landmarks):
    a1 = calculate_distance(landmarks[160], landmarks[144])
    a2 = calculate_distance(landmarks[158], landmarks[153])
    b = calculate_distance(landmarks[130], landmarks[133])
    ear = (a1 + a2) / (2.0 * b)
    return ear

def process_video_for_EAR(video_path, target_fps=30):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return []

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(actual_fps,target_fps)
    frame_interval = round(actual_fps / target_fps)
    ear_values = []
    frame_counter = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 只处理每个间隔的帧以模拟30fps

        if frame_counter % frame_interval == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                if all(landmarks[i] for i in [130, 133, 160, 159, 158, 144, 145, 153]):
                    ear = calculate_EAR(landmarks)
                    ear_values.append(ear)
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return ear_values

# 使用实际的视频路径替换以下路径，以获取EAR值列表
# video_path = "path/to/your/video.mov"
# ear_values = process_video_for_EAR(video_path)
# ear_percentages = convert_ear_to_percentage(ear_values)  # 假设你已经有了转换为百分比的函数


import numpy as np


def calculate_derivative(ear_percentages, frame_rate=30):
    """
    计算EAR百分比的一阶导数（眼睑速度）。

    参数:
    ear_percentages: 眼睑关闭信号的百分比列表。
    frame_rate: 视频的帧率（每秒帧数）。

    返回:
    derivative: 眼睑速度的列表。
    """
    # 计算时间间隔，假设每帧之间的时间是固定的
    dt = 1 / frame_rate

    # 使用NumPy的diff函数计算相邻EAR百分比之间的差分
    differences = np.diff(ear_percentages)

    # 除以时间间隔dt来得到一阶导数（速度）
    derivative = differences / dt
    #plott(derivative)
    return derivative
def convert_ear_to_percentage(ear_values):#l滤波
    if not ear_values:  # 确保EAR值列表不为空
        return []

    max_ear = max(ear_values)  # 找到最大的EAR值
    if max_ear == 0:  # 防止除以0
        return [0] * len(ear_values)

    # 计算每个EAR值占最大EAR值的百分比
    ear_percentages = [(1-(ear / max_ear)) for ear in ear_values]
    #plott(ear_percentages)
    filtered_signal = savgol_filter(ear_percentages, 13, 7)
    #plott(filtered_signal)
    return filtered_signal
def plott(x):
    # 假设ear_values是之前处理视频得到的EAR值列表
    # 创建时间轴，这里我们假设每个EAR值对应一帧，帧率为30帧/秒
    frame_rate = 30  # 帧率
    time_axis = [i / frame_rate for i in range(len(x))]
    # 绘制EAR值随时间变化的图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(time_axis, x, linestyle='-', color='b')  # 绘制EAR值曲线
    plt.title("EAR值随时间变化图")  # 设置图形标题
    plt.xlabel("时间 (秒)")  # 设置x轴标签
    plt.ylabel("EAR值")  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形


def extract_and_plot_blink(data, index_tuple):
    """
    从数据中提取特定索引范围的片段并绘制它。

    参数:
    data: 包含EAR值的列表。
    index_tuple: 包含起始和结束索引的元组，用于指定提取数据的范围。

    返回:
    None
    """
    start_index, end_index = index_tuple
    # 确保索引在数据范围内
    if start_index < 0 or end_index >= len(data):
        print("Index out of range.")
        return

    # 提取对应的数据片段
    blink_segment = data[start_index:end_index + 1]

    # 创建时间轴
    time_axis = range(start_index, end_index + 1)

    # 绘制数据
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, blink_segment, marker='o', linestyle='-', color='b')
    plt.title("眼睑闭合值的变化")
    plt.xlabel("帧索引")
    plt.ylabel("EAR值")
    plt.grid(True)
    plt.show()
    return blink_segment

def extract_features(ear_values):
    # 计算最大幅度和最小幅度
    max_amplitude = np.max(ear_values)
    min_amplitude = np.min(ear_values)

    # 计算幅度变化
    amplitude_change = max_amplitude - min_amplitude

    # 计算总持续时间
    total_duration = len(ear_values)

    # 找到最大幅度的索引
    max_index = np.argmax(ear_values)

    # 计算上升时间和下降时间
    ascending_time = max_index  # 从开始到最大值的时间
    descending_time = total_duration - max_index  # 从最大值到结束的时间

    as_des = ascending_time / descending_time  # 上升时间与下降时间的比率

    # 计算总能量
    total_energy = np.sum(np.array(ear_values) ** 2)

    # 计算速度（一阶差分）
    velocities = np.diff(ear_values) / 1  # 假设时间间隔为1
    max_velocity = np.max(np.abs(velocities))
    mean_velocity = np.mean(np.abs(velocities))  # 计算速度的平均值

    # 计算加速度（二阶差分）
    accelerations = np.diff(velocities) / 1  # 假设时间间隔为1
    max_acceleration = np.max(np.abs(accelerations))
    mean_acceleration = np.mean(np.abs(accelerations))  # 计算加速度的平均值
    threshold = 0.8 * amplitude_change + min_amplitude
    high_closure_frames = sum(1 for ear in ear_values if ear > threshold)

    # 返回所有计算的特征，包括新的特征
    return [
        max_amplitude, min_amplitude, amplitude_change, total_duration,
        ascending_time, descending_time, as_des, total_energy,
        max_velocity, mean_velocity, max_acceleration, mean_acceleration,
        high_closure_frames
    ]
def extract_all_blinks_features(ear_values, indexss):
    features_matrix = []
    for indices in indexss:
        blink_segment = ear_values[indices[0]:indices[1] + 1]
        features = extract_features(blink_segment)
        features_matrix.append(features)
    return np.array(features_matrix)
# x = process_video_for_EAR('Fold1_part1/Fold1_part1/01/0.mov')
# x = convert_ear_to_percentage(x)
# thre = find_peaks_and_classify(calculate_derivative(x))
# indexss = find_blinks(calculate_derivative(x),thre,x)
# # for indexs in indexss:
# #     extract_and_plot_blink(x,indexs)
# feature_X=extract_all_blinks_features(x,indexss)
# feature_X = pd.DataFrame(feature_X)
# feature_X.to_csv('new_blink_features.csv')
def MSPC(X,R):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

    # 选择前R个主成分

    Sigma_R = np.diag(Sigma[:R])
    VR = VT[:R, :]

    # 计算投影矩阵和单位矩阵
    projection_matrix = VR.T.dot(VR)
    I = np.eye(X.shape[1])

    # 计算T2统计量
    T2 = np.array([x.dot(VR.T).dot(np.linalg.inv(Sigma_R)).dot(VR).dot(x.T) for x in X])

    # 计算Q统计量
    Q = np.array([x.dot(I - projection_matrix).dot(x.T) for x in X])

import os
def process_all_videos_in_folder(folder_path):
    all_features = []
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if 'mp4' in file:
                video_path = os.path.join(root, file)
                print(f"Processing {video_path}...")
                try:
                # 提取EAR值
                    ear_values = process_video_for_EAR(video_path)

                # 将EAR值转换为百分比
                    ear_percentages = convert_ear_to_percentage(ear_values)

                # 计算导数并找到眨眼点
                    derivative = calculate_derivative(ear_percentages)
                    thre = find_peaks_and_classify(derivative)
                    indexss = find_blinks(derivative, thre, ear_percentages)

                # 提取特征
                    features = extract_all_blinks_features(ear_percentages, indexss)
                    all_features.extend(features)
                except:
                    pass
    # 将所有特征保存到一个CSV文件
    all_features_df = pd.DataFrame(all_features)
    output_csv_path = os.path.join(folder_path, "fatigue_drozy30fps_blink_features.csv")
    all_features_df.to_csv(output_csv_path, index=False)
folder_path = 'drozy'  # 替换为包含0.mov文件的文件夹路径
process_all_videos_in_folder(folder_path)