import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_video(video_path):
    print(f"Processing {video_path}...")
    try:
        # 假设以下函数已经定义，用于处理视频和提取特征
        ear_values = process_video_for_EAR(video_path)
        ear_percentages = convert_ear_to_percentage(ear_values)
        derivative = calculate_derivative(ear_percentages)
        thre = find_peaks_and_classify(derivative)
        indexss = find_blinks(derivative, thre, ear_percentages)
        features = extract_all_blinks_features(ear_percentages, indexss)
        return features
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []


def analyze_videos(video1, video2):
    # 处理两个视频
    features1 = process_video(video1)
    features1_df = pd.DataFrame(features1)
    features2 = process_video(video2)
    features2_df = pd.DataFrame(features2)

    # 保存特征为CSV
    csv_name1 = os.path.splitext(os.path.basename(video1))[0] + '_features.csv'
    features1_df.to_csv(csv_name1, index=False)
    csv_name2 = os.path.splitext(os.path.basename(video2))[0] + '_features.csv'
    features2_df.to_csv(csv_name2, index=False)

    # 标准化第一个视频特征
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(features1_df)

    # 奇异值分解（SVD）
    U, Sigma, VT = np.linalg.svd(X_normalized, full_matrices=False)
    R = min(8, Sigma.size)  # 选择最小的数目，防止R超过特征维度
    Sigma_R = np.diag(Sigma[:R])
    VR = VT[:R, :]

    # 计算投影矩阵和单位矩阵
    projection_matrix = VR.T @ VR
    I = np.eye(X_normalized.shape[1])

    # 计算T2和Q统计量
    T2 = np.array([(x @ VR) @ np.linalg.inv(Sigma_R) @ (x @ VR).T for x in X_normalized])
    Q = np.array([x @ (I - projection_matrix) @ x.T for x in X_normalized])

    # 标准化第二个视频特征并计算T2和Q
    Y_normalized = scaler.transform(features2_df)
    T2_Y = np.array([(y @ VR) @ np.linalg.inv(Sigma_R) @ (y @ VR).T for y in Y_normalized])
    Q_Y = np.array([y @ (I - projection_matrix) @ y.T for y in Y_normalized])

    # 保存T2和Q值
    pd.DataFrame({'T2': T2, 'Q': Q}).to_csv(csv_name1.replace('_features.csv', '_T2_Q.csv'), index=False)
    pd.DataFrame({'T2': T2_Y, 'Q': Q_Y}).to_csv(csv_name2.replace('_features.csv', '_T2_Q.csv'), index=False)

    print(f"T2 and Q values saved for {video1} and {video2}")


# 使用示例
analyze_videos('path_to_video1.mp4', 'path_to_video2.mp4')
