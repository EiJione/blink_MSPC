from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import numpy as np


def find_peaks_and_classify(derivative):
    """
    使用scipy.signal中的find_peaks函数提取一阶导数中的所有峰值，并使用k-means算法进行二分类。
    运行结束时，可视化一阶导数信号，并绘制threshold水平线和筛选后的峰值点。

    参数:
    derivative: 一阶导数的列表。
    """
    # 使用find_peaks函数寻找峰值索引
    peaks_indices, _ = find_peaks(derivative)

    # 提取峰值
    peaks = [derivative[index] for index in peaks_indices]

    # 如果没有找到峰值，直接返回并提示
    if not peaks:
        print("没有找到峰值")
        return

    # 将峰值数据转换为k-means算法所需的格式
    peaks_array = np.array(peaks).reshape(-1, 1)

    # 使用k-means算法进行二分类
    kmeans = KMeans(n_clusters=2, random_state=0).fit(peaks_array)

    # 获取两个类别的中心值
    centers = kmeans.cluster_centers_

    # 计算阈值为两个类中心的平均值
    threshold = np.mean(centers)

    # # 可视化一阶导数信号
    # plt.figure(figsize=(10, 6))
    # plt.plot(derivative, label='一阶导数信号')
    #
    # # 绘制threshold水平线
    # plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
    #
    # # 标记超过阈值的峰值点
    # for peak_idx in peaks_indices:
    #     if derivative[peak_idx] > threshold:
    #         plt.plot(peak_idx, derivative[peak_idx], 'go')  # 超过阈值的峰值点标记为绿色
    #     else:
    #         plt.plot(peak_idx, derivative[peak_idx], 'ro')  # 未超过阈值的峰值点标记为红色
    #
    # plt.legend()
    # plt.xlabel('Index')
    # plt.ylabel('Derivative Value')
    # plt.title('一阶导数信号及峰值点')
    # plt.grid(True)
    # plt.show()

    return threshold


import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


def visualize_blink_segment_modified(original_signal, start_idx, end_idx):
    """
    可视化单个眨眼过程中的原始信号片段，并将片段内的样本使用不一样的颜色进行绘制。

    参数:
    original_signal - 原始信号
    start_idx - 眨眼的起始索引
    end_idx - 眨眼的结束索引
    """
    # 创建图像
    plt.figure(figsize=(10, 4))

    # 绘制片段前的信号部分
    if start_idx > 0:  # 确保起始点之前有数据
        plt.plot(range(0, start_idx), original_signal[0:start_idx], color='grey', marker='o', label='Before Blink')

    # 绘制眨眼片段内的信号
    plt.plot(range(start_idx, end_idx + 1), original_signal[start_idx:end_idx + 1], color='blue', marker='o',
             label='Blink Segment')

    # 绘制片段后的信号部分
    if end_idx < len(original_signal) - 1:  # 确保结束点之后有数据
        plt.plot(range(end_idx + 1, len(original_signal)), original_signal[end_idx + 1:], color='grey', marker='o',
                 label='After Blink')

    # 标记起始和结束点
    plt.scatter([start_idx, end_idx], [original_signal[start_idx], original_signal[end_idx]], color='red')
    plt.text(start_idx, original_signal[start_idx], 'Start', horizontalalignment='right')
    plt.text(end_idx, original_signal[end_idx], 'End', horizontalalignment='left')

    plt.title("Blink Segment with Different Colors")
    plt.xlabel("Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.show()
def find_blinks(derivative, thre, signal):
    """
    基于一阶导数信号和给定阈值，找到所有眨眼的起始和结束索引。

    参数:
    derivative - 一阶导数信号
    thre - 确定峰值的阈值

    返回:
    blinks - 包含每次眨眼起始和结束索引的二维列表
    """
    # 找到所有超过阈值的峰值索引
    # 寻找所有峰值点的索引
    peaks_indices, _= find_peaks(derivative)

    # 筛选出超过阈值的峰值点
    valid_peaks_indices,_ = find_peaks(derivative,height=thre)

    start_end_points = []

    for peak_idx in valid_peaks_indices:
        # 向左寻找零交叉点作为起始点
        left_crossings = np.where(np.diff(np.sign(derivative[:peak_idx])))[0]
        start_idx = left_crossings[-1] + 1 if len(left_crossings) > 0 else 0

        # 向右寻找零交叉点作为结束点
        right_crossings = np.where(np.diff(np.sign(derivative[peak_idx:])))[0]
        end_idx = peak_idx + right_crossings[1] + 1 if len(right_crossings) > 0 else len(derivative) - 1

        start_end_points.append((start_idx, end_idx))
        #visualize_blink_segment_modified(signal,start_idx,end_idx)
    return start_end_points
