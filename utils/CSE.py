import numpy as np
import cv2
import math
import os


def get_maximum(img2, histsize):
    maximum2 = 0
    for i in range(3):
        range_v = 180 if i == 0 else 256  # 180 for H, 256 for S,V

        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [i], None, np.int32([range_v / histsize]), [0, range_v])

        if np.max(hist2, 0)[0] > maximum2:
            maximum2 = np.max(hist2, 0)[0]

    return maximum2


def get_comp_weight(img1, img2, histsize):
    weight = []
    for i in range(3):
        range_v = 180 if i == 0 else 256  # 180 for H, 256 for S,V
        hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)], [i], None, np.int32([range_v / histsize]), [0, range_v])
        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [i], None, np.int32([range_v / histsize]), [0, range_v])

        max1 = np.max(hist1, 0)[0]
        max2 = np.max(hist2, 0)[0]
        weight.append((max1 + max2) / 2)

    weight /= sum(weight)

    return weight


def make_diff(vector, order):
    for _ in range(order):
        vector = vector[0:-1] - vector[1:]

    return vector


def get_integral(vector):
    vector = np.abs(vector)
    integral = np.sum(vector) - 0.5 * (vector[0] + vector[-1])

    return integral


def peak_positioning(vector, Pcount):
    # Pcount should be a positive integer

    cur_peak = 0
    peak = []
    sort_v = np.argsort(-vector)
    for i in sort_v:
        if i == 0:
            if vector[i] >= vector[i + 1]:
                peak.append((i, vector[i]))
                cur_peak += 1
                if cur_peak == Pcount:
                    return peak
        elif i == len(sort_v) - 1:
            if vector[i] >= vector[i - 1]:
                peak.append((i, vector[i]))
                cur_peak += 1
                if cur_peak == Pcount:
                    return peak
        else:
            if vector[i] >= vector[i - 1] and vector[i] >= vector[i + 1]:
                peak.append((i, vector[i]))
                cur_peak += 1
                if cur_peak == Pcount:
                    return peak

    return peak


def get_peak_weight(peak1, peak2):
    weight = []
    for i in range(len(peak1)):
        weight.append((peak1[i][1] + peak2[i][1]) / 2)
    weight /= sum(weight)

    return weight


def peak_aligning(peaks1, peaks2, threshold=0.25):
    if len(peaks1) != len(peaks2):
        raise Exception('The number of peak1 and peak2 should be the same!')
    for i in range(len(peaks1) - 1):
        for j in range(i + 1, len(peaks1)):
            if math.fabs(peaks1[i][0] - peaks2[i][0]) < math.fabs(peaks1[i][0] - peaks2[j][0]) and math.fabs(
                    peaks1[j][0] - peaks2[j][0]) < math.fabs(
                peaks1[j][0] - peaks2[i][0]):  # The x-coordinate must satisfy the conditions
                continue
            difference = 0.5 * math.fabs(peaks1[i][1] - peaks1[j][1]) + 0.5 * math.fabs(peaks2[i][1] - peaks2[j][1])
            range_p = 0.25 * (peaks1[i][1] + peaks1[j][1] + peaks2[i][1] + peaks2[j][1])
            if difference / range_p <= threshold:
                peaks1[i], peaks1[j] = peaks1[j], peaks1[i]

    return peaks1, peaks2


def CSE(img1, img2, histsize=4, Pcount=2, Kdiff=1, Ithreshold=0.25):
    '''
    function: given two images, return CSE value.
    input:  range:[0,255]   type:uint8    format:[h,w,c]   BGR(Note: not RGB)
    output: a python value, i.e., color-sensitive error (CSE)
    '''
    histsize=np.int32(histsize)
    maximum2 = get_maximum(img2, histsize)
    comp_weight = get_comp_weight(img1, img2, histsize)
    error = 0
    for i in range(3):
        range_v = 180 if i == 0 else 256  # 180 for H, 256 for S,V
        hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)], [i], None, np.int32([range_v / histsize]), [0, range_v])
        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [i], None, np.int32([range_v / histsize]), [0, range_v])

        if np.max(hist2, 0)[0] < maximum2 * Ithreshold:
            error += (1 / range_v) * comp_weight[i] * np.sqrt(np.mean((hist1 - hist2) ** 2))
            continue

        peak1 = peak_positioning(hist1.squeeze(-1), Pcount)
        peak2 = peak_positioning(hist2.squeeze(-1), Pcount)
        peak1, peak2 = peak_aligning(peak1, peak2, 0.3)
        peak_weight = get_peak_weight(peak1, peak2)

        distance = 0
        for j in range(Pcount):
            distance += math.fabs(peak1[j][0] - peak2[j][0]) * peak_weight[j]

        diff1 = make_diff(hist1, Kdiff)
        diff2 = make_diff(hist2, Kdiff)
        integral = np.abs(get_integral(diff1) - get_integral(diff2))[0]

        error += (1 / range_v) * comp_weight[i] * integral * math.exp(math.sqrt(distance))

    return error


if __name__ == '__main__':
    img1 = cv2.imread('img path1')
    img2 = cv2.imread('img path2')
    cse = CSE(img1, img2)