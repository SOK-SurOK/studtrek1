import numpy as np
# import argparse
import imutils
import cv2
import math
from collections import Counter


# ap = argparse.ArgumentParser()
# ap.add_argument("-q", "--query", required=True,
#                 help="Path to the query image")
# args = vars(ap.parse_args())
def len_of_vect_2d(a, b):
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def print_numpy_arr(arr):
    for row in arr:
        print(' '.join(str(col) for col in row))


def main():
    # работа с файлом
    fn = 'p4.jpg'
    img0 = cv2.imread(fn)
    img = img0.copy()

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((38, 0, 0), np.uint8)
    h_max = np.array((255, 255, 255), np.uint8)

    # ищем маску
    img_filter = cv2.bilateralFilter(img, 10, 17, 17)
    hsv = cv2.cvtColor(img_filter, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, h_min, h_max)
    mask0 = 255 - thresh  # инвертируем маску
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask0, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=mask)

    # ищем контуры
    cnts, h = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts_norm = []
    # ищем контуры похожие на прямоугольники
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        # box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        # box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area > len(img[0]) * len(img[1]) * 0.4:
            cnts_norm.append(cnt)
            # cv2.drawContours(img, [cnt], -1, (255, 0, 0), 3)
            # cnt2 = cv2.convexHull(cnt)
            # cv2.drawContours(img, [cnt2], -1, (0, 255, 0), 3)
            # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cnt = None
    # объединяем большие прямоугольники в один
    if len(cnts_norm) > 1:
        # hull_list = []
        # for i in range(len(cnts_norm)):
        #     hull = cv2.convexHull(cnts_norm[i])
        #     hull_list.append(hull)
        approx = np.vstack((cnts_norm))
        # print(approx)
        cnt = cv2.convexHull(approx)
    else:
        cnt = cnts_norm[0]

    cnt = cv2.convexHull(cnt)  # округляем до внешнего контура
    # пытаемся сделать из многоугольника четырехугольник
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cnt_res = approx
    # если больше 4 углов, то удаляем самые близкие к центру вершины
    if len(approx) > 4:
        center = [len(img[0]) / 2, len(img[1]) / 2]
        # print('center', center)
        approx_norm = []
        for p in approx:
            approx_norm.append([p, len_of_vect_2d(center, p[0])])
        approx_norm = np.array(approx_norm)
        # print('approx_norm', approx_norm)
        # print('approx_norm[]', approx_norm[:, 1])
        while len(approx_norm) > 4:
            np_min = np.argmin(approx_norm[:, 1])
            # print('min_vect', np_min)
            approx_norm = np.delete(approx_norm, np_min, 0)
        cnt_res = []
        for p in approx_norm[:, 0]:
            cnt_res.append(p)
        cnt_res = np.int0(cnt_res)
        # print('approx_norm', approx_norm)
        # print('cnt_res', cnt_res)

    # рисуем контур
    # cv2.drawContours(res, [cnt_res], -1, (255, 0, 255), 3)

    pts = cnt_res.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    ratio = 1
    # multiply the rectangle by the original ratio
    rect *= ratio

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(res, M, (maxWidth, maxHeight))

    warp_filter = cv2.bilateralFilter(warp, 10, 17, 17)
    warp_hsv = cv2.cvtColor(warp_filter, cv2.COLOR_BGR2HSV)

    s0, s1 = warp_hsv[0], warp_hsv[1]  # перве две строчки
    s0h, s0s, s0v = s0[:, 0], s0[:, 1], s0[:, 2]
    s1h, s1s, s1v = s1[:, 0], s1[:, 1], s1[:, 2]
    sh, ss, sv = np.hstack((s0h, s1h)), np.hstack((s0s, s1s)), np.hstack((s0v, s1v))
    ch, cs, cv = Counter(sh.flat), Counter(ss.flat), Counter(sv.flat)

    print(ch, cs, cv)

    bh, bs, bv = None, None, None
    for i in range(len(ch)-1):
        bh = (ch.most_common(i + 1))[-1][0]
        if bh != 0:
            break
    for i in range(len(cs)-1):
        bs = (cs.most_common(i + 1))[-1][0]
        if bs != 0:
            break
    for i in range(len(cv)-1):
        bv = (cv.most_common(i + 1))[-1][0]
        if bv != 0:
            break

    print(bh)
    print(bs)
    print(bv)

    mask2 = cv2.inRange(warp_hsv, np.array((bh - 10, bs - 10, bv - 10), np.uint8),
                        np.array((bh + 10, bs + 10, bv + 10), np.uint8))
    # res2 = cv2.bitwise_and(warp, warp, mask=mask2)

    # resized0 = cv2.resize(mask2, (102, 102), interpolation=cv2.INTER_LINEAR_EXACT)

    # resized1 = cv2.resize(mask2, (34, 34), interpolation=cv2.INTER_LINEAR)
    # kernel = np.ones((1, 1), np.uint8)
    # kernel2 = np.ones((1, 1), np.uint8)
    # kernel3 = np.ones((1, 1), np.uint8)
    # resized2 = cv2.erode(resized1, kernel, iterations=1)
    # resized3 = cv2.dilate(resized2, kernel2, iterations=1)
    # resized4 = cv2.erode(resized3, kernel3, iterations=1)

    resized = cv2.resize(mask2, (17, 17), interpolation=cv2.INTER_BITS)
    # resized2 = cv2.resize(resized0, (17, 17), interpolation=cv2.INTER_BITS)

    result = np.where(resized > 150, 1, 0)
    print(result)
    # print(resized)
    # выводим изображения
    # cv2.imshow('img', img)
    cv2.imshow('hsv_filter', warp_hsv)
    cv2.imshow('res', res)
    cv2.imshow('mask2', mask2)
    # cv2.imshow('res2', res2)
    # cv2.imshow('resized0', resized0)
    cv2.imshow('resized', resized)
    # cv2.imshow('resized2', resized2)
    # cv2.imshow('result2', result2)
    cv2.waitKey()  # ждем нажатие любой клавиши
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
