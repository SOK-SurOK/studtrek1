import numpy as np
import argparse
import cv2
from collections import Counter
from time import time
from colorama import init, Fore, Back, Style


def len_of_vect_2d(a, b):
    """
    Вычисление длины двумерного вектора
    :param a: начальная точка (x1, y1)
    :param b: вторая точка (x2, y2)
    :return: длина двумерного вектора
    """
    return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def load_image(file_name):
    """
    Загрузка искомого файла
    :param file_name: полное имя файла
    :return: файл в формате np
    """
    return cv2.imread(file_name)


def del_background(img, hsv_min, hsv_max):
    """
    Удаление фона (за счет инвертирования маски)
    :param img: np изображение
    :param hsv_min: минимальное значение цвета фона
    :param hsv_max: максимальное значение цвета фона
    :return: np изображение без фона, маска
    """
    img_filter = cv2.bilateralFilter(img, 10, 17, 17)  # сглаживание
    hsv = cv2.cvtColor(img_filter, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    mask0 = 255 - thresh  # инвертируем маску
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask0, kernel, iterations=1)
    return cv2.bitwise_and(img, img, mask=mask), mask


def find_big_rect_cnt(mask, size=0.4):
    """
    Ищем большие похожие на прямоугольники контуры
    :param mask: маска
    :param size: минимальный процент площади контура от площади маски
    :return: похожие контуры
    """
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ищем контуры

    cnts_norm = []
    size_min = len(mask[0]) * len(mask[1]) * size
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area > size_min:
            cnts_norm.append(cnt)
    return cnts_norm


def create_rect(mask, cnts_norm):
    """
    Сделать из контур(а/ов) один хороший четырехугольник
    :param mask: маска
    :param cnts_norm: контуры
    :return: контур
    """
    cnt = None
    if len(cnts_norm) > 1:  # объединяем большие прямоугольники в один
        approx = np.vstack((cnts_norm))
        cnt = cv2.convexHull(approx)
    else:
        cnt = cnts_norm[0]

    cnt = cv2.convexHull(cnt)  # округляем до внешнего контура
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    cnt_res = cv2.approxPolyDP(cnt, epsilon, True)  # апроксимилируем

    if len(cnt_res) > 4:  # если больше 4 углов, то удаляем самые близкие к центру вершины
        center = [len(mask[0]) / 2, len(mask[1]) / 2]
        approx_norm = []
        for p in cnt_res:
            approx_norm.append([p, len_of_vect_2d(center, p[0])])
        approx_norm = np.array(approx_norm)
        while len(approx_norm) > 4:
            np_min = np.argmin(approx_norm[:, 1])
            approx_norm = np.delete(approx_norm, np_min, 0)
        cnt_res = []  # переводим в нужный формат
        for p in approx_norm[:, 0]:
            cnt_res.append(p)
        cnt_res = np.int0(cnt_res)
    return cnt_res


def create_perspective(img, cnt_res, ratio=1):
    """
    Делаем перспективу
    :param img: исходное изображение
    :param cnt_res: перспективный контур (четырехугольник)
    :param ratio: исходное соотношение
    :return: персмективное np изображение
    """
    pts = cnt_res.reshape(4, 2)  # переводим контур в нужный формат
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # верхняя левая точка имеет наименьшую сумму, тогда как
    rect[2] = pts[np.argmax(s)]  # внизу справа самая большая сумма

    diff = np.diff(pts, axis=1)  # вычислите разницу между точками
    rect[1] = pts[np.argmin(diff)]  # справа вверху будет иметь минимальную разницу
    rect[3] = pts[np.argmax(diff)]  # внизу слева будет имеют максимальную разницу

    rect *= ratio  # умножаем прямоугольник на исходное соотношение

    # теперь, когда у нас есть наш прямоугольник точек, вычислим
    # ширину нашего нового изображения
    (tl, tr, br, bl) = rect
    widthA = len_of_vect_2d(br, bl)
    widthB = len_of_vect_2d(tr, tl)

    # ...и высоту
    heightA = len_of_vect_2d(tr, br)
    heightB = len_of_vect_2d(tl, bl)

    # возьмите максимальное значение ширины и высоты для достижения наших конечных размеров
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # новые точки с высоты птичьего полета
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # вычислите матрицу преобразования перспективы и деформируйте перспективу, чтобы захватить экран
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def create_mask_most_color(img):
    """
    Создание маски по самому распространенному цвету
    :param img: np изображение
    :return:
    """
    warp_filter = cv2.bilateralFilter(img, 10, 17, 17)
    warp_hsv = cv2.cvtColor(warp_filter, cv2.COLOR_BGR2HSV)

    s0, s1 = warp_hsv[0], warp_hsv[1]  # перве две строчки
    s0h, s0s, s0v = s0[:, 0], s0[:, 1], s0[:, 2]
    s1h, s1s, s1v = s1[:, 0], s1[:, 1], s1[:, 2]
    sh, ss, sv = np.hstack((s0h, s1h)), np.hstack((s0s, s1s)), np.hstack((s0v, s1v))
    ch, cs, cv = Counter(sh.flat), Counter(ss.flat), Counter(sv.flat)

    # print(ch, cs, cv)

    bh, bs, bv = None, None, None
    for i in range(len(ch) - 1):
        bh = (ch.most_common(i + 1))[-1][0]
        if bh != 0:
            break
    for i in range(len(cs) - 1):
        bs = (cs.most_common(i + 1))[-1][0]
        if bs != 0:
            break
    for i in range(len(cv) - 1):
        bv = (cv.most_common(i + 1))[-1][0]
        if bv != 0:
            break

    # print(bh)
    # print(bs)
    # print(bv)

    return cv2.inRange(warp_hsv, np.array((bh - 10, bs - 10, bv - 10), np.uint8),
                       np.array((bh + 10, bs + 10, bv + 10), np.uint8))


def good_resize17(mask):
    """
    Сжатие маски к 17 на 17
    :param mask: маска
    :return:
    """
    resized = cv2.resize(mask, (17, 17), interpolation=cv2.INTER_BITS)
    return np.where(resized < 150, 0, resized)


def find_bad_block(mask):  # не работает
    """
    Выявление плохих блоков и их уничтожение
    :param mask: np маска
    :return:
    """
    bad = []
    for i in range(len(mask) - 1):
        for j in range(len(mask) - 1):
            if mask[i, j] == mask[i + 1, j] == mask[i, j + 1] == mask[i + 1, j + 1]:
                bad.append([i, j])
    # скорее всего плохие блоки будут на одной линии на оси ординат
    # print(bad)
    # if len(bad) > 0:
    #     j = bad[0][1]
    #     while j < len(mask) - 1:
    #         for i in range(len(mask)):
    #             mask[i, j] = mask[i, j + 1]
    #         j += 1
    #     for i in range(len(mask)):
    #         mask[i, 16] = 0


def mask_to_bit(mask):
    """
    Маску в бинарную маску
    :param mask: маска
    :return: np-маска из 0 и 1
    """
    return np.where(mask > 150, 1, 0)


def color_print(mask_bit):
    """
    Выводит красиво матрицу
    :param mask_bit:
    :return:
    """
    init()
    print(Back.GREEN, end="")
    for i in mask_bit:
        for j in i:
            if j == 0:
                print(Fore.BLACK + '0', end=" ")
            elif j == 1:
                print(Fore.RED + '1', end=" ")
        print()
    print(Style.RESET_ALL)


def main():
    parser = argparse.ArgumentParser(description='labirint')

    parser.add_argument('img', type=str, help="Path to the query image")
    args = parser.parse_args()

    img0 = load_image(args.img)
    # img0 = load_image('p5.jpg')

    time_before = time()

    img1, mask1 = del_background(img0, np.array((38, 0, 0), np.uint8), np.array((255, 255, 255), np.uint8))
    cnts1 = find_big_rect_cnt(mask1)
    cnt1 = create_rect(mask1, cnts1)
    img2 = create_perspective(img1, cnt1)
    mask2 = create_mask_most_color(img2)
    mask3 = good_resize17(mask2)
    mask4 = mask_to_bit(mask3)

    time_after = time()

    # print(mask4)
    color_print(mask4)

    print(time_after - time_before, ' seconds')

    # print('Нажмите любую клавишу, чтобы выйти')
    # cv2.imshow('mask', mask2)
    # cv2.imshow('result', mask3)
    # cv2.waitKey()  # ждем нажатие любой клавиши
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
