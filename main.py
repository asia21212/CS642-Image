import os
import cv2
import numpy as np

def stitcher():
    array = []
    for item in os.listdir('data'):
        img = cv2.imread(f'data/{item}')
        print(item)
        array.append(img)

    stitcher = cv2.Stitcher_create()
    stitcher.setPanoConfidenceThresh(0.5)
    status, result = stitcher.stitch(tuple(array))
    if status == 0:
        cv2.imwrite('test/stitcher_result9.jpg',result)
        # cv2.imshow('result',result)
        cv2.waitKey(0)
    else:
        print('fall')

def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame

def custom(thresh=0.5):
    img_ = cv2.imread('data2/feature_result12.jpg')
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    img = cv2.imread('data2/feature_result13.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,flags=2)

    img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
    # cv2.imshow("original_image_drawMatches.jpg", img3)
    # cv2.waitKey(0)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enought matches are found - %d/%d", (len(good) / MIN_MATCH_COUNT))

    dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img
    # cv2.imshow("original_image_stitched.jpg", dst)

    cv2.imwrite('data2/feature_result14.jpg',trim(dst))
    # cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    cv2.waitKey(0)


if __name__ == '__main__':
    stitcher()
    custom()