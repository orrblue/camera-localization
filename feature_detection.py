import numpy as np
import cv2
import matplotlib.pyplot as plt


pattern = cv2.imread('img/pattern.png',0)
match1 = cv2.imread('img/IMG_6725.JPG', 0)


orb = cv2.ORB_create()

keypoints_pattern, descriptors_pattern = orb.detectAndCompute(pattern, None)
keypoints_match1, descriptors_match1 = orb.detectAndCompute(match1, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors_pattern, descriptors_match1)
matches = sorted(matches, key = lambda x:x.distance)

output = cv2.drawMatches(pattern, keypoints_pattern, match1, keypoints_match1, matches[:5], None, flags=2)
cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)

output = cv2.resize(output, None, fx=0.4, fy=0.4,interpolation = cv2.INTER_CUBIC)
cv2.imshow('output', output)
#cv2.resizeWindow('output', 400, 400)



'''
cv2.imshow('pattern', pattern)
cv2.imshow('floor1', floor1)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
