import numpy as np
import cv2
import matplotlib.pyplot as plt


pattern = cv2.imread('img/pattern.png',0)
match1 = cv2.imread('img/IMG_6725.JPG', 0)
dimensions = match1.shape
print(type(dimensions))


orb = cv2.ORB_create()

keypoints_pattern, descriptors_pattern = orb.detectAndCompute(pattern, None)
keypoints_scene1, descriptors_match1 = orb.detectAndCompute(match1, None)

# print(help(keypoints_pattern[0]))
# print(help(keypoints_pattern[0].pt))
# print(keypoints_pattern)
# print(keypoints_pattern[0].pt[0])
# print(keypoints_pattern[0].pt[1])


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors_pattern, descriptors_match1)
matches = sorted(matches, key = lambda x:x.distance)
best_matches = matches[:15]
pattern_pts = []
scene_pts = []

# print('Keypoints_scene1:', keypoints_scene1[386])
# print('Keypoints_pattern:', keypoints_pattern[248])
# print('Keypoints_pattern:', keypoints_pattern[45])

for i in range(len(best_matches)):
    # print(i, best_matches[i])
    # print(i, best_matches[i].queryIdx, keypoints_pattern[ best_matches[i].queryIdx ])
    # print(i, best_matches[i].trainIdx, keypoints_scene1[ best_matches[i].trainIdx ])
    # print()

    pattern_pts.append(keypoints_pattern[ best_matches[i].queryIdx ].pt)
    scene_pts.append(keypoints_scene1[ best_matches[i].trainIdx ].pt) 

pattern_pts = np.float32(pattern_pts).reshape(-1, 1, 2) # num_elms x 1 x 2
scene_pts = np.float32(scene_pts).reshape(-1, 1, 2)

# print(pattern_pts[:3:])
print(pattern_pts.shape)
rows = pattern_pts.shape[0]
pattern_pts = np.insert(pattern_pts, 2, np.zeros((rows, 1)), 2)
# print('****')
# print(pattern_pts[:3:]) # i need to add 0s to inner vector
print(pattern_pts.shape)
# #print(scene_pts)



H, mask = cv2.findHomography(pattern_pts, scene_pts)
im_dst = cv2.warpPerspective(match1, H, (2448, 3264))
# cv2.imshow('m1', im_dst)

cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)

output = cv2.resize(im_dst, None, fx=0.4, fy=0.4,interpolation = cv2.INTER_CUBIC)
cv2.imshow('output', output)
cv2.resizeWindow('output', 400, 400)

H = np.insert(H, 2, 0, axis = 1)
print(H)

# print(scene_pts)




#print(cv2.decomposeHomographyMat(H))
#a,b,c,d,e,f = cv2.decomposeProjectionMatrix(H)


# print(type(matches[0]))
# print(help(matches[0]))
# print(matches[0].distance)
# print(matches[0].imgIdx)
# print(matches[0].queryIdx)
# print(matches[0].trainIdx)

# output = cv2.drawMatches(pattern, keypoints_pattern, match1, keypoints_scene1
# , matches[:5], None, flags=2)
# cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)

# output = cv2.resize(output, None, fx=0.4, fy=0.4,interpolation = cv2.INTER_CUBIC)
#cv2.imshow('output', output)
#cv2.resizeWindow('output', 400, 400)



'''
cv2.imshow('pattern', pattern)
cv2.imshow('floor1', floor1)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
