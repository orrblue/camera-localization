'''
calib.py

Attempt at Camera Calibration using homography/matching
AKA, no checker board!

So far, no successful calibration. So far...


'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


pattern = cv2.imread('img/pattern.png')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

images = glob.glob('img/*.JPG')
pattern_pts = []
scene_pts = []
best_matches_size = 10

for img_name in images:
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    # detect features in plain QR code, then in provided images of QR code
    keypoints_pattern, descriptors_pattern = orb.detectAndCompute(pattern, None)    
    keypoints_scene, descriptors_scene = orb.detectAndCompute(gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match features and sort based on quality. Pick out n-th best matches.
    matches = bf.match(descriptors_pattern, descriptors_scene)
    matches = sorted(matches, key = lambda x: x.distance)
    best_matches = matches[ : best_matches_size]

    #add n-th best matches to lists of object pts, and image pts
    for i in range(best_matches_size):
        pattern_pts.append(keypoints_pattern[ best_matches[i].queryIdx ].pt)
        scene_pts.append(keypoints_scene[ best_matches[i].trainIdx ].pt)



# reshape to # num_elms x best_matches_size x 2
pattern_pts = np.float32(pattern_pts).reshape(-1, best_matches_size, 2)
scene_pts = np.float32(scene_pts).reshape(-1, best_matches_size, 2)

# insert 0 as Z-coordinate since object is a plane 
rows = pattern_pts.shape[0]
pattern_pts = np.insert(pattern_pts, 2, np.zeros((rows, 1)), 2)



# Source cam_mat for calibCamera() to use as estimate. Focal length in pixels
camera_matrix_source = np.array([
    [2768, 0., 1224.], 
    [0., 2770, 1632.], 
    [0., 0., 1.]])

# camera matrix, distortion coefficients, rotation vector, translation 
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(pattern_pts, scene_pts, 
    (2448, 3264), cameraMatrix = camera_matrix_source, distCoeffs = None, rvecs = None, 
    tvecs = None, flags=1)

print()
print(rms)
print()
print(camera_matrix)
print()
print(dist_coefs)


# showing undistorted image
match1 = cv2.imread('img/IMG_6725.JPG', 0)
undist1 = cv2.undistort(match1, camera_matrix, dist_coefs, None)
output = cv2.resize(undist1, None, fx=0.3, fy=0.3,interpolation = cv2.INTER_CUBIC)
cv2.imshow('output', output)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''

=== TRIALS ===

best_matches_size = 10
RMS = 54.50464398008055

Camera Matrix =
[[  1.50512671e+02   0.00000000e+00   1.08574478e+03]
 [  0.00000000e+00   9.14708563e+01   1.66977834e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

Distortion coefficients =
[[ 0.59014476 -0.0191317   0.09039141 -0.31884398 -0.00223853]]



12
66.53174503437471

[[  1.83370933e+02   0.00000000e+00   1.16538885e+03]
 [  0.00000000e+00   1.34968339e+02   1.63702049e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

[[-0.05248756  0.00591458 -0.01091695 -0.04493884 -0.00028817]]



10 + flags = 1 'CV_CALIB_USE_INTRINSIC_GUESS'
78.64751038656593
source focal length was incorrectly in mm --> 4.15mm

[[  4.15000000e+00   0.00000000e+00   1.22400000e+03]
 [  0.00000000e+00   4.15000000e+00   1.63200000e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

[[  4.61762188e-13   1.37063213e-09  -1.37654984e-14  -8.85760712e-15
   -2.11943788e-13]]



10 + flags = 1 (correct focal length this time)
57.00925747352173

[[  3.19058750e+03   0.00000000e+00   1.33308949e+03]
 [  0.00000000e+00   1.18929582e+03   1.68754358e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

[[  1.27001459e+01   4.19172651e+02  -5.27023604e-01   1.63341525e+00
   -7.79681554e+03]]



12 + flags = 2
67.58625900752637

[[  1.64211532e+02   0.00000000e+00   1.13830361e+03]
 [  0.00000000e+00   1.64330181e+02   1.63379828e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

[[-0.08502798  0.01974622 -0.02661665 -0.03478743 -0.0013597 ]]



10 + flags = 2
54.969109577357244

[[  1.89269215e+01   0.00000000e+00   1.11073203e+03]
 [  0.00000000e+00   1.89405970e+01   1.67296973e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

[[  1.67129267e+00  -1.14127805e-01  -2.21808387e-01   2.40263807e-02
   -1.61877775e-03]]



10 + flags = 256 'fix intrinsics'
54.50464398008055

[[  1.50512671e+02   0.00000000e+00   1.08574478e+03]
 [  0.00000000e+00   9.14708563e+01   1.66977834e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

[[ 0.59014476 -0.0191317   0.09039141 -0.31884398 -0.00223853]]

'''



















