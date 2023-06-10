import numpy as np
import cv2 as cv
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import threading
import math
import os
import sys


def packKeyPoints(ratio, gameMap):
    # Check if the user has entered a valid map name
    if gameMap not in ['KC', 'WE', 'OLY', 'SP', 'BM']:
        print('Invalid map name, please enter a valid map name')
        exit()
    # Check if the user has entered a valid aspect ratio
    if ratio not in ['4by3', '16by9', '16by10']:
        print('Invalid aspect ratio, please enter a valid aspect ratio')
        exit()
    print('Packing Keypoints for ' + gameMap + ' ' + ratio + ' map')
    editedImage = cv2.imread('maps/'+ratio+'/map'+gameMap+ratio+'.jpg')
    # More allowed features does not seem to improve the accuracy of the matching, but does greatly increase the time taken to process the image
    # The same is true for the number of layers
    featureMappingAlg = cv.SIFT_create(nOctaveLayers=25, nfeatures=250000)
    # Compute the keypoints and descriptors for the image
    kp1, des1 = featureMappingAlg.detectAndCompute(editedImage, None)
    print('Saving')
    # Convert the keypoints to a numpy array for easier storage
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                    kp.angle, kp.response, kp.octave,
                    kp.class_id]
                    for kp in kp1])
    desc = np.array(des1)
    print("Number of keypoints: " + str(len(kp1)))
    # Uncomment to save an image with the keypoints drawn on it
    # plt.imsave('VerboseKeypoints.jpg', cv2.cvtColor(cv2.drawKeypoints(editedImage, kp1,
    #                                                                 editedImage, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), cv2.COLOR_BGR2RGB))
    # Save the keypoints and descriptors to a file
    np.save('packedKeypoints/'+ratio+'/'+gameMap+ratio+'KeyPoints.npy', np.hstack((kpts, desc)))


if __name__ == '__main__':
    print('Starting MiniMap Packing Tool')
    args = sys.argv
    #args = ['miniMapPlotting.py', '-mapName=WE', '-ratio=4by3']
    # Check if the user has entered the correct number of arguments
    if len(args) == 1:
        print("Command format:")
        print("\mapDataPacker.py -mapName=MAPNAME -ratio=RATIO")
        print("\t\t-mapName: Name of the map to be used, valid names are:")
        print("\t\t\t- 'KC'")
        print("\t\t\t- 'WE'")
        print("\t\t\t- 'OLY'")
        print("\t\t\t- 'SP'")
        print("\t\t\t- 'BM'")
        print("\t\t -ratio (optional): Aspect ratio of the map, valid ratios are:")
        print("\t\t\t- '4by3'")
        print("\t\t\t- '16by9'")
        print("\t\t\t- '16by10'")
    else:
        args.pop(0)
        # Run the map packing tool
        miniMapMatching = packKeyPoints(args[1].split('=')[1], args[0].split('=')[1])
