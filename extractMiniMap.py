import ffmpeg
import glob
import os
import sys


def decompMiniMap(ratio):
    print('Starting Video Decomposition Tool')
    fileName = ''
    # Grab the file name of the video
    for file in glob.glob("video/" + '/*.mp4'):
        fileName = os.path.basename(file)

    # grab only keyframes to ensure frame quality and use GPU acceleration
    stream = ffmpeg.input("video/"+fileName, skip_frame='nokey', vsync=0, hwaccel='cuda')
    if ratio == '4by3':
        # run the cropping procedure for 4by3 aspect ratio
        miniMap = ffmpeg.output(ffmpeg.crop(stream, 49, 37, 241, 181), 'input/miniMap%04d.png')
    if ratio == '16by9':
        # run the cropping procedure for 16by9 aspect ratio
        miniMap = ffmpeg.output(ffmpeg.crop(stream, 49, 44, 242, 218), 'input/miniMap%04d.png')
    if ratio == '16by10':
        # run the cropping procedure for 16by10 aspect ratio
        miniMap = ffmpeg.output(ffmpeg.crop(stream, 43, 43, 210, 212), 'input/miniMap%04d.png')

    ffmpeg.run_async(miniMap)
    sys.exit()


if __name__ == '__main__':
    print('Starting MiniMap Extraction Tool')
    args = sys.argv
    # debug args
    #args = ['extractMiniMap.py', '-ratio=4by3']
    # Check if the user has entered the correct number of arguments
    if len(args) == 1:
        print("Command format:")
        print("\textractMiniMap.py -ratio=RATIO")
        print("\t\t -ratio: Aspect ratio of the video, valid ratios are:")
        print("\t\t\t- '4by3'")
        print("\t\t\t- '16by9'")
        print("\t\t\t- '16by10'")
    else:
        args.pop(0)
        # Run the map packing tool
        decompMiniMap(args[0].split('=')[1])
