import argparse
import cv2
import dlib
import os
import urllib.request as urllib2
import hashlib

from multiprocessing import Pool

import openface
from openface.helper import mkdirP


modelDir = os.path.join('/Users/tcheng/openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

dlibFacePredictor_file = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(dlibFacePredictor)

jobs = []

list_name_dirs = os.listdir('raw')
list_next_dirs = list_name_dirs[1:]

for person, next_person in zip(list_name_dirs[:-1], list_next_dirs):
    fullPersonPath = os.path.join('raw', person)
    list_images = os.listdir(fullPersonPath)
    extract_flag = False
    count = 0
    with open('short.txt') as f2:
        for line in f2:
            if line[:-4] == person:
                print(f'Extracting images of {person[:-1]}')
                extract_flag = True
                continue 
            if line[:-4] == next_person:
                break
            if extract_flag:
                if len(list_images)==0:
                    print(f"{person[:-1]} has {count} images")
                    break
                count += 1
                img = list_images.pop(0)
                img_path = os.path.join(fullPersonPath, img)
                uid, url, l, t, r, b, pose, detection, curation = line.split()
                l, t, r, b = [int(float(x)) for x in [l, t, r, b]]
                # if int(curation) == 1:
                jobs.append((img_path, person[:-1], url, (l, t, r, b)))

last_person = list_name_dirs[-1]
fullPersonPath = os.path.join('raw', last_person)
list_images = os.listdir(fullPersonPath)
extract_flag = False
count = 0
with open('short.txt') as f2:
    for line in f2:
        if line[:-4] == person:
            print(f'Extracting images of {last_person[:-1]}')
            extract_flag = True
            continue 
        if extract_flag:
            if len(list_images)==0:
                print(f"{last_person[:-1]} has {count} images")
                break
            count += 1
            img = list_images.pop(0)
            img_path = os.path.join(fullPersonPath, img)
            uid, url, l, t, r, b, pose, detection, curation = line.split()
            l, t, r, b = [int(float(x)) for x in [l, t, r, b]]
            # if int(curation) == 1:
            jobs.append((img_path, last_person[:-1], url, (l, t, r, b)))

def align_image(img_path, person, url, bb):
    aligned_person_path = os.path.join('aligned', person)
    img_name = os.path.basename(img_path)
    aligned_img_path = os.path.join(aligned_person_path, img_name)
    
    mkdirP(aligned_person_path)
    
    if not os.path.isfile(aligned_img_path):
        bgr = cv2.imread(img_path)
        
        if bgr is None:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        dlibBB = dlib.rectangle(*bb)
        outRgb = align.align(48, rgb,
                             bb=dlibBB,
                             landmarkIndices=landmarkIndices)

        if outRgb is not None:
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aligned_img_path, outBgr)

def process_packed(args):
    try:
        align_image(*args)
    except Exception as e:
        print("\n".join((str(args), str(e))))
        pass

pool = Pool(16)
pool.map(process_packed, jobs)