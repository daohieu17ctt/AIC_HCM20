from sort_ver1 import *
import os
import pickle
import random
import cv2
from deep_sort import DeepSort
from deepsort_util import COLORS_10, draw_bboxes

bbox_path = "../bbox/"
video_path = "../videos/"
output = "../test/"
files = [f for f in os.listdir(bbox_path) if os.path.isfile(os.path.join(bbox_path, f))]
class_path = 'config/coco.names'
classes = ['class_1', 'class_2', 'class_3', 'class_4']

print(files)

def format_bbox(video_name, file_name):
    ''' prepare formatted bbox for tracking'''

    file_content = open(os.path.join(bbox_path,file_name),'rb')
    content = pickle.load(file_content)
    print("Processing:",video_name)
    data = []

    for fr_id, fr_content in enumerate(content):
        dets = []
        c0_bboxes = fr_content[0]
        c1_bboxes = fr_content[1]
        c2_bboxes = fr_content[2]
        c3_bboxes = fr_content[3]

        for bb in c0_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 0})
        for bb in c1_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 1})
        for bb in c2_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 2})
        for bb in c3_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 3})
        
        data.append(dets)
    file_content.close()
    return data


cam_video = cv2.VideoCapture(os.path.join(video_path, "cam_14.mp4"))
cam_bbox = format_bbox("cam_14", files[0])
tracker = Sort()
output_video = cv2.VideoWriter(os.path.join(output, "cam_14_result.mp4"), cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, (int(cam_video.get(3)), int(cam_video.get(4))))
# ret, frame_0 = cam_14_video.read()

track_dict = {}
# info_tracking = []
for frame_id, frame_data in enumerate(cam_bbox):
    ret, frame_image = cam_video.read()
    dets = []
    for o in frame_data:
        bbox = o["bbox"]
        dets.append((frame_id, o["score"], bbox[0], bbox[1], bbox[2], bbox[3], o["class"]))

    
    dets = np.array(dets)  
    dets = dets[dets[:, 6]==0]
    track_bb_ids = tracker.update(dets)

    for xmin, ymin, xmax, ymax, track_id in track_bb_ids:
        track_id = int(track_id)
        # class_id = int(class_id)
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        # info_tracking.append([class_id, frame_id, 0.00, track_id, xmin, ymin, xmax, ymax])
        # Visualize
        # cv2.rectangle(frame_image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
            
        # cv2.putText(frame_image, (classes[class_id] + str(track_id)).zfill(5), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        if track_id not in track_dict.keys():
            track_dict[track_id] = [(xmin, ymin, xmax, ymax, frame_id)]
        else:
            track_dict[track_id].append((xmin, ymin, xmax, ymax, frame_id))

    # output_video.write(frame_image)

print(track_dict)
# print(info_tracking[0])    
cam_video.release()
output_video.release()
# np.save(os.path.join(output, "info_cam_14.npy"), info_tracking)
