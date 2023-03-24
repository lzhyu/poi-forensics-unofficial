# no need to change audio, just change frame is enough
# no need to store bbox 
import imageio
import face_alignment
import numpy as np
import cv2
import json
import time 
import os
from pathlib import Path
from multiprocessing import Pool
from pdfdc_utils import extract_pdfdc_metadata
# test_video_path = '/mfs/lizhengyuan17-bishe/pDFDC/dfdc_preview_set/method_B/1260311/1260311_C/1255229_1260311_C_001.mp4'

jsonpath = 'YOURPATH/pDFDC/dfdc_preview_set/dataset.json'
BASE_PATH = 'YOURPATH/pDFDC/dfdc_preview_set/'


def crop_video(video_path, fa):
    # NOTE: not need to load audio here, we can deal with it seperately
    # NOTE: no need to cut to segments here
    # too slow
    # load video
    reader = imageio.get_reader(video_path)
    fps = int(reader.get_meta_data()['fps'])
    # print(reader.get_meta_data())
    # https://stackoverflow.com/questions/47775083/python-imageio-mp4-video-from-a-set-of-png-images
    output_video_path = video_path[:-4] + '_crop' + '.mp4'
    if (os.path.exists(output_video_path)):
        os.remove(output_video_path)

    # warm up?
    writer = imageio.get_writer(output_video_path, fps=fps)
    # start = time.time()
    max_frame = 0
    for i, frame in enumerate(reader):
        max_frame = i
        if i%30==0:
            bbox = fa.face_detector.detect_from_image(frame[..., ::-1])[0]
            if i==0:
                start_after_first = time.time()
            x_bound_min = max(int(bbox[1])-50, 0)
            x_bound_max = min(int(bbox[3])+50, frame.shape[0])
            y_bound_min = max(int(bbox[0])-50, 0)
            y_bound_max = min(int(bbox[2])+50, frame.shape[1])
        # crop a batch might be faster
        # for frames, estmate boxes
        # bachify
        
        # This function detects the faces present in a provided BGR(usually) image.
        # [A list of bounding boxes (x1, y1, x2, y2)]

        # crop frame
        frame = frame[x_bound_min:x_bound_max, y_bound_min: y_bound_max]
        # resize
        # https://www.cnblogs.com/lfri/p/10596530.html
        frame = cv2.resize(frame, (224,224))
        writer.append_data(frame)
        # visualize cropped frame
    # save to new video
    writer.close()
    # print(f'toltal time for {int(max_frame/fps)}s is {time.time()-start}')
    # print(f'but real time is {time.time()-start_after_first}')
    # cpu:  5000videos X 60s /3600 = ? hours
    # 10 process, 10h
    # gpu: takes ~ 150s to do the first detection 
    # 5000 videos X 15s /3600 ~= 20h 
    # 2400M GPU memory
    # we can have 3 processes in one GPU

def crop_video_list(video_path_list:list, proc_index:int):
    # 主要是每个进程需要有一个fa
    print(f'process {proc_index} initialized')
    print(f'video number is {len(video_path_list)}')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,\
        device=f'cuda:{proc_index//3}',flip_input=False)
    finished_number = 0
    start_time = time.time()
    print(f'fa initialized')
    for video_path in video_path_list:
        crop_video(video_path, fa)
        finished_number +=1
        if finished_number%100==1:
            print(f'in process {proc_index}, {finished_number} finished, {time.time()-start_time} elapsed')
        

if __name__=='__main__':
    
    metadata = extract_pdfdc_metadata(jsonpath)
    # crop_video(test_video_path)
    
    finished_number = 0 # total
    start = time.time()

    # collect video paths
    video_paths = []
    for id_name, video_list in metadata.items():
        for video_path, video_label, aug in video_list:
            filepath = Path(BASE_PATH) / video_path
            video_paths.append(str(filepath.absolute()))

    total_video_number = len(video_paths)
    print(f'total number: {total_video_number}')
    # distribute videos
    NUM_PROCESS = 3
    BLOCK_SIZE = total_video_number // NUM_PROCESS
    print('start')
    with Pool(processes=NUM_PROCESS) as pool:
        res_list = []
        for i in range(NUM_PROCESS):
            res = pool.apply_async(crop_video_list,\
             (video_paths[BLOCK_SIZE*i:min(BLOCK_SIZE*(i+1), total_video_number)],i))
            res_list.append(res)
        [res.wait(timeout=3600*20) for res in res_list]
        [print(res.successful()) for res in res_list]

    print('end')