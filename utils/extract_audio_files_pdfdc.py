# extract audio from video
from pathlib import Path
from moviepy.editor import *
import json
from pdfdc_utils import extract_pdfdc_metadata, extract_pdfdc_segs_metadata
from multiprocessing.pool import ThreadPool
jsonpath = 'YOURPATH/pDFDC/dfdc_preview_set/dataset.json'
BASE_PATH = 'YOURPATH/pDFDC/dfdc_preview_set/'

def save_audio_file(video_path: Path):
    # print(str(video_path.absolute()))
    
    audio_path = video_path.parent / f'{video_path.stem}.wav'
    if audio_path.exists():
        return
    try:
        video = VideoFileClip(str(video_path.absolute()))
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    except Exception as e:
        print(e)

def make_pdfdc_audio():
    metadata = extract_pdfdc_metadata(jsonpath)
    total_number = 0
    for id_name, video_list in metadata.items():
        total_number += len(video_list)
    print(f'total {total_number}')
    # total_number = 5250
    finished_number = 0
    for id_name, video_list in metadata.items():
        finished_number += 1
        if finished_number%100==0:
            print(f'finished number {finished_number}')

        for video_path, video_label, aug in video_list:
            filepath = Path(BASE_PATH) / video_path
            save_audio_file(video_path = filepath)

def make_pdfdc_seg_audio():
    metadata = extract_pdfdc_segs_metadata()
    total_number = 0
    for id_name, video_list in metadata.items():
        total_number += len(video_list)
    print(f'total {total_number}')
    # total_number = 5250
    finished_number = 0
    with ThreadPool(processes=8) as pool:
        results = []
        for id_name, video_list in metadata.items():
            finished_number += 1
            if finished_number%10==0:
                print(f'finished number {finished_number}')

            for video_path, video_label, aug in video_list:
                segs_path = video_path
                for seg_file in segs_path.iterdir():
                    if 'crop' not in str(seg_file) and '.mp4' in str(seg_file):
                        save_audio_file(seg_file)
                        # result = pool.apply_async(save_audio_file, (seg_file,))
                        # results.append(result)
                # if len(results)>32:
                #     [res.get(timeout=600) for res in results]
                #     results = []

if __name__=='__main__':
    make_pdfdc_seg_audio()
