# crop long videos to segments
# get video
# split video
# save and output

# https://stackoverflow.com/questions/67334379/cut-mp4-in-pieces-python
# or simple use subclip: https://www.section.io/engineering-education/video-editing-python-moviepy/
from moviepy.editor import *
from pathlib import Path
import os
from multiprocessing import Pool
import time
from vox_utils import extract_vox_dir
from pdfdc_utils import extract_pdfdc_dir, PDFDC_BASE_PATH
# Replace the filename below.
# test_video_path = '/mfs/lizhengyuan17-bishe/pDFDC/dfdc_preview_set/original_videos/643049/643049_A_001.mp4'

OUTPUT_BASE_PATH = Path('/mfs/lizhengyuan17-bishe/Voxceleb2/test_segs/mp4')
TMP_PATH = Path('/home/lizhengyuan17-bishe/tmp')
PDFDC_OUTPUT_BASE_PATH = Path('/mfs/lizhengyuan17-bishe/pDFDC/dfdc_preview_set_seg')
def cut_video(video_file_path: str, output_file_dir: Path):
    clip = VideoFileClip(video_file_path)
    duration = clip.duration
    # cut to 3 secs
    num_segs = int(duration/3)
    for seg_number in range(num_segs):
        # write to file
        seg_out_path = output_file_dir / f'{seg_number}.mp4'
        clip.subclip(seg_number*3, (seg_number+1)*3).write_videofile( \
        str(seg_out_path), verbose=False, logger=None, temp_audiofile = str(TMP_PATH/output_file_dir.name)+'.mp3')


def cut_segs_vox(metadata_dict, proc_index):
    # might be paralleled
    print(f"proc {proc_index} starts")
    finished_id = 0
    start = time.time()
    for id_child, video_dict in metadata_dict.items():
        # id_child_list: dict for one video
        id_child_dir = VOX_OUTPUT_BASE_PATH / id_child
        if not os.path.exists(str(id_child_dir)):
            os.mkdir(str(id_child_dir))
        for video_name, seg_list in video_dict.items():
            video_dir = id_child_dir / video_name
            if not os.path.exists(str(video_dir)):
                os.mkdir(str(video_dir))
            for seg in seg_list:
                input_video_path = str(seg)
                output_video_dir = video_dir / seg.stem
                if not os.path.exists(str(output_video_dir)):
                    os.mkdir(str(output_video_dir))
                cut_video(input_video_path, output_video_dir)
                
        finished_id += 1
        print(f"finished id {finished_id}/{len(list(metadata_dict.keys()))} in {proc_index}")
        print(f"time elapsed {time.time()-start}")

def cut_parallel_vox(vox_metadata):
    NUM_PROCESS = 10
    BLOCK_SIZE = int(len(list(vox_metadata.keys()))/NUM_PROCESS)+1
    print('start')
    with Pool(processes=NUM_PROCESS) as pool:
        res_list = []
        vox_metadata_list = list(vox_metadata.items())
        for i in range(NUM_PROCESS):
            block_start = BLOCK_SIZE*i
            block_end = min(BLOCK_SIZE*(i+1), len(list(vox_metadata.keys())))
            if block_end < block_start:
                continue
            print(f'start {block_start}, end {block_end}')
            metadata_split = dict(vox_metadata_list[block_start:block_end])
            res = pool.apply_async(cut_segs_vox,\
             (metadata_split,i))
            res_list.append(res)
        [res.wait(timeout=3600*20) for res in res_list]
        [print(res.successful()) for res in res_list]
    print('end')

def cut_simple_vox(vox_metadata):
    cut_segs_vox(vox_metadata, 0)

def cut_segs_pdfdc(pdfdc_metadata):
    finished_id = 0
    start = time.time()
    for id_child, clip_list in pdfdc_metadata.items():
        # id_child_list: dict for one video
        
        for clip_path in clip_list:
            relative_path = clip_path.parent.relative_to(PDFDC_BASE_PATH)
            clip_output_path = PDFDC_OUTPUT_BASE_PATH / relative_path / clip_path.stem
            # if exist continue
            if clip_output_path.exists():
                continue
            clip_output_path.mkdir(parents=True, exist_ok=False)
            
            cut_video(str(clip_path), clip_output_path)
                
        finished_id += 1
        print(f"finished id {finished_id}/{len(list(pdfdc_metadata.keys()))}")
        print(f"time elapsed {time.time()-start}")
    # {id: Paths}

if __name__=='__main__':
    # vox_metadata = extract_vox_dir()
    # cut_simple_vox(vox_metadata)
    
    # cut_parallel_vox(vox_metadata)
    pdfdc_metadata = extract_pdfdc_dir()
    cut_segs_pdfdc(pdfdc_metadata)


