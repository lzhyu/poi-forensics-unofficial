# extract audio from video
from moviepy.editor import *
from vox_utils import extract_metadata, extract_seg_metadata
from pathlib import Path
def save_audio_file(video_path: Path):

    audio_path = video_path.parent / f'{video_path.stem}.wav'
    if audio_path.exists():
        return
    try:
        video = VideoFileClip(str(video_path.absolute()))
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    except Exception as e:
        print(e)

def extract_vox_audios():
    metadata = extract_seg_metadata()
    total_number = len(list(metadata.keys()))
    print(f'total {total_number}')
    finished_number = 0
    for id_number, dic in metadata.items():
        print(f'id_number: {id_number}\n')
        finished_number += 1
        if finished_number%10==0:
            print(f'finished number {finished_number}')
        for filename in dic:
            save_audio_file(video_path = Path(filename))

extract_vox_audios()