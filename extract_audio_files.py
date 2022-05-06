# extract audio from video
from pathlib import Path
from moviepy.editor import *
BASE_PATH = '/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/'
def extract_metadata():
    # maintain a dict = {id: {video:[clips]}}
    metadata = {}
    path = Path(BASE_PATH)
    for child in path.iterdir():
        if child.is_dir():
            id_number = int(child.name[2:])
            metadata[id_number] = []
            #load videos
            for video_dirs in child.iterdir():
                for seg in video_dirs.glob("*.mp4"):
                    metadata[id_number].append(str(seg.absolute()))
        else:
            print('why here?')
            print(child.name)
    return metadata

def save_audio_file(video_path: Path):
    print(str(video_path.absolute()))
    video = VideoFileClip(str(video_path.absolute()))
    
    audio_path = video_path.parent / f'{video_path.stem}.wav'
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

metadata = extract_metadata()
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

