from pathlib import Path
BASE_PATH = Path('/mfs/lizhengyuan17-bishe/Voxceleb/official/test_videos/')
VOX_BASE_PATH = Path('/mfs/lizhengyuan17-bishe/Voxceleb2/test/mp4')
VOX_SEG_PATH = Path('/mfs/lizhengyuan17-bishe/Voxceleb2/test_segs/mp4')
# ok
def extract_metadata():
    # maintain a dict = {id: [video clips]}}
    metadata = {}
    path = BASE_PATH
    for child in path.iterdir():
        if child.is_dir():
            id_number = int(child.name[2:])
            metadata[id_number] = []
            #load videos
            for video_dirs in child.iterdir():
                for seg in video_dirs.glob("*.mp4"):
                    if 'crop' not in str(seg):
                        metadata[id_number].append(str(seg))
        else:
            print('why here?')
            print(child.name)
    total_id_number = len(list(metadata.keys()))
    print(f'total ids {total_id_number}')
    total_video_number = sum([len(ls) for ls in metadata.values()])
    print(f'total num {total_video_number}')
    return metadata

# FIXME: test after clip finished
def extract_seg_metadata():
    # maintain a dict = {id: [video clips]}}
    metadata = {}
    path = VOX_SEG_PATH
    for child in path.iterdir():
        if child.is_dir():
            id_number = int(child.name[2:])
            metadata[id_number] = []
            #load videos
            for video_dir in child.iterdir():
                for seg_dir in video_dir.iterdir():
                    for seg in seg_dir.glob("*.mp4"):
                        metadata[id_number].append(str(seg))
        else:
            print('why here?')
            print(child.name)
    total_id_number = len(list(metadata.keys()))
    print(f'total ids {total_id_number}')
    total_video_number = sum([len(ls) for ls in metadata.values()])
    print(f'total num {total_video_number}')
    return metadata
# ok
def extract_vox_dir():
    # extract dir structure
    # maintain a dict = {id_dir: {video_dir: [videos]}}
    metadata = {}
    path = VOX_BASE_PATH
    for id_child in path.iterdir():
        if id_child.is_dir():
            video_dict = {}
            for video_dirs in id_child.iterdir():
                video_dict[video_dirs.stem] = []
                for seg in video_dirs.glob("*.mp4"):
                    video_dict[video_dirs.stem].append(seg)

            metadata[id_child.stem] = video_dict
        else:
            print('why here?')
            print(id_child.name)

    total_id_number = len(list(metadata.keys()))
    print(f'total ids {total_id_number}')
    total_video_number = sum([ len(list(video_dicts.keys())) 
         for video_dicts in metadata.values()])
    print(f'total video num {total_video_number}')
    total_clip_number =  sum([ sum([len(clip_list) for clip_list in video_dict.values()])
         for video_dict in metadata.values()])
    print(f'total clip num {total_clip_number}')
    return metadata