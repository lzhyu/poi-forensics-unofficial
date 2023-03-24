import json
from pathlib import Path
JSONPATH = 'YOURPATH/pdfdc/dataset.json'
PDFDC_SEG_BASE_PATH = Path('YOURPATH/pdfdc')

def extract_pdfdc_metadata():
    with open(JSONPATH, 'r') as f:
        jsondata = f.read()

    metadata = json.loads(jsondata)
    id_dict = {}
    # {id: (video_path, label, augment:str)}
    for video_path, video_info in metadata.items():
        if video_info["label"] == "fake":
            video_label = 0
            target_id = video_info["swapped_id"]
        else:
            video_label = 1
            target_id = video_info["target_id"]

        aug = None
        if len(video_info['augmentations'])>0:
            assert(len(video_info['augmentations'])==1)
            aug = video_info['augmentations'][0]
        if target_id not in id_dict.keys():
            id_dict[target_id] = []
        id_dict[target_id].append((video_path, video_label, aug))

    return id_dict


def extract_pdfdc_segs_metadata():
    with open(JSONPATH, 'r') as f:
        jsondata = f.read()

    metadata = json.loads(jsondata)
    id_dict = {}
    # {id: (video_path, label, augment:str)}
    for video_path, video_info in metadata.items():
        if video_info["label"] == "fake":
            video_label = 0
            target_id = video_info["swapped_id"]
        else:
            video_label = 1
            target_id = video_info["target_id"]
        
        aug = None
        if len(video_info['augmentations'])>0:
            assert(len(video_info['augmentations'])==1)
            aug = video_info['augmentations'][0]
        if target_id not in id_dict.keys():
            id_dict[target_id] = []
        video_segs_path = (PDFDC_SEG_BASE_PATH / video_path).parent / Path(video_path).stem
        id_dict[target_id].append((video_segs_path, video_label, aug))

    return id_dict
    
extract_pdfdc_segs_metadata()