import json
from pathlib import Path
JSONPATH = '/home/lizhengyuan17-bishe/pdfdc/dataset.json'
# PDFDC_BASE_PATH = Path('/mfs/lizhengyuan17-bishe/pDFDC/dfdc_preview_set/')
PDFDC_SEG_BASE_PATH = Path('/home/lizhengyuan17-bishe/pdfdc')

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

# to be tested
# def extract_pdfdc_dir():
#     with open(JSONPATH, 'r') as f:
#         jsondata = f.read()

#     basepath = PDFDC_BASE_PATH
#     metadata = json.loads(jsondata)
#     id_video_paths = {}
#     # {id: (video_path, label, augment:str)}
#     for video_path, video_info in metadata.items():
#         if video_info["label"] == "fake":
#             video_label = 0
#             target_id = video_info["swapped_id"]
#         else:
#             video_label = 1
#             target_id = video_info["target_id"]
#         # 
#         id_video_paths[target_id].append(basepath / video_path)

    print(f"num ids {len(list(id_video_paths.keys()))}")
    print(f"num clips {sum([len(ls) for ls in id_video_paths.values()])}")
    return id_video_paths

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