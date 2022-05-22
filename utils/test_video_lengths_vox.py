from vox_utils import extract_metadata
import imageio
import matplotlib.pyplot as plt
import pickle

def get_video_length(video_path):
    reader = imageio.get_reader(video_path)
    return (reader.get_meta_data()['duration'])

def extract_vox_audios():
    metadata = extract_metadata()
    finished_number = 0
    lengths = []
    for id_number, dic in metadata.items():
        for filename in dic:
            lengths.append(get_video_length(filename))

        finished_number += 1
        print(finished_number)
    # plot lengths
    with open('lengths_data.pkl', 'wb') as f:
        pickle.dump(lengths, f)
    plt.hist(lengths)
    plt.savefig('lengths_hist.jpg')

def plot_lengths():
    with open('lengths_data.pkl', 'rb') as f:
        lengths = pickle.load(f)
    plt.hist(lengths, range=(0,40),bins = 20)
    plt.title("Length Distribution of Voxceleb2 Test Set")
    plt.xlabel("length/s")
    plt.ylabel("number")
    plt.savefig('lengths_hist2.jpg')

if __name__=='__main__':
    extract_vox_audios()