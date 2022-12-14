from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from statistics import median

def search_all_files(dirpath):
    assert dirpath.is_dir()
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(str(x))
        elif x.is_dir():
            file_list.extend(search_all_files(x))
    return file_list


def get_file_paths_ordered(num_speaker, test_ratio, balanced_dataset, plot_data):
    
    path_proj_root =  Path('..')
    path_dataset = path_proj_root / 'VoxCelebDataset' 
    path_train_set_orig = path_dataset / 'train_wav'
    # path_train_set_reduced = path_dataset / 'train_wav_reduced'

    # if not path_train_set_reduced.is_dir():
    #     print("New directory is created as:", path_train_set_reduced)
    #     path_train_set_reduced.mkdir()

    nb_speakers = len([x for x in path_train_set_orig.iterdir() if x.is_dir()])
    print("Total number of speakers in the original dataset: " + str(nb_speakers))

    # Get the files for all speakers in the dataset
    files_for_speakers = []
    nb_files_per_speaker = []
    files_for_speakers_sorted = []
    for speaker_dir_i in path_train_set_orig.iterdir():
        assert speaker_dir_i.is_dir()
        files_speaker_i = search_all_files(speaker_dir_i)
        files_for_speakers.append(files_speaker_i)
        nb_files_per_speaker.append(len(files_speaker_i))

    # Sorting the speakers according to the number of samples (-1 for reverse order)
    # nb_files_per_speaker_sorted = sorted(nb_files_per_speaker, reverse=True)
    sort_idx = sorted(range(len(nb_files_per_speaker)), key=lambda k: nb_files_per_speaker[k], reverse=True)
    files_for_speakers_sorted = [files_for_speakers[i] for i in sort_idx]
    nb_files_per_speaker_sorted = [nb_files_per_speaker[i] for i in sort_idx]

    # Remove unused speakers
    files_for_speakers_sorted = files_for_speakers_sorted[:num_speaker]
    nb_files_per_speaker_sorted = nb_files_per_speaker_sorted[:num_speaker]

    print("Original number of files per speaker : ", nb_files_per_speaker_sorted)

    if balanced_dataset:
        # Max number of files per speaker is limited to the median of the considered batch
        max_files_per_speaker = int(median(nb_files_per_speaker_sorted)) 
        nb_files_per_speaker_sorted_balanced = nb_files_per_speaker_sorted[:]

    # Creating the train and test dataset splits along with the labels
    paths_chained = []
    y = []
    for i in range(num_speaker):
        if balanced_dataset and nb_files_per_speaker_sorted[i] > max_files_per_speaker:
            nb_files_per_speaker_sorted_balanced[i] = max_files_per_speaker
            paths_i = files_for_speakers_sorted[i][:max_files_per_speaker]
        else:
            paths_i = files_for_speakers_sorted[i][:]

        # Save the paths
        paths_chained.extend(paths_i)
        # Save the Labels
        y.extend([i]*len(paths_i))

    paths_train, paths_test, y_train_l, y_test_l = train_test_split(paths_chained, y, test_size=test_ratio)
    print("Training number of files per speaker : ", len(paths_train))

    if plot_data:
        fig = plt.figure(figsize=(10,9))
        plt.bar(np.arange(0, num_speaker, 1), nb_files_per_speaker_sorted, label='OriginalDataset', width=1, edgecolor='k')
        if balanced_dataset:
            plt.bar(np.arange(0, num_speaker, 1), nb_files_per_speaker_sorted_balanced, label='BalancedDataset', width=1, edgecolor='k')
        
        plt.legend()
        plt.grid()
        plt.xlabel('Speakers')
        plt.ylabel('Number of Files')
        plt.show()


    return paths_train, paths_test, y_train_l, y_test_l, paths_chained
