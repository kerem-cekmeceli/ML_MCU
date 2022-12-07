from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np


def search_all_files(dirpath):
    assert dirpath.is_dir()
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(str(x))
        elif x.is_dir():
            file_list.extend(search_all_files(x))
    return file_list


def get_file_paths_ordered(num_speaker, test_ratio):
    
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

    # Creating the train and test dataset splits along with the labels
    paths_chained = []
    y = []
    for i in range(num_speaker):
        path_i = files_for_speakers_sorted[i][:]
        paths_chained.extend(path_i)
        y.extend([i]*len(path_i))

    paths_train, paths_test, y_train_one_hot, y_test_one_hot = train_test_split(paths_chained, y, test_size=test_ratio)


    return paths_train, paths_test, y_train_one_hot, y_test_one_hot

def load_data(paths_train, paths_test, y_train_one_hot, y_test_one_hot):
    x_train_list = []
    y_train_list = y_train_one_hot

    x_test_list = []
    y_test_list = y_test_one_hot

    totalSliceLength = 10 # [sec] Length to stuff the signals to

    # Ratios of samples to use for traning and testing
    train_size_ratio = 0.5
    test_size_ratio = 0.6

    trainsize = int(train_size_ratio*len(paths_train)) # Number of loaded training samples
    testsize = int(test_size_ratio*len(paths_test)) # Number of loaded testing samples

    fs = 16000 # Sampling rate of the samples
    segmentLength = 1024 # Number of samples to use per segment
    sliceLength = int(totalSliceLength * fs / segmentLength)*segmentLength # Nb samples per given slice

    for i in tqdm(range(trainsize)): #TQDM gives you progress info for the for loop
        fs_i, train_sound_data = wavfile.read(paths_train[i]) # Read wavfile to extract amplitudes
        assert fs_i == fs

        _x_train = train_sound_data.copy() # Get a mutable copy of the wavfile
        _x_train.resize(sliceLength) # Zero stuff the single to a length of sliceLength
        _x_train = _x_train.reshape(-1, int(segmentLength)) # Split slice into Segments with 0 overlap
        
        x_train_list.append(_x_train.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain
        # y_train_list.append(traindata[i]['is_hotword']) # Read label 

    for i in tqdm(range(testsize)):
        fs, test_sound_data = wavfile.read(paths_test[i])
        _x_test = test_sound_data.copy()
        _x_test.resize(sliceLength)
        _x_test = _x_test.reshape((-1,int(segmentLength)))
        x_test_list.append(_x_test.astype(np.float32))
        # y_test_list.append(testdata[i]['is_hotword'])
        
        
    # Converting to tensor
    x_train = tf.convert_to_tensor(np.asarray(x_train_list))
    y_train = tf.convert_to_tensor(np.asarray(y_train_list))
    
    print(tf.shape(x_train))

    x_test = tf.convert_to_tensor(np.asarray(x_test_list))
    y_test = tf.convert_to_tensor(np.asarray(y_test_list))
    
    return x_train, y_train, x_test, y_test

