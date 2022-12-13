import tensorflow as tf
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
from statistics import median
import matplotlib.pyplot as plt


def choose_tot_slice_len(paths, visualize):
    duration = []
    nb_samps = len(paths)
    print("Visualization of the samples ...")
    for i in tqdm(range(nb_samps)):
        fs_i, sound_data_i = wavfile.read(paths[i])
        dur_i = sound_data_i.size  # Get nb elems of thh numpy array
        dur_i /= fs_i 
        duration.append(dur_i)
    
    slice_len = round(median(duration) * 1.6)

    if visualize:
        fig = plt.figure(figsize=(10,9))
        plt.title("Sample Lengths")
        plt.xlabel("Duration [sec]")
        plt.ylabel("Samples")
        plt.barh(np.arange(0, nb_samps, 1), duration, height=0.9, label='samples')
        plt.grid()
        plt.axvline(x=slice_len, ymin=0, ymax=nb_samps, color='r', label='slice_length')
        plt.xlim((0, 35))
        plt.xticks(np.append(plt.xticks()[0], slice_len))
        plt.legend()
        plt.show()

    return slice_len


def get_data_tensors(paths_train, paths_test, y_train_l, y_test_l, tot_slice_len, used_train_sz_rat, used_test_sz_rat):
    x_train_sub_l = []
    x_test_sub_l = []

    # tot_slice_len [sec] Length to stuff the signals to  (If longer cuts, if shorter repeats)

    trainsize = int(used_train_sz_rat*len(paths_train)) # Number of loaded training samples
    testsize = int(used_test_sz_rat*len(paths_test)) # Number of loaded testing samples

    y_train_sub_l = y_train_l[:trainsize]
    y_test_sub_l = y_test_l[:testsize]

    fs, _ = wavfile.read(paths_train[0])  # fs = 16000 # Sampling rate of the samples
    segmentLength = 1024 # Number of samples to use per segment
    sliceLength = int(tot_slice_len * fs / segmentLength)*segmentLength # Nb samples per given slice

    for i in tqdm(range(trainsize)): #TQDM gives you progress info for the for loop
        fs_i, train_sound_data = wavfile.read(paths_train[i]) # Read wavfile to extract amplitudes
        assert fs_i == fs

        _x_train = train_sound_data.copy() # Get a mutable copy of the wavfile
        _x_train = np.resize(_x_train, sliceLength) # Repeat or cut if necessary to a length of sliceLength [sec]
        _x_train = _x_train.reshape(-1, int(segmentLength)) # Split slice into Segments with 0 overlap
        
        x_train_sub_l.append(_x_train.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain

    for i in tqdm(range(testsize)):
        fs_i, test_sound_data = wavfile.read(paths_test[i])
        assert fs_i == fs

        _x_test = test_sound_data.copy()
        _x_test = np.resize(_x_test, sliceLength)
        _x_test = _x_test.reshape((-1,int(segmentLength)))
        x_test_sub_l.append(_x_test.astype(np.float32))
        
    # Converting to tensor - data
    x_train = tf.convert_to_tensor(np.asarray(x_train_sub_l))
    x_test = tf.convert_to_tensor(np.asarray(x_test_sub_l))

    y_train = tf.convert_to_tensor(np.asarray(y_train_sub_l))
    y_test = tf.convert_to_tensor(np.asarray(y_test_sub_l))
    
    return x_train, y_train, x_test, y_test

def compute_mfccs(tensor):
    sample_rate = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13

    stfts = tf.signal.stft(tensor, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    # Short time fourier transofrm
    spectrograms = tf.abs(stfts) # [frame_len, fft_len//2+1]
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1)) #flatten last dim
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)  # Vector multi
    # The matrix can be used with tf.tensordot to convert an arbitrary rank Tensor of linear-scale spectral bins into the mel scale.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    
    return tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))

