import numpy as np
import random
import os
import glob
import soundfile
import tensorflow as tf
from scipy import signal


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length : 32240
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor : sample_rate = 16000
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats,axis=0).astype(float)

    return feat
    

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


# TO DO : Make a tf.keras.utils.Sequence ?
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c 

class train_dataset_loader():
    def __init__(self, train_list="data/train_list.txt", augment=False, musan_path="data/musan_split", \
                 rir_path="data/RIRS_NOISES/simulated_rirs", max_frames=200, train_path="data", **kwargs):

        """
        Args:
            train_list:     type=str,   default="data/train_list.txt",  help='Train list')
            augment:        type=bool,  default=False,  help='Augment input')
            musan_path':    type=str,   default="data/musan_split", help='Absolute path to the test set')
            rir_path:       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')
            max_frames:     type=int,   default=200,    help='Input length to the network for training')
            train_path:     type=str,   default="data/voxceleb2", help='Absolute path to the train set')

        """

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.train_list = train_list
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment
        
        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        self.speaker_list = { key : ii for ii, key in enumerate(dictkeys) }
        print("Speakers : ", self.speaker_list)

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = self.speaker_list[data[0]]
            filename = os.path.join(train_path,data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):

        feat = []
        for index in range(len(indices)):

            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            
            if self.augment:
                augtype = random.randint(0,4)
                if augtype == 1:
                    audio   = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio   = self.augment_wav.additive_noise('music',audio)
                elif augtype == 3:
                    audio   = self.augment_wav.additive_noise('speech',audio)
                elif augtype == 4:
                    audio   = self.augment_wav.additive_noise('noise',audio)
                    
            feat.append(audio)

        feat = np.concatenate(feat, axis=0)
        return feat

    def get_loader(self):
        x_train = self.__getitem__(self.data_label)
        return tf.convert_to_tensor(x_train), tf.convert_to_tensor((self.data_label)) 
        
    def __len__(self):
        return len(self.data_list)


class test_dataset_loader():
    def __init__(self, test_list='data/test_list.txt', test_path="data", eval_frames=200, num_eval=10, **kwargs):
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

        # Read testing files
        with open(test_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        self.speaker_list = { key : ii for ii, key in enumerate(dictkeys) }
        print(self.speaker_list)

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = self.speaker_list[data[0]]
            filename = os.path.join(test_path,data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):
        # audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval) 
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False, num_eval=self.num_eval)
        return audio

    def __len__(self):
        return len(self.test_list)

    def get_loader(self):
        x_test = []
        for i in range(len(self.data_list)):
            x_test.append(self.__getitem__(i))
        x_test = np.concatenate(x_test, axis=0)
        return tf.convert_to_tensor(x_test), tf.convert_to_tensor((self.data_label)) 