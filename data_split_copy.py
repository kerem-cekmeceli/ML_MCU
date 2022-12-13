import os
import shutil

BASE_DIR = "C:/Users/kerem/Documents/Coding/Python/ML_MCU" # os.getcwd()
RAW_DATA_DIR = "C:/Users/kerem/Documents/Coding/Python/ML_MCU/VoxCelebDataset/train_wav/" # Directory of all data files
DATA_DIR = "C:/Users/kerem/Documents/Coding/Python/ML_MCU/VoxCelebDataset/train_reduced" # Directory of extracted data files
print("Base directory : ", BASE_DIR)
print("Data directory : ", RAW_DATA_DIR)
print("New data directory : ", DATA_DIR)

SIZE_LIMIT = 70 # Number of speakers to extract


def extract_dataset():
    # get list of all speakers in the raw dataset
    id_list = os.listdir(RAW_DATA_DIR)

    # count number of files in directory
    file_count = dict(zip(id_list, [None]*len(id_list)))
    for i,id in enumerate(id_list):
        file_count[id] = sum(len(files) for _, _, files in os.walk(os.path.join(RAW_DATA_DIR,id_list[i])))

    # sort the list by number of files
    file_count = sorted(file_count.items(), key=lambda x:x[1], reverse=True)
    print(file_count[:SIZE_LIMIT])

    for i in range(SIZE_LIMIT):
        src = os.path.join(RAW_DATA_DIR, file_count[i][0])
        dst = os.path.join(DATA_DIR, file_count[i][0])
        # shutil.copytree(src, dst, symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=True)
        if not os.path.exists(dst): 
            os.makedirs(dst)

        youtube_keys = os.listdir(src)
        wav = 0
        
        # copy files from each youtube key to new data directory
        for key in youtube_keys:
            print("Key : ", key)
            src_key = os.path.join(src, key)
            for file in os.listdir(src_key):
                src_file = os.path.join(src_key, file)
                dst_file = os.path.join(dst, '{}'.format(str(wav)).zfill(4) + '.wav')
                shutil.copy(src_file, dst_file)
                wav += 1


# create a list of all training and testing files
def make_train_test_list_file():
    id_list = os.listdir(DATA_DIR)
    with open('data/train_list.txt', 'w') as train_file, open('data/test_list.txt', 'w') as test_file:
        for id in id_list:
            wav_list = os.listdir(os.path.join(DATA_DIR, id))
            test_size = int(len(wav_list)*0.2)
            for wav in wav_list[:-test_size]:
                train_file.write(id + ' ' + os.path.join(id, wav) + '\n')
            for wav in wav_list[-test_size:]:
                test_file.write(id + ' ' + os.path.join(id, wav) + '\n')

extract_dataset()