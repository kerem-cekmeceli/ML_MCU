import os
import shutil

BASE_DIR = os.getcwd()
# Directory of all data files
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
# Directory of extracted data files
NEW_DATA_DIR = os.path.join(BASE_DIR, 'data') 
SIZE_LIMIT = 20
print("Base directory : ", BASE_DIR)
print("Data directory : ", RAW_DATA_DIR)
print("New data directory : ", NEW_DATA_DIR)
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
    dst = os.path.join(NEW_DATA_DIR, file_count[i][0])
    shutil.copytree(src, dst, symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=True)