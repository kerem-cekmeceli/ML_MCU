from data_paths import get_file_paths_ordered
from pre_process import *
from models import *

paths_train, paths_test, y_train_l, y_test_l, all_paths_l = get_file_paths_ordered(num_speaker=8, test_ratio=0.2,
                                                                                   balanced_dataset=True, plot_data=True)

seg_len = choose_tot_slice_len(paths=all_paths_l, visualize=True)

x_train, y_train, x_test, y_test = get_data_tensors(paths_train=paths_train, paths_test=paths_test, 
                                                    y_train_l=y_train_l, y_test_l=y_test_l,
                                                    tot_slice_len = seg_len,
                                                    used_train_sz_rat=0.45, used_test_sz_rat=0.45)

x_train_mfcc = compute_mfccs(x_train)
x_test_mfcc  = compute_mfccs(x_test)

batchSize = 10 # nb of togetherly processed segments(of 1024 samples each) 
epochs = 30 # nb of back propagations

get_model = get_model(input_shape=(x_train_mfcc[0].shape), nb_classes=10, model_idx=0)
get_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
get_model.fit(x=x_train_mfcc, y=y_train, batch_size=batchSize, epochs=epochs)
get_model.summary()

score = get_model.evaluate(x_test_mfcc, y_test)

print(score)

get_model.save("test_model.h5")