from data_load import get_file_paths_ordered, load_data

paths_train, paths_test, y_train_one_hot, y_test_one_hot = get_file_paths_ordered(num_speaker=30, test_ratio=0.2)

x_train, y_train, x_test, y_test = load_data(paths_train=paths_train, paths_test=paths_test, 
                                             y_train_one_hot=y_train_one_hot, y_test_one_hot=y_test_one_hot)

print()