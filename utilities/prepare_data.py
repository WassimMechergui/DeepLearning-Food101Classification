import tensorflow_datasets as tfds

def prepare_data(data_dir = "./data"):
    (train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                                split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                                shuffle_files=True, # shuffle files on download?
                                                as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                                with_info=True,
                                                data_dir = data_dir) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)
    return (train_data, test_data), ds_info.features["label"].names