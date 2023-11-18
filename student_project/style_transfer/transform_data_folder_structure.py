import general as g
import os
import shutil

path_val = os.path.join(g.IMAGENET_PATH, 'val_raw/')
src_path_val = os.path.join(path_val, 'images')
trg_path_val = os.path.join(g.IMAGENET_PATH, 'val/')

path_train = os.path.join(g.IMAGENET_PATH, 'train_raw/')
src_path_train = path_train
trg_path_train = os.path.join(g.IMAGENET_PATH, 'train/')

def get_labels_from_annotations(path_annotations: str, sep='\t'):
    '''returns dict of labels for each image from annotations file'''
    label_to_filenames = {}

    with open(path_annotations, mode='r') as f:
        content = [line for line in f]

    for line in content:
        img_name = line.split(sep)[0]
        label = line.split(sep)[1]

        if label in label_to_filenames:
            label_to_filenames[label].append(img_name)
        else:
            label_to_filenames[label] = [img_name]

    return label_to_filenames

def create_class_folders(label_dict: dict, src_path: str, trg_path: str):
    '''creates one subfolder per class in the same style as train set'''
    cnt_files = 0
    for label in label_dict:
        file_names = label_dict[label]
        cnt_files += len(file_names)
        if not os.path.exists(os.path.join(trg_path, label)):
            os.makedirs(os.path.join(trg_path, label))
        for file in file_names:
            if not os.path.exists(os.path.join(trg_path, label, file)):
                shutil.copyfile(os.path.join(src_path, file), os.path.join(trg_path, label, file))

    assert cnt_files==len([name for name in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, name))]), "Warning: Not all files were assigned to one of the classes!"


def transform_train_folders(src_path: str, trg_path: str):
    for folder in os.listdir(src_path):
        # is actually a folder corresponding to a class label
        if folder[0] == 'n':
            if not os.path.exists(os.path.join(trg_path, folder)):
                os.makedirs(os.path.join(trg_path, folder))
            for file in os.listdir(os.path.join(src_path, folder, 'images')):
                shutil.copyfile(os.path.join(src_path, folder, 'images', file), os.path.join(trg_path, folder, file))

if __name__ == '__main__':
    # validation data
    label_dict = get_labels_from_annotations( os.path.join(path_val, 'val_annotations.txt'))
    create_class_folders(label_dict, src_path_val, trg_path_val)

    # Train data
    transform_train_folders(src_path_train, trg_path_train)