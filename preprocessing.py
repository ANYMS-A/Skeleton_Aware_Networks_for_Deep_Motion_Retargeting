import os
from tqdm import tqdm
from utils.BVH_FILE import read_bvh


def to_unix_name_format(file_names):
    file_list = [each.replace(' ', '\ ') for each in file_names]
    file_list = [each.replace('(', '\(') for each in file_list]
    file_list = [each.replace(')', '\)') for each in file_list]
    return file_list


def move_bvh_files(src, dst):
    """
    :param src:  source folder e.g. "./datasets/Mixamo_fbx/Aj"
    :param dst:  destination folder e.g. "./datasets/Mixamo/Aj"
    :return: None
    """
    file_list = os.listdir(src)
    file_list = [each for each in file_list if '.bvh' in each]
    file_list = to_unix_name_format(file_list)
    pbar = tqdm(file_list)
    for file in pbar:
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        cmd = f'cp {src_path} {dst_path}'
        os.system(cmd)
        pbar.set_description('moving bvh files')
    return


def delete_files(folder: str, file_list: list):
    file_list = to_unix_name_format(file_list)
    for file in file_list:
        del_path = os.path.join(folder, file)
        cmd = f'rm {del_path}'
        os.system(cmd)
    return


def has_equal_frame(path1: str, path2: str) -> bool:
    anim1, names1, _ = read_bvh(path1)
    anim2, names2, _ = read_bvh(path2)
    frame1 = anim1.positions.shape[0]
    frame2 = anim2.positions.shape[0]
    if frame1 != frame2:
        # print('*'*10)
        # print('Frame nor equal')
        # print(path1)
        # print(path2)
        return False
    elif anim1.positions.shape[0] < 32 or anim2.positions.shape[0] < 32:
        # print('*' * 10)
        # print("Frame too short")
        return False
    else:
        return True


def write_txt_file(mode: str, file_names: list):
    save_name = f'./datasets/Mixamo/{mode}_bvh_files.txt'
    file_names = [each + '\n' for each in file_names]
    with open(save_name, 'w+') as f:
        for file_name in file_names:
            f.write(file_name)
    return


def split_train_val_files(folder_1, folder_2):
    file_list1 = os.listdir(folder_1)
    file_list2 = os.listdir(folder_2)
    # take the union set
    file_list = list(set(file_list1) & set(file_list2))
    # take the rest set
    del_files_1 = list(set(file_list1) - set(file_list))
    del_files_2 = list(set(file_list2) - set(file_list))
    # check whether the frame number are equal
    bad_files = []
    for idx, file in enumerate(file_list):
        # define the path to the bvh files
        path_1 = os.path.join(folder_1, file)
        path_2 = os.path.join(folder_2, file)
        if not has_equal_frame(path_1, path_2):
            del_files_1.append(file)
            del_files_2.append(file)
            bad_files.append(file)
    file_list = list(set(file_list) - set(bad_files))
    # delete the bad data, bad means unpaired or have different number of frame
    delete_files(folder_1, del_files_1)
    delete_files(folder_2, del_files_2)
    # create 2 txt file contains the train and val bvh files name
    print("There are ", len(file_list), " files")
    val_names = file_list[::10]
    train_names = list(set(file_list) - set(val_names))
    write_txt_file('train', train_names)
    write_txt_file('validate', val_names)
    return


if __name__ == "__main__":
    # convert the fbx file into bvh file
    cmd_blender = 'blender -b -P ./datasets/fbx2bvh.py'
    os.system(cmd_blender)
    # move the bvh files into .dataset/Mixamo folder
    src = "./datasets/Mixamo_fbx/Aj"
    dst = "./datasets/Mixamo/Aj"
    move_bvh_files(src, dst)
    src = "./datasets/Mixamo_fbx/BigVegas"
    dst = "./datasets/Mixamo/BigVegas"
    move_bvh_files(src, dst)
    # create train and validation files list
    split_train_val_files(folder_1="./datasets/Mixamo/Aj", folder_2="./datasets/Mixamo/BigVegas")




