"""
For dataset, it should convert the data of .bvh file into tensors.

For static data, it should be in shape B * 3 * num_joints * 1, where 3 represent the 3D offsets which describe the
skeleton in some arbitrary initial poses

For dynamic data, it should be in shape B * 7 * num_joints * time_length, where 7=4+3 represent rotation of each frame
by a unit quaternions
"""
import numpy as np
import torch


def get_bvh_file_names(is_train):
    if is_train:
        file = open('./datasets/Mixamo/train_bvh_files.txt', 'r')
    else:
        file = open('./datasets/Mixamo/validate_bvh_files.txt', 'r')
    files_list = file.readlines()
    files_list = [f[:-1][:-4] for f in files_list]
    return files_list


def data_loader_collate_function(batch: list):
    """
    ONLY USED DURING TRAINING STAGE
    collate function used by the DataLoader initialization
    :param batch: A list contains a batch of tensor in shape [Ni, 7, simple_joint_num, a_window_length_frame]
    :return: A tensor with shape [B, 7, simple_joint_num, a_window_length_frame]
    Where B = sum(N1, N2..., Ni, N_length_of_batch), i is the index of batch list
    """
    character_data_a = [each[0] for each in batch]
    character_data_b = [each[1] for each in batch]
    return torch.cat(character_data_a, dim=0), torch.cat(character_data_b, dim=0)


def convert_mean_var_shape(mean: np.array, var: np.array):
    """
    convert the mean of rotation and root joint's position from the origin author's shape into our needed shape
    :param mean: in shape[(J-1)*4+3, 1]
    :param var: in shape[[(J-1)*4+3, 1]
    :return: 4 numpy Arrays
    """
    # in shape [1, (J-1), 4]
    rotation_mean = mean[:-3, :].reshape(1, -1, 4)
    # in shape [1, 1, 3]
    position_mean = mean[-3:, :].reshape(1, -1, 3)
    rotation_var = var[:-3, :].reshape(1, -1, 4)
    position_var = var[-3:, :].reshape(1, -1, 3)
    return rotation_mean, position_mean, rotation_var, position_var
