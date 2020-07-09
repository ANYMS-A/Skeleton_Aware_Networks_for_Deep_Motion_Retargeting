"""
The purpose of our task is retarget the skeleton A to skeleton B
So the dataset will load both A & B's .bvh motion data
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import get_bvh_file_names, convert_mean_var_shape
from .bvh_parser import BvhData


class MotionDataset(Dataset):
    def __init__(self, args, mode):
        super(MotionDataset, self).__init__()
        if not args.is_train and args.batch_size != 1:
            raise Exception("If not in train mode, batch_size must be 1")
        self.args = args
        self.character_a_name = args.character_A
        self.character_b_name = args.character_B
        self.bvh_file_list = get_bvh_file_names(mode=mode)
        # topologies and edges are needed when initialize the neural network
        # it's used for calculate the neighboring matrix
        self.topologies = []
        self.edges = []
        self.names = []
        self.offsets = []
        self.ee_ids = []
        # contains numpy arrays in shape[1, (J - 1), 4]
        self.rot_means = []
        self.rot_vars = []
        # contains numpy arrays in shape [1, 1, 3]
        self.pos_means = []
        self.pos_vars = []
        # used for define the kernel size of skeleton convolution
        self.joint_nums = []

        for character_name in [self.character_a_name, self.character_b_name]:
            std_bvh_data = BvhData(character_name, motion_file_name=character_name)
            self.topologies.append(std_bvh_data.topology)
            self.edges.append(std_bvh_data.edges)
            self.ee_ids.append(std_bvh_data.get_ee_id())
            self.names.append(std_bvh_data.names)
            # the offset now in shape [simple_joint_num, 3]
            offset = torch.from_numpy(std_bvh_data.offset).float()
            offset = offset.to(self.args.cuda_device) if self.args.use_gpu else offset
            self.offsets.append(offset)
            self.joint_nums.append(offset.shape[0])
            mean = np.load(os.path.join(args.data_dir, "mean_var", f"{character_name}_mean.npy"))
            var = np.load(os.path.join(args.data_dir, "mean_var", f"{character_name}_var.npy"))
            rot_mean, pos_mean, rot_var, pos_var = convert_mean_var_shape(mean, var)
            self.rot_means.append(rot_mean)
            self.rot_vars.append(rot_var)
            self.pos_means.append(pos_mean)
            self.pos_vars.append(pos_var)
        return

    def __len__(self):
        return len(self.bvh_file_list)

    def __getitem__(self, idx):
        """
        :return: the dynamic part of the .bvh data with shape[B, 4, ]
        """
        bvh_name = self.bvh_file_list[idx]
        character_list = []
        for idx,  character_name in enumerate([self.character_a_name, self.character_b_name]):
            bvh_data = BvhData(character_name, motion_file_name=bvh_name)
            # [frame, simple_joint_num - 1, 4]
            rotation = bvh_data.get_rotation()
            # [frame, 1, 3]
            root_position = bvh_data.get_root_position()
            # normalize the data
            rotation, root_position = self._normalize(rotation, root_position, idx)
            # concatenate as final input form [frame, simple_joint_num, 4]
            concat_array = self._concat_together(rotation, root_position)
            # convert to tensor [4, simple_joint_num, frame]
            final_output = self._to_tensor(concat_array)
            # if in training mode, the tensor should be sliced into equal frame length
            # which is convenient for mini-batch training
            if self.args.is_train:
                final_output = self._slice_to_equal_frame_len(final_output)
            character_list.append(final_output)
        return character_list[0].float(), character_list[1].float()

    def _normalize(self, rot, root_pos, character_idx: int):
        """
        :param rot: [frame, simple_joint_num - 1, 4]
        :param root_pos: # [frame, 1, 3]
        :param character_idx: idx for get mean and var for different character
        """
        norm_rot = (rot - self.rot_means[character_idx]) / self.rot_vars[character_idx]
        norm_root_pos = (root_pos - self.pos_means[character_idx]) / self.pos_vars[character_idx]
        return norm_rot, norm_root_pos

    def de_normalize(self, raw: torch.tensor, character_idx: int):
        """
        This function is called during both train and inference stage
        :param raw: The output of Decoder with shape[B, 4, simple_joint_num, frame]
        So we need to separate it into the rotations of joints and the position part, and de-normalize them
        :param character_idx: idx for get mean&var of different character
        :return The de-normalized root position and rotation data with shape [B, 3, 1, frame] and [B, 4, J-1, frame]
        """
        device = raw.device
        # numpy arrays in shape[1, (J - 1), 4]
        rot_mean = self.rot_means[character_idx]
        rot_var = self.rot_vars[character_idx]
        # convert to tensor with shape [1, 4, J-1, 1]
        rot_mean = torch.from_numpy(rot_mean).float().permute(0, 2, 1).unsqueeze(3).to(device)
        rot_var = torch.from_numpy(rot_var).float().permute(0, 2, 1).unsqueeze(3).to(device)
        # numpy arrays in shape [1, 1, 3]
        pos_mean = self.pos_means[character_idx]
        pos_var = self.pos_vars[character_idx]
        # convert to tensor with shape [1, 3, 1, 1]
        pos_mean = torch.from_numpy(pos_mean).float().permute(0, 2, 1).unsqueeze(3).to(device)
        pos_var = torch.from_numpy(pos_var).float().permute(0, 2, 1).unsqueeze(3).to(device)
        # separate the raw tensor
        # [B, 4, J-1, frame]
        rot_part = raw[:, :, 1:, :]
        # [B, 4, 1, frame] -> [B, 3, 1, frame]
        pos_part = raw[:, 0:-1, 0:1, :]
        # denormalize
        rot_part = (rot_part * rot_var) + rot_mean
        pos_part = (pos_part * pos_var) + pos_mean
        return rot_part, pos_part

    def _concat_together(self, rot, root_pos):
        """
        concatenate the rotation, root_position together as the dynamic input of the
        neural network
        :param rot: rotation matrix with shape [frame, simple_joint_num - 1, 4]
        :param root_pos: with shape [frame, 1, 3], pad a 0 in axis=2, to make the position with shape
        [frame, 1, 4]
        :return: numpy Array with shape [frame, simple_joint_num, 4]
        """
        frame_num = root_pos.shape[0]
        # pad 0 make root_pos with shape [frame, 1, 4]
        pad_root_pos = np.zeros([frame_num, 1, 4])
        pad_root_pos[:, :, 0:3] = root_pos
        # concatenate all together
        result = np.concatenate([pad_root_pos, rot], axis=1)
        return result

    def _to_tensor(self, np_array):
        """
        :return: Tensor with shape [4, simple_joint_num, frame]
        """
        result = torch.from_numpy(np_array)
        result = result.permute(2, 1, 0)
        result = result.to(self.args.cuda_device) if self.args.use_gpu else result
        return result

    def _slice_to_equal_frame_len(self, input_tensor):
        """
        ONLY USED DURING TRAINING STAGE
        :param input_tensor: tensor in shape [7, simple_joint_num, frame]
        :return:tensor in shape [B, 7, simple_joint_num, args.window_size]
        Where B depends on frame and args.window_size
        """
        win_size = self.args.window_size
        total_frame = input_tensor.size(2)
        win_num = total_frame // win_size
        if win_num == 0:
            raise Exception("The total frame is less than window_size!")
        result_list = []
        for i in range(win_num):
            tmp_frame_idx = range(i*win_size, (i+1)*win_size)
            tmp_tensor = input_tensor[:, :, tmp_frame_idx]
            # expand dim to [1, 7, simple_joint_num, args.window_size]
            tmp_tensor.unsqueeze_(0)
            result_list.append(tmp_tensor)
        return torch.cat(result_list, dim=0)

    def convert_to_bvh_write_format(self, raw: torch.tensor, character_idx):
        """
        :param raw: in shape [B, 4, J, frame], since this function only called during inference stage,
        the B is always equal to 1
        :param character_idx: an int number
        :return: tensor with shape
        """
        # denormalize first
        # [1, 4, J-1, frame], [1, 3, 1, frame]
        denorm_rot, denorm_root_pos = self.de_normalize(raw, character_idx)
        # make rotation from [1, 4, J-1, frame] to [frame, J-1, 4]
        rotation = denorm_rot.squeeze(0).permute(2, 1, 0)
        # make root position from [1, 3, 1, frame] to [frame, 1, 3]
        root_pos = denorm_root_pos.squeeze(0).permute(2, 1, 0)
        # into [frame, (simple_joint_num - 1) * 4]
        rotation = rotation.reshape(rotation.size(0), -1)
        # into [frame, 3]
        root_pos = root_pos.reshape(root_pos.size(0), -1)
        # into [frame, (simple_joint_num - 1) * 4 + 3]
        result = torch.cat([rotation, root_pos], dim=1)
        # into [(simple_joint_num - 1) * 4 + 3, frame]
        result = result.permute(1, 0)
        return result





