import torch.nn as nn
import torch
import numpy as np

"""
Since I didn't find an elegant way to find the topology every time after Skeleton pooling
I use a dirty trick to hard-encode the topologies after pooling
So, this means we could only apply our model on Aj and Bigvegas's skeletons
"""
topology_after_1_pool = np.array([0, 0, 1, 0, 3, 0, 5, 6, 6, 8, 6, 10])
ee_id_after_1_pool = [2, 4, 7, 9, 11]

topology_after_2_pool = np.array([0, 0, 0, 0, 3, 3, 3])
ee_id_after_2_pool = [1, 2, 4, 5, 6]


class SkeletonConvolution(nn.Module):
    """
    The skeleton convolution based on the paper
    Use a more intuitive 2D convolution than the 1D convolution in the original source code
    """

    def __init__(self, in_channels, out_channels, k_size: tuple, stride, pad_size: tuple,
                 topology: np.ndarray, neighbor_dist: int, ee_id: list):
        """
        :param k_size: should be (simple_joint_num, a_short_time_length) !!!
        :param topology: A numpy array, the value of the array is parent idx, the idx of the array is child node idx
        it could tell us the topology of the initial simplified skeleton or the pooled skeleton
        """
        super(SkeletonConvolution, self).__init__()
        self.neighbor_list = find_neighbor(topology, neighbor_dist, ee_id)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad_size)

    def forward(self, x):
        """
        :param x: The input tensor should have size [B , IN_C , simple_joint_num , total_time_length]
        :return:
        """
        result_list = []
        for neighbors in self.neighbor_list:
            binary_mask = torch.zeros_like(x, dtype=torch.float)
            # only neighboring joint can has mask value 1
            binary_mask[:, :, neighbors, :] = 1
            tmp_x = x * binary_mask
            # tmp_result should have size [B * OUT_C * 1 * total_time_length]
            tmp_result = self.conv(tmp_x)
            result_list.append(tmp_result)
        return torch.cat(result_list, dim=2)


class SkeletonPool(nn.Module):
    """
    Apply average skeleton pooling on the output of the skeleton convolution layers
    The pooling layer should tell that: what's the topology of the next convolution layer!.
    """
    def __init__(self, topology, ee_id, layer_idx):
        """
        :param topology: 1D numpy array
        :param ee_id: A list
        """
        super(SkeletonPool, self).__init__()
        self.old_topology = topology
        self.old_ee_id = ee_id
        # store the topology after pooling and merge joints
        self.new_topology = topology_after_1_pool if layer_idx == 1 else topology_after_2_pool
        # store the ee_ids after pooling and merge joints
        self.new_ee_id = ee_id_after_1_pool if layer_idx == 1 else ee_id_after_2_pool
        self.seq_list = []
        self.pooling_list = []
        self.old_joint_num = len(self.old_topology.tolist())
        self.new_joint_num = len(self.new_topology.tolist())
        # calculate the degree of each joint in the skeleton graph
        # 经过下面的操作，计算各个joint的degree
        self.degree = calculate_degree(topology)
        # separate the skeleton into multiple sequence
        self.pooling_seq = find_pool_seq(self.degree)  # a list
        self.merge_pairs = self._get_merge_pairs()  # a list
        self.merge_nums = [len(each) for each in self.merge_pairs]

    def _get_merge_pairs(self):
        merge_pair_list = []
        for seq in self.pooling_seq:
            if len(seq) == 1:
                single_joint = [seq.pop(0)]
                merge_pair_list.append(single_joint)
                continue
            elif len(seq) % 2 != 0:
                single_joint = [seq.pop(0)]
                merge_pair_list.append(single_joint)
            for i in range(0, len(seq), 2):
                tmp_pair = [seq[i], seq[i+1]]
                merge_pair_list.append(tmp_pair)
        return merge_pair_list

    def forward(self, x):
        result_list = []
        result_list.append(x[:, :, 0:1, :])  # add the root joint's data into result
        for merge_pair in self.merge_pairs:
            tmp_result = torch.zeros_like(x[:, :, 0:1, :])
            for merge_idx in merge_pair:
                tmp_result += x[:, :, merge_idx: merge_idx+1, :]
            tmp_result /= len(merge_pair)
            result_list.append(tmp_result)
        result = torch.cat(result_list, dim=2)
        if result.shape[2] != self.new_joint_num:
            raise Exception('Joint num does not match after pooling')
        return result


class SkeletonUnPool(nn.Module):
    def __init__(self, un_pool_expand_nums: list):
        """
        :param un_pool_expand_nums: 一个列表，记录着对应的pooling层，每个merge后的joint是由几个关节合并得到的。
        由几个关节merge得到，在UnPool的时候，就把该关节的tensor复制几次。
        需要注意的是，root joint是从未被merge过的，所以也不duplicate它。
        """
        super(SkeletonUnPool, self).__init__()
        self.un_pool_expand_nums = un_pool_expand_nums

    def forward(self, x):
        result_list = []
        result_list.append(x[:, :, 0:1, :])  # add root joint's feature tensor first
        for idx, expand_num in enumerate(self.un_pool_expand_nums):
            tmp_idx = idx + 1
            tmp_x = x[:, :, tmp_idx: tmp_idx + 1, :]
            tmp_x = tmp_x.repeat(1, 1, expand_num, 1)
            result_list.append(tmp_x)
        out = torch.cat(result_list, dim=2)
        return out





def build_bone_topology(topology):
    # The topology is simplified already!
    # get all edges (parents_bone_idx, current_bone_idx)
    # edges 要比topology的个数少1
    edges = []
    joint_num = len(topology)
    # 舍去了root joint
    for i in range(1, joint_num):
        # i 指的是简化后骨骼的index, topology[i] is i's parents bone
        edges.append((topology[i], i))
    return edges


def calculate_neighbor_matrix(topology):
    topology = topology.tolist()
    joint_num = len(topology)
    mat = [[100000] * joint_num for _ in range(joint_num)]
    for i, j in enumerate(topology):
        mat[i][j] = 1
        mat[j][i] = 1
    for i in range(joint_num):
        mat[i][i] = 0
    # Floyd algorithm to calculate distance between nodes of the skeleton graph
    for k in range(joint_num):
        for i in range(joint_num):
            for j in range(joint_num):
                mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
    return mat


def calculate_degree(topology):
    topology = topology.tolist()
    joint_num = len(topology)
    mat = [[0] * joint_num for _ in range(joint_num)]
    for i, j in enumerate(topology):
        mat[i][j] = 1
        mat[j][i] = 1
    for i in range(joint_num):
        mat[i][i] = 0
    degree_list = [sum(each) for each in mat]
    return degree_list


def find_neighbor(topology, dist, ee_id):
    distance_mat = calculate_neighbor_matrix(topology)
    neighbor_list = []
    joint_num = len(distance_mat)
    for i in range(joint_num):
        neighbor = []
        for j in range(joint_num):
            # 距离小于d的，就加入到该骨骼的领接列表中来
            if distance_mat[i][j] <= dist:
                neighbor.append(j)
        # 将每根骨骼的邻接列表插入到一个总的列表中
        neighbor_list.append(neighbor)

    # add neighbor for global part(the root joint's neighbors' index)
    global_part_neighbor = neighbor_list[0].copy()
    # based on the paper, the end_effector should also be regarded as the neighbor of the root joint
    # so we need to add end_effector's index to root joint's neighbor list
    global_part_neighbor = list(set(global_part_neighbor).union(set(ee_id)))
    for i in global_part_neighbor:
        # the index of root joint is 0!
        if 0 not in neighbor_list[i]:
            neighbor_list[i].append(0)
    neighbor_list[0] = global_part_neighbor
    return neighbor_list


def find_pool_seq(degree):
    num_joint = len(degree)
    seq_list = [[]]
    for joint_idx in range(1, num_joint):
        if degree[joint_idx] == 2:
            seq_list[-1].append(joint_idx)
        else:
            seq_list[-1].append(joint_idx)
            seq_list.append([])
            continue
    seq_list = [each for each in seq_list if len(each) != 0]
    return seq_list

