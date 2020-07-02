import sys
import torch
import numpy as np
from utils.Quaternions import Quaternions
from utils.BVH_FILE import read_bvh, save_bvh
from utils.FKinematics import ForwardKinematics
from model.skeleton import build_bone_topology
from option_parser import get_bvh

# our experiment will retarget the skeleton of BigVegas to Mousy_m's
# define 2 "SIMPLIFIED" skeletons
big_vegas_skeleton = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
                      'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
                      'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']

aj_skeleton = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
               'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
               'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']

ee_name_big_vegas = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_aj = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']

simple_skeletons = [aj_skeleton,big_vegas_skeleton]
ee_names = [ee_name_aj, ee_name_big_vegas]


class BvhData(object):
    def __init__(self, character_name, motion_file_name, args=None):
        self.skeleton_types = ["Aj", "BigVegas"]
        if character_name not in self.skeleton_types:
            raise Exception('Unknown skeleton')

        file_path = get_bvh(character_name, bvh_file_name=motion_file_name)
        self.anim, self._names, self.frame_time = read_bvh(file_path)
        self.complete_joint_num = self.anim.shape[1]
        self.skeleton_type = self.skeleton_types.index(character_name)
        self.edges = []
        self.edge_mat = []  # neighboring matrix
        self.edge_num = 0
        self._topology = None
        self.ee_length = []
        # eliminate the ":" in the JOINT name
        for i, name in enumerate(self._names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                self._names[i] = name

        self.set_new_root(1)

        self.simplified_name = simple_skeletons[self.skeleton_type]
        # self.corps store the index in complete skeleton
        self.corps = []
        for name in self.simplified_name:
            j = self._names.index(name)
            self.corps.append(j)
        self.simplify_joint_num = len(self.corps)

        # ee_id is the end_effector's index list in the simplified skeleton
        self.ee_id = []
        for ee_name in ee_names[self.skeleton_type]:
            self.ee_id.append(self.simplified_name.index(ee_name))

        # 2 dicts map the index between simple & complete skeletons
        self.simplify_map = {}
        self.inverse_simplify_map = {}
        for simple_idx, complete_idx in enumerate(self.corps):
            self.simplify_map[complete_idx] = simple_idx
            self.inverse_simplify_map[simple_idx] = complete_idx
        # TODO why set -1 here ???
        self.inverse_simplify_map[0] = -1
        self.edges = build_bone_topology(self.topology)
        return

    def scale(self, alpha):
        self.anim.offsets *= alpha
        global_position = self.anim.positions[:, 0, :]
        global_position[1:, :] *= alpha
        global_position[1:, :] += (1 - alpha) * global_position[0, :]

    def rotate(self, theta, axis):
        q = Quaternions(np.hstack((np.cos(theta/2), np.sin(theta/2) * axis)))
        position = self.anim.positions[:, 0, :].copy()
        rotation = self.anim.rotations[:, 0, :]
        position[1:, ...] -= position[0:-1, ...]
        q_position = Quaternions(np.hstack((np.zeros((position.shape[0], 1)), position)))
        q_rotation = Quaternions.from_euler(np.radians(rotation))
        q_rotation = q * q_rotation
        q_position = q * q_position * (-q)
        self.anim.rotations[:, 0, :] = np.degrees(q_rotation.euler())
        position = q_position.imaginaries
        for i in range(1, position.shape[0]):
            position[i] += position[i-1]
        self.anim.positions[:, 0, :] = position

    @property
    def topology(self):
        if self._topology is None:
            # 得到每个简化骨骼的parents骨骼的index
            self._topology = self.anim.parents[self.corps].copy()
            for i in range(self._topology.shape[0]):
                # 把parents骨骼的index映射为simplify后的index
                # 所以topology[i]表示，简化后的第i个骨骼，它的parents的index是多少
                if i >= 1:
                    self._topology[i] = self.simplify_map[self._topology[i]]
        # return a np.array
        return self._topology

    def get_ee_id(self):
        return self.ee_id

    def to_reconstruction_tensor(self):
        rotations = self.get_rotation()
        root_pos = self.get_root_position()
        # reshape the rotation into [frame, (simple_joint_num - 1) * 4]
        rotations = rotations.reshape(rotations.shape[0], -1)
        # reshape the position into [frame, 3]
        root_pos = root_pos.reshape(root_pos.shape[0], -1)
        result = np.concatenate([rotations, root_pos], axis=1)
        # convert to tensor type
        res = torch.tensor(result, dtype=torch.float)
        # in shape [(simple_joint_num - 1) * 4 + 3, frame]
        res = res.permute(1, 0)
        res = res.reshape((-1, res.shape[-1]))
        return res

    def get_root_position(self):
        """
        Get the position of the root joint
        :return: A numpy Array in shape [frame, 1, 3]
        """
        # position in shape[frame, 1, 3]
        position = self.anim.positions[:, 0:1, :]
        return position

    def get_root_rotation(self):
        """
        Get the rotation of the root joint
        :return: A numpy Array in shape [frame, 1, 4]
        """
        # rotation in shape[frame, 1, 4]
        rotation = self.anim.rotations[:, self.corps, :]
        rotation = rotation[:, 0:1, :]
        # convert to quaternion format
        rotation = Quaternions.from_euler(np.radians(rotation)).qs
        return rotation

    def get_rotation(self):
        """
        Get the rotation of each joint's parent except the root joint
        :return: A numpy Array in shape [frame, simple_joint_num - 1, 4]]
        """
        # rotation in shape[frame, simple_joint_num, 3]
        rotations = self.anim.rotations[:, self.corps, :]
        # transform euler degree into radians then into quaternion
        # in shape [frame, simple_joint_num, 4]
        rotations = Quaternions.from_euler(np.radians(rotations)).qs
        # 除去root的剩下所有joint信息都存储在edges中
        # 因为.bvh中每个joint的rotation指的是施加在child joint上的旋转
        # 而此时简化的骨架的旋转是完整骨架中每根骨骼的parent的旋转，所以要把这些旋转信息替换成
        # 简化骨骼中对应的parent的旋转
        # 又因为使用了edges的信息，所以一定程度上可以看作这些edges自身旋转了多少
        index = []
        for e in self.edges:
            index.append(e[0])
        # now rotation is in shape[frame, simple_joint_num - 1, 4]
        rotations = rotations[:, index, :]
        return rotations

    @property
    def offset(self) -> np.ndarray:
        # in shape[simple_joint_num, 3]
        return self.anim.offsets[self.corps]

    @property
    def names(self):
        return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology
        res = 0

        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        p = self.ee_id[2]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]
        return res

    def write(self, file_path):
        save_bvh(file_path, self.anim, self.names, self.frame_time, order='xyz')
        return

    def set_new_root(self, new_root):
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[new_root], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        self.anim.offsets[0] = -self.anim.offsets[new_root]
        self.anim.offsets[new_root] = np.zeros((3, ))
        self.anim.positions[:, new_root, :] = new_pos
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, new_root, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_rot0 = (-rot1)
        new_rot0 = np.degrees(new_rot0.euler())
        new_rot1 = np.degrees(new_rot1.euler())
        self.anim.rotations[:, 0, :] = new_rot0
        self.anim.rotations[:, new_root, :] = new_rot1

        new_seq = []
        vis = [0] * self.anim.rotations.shape[1]
        new_idx = [-1] * len(vis)
        new_parent = [0] * len(vis)

        def relabel(x):
            nonlocal new_seq, vis, new_idx, new_parent
            new_idx[x] = len(new_seq)
            new_seq.append(x)
            vis[x] = 1
            for y in range(len(vis)):
                if not vis[y] and (self.anim.parents[x] == y or self.anim.parents[y] == x):
                    relabel(y)
                    new_parent[new_idx[y]] = new_idx[x]

        relabel(new_root)
        self.anim.rotations = self.anim.rotations[:, new_seq, :]
        self.anim.offsets = self.anim.offsets[new_seq]
        names = self._names.copy()
        for i, j in enumerate(new_seq):
            self._names[i] = names[j]
        self.anim.parents = np.array(new_parent, dtype=np.int)
