import numpy as np
from utils.Quaternions import Quaternions
from utils.Filtering import gaussian_smooth


# rotation with shape frame * J * 3
def write_bvh(parent, offset, rotation, position, names, frametime, order, path):
    file = open(path, 'w')
    frame = rotation.shape[0]
    joint_num = rotation.shape[1]
    order = order.upper()

    file_string = 'HIERARCHY\n'

    def write_static(idx, prefix):
        nonlocal parent, offset, rotation, names, order, file_string
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(*order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*order)
        offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx+1, rotation.shape[1]):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frame) + 'Frame Time: %.8f\n' % frametime
    for i in range(frame):
        file_string += '%.6f %.6f %.6f ' % (position[i][0], position[i][1], position[i][2])
        for j in range(joint_num):
            file_string += '%.6f %.6f %.6f ' % (rotation[i][j][0], rotation[i][j][1], rotation[i][j][2])
        file_string += '\n'

    file.write(file_string)
    return file_string


class BvhWriter(object):
    def __init__(self, edges: list, names: list, offset):
        """
        Write the simplified skeleton into bvh file
        :param edges: a list contains the parents-child relationships of the skeleton
        :param names: a list contains the names of the simplified skeleton(bones) names
        :param offset: a torch.Tensor with shape [3, simplified_num_joints]
        """
        offset = offset.permute(1, 0).cpu().detach().numpy()
        self.edges = edges
        self.names = names
        self.offset = offset
        self.joint_num = len(self.names)

    def edge_rotation_to_joint_rotation(self, rotation):
        """
        Since the rotation of edges are parents' joint rotation, we need to convert them into the bvh file format
        :param rotation: numpy array with shape [frame, J-1, 3], use euler angle not quaternion
        :return: numpy array with shape [frame, J, 3]
        """
        f, j_mone, _ = rotation.shape
        children_of_joint = [list() for each in range(self.joint_num)]
        parent_of_joint = [-1] * self.joint_num
        for edge in self.edges:
            parent_idx = edge[0]
            child_idx = edge[1]
            children_of_joint[parent_idx].append(child_idx)
            parent_of_joint[child_idx] = parent_idx
        # set root's parent be itself
        parent_of_joint[0] = 0

        rotation_new = np.zeros([f, j_mone+1, 3])
        for idx in range(self.joint_num):
            tmp_children = children_of_joint[idx]
            tmp_children = [each - 1 for each in tmp_children]
            if len(tmp_children) == 0:
                continue
            else:
                tmp_rotations = rotation[:, tmp_children, :]
                tmp_rotations = tmp_rotations.mean(axis=1)
                rotation_new[:, idx, :] = tmp_rotations
        return rotation_new, parent_of_joint

    def write(self, rotations, positions, order, path, frametime=1.0/30):
        # position in shape [frame, 3]
        # rotation in shape [frame, J-1, 4]
        if order == 'quaternion':
            norm = rotations[:, :, 0] ** 2 + rotations[:, :, 1] ** 2 + rotations[:, :, 2] ** 2 + rotations[:, :, 3] ** 2
            norm = np.repeat(norm[:, :, np.newaxis], 4, axis=2)
            rotations /= norm
            rotations = Quaternions(rotations)
            rotations = np.degrees(rotations.euler())
            order = 'xyz'
        # convert the rotation, [frame, J, 3]
        rotations, parent_of_joint = self.edge_rotation_to_joint_rotation(rotations)
        return write_bvh(parent_of_joint, self.offset, rotations, positions, self.names, frametime, order, path)

    def write_raw(self, motion, order, path, frametime=1.0/30):
        """
        :param motion: A tensor with shape [(simple_joint_num - 1) * 4 + 3, frame]
        :return:
        """
        # motion now in shape [frame, (simple_joint_num - 1) * 4 + 3]
        motion = motion.permute(1, 0).detach().cpu().numpy()
        positions = motion[:, -3:]
        rotations = motion[:, :-3]
        # smooth the position and rotation info
        rotations, positions = gaussian_smooth(rot=rotations, pos=positions)
        if order == 'quaternion':
            # [frame, J-1, 4]
            rotations = rotations.reshape((motion.shape[0], -1, 4))
        else:
            rotations = rotations.reshape((motion.shape[0], -1, 3))
        # position in shape [frame, 3]
        return self.write(rotations, positions, order, path, frametime)
