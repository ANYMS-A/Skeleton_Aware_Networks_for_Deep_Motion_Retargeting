import re
import numpy as np
from .Quaternions import Quaternions
from .Animation import Animation


channel_map = {'Xrotation': 'x',
               'Yrotation': 'y',
               'Zrotation': 'z'}

channel_map_inv = {'x': 'Xrotation',
                   'y': 'Yrotation',
                   'z': 'Zrotation'}

order_map = {'x': 0,
             'y': 1,
             'z': 2}


def match_one_line(line):
    result = dict()
    result["r_match"] = re.match(r"ROOT (\w+:?\w+)", line)
    result["off_match"] = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
    result["chan_match"] = re.match(r"\s*CHANNELS\s+(\d+)", line)
    result["j_match"] = re.match("\s*JOINT\s+(\w+:?\w+)", line)
    result["f_match"] = re.match("\s*Frames:\s+(\d+)", line)
    result["ft_match"] = re.match("\s*Frame Time:\s+([\d\.]+)", line)
    result["d_match"] = line.strip().split()
    return result


def read_bvh(file_name: str, start=None, end=None,order=None, world=False, need_quater=False):
    """
    Reads a BVH file and constructs an animation
    Parameters
    ----------
    file_name: str
        File to be opened

    start : int
        Optional Starting Frame

    end : int
        Optional Ending Frame

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """
    f = open(file_name, "r")
    idx = 0
    active = -1
    end_site = False
    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    # if string in this list is detected, just continue, do nothing
    continue_list = ["HIERARCHY", "MOTION", "{"]

    for line in f:
        whether_continue = [each in line for each in continue_list]
        if any(whether_continue):
            continue

        match_dict = match_one_line(line)

        if match_dict["r_match"]:
            names.append(match_dict["r_match"].group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        if match_dict["off_match"]:
            if not end_site:
                offsets[active] = np.array([list(map(float, match_dict["off_match"].groups()))])
            continue

        if match_dict["chan_match"]:
            channels = int(match_dict["chan_match"].group(1))
            if order is None:
                channel_start = 0 if channels == 3 else 3
                channel_end = 3 if channels == 3 else 6
                parts = line.split()[2+channel_start:2+channel_end]
                if any([p not in channel_map for p in parts]):
                    continue
                order = "".join([channel_map[p] for p in parts])
            continue

        if match_dict["j_match"]:
            names.append(match_dict["j_match"].group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "End Site" in line:
            end_site = True
            continue

        if match_dict["f_match"]:
            if start and end:
                fnum = (end - start)-1
            else:
                fnum = int(match_dict["f_match"].group(1))
            j_num = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        if match_dict["ft_match"]:
            frame_time = float(match_dict["ft_match"].group(1))
            continue

        if (start and end) and (idx < start or idx >= end-1):
            idx += 1
            continue

        if match_dict["d_match"]:
            data_block = np.array(list(map(float, match_dict["d_match"])))
            N = len(parents)
            fi = idx - start if start else idx
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)
            idx += 1

    f.close()
    if need_quater:
        rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
    elif order != 'xyz':
        rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
        rotations = np.degrees(rotations.euler())

    return Animation(rotations, positions, orients, offsets, parents), names, frame_time


def save_bvh(file_name, anim, names=None, frame_time=1.0/24.0, order='zyx', positions=False, orients=True, mask=None,
             use_quaternion=False):
    """
        Saves an Animation to file as BVH
        Parameters
        ----------
        file_name: str
            File to be saved to
        anim : Animation
            Animation to save
        names : [str]
            List of joint names
        order : str
            Optional Specifier for joint order.
            Given as string E.G 'xyz', 'zxy'
        frame_time : float
            Optional Animation Frame time
        positions : bool
            Optional specfier to save bone
            positions for each frame
        orients : bool
            Multiply joint orients to the rotations
            before saving.
        use_quaternion : bool
    """
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]

    with open(file_name, 'w') as f:
        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channel_map_inv[order[0]], channel_map_inv[order[1]], channel_map_inv[order[2]]))

        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(f, anim, names, t, i, order=order, positions=positions)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frame_time);

        if use_quaternion:
            rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        else:
            rots = anim.rotations
        poss = anim.positions

        for i in range(anim.shape[0]):
            for j in range(anim.shape[1]):

                if positions or j == 0:

                    f.write("%f %f %f %f %f %f " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, order_map[order[0]]], rots[i, j, order_map[order[1]]], rots[i, j, order_map[order[2]]]))
                else:
                    if mask is None or mask[j] == 1:
                        f.write("%f %f %f " % (
                            rots[i, j, order_map[order[0]]], rots[i, j, order_map[order[1]]], rots[i, j, order_map[order[2]]]))
                    else:
                        f.write("%f %f %f " % (0, 0, 0))

            f.write("\n")


def save_joint(f, anim, names, t, i, order='zyx', positions=False):
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))

    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                                                                            channel_map_inv[order[0]],
                                                                            channel_map_inv[order[1]],
                                                                            channel_map_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                             channel_map_inv[order[0]], channel_map_inv[order[1]],
                                             channel_map_inv[order[2]]))

    end_site = True

    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t = save_joint(f, anim, names, t, j, order=order, positions=positions)
            end_site = False

    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)
    return t




