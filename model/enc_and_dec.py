from .skeleton import SkeletonConvolution, SkeletonPool, SkeletonUnPool
import torch.nn as nn
import torch


class EncBasicBlock(nn.Module):
    """
    The Convolution + ReLU + Pooling block for building Encoder(both dynamic or static)
    """
    def __init__(self, args, in_channel, out_channel, topology, ee_id, layer_idx, dynamic=True):
        super(EncBasicBlock, self).__init__()
        joint_num = len(topology.tolist())

        kernel_size = (joint_num, args.dynamic_kernel_size) if dynamic else (joint_num, args.static_kernel_size)
        hidden_channel = out_channel // 2
        self.conv1by1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

        self.conv1 = SkeletonConvolution(in_channels=in_channel,
                                         out_channels=hidden_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args.neighbor_dist_thresh,
                                         ee_id=ee_id)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lkrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = SkeletonConvolution(in_channels=hidden_channel,
                                         out_channels=out_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args.neighbor_dist_thresh,
                                         ee_id=ee_id)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.lkrelu2 = nn.LeakyReLU(inplace=True)

        self.pool = SkeletonPool(topology=topology, ee_id=ee_id, layer_idx=layer_idx)
        self.new_topology = self.pool.new_topology
        self.new_ee_id = self.pool.new_ee_id
        # this attribute is for Decoder to UpSampling
        self.expand_num = self.pool.merge_nums

    def forward(self, x):
        identity = self.conv1by1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lkrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.lkrelu2(out)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    """
    The encoder which encodes the dynamic&static part of the animation data as mentioned in the paper.
    """
    def __init__(self, args, init_topology, init_ee_id):
        """
        :param args: options arguments
        :param init_topology: After parsing the bvh file, we'll get the init topology info of the simplified skeleton
        edges are a list obj which consists of many lists: [parent_node_idx, current_node_idx], it could tell us the
        init topology of the simplified skeleton
        :param init_ee_id: the end_effector index of the initial simplified skeleton
        """
        super(Encoder, self).__init__()
        # store topologies for every SkeletonConvolution layer after SkeletonPooling
        self.topologies = [init_topology]
        self.ee_id_list = [init_ee_id]
        self.expand_num_list = []
        self.in_channels = [7, 64]
        self.out_channels = [32, 128]
        # build the 1st Encoder layer
        self.enc_layer1 = EncBasicBlock(args=args, in_channel=self.in_channels[0],
                                        out_channel=self.out_channels[0], topology=self.topologies[0],
                                        ee_id=self.ee_id_list[0], layer_idx=1)
        self.topologies.append(self.enc_layer1.new_topology)
        self.ee_id_list.append(self.enc_layer1.new_ee_id)
        self.expand_num_list.append(self.enc_layer1.expand_num)
        # 2nd Encoder layer
        self.enc_layer2 = EncBasicBlock(args=args, in_channel=self.in_channels[1],
                                        out_channel=self.out_channels[1], topology=self.topologies[1],
                                        ee_id=self.ee_id_list[1], layer_idx=2)
        self.expand_num_list.append(self.enc_layer2.expand_num)
        # init weights
        self.init_weights()
        # move to device
        self.to(args.cuda_device)

    def forward(self, x, s_latent):
        """
        :param x: The dynamic & static concatenate input[B, 7, joint_num, frame]
        :param s_latent: The static latent feature input[B, 32, joint_num, frame]
        :return: tensor with shape [B, 128, joint_num_after_2_pooling, frame]
        """
        # [B, 7, joint_num, frame] -> [B, 32, joint_num, frame]
        out = self.enc_layer1(x)
        # [B, 32, joint_num, frame] -> [B, 64, joint_num, frame]
        out = torch.cat([out, s_latent], dim=1)
        # [B, 64, joint_num, frame] -> [B, 128, joint_num, frame]
        out = self.enc_layer2(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)


class DecBasicBlock(nn.Module):
    """
    Unpool + Convolution(no non-linear activation at the end of layer)
    """
    def __init__(self, args, in_channel, out_channel, topology, ee_id, expand_nums):
        super(DecBasicBlock, self).__init__()
        kernel_size = (len(topology.tolist()), args.dynamic_kernel_size)
        hidden_channel = out_channel // 2
        self.un_pool = SkeletonUnPool(un_pool_expand_nums=expand_nums)
        self.conv1by1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.conv1 = SkeletonConvolution(in_channels=in_channel,
                                         out_channels=hidden_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args.neighbor_dist_thresh,
                                         ee_id=ee_id)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lkrelu = nn.LeakyReLU(inplace=True)

        self.conv2 = SkeletonConvolution(in_channels=hidden_channel,
                                         out_channels=out_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args.neighbor_dist_thresh,
                                         ee_id=ee_id)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

    def forward(self, x, s_latent):
        out = self.un_pool(x)
        out = torch.cat([out, s_latent], dim=1)
        identity = self.conv1by1(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.lkrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return out


class Decoder(nn.Module):
    def __init__(self, args, topologies: list, ee_ids: list, expand_nums: list):
        super(Decoder, self).__init__()
        self.in_channels = [128+32, 64+3]
        self.out_channels = [64, 4]
        self.dec_layer1 = DecBasicBlock(args=args, in_channel=self.in_channels[0], out_channel=self.out_channels[0],
                                        topology=topologies[-1], ee_id=ee_ids[-1], expand_nums=expand_nums[-1])

        self.lkrelu = nn.LeakyReLU(inplace=True)

        self.dec_layer2 = DecBasicBlock(args=args, in_channel=self.in_channels[1], out_channel=self.out_channels[1],
                                        topology=topologies[-2], ee_id=ee_ids[-2], expand_nums=expand_nums[-2])
        #self.tan_shrink = nn.Tanhshrink()

        self.init_weights()
        # move to device
        self.to(args.cuda_device)

    def forward(self, d_latent, s_latent, sx):
        out = self.dec_layer1(d_latent, s_latent)
        out = self.lkrelu(out)
        out = self.dec_layer2(out, sx)
        #out = self.tan_shrink(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)


class StaticEncoder(nn.Module):
    """
    Encode the static offset with shape [B, 3, J, frame_num] into
    latent static tensor with shape [B, 32, J, frame_num]
    """
    def __init__(self, args, init_topology, init_ee_id):
        super(StaticEncoder, self).__init__()

        self.in_channel = 3
        self.out_channel = 32
        self.enc_layer = EncBasicBlock(args=args, in_channel=self.in_channel,
                                       out_channel=self.out_channel, topology=init_topology,
                                       ee_id=init_ee_id, layer_idx=1, dynamic=False)
        # init weights
        self.init_weights()
        # move to device
        self.to(args.cuda_device)

    def forward(self, x):
        out = self.enc_layer(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)













