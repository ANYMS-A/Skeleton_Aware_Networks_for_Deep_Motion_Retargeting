from datasets.motion_dataset import MotionDataset
from torch.utils.data import DataLoader
from datasets import data_loader_collate_function
import torch
from option_parser import get_args
from model.skeleton import SkeletonConvolution, SkeletonPool
from model.enc_and_dec import Encoder, Decoder


if __name__ == "__main__":
    args = get_args()
    print(args)
    args.batch_size = 1
    args.is_train = 0
    dataset = MotionDataset(args)
    print(len(dataset))

    top_a, top_b = dataset.topologies
    ee_id_a, ee_id_b = dataset.ee_ids
    offsets = dataset.offsets
    offsets = [each.permute(1, 0) for each in offsets]
    offset_a, offset_b = offsets

    enc_a = Encoder(args=args, init_topology=top_a, init_ee_id=ee_id_a)


    dec_tops = enc_a.topologies
    dec_ee_ids = enc_a.ee_id_list
    dec_expand_nums = enc_a.expand_num_list
    dec_a = Decoder(args=args, topologies=dec_tops, ee_ids=dec_ee_ids, expand_nums=dec_expand_nums)

    if args.is_train:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_loader_collate_function)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
    for dx_a, dx_b in dataloader:
        print(dx_a.size(), dx_b.size())
        # if dx_a.size(3) != dx_b.size(3):
        #     print("Exception")
        # tmp_batch = dx_a.size(0)
        # tmp_frame = dx_a.size(3)
        # sx_a = offset_a.unsqueeze(2).repeat(1, 1, tmp_frame)
        # sx_a = sx_a.unsqueeze(0).repeat(tmp_batch, 1, 1, 1)
        # sx_a = sx_a
        # dx_a = dx_a
        #
        # d_latent, s_latent = enc_a(dx_a, sx_a)
        #
        # out = dec_a(d_latent, s_latent, sx_a)
        # print(out.size())






