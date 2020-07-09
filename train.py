from datasets.motion_dataset import MotionDataset
from torch.utils.data import DataLoader
from datasets import data_loader_collate_function
from option_parser import get_args
from model.enc_and_dec import Encoder, Decoder, StaticEncoder
from tqdm import tqdm
from utils.FKinematics import ForwardKinematics
import torch
from itertools import chain
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    args = get_args()
    print(args)
    # init dataset&dataLoader for training
    dataset = MotionDataset(args)
    if args.is_train:
        """
        since the training stage needs the data to have same tensor size except the batch dim
        so we use the data_loader_collate_function to concatenate the sliced frames along batch dim
        """
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_loader_collate_function, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # topologies&ee_ids for init the neural networks
    top_a, top_b = dataset.topologies
    ee_id_a, ee_id_b = dataset.ee_ids
    offsets = dataset.offsets
    offsets = [each.permute(1, 0) for each in offsets]
    offset_a, offset_b = offsets
    # init ForwardKinematic transform
    edges_a, edges_b = dataset.edges
    fk_transform_a = ForwardKinematics(args=args, edges=edges_a)
    fk_transform_b = ForwardKinematics(args=args, edges=edges_b)
    # init Encoder for A and B character
    enc_a = Encoder(args=args, init_topology=top_a, init_ee_id=ee_id_a)
    enc_b = Encoder(args=args, init_topology=top_b, init_ee_id=ee_id_b)
    # init static Encoder for A and B character
    static_enc_a = StaticEncoder(args=args, init_topology=top_a, init_ee_id=ee_id_a)
    static_enc_b = StaticEncoder(args=args, init_topology=top_b, init_ee_id=ee_id_b)
    # get topologies and ee_id info of encoder to init decoder
    # A's decoder info
    dec_a_tops = enc_a.topologies
    dec_a_ee_ids = enc_a.ee_id_list
    dec_a_expand_nums = enc_a.expand_num_list
    # B's decoder info
    dec_b_tops = enc_b.topologies
    dec_b_ee_ids = enc_b.ee_id_list
    dec_b_expand_nums = enc_b.expand_num_list
    # init decoders
    dec_a = Decoder(args=args, topologies=dec_a_tops, ee_ids=dec_a_ee_ids, expand_nums=dec_a_expand_nums)
    dec_b = Decoder(args=args, topologies=dec_b_tops, ee_ids=dec_b_ee_ids, expand_nums=dec_b_expand_nums)
    # init optimizer
    optimizer_a = torch.optim.Adam(chain(enc_a.parameters(), dec_a.parameters(), static_enc_a.parameters()),
                                   lr=args.lr, betas=(0.9, 0.999), amsgrad=True)
    optimizer_b = torch.optim.Adam(chain(enc_b.parameters(), dec_b.parameters(), static_enc_b.parameters()),
                                   lr=args.lr, betas=(0.9, 0.999), amsgrad=True)
    # init criterion
    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    # init TensorBoard Summary Writer
    writer = SummaryWriter()
    # training loop
    enc_a.train()
    enc_b.train()
    static_enc_a.train()
    static_enc_b.train()
    dec_a.train()
    dec_b.train()
    """
    START TRAINING
    """
    for epoch in range(1, args.epoch_num+1):
        p_bar = tqdm(dataloader)
        rec1_list = []
        rec2_list = []
        rec3_list = []
        rec4_list = []
        for dynamic_a, dynamic_b in p_bar:
            '''
            prepare needed data
            '''
            tmp_batch = dynamic_a.size(0)
            tmp_frame = dynamic_a.size(3)
            static_a = offset_a.unsqueeze(2).repeat(1, 1, tmp_frame).unsqueeze(0).repeat(tmp_batch, 1, 1, 1).detach()
            static_b = offset_b.unsqueeze(2).repeat(1, 1, tmp_frame).unsqueeze(0).repeat(tmp_batch, 1, 1, 1).detach()
            # cat dynamic and static data together to shape[B, 7, J, frame]
            mix_a = torch.cat([dynamic_a, static_a], dim=1)
            # mix_b = torch.cat([dynamic_b, static_b], dim=1)
            '''
            forward
            '''
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            # encode A's motion
            s_latent_a = static_enc_a(static_a)
            d_latent_a = enc_a(mix_a, s_latent_a)
            # encode B's static motion first
            s_latent_b = static_enc_b(static_b)
            # decode the latent cross A&B's domain
            pred_dynamic_b = dec_b(d_latent_a, s_latent_b, static_b)
            '''
            calculate losses
            '''
            # denormalize the outputs and targets
            denorm_pred_rot_b, denorm_pred_root_pos_b = dataset.de_normalize(raw=pred_dynamic_b, character_idx=1)
            denorm_rot_b, denorm_root_pos_b = dataset.de_normalize(raw=dynamic_b, character_idx=1)
            # 1st part of reconstruction loss(rotation of joints)
            rec_loss_1 = criterion_mse(denorm_pred_rot_b, denorm_rot_b)
            # 2nd part of reconstruction loss(the root position)
            rec_loss_2 = criterion_mse(denorm_pred_root_pos_b / 236.57, denorm_root_pos_b / 236.57)
            # calculate positions of all joints by forward kinematics
            pred_pos_b = fk_transform_b.forward_from_raw(denorm_rot=denorm_pred_rot_b,
                                                         de_norm_pos=denorm_pred_root_pos_b,
                                                         offset=offset_b.permute(1, 0).unsqueeze(0).repeat(tmp_batch, 1, 1))
            pos_b = fk_transform_b.forward_from_raw(denorm_rot=denorm_rot_b,
                                                    de_norm_pos=denorm_root_pos_b,
                                                    offset=offset_b.permute(1, 0).unsqueeze(0).repeat(tmp_batch, 1, 1))
            # convert pos to global world pos
            pred_pos_b = fk_transform_b.from_local_to_world(pred_pos_b / 236.57)
            pos_b = fk_transform_b.from_local_to_world(pos_b / 236.57)
            rec_loss_3 = 20 * criterion_mse(pred_pos_b, pos_b)

            rec_loss_4 = criterion_mse(pred_dynamic_b, dynamic_b)

            total_loss = rec_loss_1 + rec_loss_2 + rec_loss_3 + rec_loss_4
            total_loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            p_bar.set_description('Epoch: %s, rec1: %s, rec_2: %s, rec_3: %s, rec_4: %s'%
                                  (epoch, round(rec_loss_1.item(), 4), round(rec_loss_2.item(), 4),
                                   round(rec_loss_3.item(), 4), round(rec_loss_4.item(), 4)))
            rec1_list.append(rec_loss_1.item())
            rec2_list.append(rec_loss_2.item())
            rec3_list.append(rec_loss_3.item())
            rec4_list.append(rec_loss_4.item())

        avg_rec1 = sum(rec1_list) / len(rec1_list)
        avg_rec2 = sum(rec2_list) / len(rec2_list)
        avg_rec3 = sum(rec3_list) / len(rec3_list)
        avg_rec4 = sum(rec4_list) / len(rec4_list)

        # add loss to TensorBoard
        writer.add_scalar('rec1_loss', avg_rec1, epoch)
        writer.add_scalar('rec2_loss', avg_rec2, epoch)
        writer.add_scalar('rec3_loss', avg_rec3, epoch)
        writer.add_scalar('rec4_loss', avg_rec4, epoch)

        if epoch % 200 == 0:
            torch.save(enc_a.state_dict(), f'./pretrained/enc_a_{epoch}.pt')
            torch.save(dec_b.state_dict(), f'./pretrained/dec_b_{epoch}.pt')
            torch.save(static_enc_a.state_dict(), f'./pretrained/static_enc_a_{epoch}.pt')
            torch.save(static_enc_b.state_dict(), f'./pretrained/static_enc_b_{epoch}.pt')









