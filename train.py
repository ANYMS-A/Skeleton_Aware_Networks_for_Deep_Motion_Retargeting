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


def train_one_epoch(args, encoder_a, encoder_b, static_encoder_a, static_encoder_b,
                    decoder_a, decoder_b, optim_a, optim_b, dataset):
    # switch models to train mode
    encoder_a.train()
    encoder_b.train()
    static_encoder_a.train()
    static_encoder_b.train()
    decoder_a.train()
    decoder_b.train()
    # init criterion
    criterion_mse = torch.nn.MSELoss()
    # init data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_loader_collate_function, shuffle=True)
    p_bar = tqdm(data_loader)
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
        optim_a.zero_grad()
        optim_b.zero_grad()
        # encode A's motion
        s_latent_a = static_encoder_a(static_a)
        d_latent_a = encoder_a(mix_a, s_latent_a)
        # encode B's static motion first
        s_latent_b = static_encoder_b(static_b)
        # decode the latent cross A&B's domain
        pred_dynamic_b = decoder_b(d_latent_a, s_latent_b, static_b)
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
        optim_a.step()
        optim_b.step()

        p_bar.set_description('Train_Epoch: %s, rot: %s, root_pos: %s, world_pos: %s, rot&root_pos: %s' %
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
    return avg_rec1, avg_rec2, avg_rec3, avg_rec4


def validation(args, encoder_a, encoder_b, static_encoder_a, static_encoder_b, decoder_a, decoder_b, dataset):
    # switch models to eval mode
    encoder_a.eval()
    encoder_b.eval()
    static_encoder_a.eval()
    static_encoder_b.eval()
    decoder_a.eval()
    decoder_b.eval()
    # init data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_loader_collate_function, shuffle=False)
    p_bar = tqdm(data_loader)
    # init criterion
    criterion_mse = torch.nn.MSELoss()
    rec1_list = []
    rec2_list = []
    rec3_list = []
    rec4_list = []
    with torch.no_grad():
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
            # encode A's motion
            s_latent_a = static_encoder_a(static_a)
            d_latent_a = encoder_a(mix_a, s_latent_a)
            # encode B's static motion first
            s_latent_b = static_encoder_b(static_b)
            # decode the latent cross A&B's domain
            pred_dynamic_b = decoder_b(d_latent_a, s_latent_b, static_b)
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
                                                         offset=offset_b.permute(1, 0).unsqueeze(0).repeat(tmp_batch, 1,
                                                                                                           1))
            pos_b = fk_transform_b.forward_from_raw(denorm_rot=denorm_rot_b,
                                                    de_norm_pos=denorm_root_pos_b,
                                                    offset=offset_b.permute(1, 0).unsqueeze(0).repeat(tmp_batch, 1, 1))
            # convert pos to global world pos
            pred_pos_b = fk_transform_b.from_local_to_world(pred_pos_b / 236.57)
            pos_b = fk_transform_b.from_local_to_world(pos_b / 236.57)
            rec_loss_3 = 20 * criterion_mse(pred_pos_b, pos_b)
            rec_loss_4 = criterion_mse(pred_dynamic_b, dynamic_b)
            p_bar.set_description('Validate_Epoch: %s, rot: %s, root_pos: %s, world_pos: %s, rot&root_pos: %s' %
                                  (epoch, round(rec_loss_1.item(), 4), round(rec_loss_2.item(), 4),
                                   round(rec_loss_3.item(), 4), round(rec_loss_4.item(), 4)))
            rec1_list.append(rec_loss_1.item())
            rec2_list.append(rec_loss_2.item())
            rec3_list.append(rec_loss_3.item())
            rec4_list.append(rec_loss_4.item())
        # calculate average loss
        avg_rec1 = sum(rec1_list) / len(rec1_list)
        avg_rec2 = sum(rec2_list) / len(rec2_list)
        avg_rec3 = sum(rec3_list) / len(rec3_list)
        avg_rec4 = sum(rec4_list) / len(rec4_list)
        return avg_rec1, avg_rec2, avg_rec3, avg_rec4


if __name__ == "__main__":
    train_args = get_args()
    print(train_args)
    # init dataset&dataLoader for training
    train_dataset = MotionDataset(train_args, mode='train')
    validate_dataset = MotionDataset(train_args, mode='validate')

    # topologies&ee_ids for init the neural networks
    top_a, top_b = train_dataset.topologies
    ee_id_a, ee_id_b = train_dataset.ee_ids
    offsets = train_dataset.offsets
    offsets = [each.permute(1, 0) for each in offsets]
    offset_a, offset_b = offsets
    # init ForwardKinematic transform
    edges_a, edges_b = train_dataset.edges
    fk_transform_a = ForwardKinematics(args=train_args, edges=edges_a)
    fk_transform_b = ForwardKinematics(args=train_args, edges=edges_b)
    # init Encoder for A and B character
    enc_a = Encoder(args=train_args, init_topology=top_a, init_ee_id=ee_id_a)
    enc_b = Encoder(args=train_args, init_topology=top_b, init_ee_id=ee_id_b)
    # init static Encoder for A and B character
    static_enc_a = StaticEncoder(args=train_args, init_topology=top_a, init_ee_id=ee_id_a)
    static_enc_b = StaticEncoder(args=train_args, init_topology=top_b, init_ee_id=ee_id_b)
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
    dec_a = Decoder(args=train_args, topologies=dec_a_tops, ee_ids=dec_a_ee_ids, expand_nums=dec_a_expand_nums)
    dec_b = Decoder(args=train_args, topologies=dec_b_tops, ee_ids=dec_b_ee_ids, expand_nums=dec_b_expand_nums)
    # init optimizer
    optimizer_a = torch.optim.Adam(chain(enc_a.parameters(), dec_a.parameters(), static_enc_a.parameters()),
                                   lr=train_args.lr, betas=(0.9, 0.999), amsgrad=True)
    optimizer_b = torch.optim.Adam(chain(enc_b.parameters(), dec_b.parameters(), static_enc_b.parameters()),
                                   lr=train_args.lr, betas=(0.9, 0.999), amsgrad=True)

    # init TensorBoard Summary Writer
    writer = SummaryWriter()
    # training loop
    """
    START TRAINING
    """
    for epoch in range(1, train_args.epoch_num+1):
        """
        TRAIN
        """
        loss1_train, loss2_train, loss3_train, loss4_train = train_one_epoch(args=train_args,
                                                                             encoder_a=enc_a, encoder_b=enc_b,
                                                                             static_encoder_a=static_enc_a,
                                                                             static_encoder_b=static_enc_b,
                                                                             decoder_a=dec_a, decoder_b=dec_b,
                                                                             optim_a=optimizer_a, optim_b=optimizer_b,
                                                                             dataset=train_dataset)
        # add train loss to TensorBoard
        writer.add_scalar('rot_loss/train', loss1_train, epoch)
        writer.add_scalar('root_pos_loss/train', loss2_train, epoch)
        writer.add_scalar('world_pos_loss/train', loss3_train, epoch)
        writer.add_scalar('rot_and_root_pos_loss/train', loss4_train, epoch)
        """
        VALIDATE
        """
        loss1_val, loss2_val, loss3_val, loss4_val = validation(args=train_args,
                                                                encoder_a=enc_a, encoder_b=enc_b,
                                                                static_encoder_a=static_enc_a,
                                                                static_encoder_b=static_enc_b,
                                                                decoder_a=dec_a, decoder_b=dec_b,
                                                                dataset=validate_dataset)
        # add validate loss to TensorBoard
        writer.add_scalar('rot_loss/validate', loss1_val, epoch)
        writer.add_scalar('root_pos_loss/validate', loss2_val, epoch)
        writer.add_scalar('world_pos_loss/validate', loss3_val, epoch)
        writer.add_scalar('rot_and_root_pos_loss/validate', loss4_val, epoch)

        if epoch % 200 == 0:
            torch.save(enc_a.state_dict(), f'./pretrained/enc_a_{epoch}.pt')
            torch.save(dec_b.state_dict(), f'./pretrained/dec_b_{epoch}.pt')
            torch.save(static_enc_a.state_dict(), f'./pretrained/static_enc_a_{epoch}.pt')
            torch.save(static_enc_b.state_dict(), f'./pretrained/static_enc_b_{epoch}.pt')









