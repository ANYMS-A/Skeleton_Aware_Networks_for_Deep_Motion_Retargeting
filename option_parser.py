import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_A', type=str, default='Aj', help='The skeleton to be retargeted')
    parser.add_argument('--character_B', type=str, default='BigVegas', help='The ground truth skeleton')
    parser.add_argument('--data_dir', type=str, default='./datasets/Mixamo', help='directory for all savings')
    parser.add_argument('--save_dir', type=str, default='./pretrained', help='directory for all savings')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--use_gpu', type=int, default=1, help='whether use GPU acceleration')
    parser.add_argument('--epoch_num', type=int, default=20001, help='epoch_num')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0, help='penalty of sparsity')
    parser.add_argument('--window_size', type=int, default=32, help='length of time axis per window')
    parser.add_argument('--dynamic_kernel_size', type=int, default=15, help='kernel size of dynamic convolution layer')
    parser.add_argument('--static_kernel_size', type=int, default=1, help='static size of static convolution layer')
    parser.add_argument('--neighbor_dist_thresh', type=int, default=2, help='Threshold for determine neighbor list for every joint')
    parser.add_argument('--dataset', type=str, default='Mixamo')
    parser.add_argument('--log_path', type=str, default='./log/train.log')

    parser.add_argument('--fk_world', type=int, default=1)


    parser.add_argument('--pos_repr', type=str, default='3d')
    parser.add_argument('--D_global_velo', type=int, default=0)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--model', type=str, default='mul_top_mul_ske')
    parser.add_argument('--lambda_rec', type=float, default=5)
    parser.add_argument('--lambda_cycle', type=float, default=5)
    parser.add_argument('--lambda_ee', type=float, default=10)
    parser.add_argument('--lambda_global_pose', type=float, default=2.5)
    parser.add_argument('--lambda_position', type=float, default=1)
    parser.add_argument('--ee_velo', type=int, default=1)
    parser.add_argument('--ee_from_root', type=int, default=1)
    parser.add_argument('--rec_loss_mode', type=str, default='extra_global_pos')
    parser.add_argument('--adaptive_ee', type=int, default=0)
    parser.add_argument('--simple_operator', type=int, default=0)
    parser.add_argument('--use_sep_ee', type=int, default=0)
    parser.add_argument('--eval_seq', type=int, default=0)
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def get_bvh(character_name, bvh_file_name):
    bvh_path = './datasets/Mixamo/{}/{}.bvh'.format(character_name, bvh_file_name)
    return bvh_path


def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
