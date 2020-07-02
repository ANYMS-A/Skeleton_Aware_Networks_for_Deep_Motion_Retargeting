from datasets.motion_dataset import MotionDataset
from option_parser import get_args
from torch.utils.data import DataLoader

if __name__ == "__main__":
    args = get_args()
    print(args)
    args.is_train = 0
    args.batch_size = 1
    dataset = MotionDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    total_rot_max = -1000
    total_rot_min = 1000
    total_pos_max = -1000
    total_pos_min = 1000

    for dynamic_a, dynamic_b in dataloader:
        rot_a = dynamic_a[:, :-3, :, :]
        rot_b = dynamic_b[:, :-3, :, :]
        print(rot_a.max(), rot_a.min())
        print(rot_b.max(), rot_b.min())

        rot_max = rot_a.max() if rot_a.max() > rot_b.max() else rot_b.max()
        rot_min = rot_a.min() if rot_a.min() < rot_b.min() else rot_b.min()
        total_rot_max = rot_max if rot_max > total_rot_max else total_rot_max
        total_rot_min = rot_min if rot_min < total_rot_min else total_rot_min

        pos_a = dynamic_a[:, -3:, 0:1, :]
        pos_b = dynamic_b[:, -3:, 0:1, :]
        print(pos_a.max(), pos_a.min())
        print(pos_b.max(), pos_b.min())
        pos_max = pos_a.max() if pos_a.max() > pos_b.max() else pos_b.max()
        pos_min = pos_a.min() if pos_a.min() < pos_b.min() else pos_b.min()
        total_pos_max = pos_max if pos_max > total_pos_max else total_pos_max
        total_pos_min = pos_min if pos_min < total_pos_min else total_pos_min



        print("*"*10)
    print(total_rot_max, total_rot_min)
    print(total_pos_max, total_pos_min)

