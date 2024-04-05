import argparse
import torch
import sys
from torchvision import transforms
from tqdm import tqdm

# DPC repositories and dependencies from https://github.com/TengdaHan/DPC needed to build self-supervised model architecture
# Models based on https://arxiv.org/abs/1909.04656
sys.path.append('./DPC_repo/utils') # set path to repo individually
sys.path.append('./DPC_repo/backbone')
from DPC_repo.eval.model_3d_lc import *
from DPC_repo.eval.dataset_3d_lc import UCF101_3d
from DPC_repo.eval.dataset_3d_lc import HMDB51_3d
from DPC_repo.dpc.dataset_3d import Kinetics400_full_3d
from augmentation import *


parser = argparse.ArgumentParser(description='Compute features of pretrained Dense Predictive Coding model')
parser.add_argument('--dataset', type=str, choices=["UCF101", "HMDB51", "Kinetics400"], help='dataset to compute features for')
parser.add_argument('--output_path', type=str, default="./", help='path where to store saved features')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

# For DPC, the datasets need to be set up according to https://github.com/TengdaHan/DPC/tree/master/process_data
def get_data(transform, args, batch_size=16, seq_len=5, num_seq=8):
    if args.dataset == "UCF101": 
        dataset_train = UCF101_3d(mode='train', 
                            transform=transform, 
                            seq_len=seq_len,
                            num_seq=num_seq)
        dataset_val = UCF101_3d(mode='val', 
                            transform=transform, 
                            seq_len=seq_len,
                            num_seq=num_seq)
        dataset_test = UCF101_3d(mode='val', 
                            transform=transform, 
                            seq_len=seq_len,
                            num_seq=num_seq)
    if args.dataset == "HMDB51":
        dataset_train = HMDB51_3d(mode='train', 
                            transform=transform, 
                            seq_len=seq_len,
                            num_seq=num_seq)
        dataset_val = HMDB51_3d(mode='val', 
                            transform=transform, 
                            seq_len=seq_len,
                            num_seq=num_seq)
        dataset_test = HMDB51_3d(mode='val', 
                            transform=transform, 
                            seq_len=seq_len,
                            num_seq=num_seq)
    if args.dataset == "Kinetics400":
        dataset_train = Kinetics400_full_3d(mode='train',
                                    transform=transform,
                                    seq_len=seq_len,
                                    num_seq=num_seq,
                                    big=True)
        dataset_val = Kinetics400_full_3d(mode='val',
                                    transform=transform,
                                    seq_len=seq_len,
                                    num_seq=num_seq,
                                    big=True)
        dataset_test = Kinetics400_full_3d(mode='test',
                                    transform=transform,
                                    seq_len=seq_len,
                                    num_seq=num_seq,
                                    big=True)

    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val, dataset_test])
    loader = torch.utils.data.DataLoader(dataset, batch_size)
    return loader

def main():
    args = parser.parse_args()

    print(f'DPC ResNet-34 model and Dataset: {args.dataset} chosen')

    # See https://github.com/TengdaHan/DPC#dpc-pretrained-weights for pretrained model weights on Kinetics-400 with 3D ResNet-34 (runningStats)
    checkpoint_path = "/path/to/k400_224_r34_dpc-rnn_runningStats.pth.tar"
    sample_size = 224
    num_seq = 5
    seq_len = 5
    network = 'resnet34'
    batch_size = 16

    model = LC(sample_size=sample_size, 
                    num_seq=num_seq, 
                    seq_len=seq_len, 
                    network=network )

    model_dict = torch.load(checkpoint_path)

    state_dict = model_dict['state_dict']
    state_dict = {
                key.replace("module.", ""): value
                for (key, value) in state_dict.items()
            }
    model.load_state_dict(state_dict, strict=False)

    device = "cuda"
    model = model.eval()
    model = model.to(device)

    transform = transforms.Compose([
                RandomSizedCrop(consistent=True, size=sample_size, p=0.0),
                Scale(size=sample_size),
                ToTensor(),
                Normalize()
            ])

    loader = get_data(transform, args, batch_size=batch_size, seq_len=seq_len, num_seq=num_seq)

    print(f'Computing features of DPC model and chosen dataset...')
    reps = torch.empty((0,256)).to(device)
    with torch.no_grad():

        for batch in tqdm(loader):

            input_seq = batch[0].to(device)

            _, features = model(input_seq)
            reps = torch.concat((reps, features.squeeze()), dim=0)

    print(reps.shape)
    torch.save(reps, args.output_path + 'DPC' + '_'+ args.dataset + '_' + 'representations.pt')
    print(f'...stored features of DPC model and chosen dataset.')

if __name__ == '__main__':
    main()




