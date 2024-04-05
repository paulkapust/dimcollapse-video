import torch
import pytorchvideo.data
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import argparse
from tqdm import tqdm

# PySlowFast packages from https://github.com/facebookresearch/SlowFast needed to build self-supervised models
# Models based on https://arxiv.org/abs/2104.14558
from slowfast.models.build import build_model
from slowfast.utils.parser import load_config
from slowfast.config.defaults import assert_and_infer_cfg

parser = argparse.ArgumentParser(description='Compute features of selected self-supervised method and stored model')
parser.add_argument('--method', type=str, choices=["SimCLR", "MoCo", "SwAV", "BYOL"], help='list of method names')
parser.add_argument('--checkpoint', type=str, required=True, help='path to stored model')
parser.add_argument('--dataset', type=str, choices=["UCF101", "HMDB51", "Kinetics400"], help='dataset to compute features for')
parser.add_argument('--output_path', type=str, default="./", help='path where to store saved features')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

# Based on slowfast/utils/parser.py arguments (Defaults used)
parser.add_argument("--cfg", dest="cfg_files", required=True, nargs="+", help="Path to the config files of stored model")
parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1", default=0, type=int)
parser.add_argument("--num_shards", help="Number of shards using by the job", default=1, type=int)
parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system", default="tcp://localhost:9999", type=str)
parser.add_argument("--opts", help="See slowfast/config/defaults.py for all options", default=None, nargs=argparse.REMAINDER)


def main():
    args = parser.parse_args()

    print(f'Config file: {args.cfg_files} and Dataset: {args.dataset} chosen')
    for path_to_config in args.cfg_files:
        print(path_to_config)
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    # Building model based on method config file 
    model = build_model(cfg)
    # Loading self-supervised pretrained model state dictionary
    model_dict = torch.load(args.checkpoint)
    # For both look up https://github.com/facebookresearch/SlowFast/tree/main/projects/contrastive_ssl

    state_dict = model_dict['model_state']
    if args.method == "SimCLR" or args.method == "SwAV" or args.method == "MoCo":
        state_dict = {
                    key.replace("backbone.ssl_128.", "backbone.head."): value
                    for (key, value) in state_dict.items()
                }

    if args.method == "MoCo":
        state_dict = {
                    key.replace("backbone_hist.ssl_128.", "backbone_hist.head."): value
                    for (key, value) in state_dict.items()
                }

    if args.method == "BYOL":
        state_dict = {
                    key.replace("backbone.ssl_256.", "backbone.head."): value
                    for (key, value) in state_dict.items()
                }
        state_dict = {
                    key.replace("backbone_hist.ssl_256.", "backbone_hist.head."): value
                    for (key, value) in state_dict.items()
                }

    model.load_state_dict(state_dict)

    device = args.device
    model = model.eval()
    model = model.to(device)

    side_size = 256
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    crop_size = 224
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std)           
            ]
        ),
    )

    clip_duration = (num_frames * sampling_rate)/frames_per_second

    if args.dataset == "UCF101": 
        dataset = pytorchvideo.data.Ucf101(
                    data_path="/path/to/UCF101/videos",
                    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
                    video_sampler=torch.utils.data.SequentialSampler,
                    transform=transform,
                    decode_audio=False
                )

    if args.dataset == "Kinetics400": 
        dataset = pytorchvideo.data.Kinetics(
                    data_path="/path/to/Kinetics400/videos/val",
                    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
                    video_sampler=torch.utils.data.SequentialSampler,
                    transform=transform,
                    decode_audio=False
                )
    
    if args.dataset == "HMDB51":
        dataset_train = pytorchvideo.data.Hmdb51(
                    data_path="/path/to/HMDB51/split/testTrainMulti_7030_splits",
                    video_path_prefix="/path/to/HMDB51/videos",
                    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
                    video_sampler=torch.utils.data.SequentialSampler,
                    transform=transform,
                    decode_audio=False
                )
        dataset_test = pytorchvideo.data.Hmdb51(
                    data_path="/path/to/HMDB51/split/testTrainMulti_7030_splits",
                    video_path_prefix="/path/to/datasets/HMDB51/videos",
                    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
                    video_sampler=torch.utils.data.SequentialSampler,
                    transform=transform,
                    decode_audio=False,
                    split_type="test"
                )
        dataset_unused = pytorchvideo.data.Hmdb51(
                    data_path="/path/to/HMDB51/split/testTrainMulti_7030_splits",
                    video_path_prefix="/path/to/HMDB51/videos",
                    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
                    video_sampler=torch.utils.data.SequentialSampler,
                    transform=transform,
                    decode_audio=False,
                    split_type="unused"
                )
        dataset = torch.utils.data.ChainDataset([dataset_train, dataset_test, dataset_unused])

    loader = torch.utils.data.DataLoader(dataset, batch_size = 16)

    if args.method == "BYOL":
        projs = torch.empty((0,256)).to(device)
    else: 
        projs = torch.empty((0,128)).to(device)
    reps = []
    def hook_feat_map(mod, inp, out):
        reps.append(out.squeeze())
    model.backbone.head.pathway0_avgpool.register_forward_hook(hook_feat_map)
        
    print(f'Computing features of chosen self-supervised pretrained model and dataset...')
    with torch.no_grad():

        for batch in tqdm(loader):
                
            inputs = batch["video"]
            inputs = inputs.to(device)

            projs = torch.concat((projs, model([inputs])), dim=0)

    print(projs.shape)
    reps = torch.cat(reps, dim=0)
    print(reps.shape)
    torch.save(projs, args.output_path + args.method + '_'+ args.dataset + '_' + 'embeddings.pt')
    torch.save(reps, args.output_path + args.method + '_'+ args.dataset + '_' + 'representations.pt')
    print(f'...stored features of chosen model and dataset.')

if __name__ == '__main__':
    main()

