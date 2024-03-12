import torch
import pytorchvideo.data
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Compute features of pretrained supervised Slow 3D ResNet-50 model')
parser.add_argument('--dataset', type=str, choices=["UCF101", "HMDB51", "Kinetics400"], help='dataset to compute features for')
parser.add_argument('--output_path', type=str, default="./", help='path where to store saved features')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])


def main():
    args = parser.parse_args()

    print(f'Slow 3D ResNet-50 model and Dataset: {args.dataset} chosen')

    # Following pipeline and model architecture from https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    model.blocks[-1].proj = torch.nn.Identity()


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
                    video_path_prefix="/path/to/HMDB51/videos",
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

    print(f'Computing features of Slow50 model and chosen dataset...')
    reps = torch.empty((0,2048)).to(device)
    with torch.no_grad():
        for batch in tqdm(loader):
            
            inputs = batch["video"]
            inputs = inputs.to(device)

            reps = torch.concat((reps, model(inputs)), dim=0)

    print(reps.shape)
    torch.save(reps, args.output_path + 'Slow50' + '_'+ args.dataset + '_' + 'representations.pt')
    print(f'...stored features of Slow50 model and chosen dataset.')

if __name__ == '__main__':
    main()
