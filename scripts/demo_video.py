import torch
import numpy as np
import os
import os.path as osp
import argparse
import subprocess
import cv2
from pytube import YouTube

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('test_image/input/tmp', osp.basename(vid_file).replace('.', '_'))
    os.makedirs(img_folder, exist_ok=True)
    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    print(f'Images saved to \"{img_folder}\"')
    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape
    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.first().download(output_path=download_folder)

def main(args):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    video_file = args.vid_file
    # # download the youtube video
    # if video_file.startswith('https://www.youtube.com'):
    #     print(f'Downloading YouTube video \"{video_file}\"')
    #     video_file = download_youtube_clip(video_file, '/tmp')

    #     if video_file is None:
    #         exit('Youtube url is not valid!')
        
    #     print(f'Youtube Video has been downloaded to {video_file}...')
    
    # video to image
    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    print(f'Input video numbber of frames {num_frames}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', default='vibe_data/vibe_data/sample_video.mp4', type=str, help='input video path or youtube link')

    args = parser.parse_args()
    main(args)