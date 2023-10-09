import torch
import numpy as np
import os
import os.path as osp
import argparse
import subprocess
import cv2
from pytube import YouTube

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/image_%05d.jpg', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

img_folder = 'test_image/output/outdoors_golf_00'
output_vid_file = 'test_image/output/video/outdoors_golf_00.mp4'
images_to_video(img_folder, output_vid_file)