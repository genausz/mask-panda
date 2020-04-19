import cv2
from glob import glob
import os
import numpy as np

class Video2Image():

    def __init__(self, video_dir, image_dir, samples_per_video=10):
        self.panda_videos = sorted(glob(video_dir+'/*.mp4'))
        self.image_dir = image_dir
        self.image_idx = 0
        self.samples_per_video = samples_per_video
        self.num_of_videos = len(self.panda_videos)
        
    def _fetch_frame_from_video(self, video_idx, phase):
        video_file = self.panda_videos[video_idx]
        print(phase, video_file)
        video_cap = cv2.VideoCapture(video_file)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.random.choice(total_frames, self.samples_per_video)
        for frame_idx in frame_idxs:            
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, image = video_cap.read()
            if success:
                image_dir = os.path.join(self.image_dir, phase)
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                cv2.imwrite('{}/{:04d}.jpg'.format(image_dir, self.image_idx), image)
                self.image_idx += 1
            else:
                print('[*] UNSECCESS SAMPLING in ', video_file)
            
    
    def generate(self):
        for idx in range(len(self.panda_videos)):
            if idx < 0.8 * self.num_of_videos:
                self._fetch_frame_from_video(idx, 'train')
            elif idx < self.num_of_videos - 1:
                self._fetch_frame_from_video(idx, 'val')
            else:
                self._fetch_frame_from_video(idx, 'test')

if __name__ == "__main__":
    VIDEO_DIR = './videos'
    IMAGE_DIR = './images'          
    video2Image = Video2Image(VIDEO_DIR, IMAGE_DIR)
    video2Image.generate()