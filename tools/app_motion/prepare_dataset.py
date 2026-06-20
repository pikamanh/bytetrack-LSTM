import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T

class MOTReIDMotionDataset(Dataset):
    def __init__(self, annotation_file, img_dir, K=5, transform=None):
        """
        annotation_file: Đường dẫn đến train_half.json hoặc val_half.json
        img_dir: Đường dẫn đến thư mục chứa ảnh MOT17
        K: Độ dài chuỗi lịch sử (số frame)
        """
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.K = K
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.tracks = self._build_tracks()
        self.samples = self._generate_samples()

    def _build_tracks(self):
        # Nhóm tất cả các annotation theo track_id và sắp xếp theo frame_id
        tracks = {}
        for ann_id in self.coco.getAnnIds():
            ann = self.coco.loadAnns(ann_id)[0]
            track_id = ann.get('track_id', ann.get('attributes', {}).get('track_id', -1))
            if track_id == -1: continue
            
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(ann)
            
        # Sắp xếp các bbox trong mỗi track theo image_id (frame_id)
        for track_id in tracks:
            tracks[track_id].sort(key=lambda x: x['image_id'])
            
        return tracks

    def _generate_samples(self):
        samples = []
        for track_id, anns in self.tracks.items():
            if len(anns) < self.K:
                continue
            
            # Trượt cửa sổ K frame
            for i in range(len(anns) - self.K + 1):
                window = anns[i : i + self.K]
                samples.append({
                    'track_id': track_id,
                    'window': window
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = sample['window']
        target_ann = window[-1] # Frame hiện tại t
        
        # 1. LẤY ẢNH CROP CỦA FRAME HIỆN TẠI (APPEARANCE)
        img_info = self.coco.loadImgs(target_ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        x, y, w, h = map(int, target_ann['bbox'])
        # Crop ảnh và xử lý tràn viền
        crop_img = img[max(0, y):min(img.shape[0], y+h), max(0, x):min(img.shape[1], x+w)]
        if self.transform:
            crop_img = self.transform(crop_img)

        # 2. TÍNH TOÁN LỊCH SỬ CHUYỂN ĐỘNG (MOTION)
        motion_history = []
        for i in range(1, self.K):
            prev_bbox = window[i-1]['bbox']
            curr_bbox = window[i]['bbox']
            
            # Tính delta (vận tốc) giữa các frame liên tiếp
            delta_x = curr_bbox[0] - prev_bbox[0]
            delta_y = curr_bbox[1] - prev_bbox[1]
            delta_w = curr_bbox[2] - prev_bbox[2]
            delta_h = curr_bbox[3] - prev_bbox[3]
            
            motion_history.append([delta_x, delta_y, delta_w, delta_h])
            
        motion_history = torch.tensor(motion_history, dtype=torch.float32)

        return crop_img, motion_history, sample['track_id']