import torch
import torch.nn.functional as F
import numpy as np
import copy

# BẮT BUỘC: Import class TrajectorySiameseKAN từ file train_KAN của bạn
# (Hãy đảm bảo đường dẫn import đúng với project của bạn)
from yolox.tracker.KAN.train_KAN import TrajectorySiameseKAN

class KANMatcher:
    def __init__(self, model_path, seq_len=10, img_w=1920.0, img_h=1080.0, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.img_w = img_w
        self.img_h = img_h
        
        # Khởi tạo model và load weights
        input_dim = seq_len * 7
        self.model = TrajectorySiameseKAN(model_name='faster_kan', input_dim=input_dim)
        
        # Load weights an toàn
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() # Chuyển sang chế độ inference
        print(f"[*] Đã load thành công KAN Model từ: {model_path}")

    def _get_track_history(self, track):
        """
        Trích xuất 10 frame gần nhất từ track. 
        Nếu track chưa đủ 10 frame, ta copy frame đầu tiên để bù vào (padding).
        """
        # Giả sử trong class STrack của bạn có lưu lịch sử: track.history = [{'cx':.., 'cy':.., 'vx':..}]
        # Nếu chưa có, bạn phải tự thêm một list history vào class STrack.
        history = getattr(track, 'history', []) 
        
        if len(history) == 0:
            # Nếu track hoàn toàn mới, tạo data rỗng (sẽ bị loại trừ lúc matching)
            return np.zeros((self.seq_len, 7), dtype=np.float32)

        # Cắt lấy seq_len frame cuối
        recent = history[-self.seq_len:]
        
        # Nếu thiếu frame, pad bằng frame cũ nhất
        while len(recent) < self.seq_len:
            recent.insert(0, recent[0])
            
        # Format thành numpy array và chuẩn hóa [0, 1]
        formatted = []
        for feat in recent:
            formatted.append([
                feat['frame_id'],
                feat['cx'] / self.img_w,
                feat['cy'] / self.img_h,
                feat['w'] / self.img_w,
                feat['h'] / self.img_h,
                feat['vx'] / self.img_w,
                feat['vy'] / self.img_h
            ])
        return np.array(formatted, dtype=np.float32).flatten()

    def _get_detection_history(self, track, det):
        """
        Đây là phần cực kỳ thông minh:
        Detection (bbox mới) không có lịch sử. Nên ta sẽ lấy lịch sử 9 frame của Track, 
        cộng thêm Detection này làm frame thứ 10 để tạo thành một 'Quỹ đạo dự kiến'.
        """
        history = getattr(track, 'history', [])
        recent = history[-(self.seq_len - 1):] if len(history) > 0 else []
        
        # Pad nếu thiếu
        if len(recent) > 0:
            while len(recent) < self.seq_len - 1:
                recent.insert(0, recent[0])
        else:
            # Fake lịch sử nếu track chưa có gì
            fake_feat = {'frame_id': 0, 'cx': det[0], 'cy': det[1], 'w': det[2], 'h': det[3], 'vx': 0, 'vy': 0}
            recent = [fake_feat] * (self.seq_len - 1)

        # Tính toán vận tốc tạm thời cho detection (so với frame cuối của track)
        last_track_frame = recent[-1]
        det_cx, det_cy, det_w, det_h = det[0], det[1], det[2], det[3]
        vx = det_cx - last_track_frame['cx']
        vy = det_cy - last_track_frame['cy']

        det_feat = [
            last_track_frame['frame_id'] + 1,
            det_cx / self.img_w, det_cy / self.img_h,
            det_w / self.img_w, det_h / self.img_h,
            vx / self.img_w, vy / self.img_h
        ]
        
        # Format 9 frames track
        formatted = []
        for feat in recent:
            formatted.append([
                feat['frame_id'], feat['cx']/self.img_w, feat['cy']/self.img_h,
                feat['w']/self.img_w, feat['h']/self.img_h, feat['vx']/self.img_w, feat['vy']/self.img_h
            ])
        
        # Append frame thứ 10 (Detection)
        formatted.append(det_feat)
        return np.array(formatted, dtype=np.float32).flatten()

    @torch.no_grad()
    def get_kan_cost_matrix(self, tracks, detections):
        """
        Tính ma trận chi phí KAN (Cosine Distance) giữa N tracks và M detections
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        # Duyệt qua từng cặp (Tối ưu hơn là dùng batching, nhưng để test thì vòng lặp for là an toàn nhất)
        for i, track in enumerate(tracks):
            # Lấy vector quỹ đạo chuẩn của track
            t1_np = self._get_track_history(track)
            t1_tensor = torch.tensor(t1_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            feat_track = self.model.extract_feature(t1_tensor) # Shape: [1, 16]

            for j, det in enumerate(detections):
                # Det.tlwh là dạng [x_topleft, y_topleft, w, h] -> Chuyển sang cx, cy
                det_cx = det.tlwh[0] + det.tlwh[2] / 2
                det_cy = det.tlwh[1] + det.tlwh[3] / 2
                det_box = [det_cx, det_cy, det.tlwh[2], det.tlwh[3]]
                
                # Tạo vector 'quỹ đạo dự kiến' nếu ghép Det vào Track
                t2_np = self._get_detection_history(track, det_box)
                t2_tensor = torch.tensor(t2_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                feat_det = self.model.extract_feature(t2_tensor) # Shape: [1, 16]

                # Tính khoảng cách Cosine (1 - Cosine Similarity)
                # Similarity chạy từ -1 đến 1. Cost chạy từ 0 đến 2 (Càng nhỏ càng giống nhau)
                similarity = F.cosine_similarity(feat_track, feat_det, dim=1).item()
                cost_matrix[i, j] = 1.0 - similarity 

        return cost_matrix