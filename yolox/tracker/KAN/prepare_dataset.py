import json
import numpy as np
from collections import defaultdict

# BẮT BUỘC: Import KalmanFilter từ source code ByteTrack của bạn
try:
    from yolox.tracker.kalman_filter import KalmanFilter
except ImportError:
    print("Lỗi: Không tìm thấy thư mục yolox. Hãy đảm bảo bạn chạy script này ở thư mục gốc của ByteTrack!")
    exit(1)


def smooth_trajectory_with_kalman(tracklet_raw, img_w, img_h):
    """
    Làm mượt quỹ đạo và trích xuất vận tốc bằng Kalman Filter.
    """
    kf = KalmanFilter()
    mean, covariance = None, None
    smoothed_features = []

    for i, current in enumerate(tracklet_raw):
        cx = current['cx']
        cy = current['cy']
        w = current['w']
        h = current['h']
        
        # Kalman Filter của ByteTrack nhận input dạng: [cx, cy, aspect_ratio, height]
        measurement = np.array([cx, cy, w / h, h])

        # Chạy dự đoán và cập nhật KF
        if i == 0:
            mean, covariance = kf.initiate(measurement)
        else:
            mean, covariance = kf.predict(mean, covariance)
            mean, covariance = kf.update(mean, covariance, measurement)

        # Trích xuất dữ liệu từ trạng thái (mean) của KF
        # Trạng thái mean có 8 chiều: [cx, cy, a, h, vx, vy, va, vh]
        cx_kf = mean[0]
        cy_kf = mean[1]
        w_kf = mean[2] * mean[3] # width = ratio * height
        h_kf = mean[3]
        vx_kf = mean[4]  # Vận tốc X
        vy_kf = mean[5]  # Vận tốc Y

        # CHUẨN HÓA TOÀN BỘ VỀ KHOẢNG [0, 1] hoặc [-1, 1] CHO MẠNG KAN
        cx_norm = cx_kf / img_w
        cy_norm = cy_kf / img_h
        w_norm = w_kf / img_w
        h_norm = h_kf / img_h
        
        # Vận tốc chuẩn hóa
        vx_norm = vx_kf / img_w
        vy_norm = vy_kf / img_h

        # Nén thành vector 7 chiều: [frame, cx, cy, w, h, vx, vy]
        smoothed_features.append([
            current['frame_id'], 
            cx_norm, cy_norm, w_norm, h_norm, vx_norm, vy_norm
        ])

    return smoothed_features


def extract_and_smooth_trajectories(json_path):
    print(f"1. Đang đọc dữ liệu từ {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Lập bản đồ (mapping) image_id sang thông tin ảnh
    image_dict = {}
    for img in data['images']:
        image_dict[img['id']] = {
            'frame_id': img['frame_id'],
            'video_id': img['video_id'],
            'width': float(img['width']),
            'height': float(img['height'])
        }

    # Gom nhóm dữ liệu theo video và ID người
    tracks = defaultdict(lambda: defaultdict(list))
    
    print("2. Đang phân loại Bounding Box (chỉ lấy Pedestrian)...")
    for ann in data['annotations']:
        # Chỉ lấy đối tượng là Pedestrian (category_id = 1) và không nhiễu
        if ann.get('category_id') != 1:
            continue
        if 'track_id' not in ann or ann['track_id'] <= 0:
            continue
            
        img_info = image_dict[ann['image_id']]
        video_id = img_info['video_id']
        frame_id = img_info['frame_id']
        
        x, y, w, h = ann['bbox']
        
        # Chuyển sang hệ tọa độ tâm (GIỮ NGUYÊN PIXEL THỰC ĐỂ ĐƯA VÀO KALMAN)
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        tracks[video_id][ann['track_id']].append({
            'frame_id': frame_id,
            'cx': cx, 
            'cy': cy, 
            'w': w, 
            'h': h,
            'img_w': img_info['width'],
            'img_h': img_info['height']
        })

    final_track_dict = {}
    total_valid_tracks = 0
    
    print("3. Đang áp dụng Kalman Filter để khử nhiễu và tính vận tốc...")
    for vid, video_tracks in tracks.items():
        for tid, tracklet in video_tracks.items():
            # Quan trọng: Bắt buộc phải sắp xếp theo thứ tự thời gian
            tracklet = sorted(tracklet, key=lambda item: item['frame_id'])
            
            # Chỉ lấy những ID xuất hiện đủ lâu để Kalman Filter hội tụ (ví dụ >= 15 frames)
            if len(tracklet) < 10:
                continue
                
            img_w = tracklet[0]['img_w']
            img_h = tracklet[0]['img_h']
            
            # Khử nhiễu bằng hàm Kalman đã viết ở trên
            smoothed_features = smooth_trajectory_with_kalman(tracklet, img_w, img_h)
            
            # Nối video_id và track_id để đảm bảo ID không bị trùng giữa các video
            unique_track_id = f"video{vid}_track{tid}"
            final_track_dict[unique_track_id] = smoothed_features
            total_valid_tracks += 1

    print(f"-> Hoàn tất! Đã tạo được {total_valid_tracks} chuỗi quỹ đạo siêu mượt.\\n")
    return final_track_dict


if __name__ == "__main__":
    import torch # Import thêm torch để lưu file
    
    # 1. Xử lý tập huấn luyện (Train)
    print("=== ĐANG XỬ LÝ TẬP TRAIN ===")
    train_dict = extract_and_smooth_trajectories("/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/datasets/mot/annotations/train_half.json")
    
    # 2. Xử lý tập kiểm thử (Validation)
    print("=== ĐANG XỬ LÝ TẬP VALIDATION ===")
    # Sửa tên file ở đây cho đúng với file của bạn (ví dụ: val_half.json)
    val_dict = extract_and_smooth_trajectories("/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/datasets/mot/annotations/val_half.json")
    
    # 3. Lưu toàn bộ dữ liệu ra file để xài dần cho bước Training KAN sau này
    print("Đang lưu dữ liệu ra file...")
    torch.save(train_dict, '/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/datasets/KAN_dataset/kan_train_data.pt')
    torch.save(val_dict, '/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/datasets/KAN_dataset/kan_val_data.pt')
    print("Hoàn tất! Dữ liệu đã sẵn sàng để train.")