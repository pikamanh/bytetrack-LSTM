import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

from load_model import TrajectorySiameseKAN

# ---------------------------------------------------------
# 2. CONTRASTIVE LOSS
# ---------------------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        euclidean_distance = F.pairwise_distance(feat1, feat2, keepdim=True)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# ---------------------------------------------------------
# 3. DATASET (Kết hợp logic cắt ghép Pairs thật)
# ---------------------------------------------------------
class TrajectoryContrastiveDataset(Dataset):
    def __init__(self, data_path=None, seq_len=10, feature_dim=7, num_samples=5000):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.input_size = seq_len * feature_dim
        self.samples = []
        
        if data_path and os.path.exists(data_path):
            print(f"Đang load dữ liệu từ: {data_path}...")
            self.track_dict = torch.load(data_path, weights_only=False)
            self._generate_real_pairs(num_samples)
        else:
            print("Không tìm thấy file data. Khởi tạo dữ liệu giả lập (Mock Data)...")
            self._generate_mock_data(num_samples)

    def _generate_real_pairs(self, num_samples):
        # Logic sinh Positive và Negative pairs (đã viết ở bài trước)
        track_ids = list(self.track_dict.keys())
        video_groups = defaultdict(list)
        for tid in track_ids:
            vid = tid.split('_')[0] 
            video_groups[vid].append(tid)

        target_positive = num_samples // 2
        
        # 1. POSITIVE PAIRS
        pos_count = 0
        while pos_count < target_positive and track_ids:
            tid = np.random.choice(track_ids)
            tracklet = self.track_dict[tid]
            
            # Cần tối thiểu độ dài cho 2 chunk + ít nhất 1 frame bị che khuất (gap)
            min_len = 2 * self.seq_len + 1
            if len(tracklet) <= min_len: 
                continue
                
            # 1. Tính toán gap an toàn (để không bao giờ cắt mảng bị lố)
            max_possible_gap = len(tracklet) - 2 * self.seq_len
            max_gap = min(10, max_possible_gap) # Max gap là 10, hoặc số frame còn dư
            gap = np.random.randint(1, max_gap + 1)
            
            # 2. Tính toán điểm bắt đầu an toàn
            max_start1 = len(tracklet) - 2 * self.seq_len - gap
            start1 = np.random.randint(0, max_start1 + 1)
            
            # 3. Cắt mảng
            chunk1 = tracklet[start1 : start1 + self.seq_len]
            start2 = start1 + self.seq_len + gap
            chunk2 = tracklet[start2 : start2 + self.seq_len]
            
            # Chốt chặn an toàn cuối cùng: Đảm bảo cả 2 chunk đều đủ 10 frames
            if len(chunk1) != self.seq_len or len(chunk2) != self.seq_len:
                continue
            
            t1_tensor = torch.tensor(chunk1, dtype=torch.float32).flatten()
            t2_tensor = torch.tensor(chunk2, dtype=torch.float32).flatten()
            self.samples.append((t1_tensor, t2_tensor, torch.tensor([1.0], dtype=torch.float32)))
            pos_count += 1

        # 2. NEGATIVE PAIRS
        neg_count = 0
        for vid, tids in video_groups.items():
            if len(tids) < 2: continue
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    if neg_count >= target_positive: break
                    trackA, trackB = self.track_dict[tids[i]], self.track_dict[tids[j]]
                    dictA, dictB = {f[0]: f for f in trackA}, {f[0]: f for f in trackB}
                    common_frames = sorted(list(set(dictA.keys()) & set(dictB.keys())))
                    if len(common_frames) < self.seq_len: continue
                        
                    chunkA = [dictA[f] for f in common_frames[:self.seq_len]]
                    chunkB = [dictB[f] for f in common_frames[:self.seq_len]]
                    dist = np.sqrt((chunkA[0][1] - chunkB[0][1])**2 + (chunkA[0][2] - chunkB[0][2])**2)
                    
                    if dist < 0.1:
                        t1_tensor = torch.tensor(chunkA, dtype=torch.float32).flatten()
                        t2_tensor = torch.tensor(chunkB, dtype=torch.float32).flatten()
                        self.samples.append((t1_tensor, t2_tensor, torch.tensor([0.0], dtype=torch.float32)))
                        neg_count += 1
                        
        np.random.shuffle(self.samples)
        print(f"-> Tạo thành công {len(self.samples)} cặp mẫu.")

    def _generate_mock_data(self, num_samples):
        for _ in range(num_samples):
            t1 = torch.rand(self.input_size, dtype=torch.float32)
            label = float(np.random.randint(0, 2))
            t2 = t1 + torch.randn(self.input_size) * 0.05 if label == 1.0 else torch.rand(self.input_size, dtype=torch.float32)
            self.samples.append((t1, t2, torch.tensor([label], dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------------------------------------------------
# 4. TRAINING LOOP VỚI VALIDATION & EARLY STOPPING
# ---------------------------------------------------------
def train(args):
    # 1. Tạo thư mục lưu checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n--- BẮT ĐẦU TRAINING ---")
    print(f"Model: {args.model} | Device: {device} | Thư mục lưu: {args.save_dir}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Early Stop Patience: {args.patience}\n")
    
    INPUT_DIM = args.seq_len * 7  
    
    # Khởi tạo mô hình
    model = TrajectorySiameseKAN(
        model_name=args.model, input_dim=INPUT_DIM, 
        hidden_dim=args.hidden_dim, output_dim=args.output_dim
    ).to(device)
    
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 2. Load Dataset
    print("[TRAIN DATASET]")
    train_dataset = TrajectoryContrastiveDataset(data_path=args.train_data_path, seq_len=args.seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_dataloader = None
    if args.val_data_path:
        print("[VALIDATION DATASET]")
        val_dataset = TrajectoryContrastiveDataset(data_path=args.val_data_path, seq_len=args.seq_len, num_samples=1000) # Lấy số sample ít hơn cho val
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        print("Cảnh báo: Không có tập validation (--val-data-path). Early Stopping sẽ dựa trên Training Loss.")
    
    # Các biến phục vụ Early Stopping và Best Model
    best_loss = float('inf')
    patience_counter = 0
    
    # 3. Vòng lặp Epoch
    for epoch in range(args.epochs):
        # ----- PHẦN TRAIN -----
        model.train()
        total_train_loss = 0
        
        for t1, t2, labels in train_dataloader:
            t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            feat1, feat2 = model(t1, t2)
            loss = criterion(feat1, feat2, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # ----- PHẦN VALIDATION -----
        avg_val_loss = avg_train_loss # Mặc định lấy train loss làm chuẩn nếu ko có val data
        
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad(): # Bắt buộc phải có để ko lưu gradient tốn VRAM
                for t1, t2, labels in val_dataloader:
                    t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
                    feat1, feat2 = model(t1, t2)
                    loss = criterion(feat1, feat2, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            
            print(f"Epoch [{epoch+1:03d}/{args.epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1:03d}/{args.epochs}] - Train Loss: {avg_train_loss:.4f}")
            
        # ----- CƠ CHẾ LƯU VÀ EARLY STOPPING -----
        
        # Lưu định kỳ mỗi N epoch (Yêu cầu 3)
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
        
        # Lưu Best Model và Reset Patience (Yêu cầu 4 & 5)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0 # Reset bộ đếm
            
            best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   => Đã cập nhật best model (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   ! Không cải thiện (Patience: {patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n[!] EARLY STOPPING KÍCH HOẠT TẠI EPOCH {epoch+1}.")
                print(f"Đã ngừng huấn luyện do loss không giảm trong {args.patience} epoch liên tiếp.")
                break # Thoát vòng lặp

    print(f"\nHuấn luyện hoàn tất. Trọng số tốt nhất được lưu tại: '{os.path.join(args.save_dir, f'{args.model}_best.pth')}'")

# ---------------------------------------------------------
# 5. ARGPARSE CẤU HÌNH COMMAND LINE
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện KAN chống Switch ID cho ByteTrack")
    
    # Cấu hình Model
    parser.add_argument('--model', type=str, default='faster_kan', choices=['faster_kan', 'fast_kan', 'bsrbf_kan', 'efficient_kan'])
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--output-dim', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=10)
    
    parser.add_argument('--train-data-path', type=str, default='', help="File .pt của tập Train")
    parser.add_argument('--val-data-path', type=str, default='', help="File .pt của tập Validation")
    parser.add_argument('--save-dir', type=str, default='./weights', help="Thư mục để lưu file weights (.pth)")
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--save-every', type=int, default=10, help="Lưu model phụ mỗi N epoch")
    parser.add_argument('--patience', type=int, default=10, help="Số epoch tối đa chờ đợi Early Stopping")
    
    # Hệ thống
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    train(args)

#python new_pipeline/KAN/train_KAN.py \
#    --train-data-path "/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/datasets/KAN_dataset/kan_train_data.pt" \
#    --val-data-path "/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/datasets/KAN_dataset/kan_val_data.pt" \
#    --save-dir "/media/hung/Work/Project/AI Engineer/CPV/bytetrack-LSTM/checkpoints/kan" \
#    --epochs 100 \
#    --patience 15 \
#    --save-every 10