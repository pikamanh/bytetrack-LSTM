import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_reid import TwoStreamReID
from prepare_dataset import MOTReIDMotionDataset
from sampler import RandomIdentitySampler
from hard_mining import BatchHardTripletLoss

def make_parser():
    parser = argparse.ArgumentParser("Training ReID (Appearance + Motion) cho ByteTrack")

    # Arguments cho Dataset
    parser.add_argument("--train-json", default="datasets/mot/annotations/train_half.json", type=str, help="Đường dẫn file COCO train")
    parser.add_argument("--valid-json", default="datasets/mot/annotations/val_half.json", type=str, help="Đường dẫn file COCO valid")
    parser.add_argument("--img-dir", default="datasets/mot/train", type=str, help="Thư mục chứa ảnh gốc")
    
    # Arguments cho Model & Cấu hình chuỗi
    parser.add_argument("--k-frames", default=5, type=int, help="Độ dài chuỗi lịch sử bounding box")
    parser.add_argument("--app-dim", default=128, type=int, help="Kích thước vector ngoại hình")
    parser.add_argument("--mot-dim", default=64, type=int, help="Kích thước vector chuyển động")
    parser.add_argument("--final-dim", default=128, type=int, help="Kích thước embedding đầu ra cuối cùng")
    
    # Arguments cho Training
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate (Adam)")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--patience", default=10, type=int, help="Earlystop training")
    
    # Arguments cho Output
    parser.add_argument("--output-dir", default="checkpoints/ReID", type=str)
    
    return parser

def validate_reid(
    model,
    valid_loader,
    device
):

    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(
            valid_loader,
            desc="Validation",
            leave=False
        )

        for imgs, motions, track_ids in pbar:
            imgs = imgs.to(device)
            motions = motions.to(device)

            embeddings = model(
                imgs,
                motions
            )

            all_embeddings.append(
                embeddings.cpu()
            )

            all_labels.append(
                track_ids.cpu()
            )

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    sim_mat = embeddings @ embeddings.T
    N = len(labels)

    positive_scores = []
    negative_scores = []

    positive_scores = []
    negative_scores = []

    for i in tqdm(range(N), desc="Computing Similarity", leave=False):
        same_mask = (
            labels == labels[i]
        )
        diff_mask = (
            labels != labels[i]
        )

        same_mask[i] = False

        positive_scores.extend(
            sim_mat[i][same_mask].tolist()
        )
        negative_scores.extend(
            sim_mat[i][diff_mask].tolist()
        )

    avg_pos = (
        sum(positive_scores)
        / len(positive_scores)
    )
    avg_neg = (
        sum(negative_scores)
        / len(negative_scores)
    )

    return avg_pos, avg_neg

def main():
    args = make_parser().parse_args()
    args = vars(args)
    os.makedirs(args['output_dir'], exist_ok=True)

    counter = 0 #Earlystop
    best_score = float("-inf")
    
    print(f"🚀 Bắt đầu setup training trên thiết bị: {args['device']}")

    # 1. KHỞI TẠO DATASET
    print("Loading datasets...")
    train_dataset = MOTReIDMotionDataset(args['train_json'], args['img_dir'], K=args['k_frames'])
    valid_dataset = MOTReIDMotionDataset(args['valid_json'], args['img_dir'], K=args['k_frames'])
    
    # LƯU Ý QUAN TRỌNG: track_id trong MOT thường không liên tục (VD: 1, 3, 5, 10).
    # Ta cần ánh xạ (map) chúng về dạng liên tục (0, 1, 2... N-1) để dùng cho CrossEntropyLoss
    unique_ids = list(train_dataset.tracks.keys())
    num_classes = len(unique_ids)
    id_to_class = {track_id: i for i, track_id in enumerate(unique_ids)}
    print(f"Đã tìm thấy {num_classes} ID (người) khác nhau trong tập Train.")

    sampler = RandomIdentitySampler(
        train_dataset,
        batch_size=args['batch_size'],
        num_instances=4
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        sampler=sampler,
        num_workers=args['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers']
    )

    # 2. KHỞI TẠO MÔ HÌNH
    model = TwoStreamReID(
        k_frames=args['k_frames'], 
        appearance_dim=args['app_dim'], 
        motion_dim=args['mot_dim'], 
        final_dim=args['final_dim'], 
        num_classes=num_classes  # Phải truyền số lượng class vào để tạo bộ Classifier
    ).to(args['device'])

    # 3. HÀM LOSS VÀ OPTIMIZER
    criterion_ce = nn.CrossEntropyLoss()
    # criterion_triplet = nn.TripletMarginLoss(margin=0.3, p=2) # Margin 0.3 là chuẩn cho ReID
    criterion_triplet = BatchHardTripletLoss(margin=0.3)
    
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 4. VÒNG LẶP TRAINING
    for epoch in range(args['epochs']):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_triplet_loss = 0.0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        for imgs, motions, raw_track_ids in pbar:
            # Map raw track_id sang class index liên tục
            labels = torch.tensor([id_to_class[tid.item()] for tid in raw_track_ids], dtype=torch.long)
            
            imgs = imgs.to(args['device'])
            motions = motions.to(args['device'])
            labels = labels.to(args['device'])

            optimizer.zero_grad()

            # Forward: Trả về embedding và dự đoán phân loại
            embeddings, logits = model(imgs, motions)

            # Tính Loss
            # --- 4.1 Cross Entropy Loss ---
            loss_ce = criterion_ce(logits, labels)
            
            # --- 4.2 Triplet Loss ---
            loss_triplet, nonzero_ratio = criterion_triplet(embeddings, labels)

            # Tổng hợp Loss (Có thể đánh trọng số, VD: 1.0 * CE + 1.0 * Triplet)
            loss = loss_ce + loss_triplet
            
            # Backward
            loss.backward()
            optimizer.step()

            # Tính accuracy tạm thời
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_triplet_loss += loss_triplet.item()
            
            # Update Progress Bar
            pbar.set_postfix({
                'Loss CE': f"{loss_ce.item():.4f}",
                'Loss Triplet': f"{loss_triplet.item():.4f}",
                'Nonzero Ratio': f"{nonzero_ratio.item():.4f}",
                'Loss': f"{loss.item():.4f}", 
                'Acc': f"{(correct/total_samples)*100:.1f}%"
            })
            
        scheduler.step()

        avg_pos, avg_neg = validate_reid(
            model,
            valid_loader,
            args['device']
        )

        score = avg_pos - avg_neg
        
        # In tổng kết Epoch
        avg_loss = total_loss / len(train_loader)
        avg_acc = (correct / total_samples) * 100
        print(
            f"=> Epoch {epoch+1}"
            f" | Train Loss {avg_loss:.4f}"
            f" | Train Acc {avg_acc:.2f}%"
            f" | Pos Cos {avg_pos:.4f}"
            f" | Neg Cos {avg_neg:.4f}"
        )

        if score > best_score:
            best_score = score
            counter = 0

            save_path = os.path.join(
                args['output_dir'],
                "best_reid.pth"
            )

            torch.save(model.state_dict(), save_path)

            print(
                f"🔥 New Best Model Saved!"
                f" Score={score:.4f}"
            )
        else:
            counter += 1
            print(
                f"⏳ No improvement "
                f"({counter}/{args['patience']})"
            )

        if counter >= args['patience']:
            print(
                f"\n🛑 Early Stopping!"
                f"\nBest Score: {best_score:.4f}"
                f"\nStopped at Epoch {epoch+1}"
            )

            break

        # Lưu checkpoint mỗi 5 epoch
        if (epoch + 1) % 5 == 0 or epoch == args['epochs'] - 1:
            save_path = os.path.join(args['output_dir'], f"reid_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Đã lưu weights tại {save_path}")

if __name__ == "__main__":
    main()