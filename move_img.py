from pathlib import Path
import shutil
from tqdm import tqdm

src_dir = Path(r"datasets/crowdhuman/val/images")
dst_dir = Path(r"datasets/crowdhuman/val")

dst_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

count = 0

for file in tqdm(src_dir.iterdir(), desc="Moving:"):
    if file.is_file() and file.suffix.lower() in image_exts:
        shutil.move(str(file), str(dst_dir / file.name))
        count += 1

print(f"Đã move {count} ảnh sang: {dst_dir}")