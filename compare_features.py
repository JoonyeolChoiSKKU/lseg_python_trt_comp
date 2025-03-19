import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import argparse

# 입력 파라미터 설정
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Input image file (e.g., cat.jpeg, cat2.jpeg, cat3.jpeg)")
parser.add_argument("--sizes", type=int, nargs='+', required=True, help="List of image sizes (e.g., 480 384 320 128)")
args = parser.parse_args()

# Cosine Similarity 계산
def cosine_similarity(feature1, feature2):
    f1 = torch.tensor(feature1).flatten().float()
    f2 = torch.tensor(feature2).flatten().float()
    return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()

# L2 Distance 계산
def l2_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# 비교할 파일 리스트
sizes = args.sizes
image_name = os.path.basename(args.image).split('.')[0]
results = []

for size in sizes:
    pytorch_feature = np.load(f"outputs/pytorch_feature_{image_name}_{size}.npy")
    trt_feature = np.load(f"outputs/trt_feature_{size}_{image_name}.npy")

    cos_sim = cosine_similarity(pytorch_feature, trt_feature)
    l2_dist = l2_distance(pytorch_feature, trt_feature)
    results.append((size, cos_sim, l2_dist))

# 비교 결과 출력
print(f"\nFeature Map Comparison Results for {args.image}:")
print("Size | Cosine Similarity | L2 Distance")
print("---------------------------------------")
for size, cos_sim, l2_dist in results:
    print(f"{size}  | {cos_sim:.6f}         | {l2_dist:.6f}")


# 시각화를 위한 subplot grid 준비 (2행 x 4열 = 총 8개의 subplot)
num_sizes = len(args.sizes)
fig, axes = plt.subplots(2, num_sizes, figsize=(num_sizes * 4, 2 * 4))

for col_idx, size in enumerate(args.sizes):
    pytorch_feature_path = f"outputs/pytorch_feature_{image_name}_{size}.npy"
    trt_feature_path = f"outputs/trt_feature_{size}_{image_name}.npy"
    
    if not os.path.exists(pytorch_feature_path) or not os.path.exists(trt_feature_path):
        print(f"파일이 존재하지 않습니다: {pytorch_feature_path} 또는 {trt_feature_path}")
        continue

    pytorch_feature = np.load(pytorch_feature_path)
    trt_feature = np.load(trt_feature_path)
    
    # (N, C, H, W) 형태에서 첫 번째 샘플의 첫 번째 채널 사용
    original_map = pytorch_feature[0, 0]
    trt_map = trt_feature[0, 0]
    
    # 상단 subplot: 원본 Feature Map
    axes[0, col_idx].imshow(original_map, cmap="viridis")
    axes[0, col_idx].set_title(f"Original {size}")
    axes[0, col_idx].axis("off")
    
    # 하단 subplot: TRT Feature Map
    axes[1, col_idx].imshow(trt_map, cmap="viridis")
    axes[1, col_idx].set_title(f"TRT {size}")
    axes[1, col_idx].axis("off")

plt.tight_layout()
plt.show()