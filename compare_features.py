import numpy as np
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
