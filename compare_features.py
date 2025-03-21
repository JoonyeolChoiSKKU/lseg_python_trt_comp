import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import argparse
from sklearn.decomposition import PCA

# Cosine Similarity 계산
def cosine_similarity(feature1, feature2):
    f1 = torch.tensor(feature1).flatten().float()
    f2 = torch.tensor(feature2).flatten().float()
    return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()

# L2 Distance 계산
def l2_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)


def reduce_feature_map(feature_map):
    """
    feature_map: numpy array of shape (C, H, W)
    모든 채널의 정보를 이용해 각 픽셀의 C차원 벡터를 1차원 값으로 투영하고,
    결과를 HxW 이미지로 반환합니다.
    scikit-learn의 PCA를 활용하여 최적화된 속도로 차원 축소를 수행합니다.
    """
    C, H, W = feature_map.shape
    # 각 픽셀의 feature를 행(row)로 두기 위해 (H*W, C) 형태로 변환합니다.
    X = feature_map.reshape(C, -1).T  # shape: (H*W, C)
    pca = PCA(n_components=1, svd_solver='randomized')
    projection = pca.fit_transform(X).squeeze()  # shape: (H*W,)
    projection = projection.reshape(H, W)
    # [0,1] 범위로 정규화
    proj_min, proj_max = projection.min(), projection.max()
    if proj_max - proj_min > 0:
        projection_norm = (projection - proj_min) / (proj_max - proj_min)
    else:
        projection_norm = projection
    return projection_norm


if __name__ == '__main__':
    # 입력 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image file (e.g., cat.jpeg, cat2.jpeg, cat3.jpeg)")
    parser.add_argument("--sizes", type=int, nargs='+', required=True, help="List of image sizes (e.g., 480 384 320 128)")
    args = parser.parse_args()
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
        
        # 첫 번째 샘플의 전체 채널 사용 (shape: (C, H, W))
        original_feature_map = pytorch_feature[0]
        trt_feature_map = trt_feature[0]
        
        original_map = reduce_feature_map(original_feature_map)
        trt_map = reduce_feature_map(trt_feature_map)
        
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