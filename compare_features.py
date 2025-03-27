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
    tags = ["ade20k", "fss"]
    image_name = os.path.basename(args.image).split('.')[0]

    for tag in tags:
        print(f"\n▶ Feature Map Comparison for tag: `{tag}`")
        results = []

        for size in sizes:
            pytorch_path = f"outputs/pt_vit_{tag}_{size}_fMap_{image_name}.npy"
            trt_path = f"outputs/trt_vit_{tag}_{size}_fMap_{image_name}.npy"

            if not os.path.exists(pytorch_path) or not os.path.exists(trt_path):
                print(f"[경고] 파일 없음: {pytorch_path} 또는 {trt_path}")
                continue
            pytorch_feature = np.load(pytorch_path)
            trt_feature = np.load(trt_path)

            cos_sim = cosine_similarity(pytorch_feature, trt_feature)
            l2_dist = l2_distance(pytorch_feature, trt_feature)
            results.append((size, cos_sim, l2_dist))

        # 비교 결과 출력
        print(f"\nFeature Map Comparison Results for {args.image}:")
        print("Size | Cosine Similarity | L2 Distance")
        print("---------------------------------------")
        for size, cos_sim, l2_dist in results:
            print(f"{size}  | {cos_sim:.6f}         | {l2_dist:.6f}")

    
     # 🔁 Cross-Tag 비교: 각 size에 대해 ade20k vs fss 비교 (PyTorch & TRT 기준)
    if len(tags) >= 2:
        print("\n📊 Cross-Tag Feature Map Comparison (ade20k vs fss):")

        for size in sizes:
            cross_results = []

            pt_ade_path = f"outputs/pt_vit_ade20k_{size}_fMap_{image_name}.npy"
            pt_fss_path = f"outputs/pt_vit_fss_{size}_fMap_{image_name}.npy"
            trt_ade_path = f"outputs/trt_vit_ade20k_{size}_fMap_{image_name}.npy"
            trt_fss_path = f"outputs/trt_vit_fss_{size}_fMap_{image_name}.npy"

            # 파일 존재 확인
            if not all(map(os.path.exists, [pt_ade_path, pt_fss_path, trt_ade_path, trt_fss_path])):
                print(f"[경고] {size}px 비교를 위한 파일 중 일부가 존재하지 않습니다.")
                continue

            # PyTorch Feature 비교
            pt_ade_feat = np.load(pt_ade_path)
            pt_fss_feat = np.load(pt_fss_path)
            cos_sim_pt = cosine_similarity(pt_ade_feat, pt_fss_feat)
            l2_dist_pt = l2_distance(pt_ade_feat, pt_fss_feat)

            # TRT Feature 비교
            trt_ade_feat = np.load(trt_ade_path)
            trt_fss_feat = np.load(trt_fss_path)
            cos_sim_trt = cosine_similarity(trt_ade_feat, trt_fss_feat)
            l2_dist_trt = l2_distance(trt_ade_feat, trt_fss_feat)

            print(f"\n▶ Size {size}")
            print("Method   | Cosine Similarity | L2 Distance")
            print("------------------------------------------")
            print(f"PyTorch  | {cos_sim_pt:.6f}         | {l2_dist_pt:.6f}")
            print(f"TRT      | {cos_sim_trt:.6f}         | {l2_dist_trt:.6f}")


    # ✅ 시각화 (각 태그마다 행, 사이즈마다 열)
    fig, axes = plt.subplots(len(tags) * 2, len(sizes), figsize=(len(sizes) * 4, len(tags) * 2 * 4))

    for tag_idx, tag in enumerate(tags):
        for col_idx, size in enumerate(sizes):
            pytorch_path = f"outputs/pt_vit_{tag}_{size}_fMap_{image_name}.npy"
            trt_path = f"outputs/trt_vit_{tag}_{size}_fMap_{image_name}.npy"

            if not os.path.exists(pytorch_path) or not os.path.exists(trt_path):
                continue

            pytorch_feature = np.load(pytorch_path)[0]
            trt_feature = np.load(trt_path)[0]

            pytorch_map = reduce_feature_map(pytorch_feature)
            trt_map = reduce_feature_map(trt_feature)

            row_base = tag_idx * 2

            axes[row_base, col_idx].imshow(pytorch_map, cmap="viridis")
            axes[row_base, col_idx].set_title(f"[{tag}] PyTorch {size}")
            axes[row_base, col_idx].axis("off")

            axes[row_base + 1, col_idx].imshow(trt_map, cmap="viridis")
            axes[row_base + 1, col_idx].set_title(f"[{tag}] TRT {size}")
            axes[row_base + 1, col_idx].axis("off")

    plt.tight_layout()
    plt.show()