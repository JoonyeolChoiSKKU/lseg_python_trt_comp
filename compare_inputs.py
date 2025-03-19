import os
import numpy as np

# 비교할 이미지 이름과 사이즈 리스트 (파일명에 맞게)
images = ["cat", "cat2", "cat3"]
#images = ["cat"]
sizes = [256, 320, 384, 480]
#sizes = [256]

input_dir = "inputs"

for image in images:
    for size in sizes:
        # PyTorch 파일명: pytorch_input_{image}_{size}.npy
        # TRT 파일명: trt_input_{size}_{image}.npy
        pytorch_file = os.path.join(input_dir, f"pytorch_input_{image}_{size}.npy")
        trt_file = os.path.join(input_dir, f"trt_input_{size}_{image}.npy")
        
        if not os.path.exists(pytorch_file):
            print(f"[WARN] 파일이 존재하지 않습니다: {pytorch_file}")
            continue
        if not os.path.exists(trt_file):
            print(f"[WARN] 파일이 존재하지 않습니다: {trt_file}")
            continue

        pytorch_input = np.load(pytorch_file)
        trt_input = np.load(trt_file)
        
        # PyTorch의 경우 (1, 3, H, W)면 squeeze 처리하여 (3, H, W)로 변경
        if pytorch_input.ndim == 4 and pytorch_input.shape[0] == 1:
            pytorch_input = np.squeeze(pytorch_input, axis=0)
        
        # 텐서의 shape 비교
        if pytorch_input.shape != trt_input.shape:
            print(f"[ERROR] {image} - {size}: Shape mismatch -> PyTorch: {pytorch_input.shape}, TRT: {trt_input.shape}")
            continue

        # 값 차이 계산
        diff = np.abs(pytorch_input - trt_input)
        max_diff = diff.max()
        mean_diff = diff.mean()
        is_close = np.allclose(pytorch_input, trt_input, rtol=1e-5, atol=1e-8)
        
        print(f"비교 결과 for {image} at size {size}:")
        print(f"  동일 여부 (np.allclose): {is_close}")
        print(f"  최대 차이 (max): {max_diff}")
        print(f"  평균 차이 (mean): {mean_diff}\n")

        print("첫 번째 행의 처음 6개 값 출력:")
        for c in range(pytorch_input.shape[0]):
            # 각 채널의 첫 번째 행의 처음 6개 값을 가져옴
            pyt_values = pytorch_input[c, 0, :6]
            trt_values = trt_input[c, 0, :6]
            
            print(f"Channel {c} PyT : {pyt_values}")
            print(f"Channel {c} TRT : {trt_values}\n")
