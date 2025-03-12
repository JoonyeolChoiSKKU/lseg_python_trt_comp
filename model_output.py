import torch
from lseg.image_encoder import LSegImageEncoder
from lseg.lseg_module import LSegModule
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse

# 입력 파라미터 설정
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Input image file (e.g., cat.jpeg, cat2.jpeg, cat3.jpeg)")
parser.add_argument("--sizes", type=int, nargs='+', required=True, help="List of image sizes (e.g., 480 384 320 128)")
args = parser.parse_args()

# 경로 설정
CKPT_PATH = "models/demo_e200.ckpt"
os.makedirs("outputs", exist_ok=True)

# 이미지 로딩 및 전처리
def load_image(image_path, size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 모델 로드
lseg_module = LSegModule.load_from_checkpoint(
    checkpoint_path=CKPT_PATH,
    backbone='clip_vitl16_384',
    num_features=256
)
lseg_net = lseg_module.net if hasattr(lseg_module, 'net') else lseg_module
image_encoder = LSegImageEncoder(lseg_net).eval().cuda()

# 입력받은 이미지 경로 및 크기 리스트
image_path = args.image
sizes = args.sizes

with torch.no_grad():
    for size in sizes:
        # 이미지 로드
        input_tensor = load_image(image_path, size).cuda()

        # 추론 실행
        output = image_encoder(input_tensor)

        # numpy로 변환 후 CPU로 이동
        output_np = output.cpu().numpy()

        # 결과 저장
        output_filename = f"pytorch_feature_{os.path.basename(image_path).split('.')[0]}_{size}.npy"
        output_path = os.path.join("outputs", output_filename)

        np.save(output_path, output_np)
        print(f"[INFO] 크기 {size} Feature map 저장 완료 -> {output_path}")

print("[INFO] 모든 크기의 피처맵 저장 완료!")