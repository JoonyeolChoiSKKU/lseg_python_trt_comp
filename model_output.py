import torch
from modules.lseg_module import LSegModule
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse

# ✅ 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# ✅ 모델 로드
checkpoint_path = "modules/demo_e200.ckpt"
model = LSegModule.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    backbone="clip_vitl16_384",
    aux=False,
    num_features=256,
    readout="project",
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    ignore_index=255,
    dropout=0.0,
    scale_inv=False,
    augment=False,
    no_batchnorm=False,
    widehead=True,
    widehead_hr=False,
    map_location=device,
    arch_option=0,
    block_depth=0,
    activation="lrelu"
).net.to(device)

# ✅ 모델을 평가 모드로 설정
model.eval()

# 입력받은 이미지 경로 및 크기 리스트
image_path = args.image
sizes = args.sizes

with torch.no_grad():
    for size in sizes:
        # 이미지 로드
        input_tensor = load_image(image_path, size).cuda()

        # 추론 실행
        output = model(input_tensor)

        # numpy로 변환 후 CPU로 이동
        output_np = output.cpu().numpy()

        # 결과 저장
        output_filename = f"pytorch_feature_{os.path.basename(image_path).split('.')[0]}_{size}.npy"
        output_path = os.path.join("outputs", output_filename)

        np.save(output_path, output_np)
        print(f"[INFO] 크기 {size} Feature map 저장 완료 -> {output_path}")

print("[INFO] 모든 크기의 피처맵 저장 완료!")