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
parser.add_argument("--weights", type=str, default="modules/demo_e200.ckpt", help="Path to checkpoint")
args = parser.parse_args()

# 경로 설정
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

# ✅ 체크포인트 경로 (데이터셋 관련 매개변수는 필요 없으므로 제거 가능)
checkpoint_path = args.weights

# ✅ 체크포인트 경로에서 태그 추출 (예: demo_e200.ckpt → ade20k, fss_l16.ckpt → fss)
checkpoint_filename = os.path.basename(checkpoint_path)
if "ade20k" in checkpoint_filename or "demo" in checkpoint_filename:
    tag = "ade20k"
elif "fss" in checkpoint_filename:
    tag = "fss"
else:
    tag = "custom"

# ✅ 모델 로드
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

        # # --> 여기서 input tensor 저장 (numpy array로 변환)
        # input_np = input_tensor.cpu().numpy()
        # input_filename = f"pytorch_input_{os.path.basename(image_path).split('.')[0]}_{size}.npy"
        # np.save(os.path.join("inputs", input_filename), input_np)
        # print(f"[INFO] 크기 {size}의 input tensor 저장 완료 -> {os.path.join('inputs', input_filename)}")

        # 추론 실행
        output = model(input_tensor)

        # numpy로 변환 후 CPU로 이동
        output_np = output.cpu().numpy()

        # 결과 저장
        output_filename = f"pt_vit_{tag}_{size}_fMap_{os.path.basename(image_path).split('.')[0]}.npy"
        output_path = os.path.join("outputs", output_filename)

        np.save(output_path, output_np)
        print(f"[INFO] 크기 {size} Feature map 저장 완료 -> {output_path}")

print("[INFO] 모든 크기의 피처맵 저장 완료!")