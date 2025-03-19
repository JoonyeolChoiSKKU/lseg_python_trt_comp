#!/bin/bash

# 이미지 파일 리스트
IMAGES=("images/cat.jpeg" "images/cat2.jpeg" "images/cat3.jpeg")

# 사용할 이미지 크기 (내림차순 정렬)
#SIZES=(480 384 320 128)
SIZES=(480 384 320 256)

for IMAGE_FILE in "${IMAGES[@]}"; do
    echo "✅ Running Pytorch Model for image: $IMAGE_FILE with sizes: ${SIZES[*]}"
    python3 model_output.py --image "$IMAGE_FILE" --sizes ${SIZES[*]}

    echo "✅ Running TensorRT Inference for image: $IMAGE_FILE"
    ./build/trt_feature_extractor "$IMAGE_FILE" ${SIZES[*]}

    echo "✅ Comparing Features for image: $IMAGE_FILE"
    python3 compare_features.py --image "$IMAGE_FILE" --sizes ${SIZES[*]}

done

echo "✅ All image comparisons completed!"