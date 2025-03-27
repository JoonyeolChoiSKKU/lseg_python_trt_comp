#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime.h"
#include "cnpy.h"
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

ICudaEngine* loadEngine(const std::string& enginePath, ILogger& logger) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    delete runtime;

    return engine;
}

void preprocessImage(const cv::Mat& img, float* inputBuffer, int inputWidth, int inputHeight) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);  // BGR -> RGB 변환
    
    cv::Mat resized, floatImg;
    cv::resize(rgb, resized, cv::Size(inputWidth, inputHeight));
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float std[3]  = {0.5f, 0.5f, 0.5f};

    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                float pixel = floatImg.at<cv::Vec3f>(i, j)[c];
                inputBuffer[index++] = (pixel - mean[c]) / std[c];
            }
        }
    }
}

std::vector<float> run_trt_inference(cv::Mat& img, std::string imgName, ICudaEngine* engine, IExecutionContext* context, int inputWidth, int inputHeight) {
    std::string inputTensorName = engine->getIOTensorName(0);
    std::string outputTensorName = engine->getIOTensorName(1);

    int inputSize = 3 * inputWidth * inputHeight * sizeof(float);
    int outputSize = (inputWidth) * (inputHeight) * 512 * sizeof(float);

    void* d_input;
    void* d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    std::vector<float> inputData(3 * inputWidth * inputHeight);
    preprocessImage(img, inputData.data(), inputWidth, inputHeight);

    // // --> 여기서 preprocessed input 저장 (예: 3 x H x W 배열로 저장)
    // std::vector<size_t> inputShape = {3, static_cast<size_t>(inputHeight), static_cast<size_t>(inputWidth)};
    // std::string input_output_path = "inputs/trt_input_" + std::to_string(inputWidth) + "_" + imgName + ".npy";
    // cnpy::npy_save(input_output_path, inputData.data(), inputShape);
    // std::cout << "[INFO] Saved TRT input: " << input_output_path << std::endl;


    cudaMemcpy(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice);

    context->setTensorAddress(inputTensorName.c_str(), d_input);
    context->setTensorAddress(outputTensorName.c_str(), d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::vector<float> outputData((inputWidth) * (inputHeight) * 512);
    cudaMemcpy(outputData.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    return outputData;
}

void process_and_save_feature(const std::string& enginePath, const std::string& img_path, 
                              int size, const std::string& tag) {
    Logger logger;
    ICudaEngine* engine = loadEngine(enginePath, logger);
    if (!engine) {
        std::cerr << "Failed to load engine: " << enginePath << std::endl;
        return;
    }

    IExecutionContext* context = engine->createExecutionContext();
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Error loading image: " << img_path << std::endl;
        return;
    }

    // ✅ 이미지 파일명만 추출
    std::string image_name = img_path.substr(img_path.find_last_of("/") + 1); // 파일명만 추출
    image_name = image_name.substr(0, image_name.find_last_of(".")); // 확장자 제거


    std::vector<float> featureMap = run_trt_inference(img,image_name, engine, context, size, size);
    
    // ✅ outputs 폴더 체크 후 생성
    if (mkdir("outputs", 0777) && errno != EEXIST) {
        std::cerr << "[ERROR] Failed to create output directory!" << std::endl;
        return;
    }

    // ✅ 올바른 경로로 저장
    std::vector<size_t> shape = {1, 512, static_cast<size_t>(size), static_cast<size_t>(size)};
    std::string output_path = "outputs/trt_vit_" + tag + "_" + std::to_string(size) + "_fMap_" + image_name + ".npy";
    cnpy::npy_save(output_path, featureMap.data(), shape);

    std::cout << "[INFO] Saved: " << output_path << std::endl;
    
    delete context;
    delete engine;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <engine.trt> <image_file> <sizes>" << std::endl;
        return -1;
    }
    std::string enginePath = argv[1];
    std::string image_path = argv[2];

    // ✅ tag 추출
    std::string engineFile = enginePath.substr(enginePath.find_last_of("/") + 1);
    std::string tag = "custom";
    if (engineFile.find("ade20k") != std::string::npos)
        tag = "ade20k";
    else if (engineFile.find("fss") != std::string::npos)
        tag = "fss";


    std::vector<int> sizes;
    for (int i = 3; i < argc; ++i) {
        sizes.push_back(std::stoi(argv[i]));
    }

    for (int size : sizes) {
        // std::string enginePath = "models/lseg_image_encoder_" + std::to_string(size) + ".trt";
        process_and_save_feature(enginePath, image_path, size, tag);
    }
    return 0;
}