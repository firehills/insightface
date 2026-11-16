#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace {

constexpr int kInputSize = 80;
constexpr double kMaxOutOfBoundsRatio = 0.30;
constexpr float kLivenessThreshold = 0.8f;

const std::array<cv::Point2d, 5> kDestinationKeypoints = {{
    {32.03, 38.06},
    {47.89, 37.98},
    {40.01, 47.08},
    {33.50, 56.36},
    {46.63, 56.29},
}};

}  // namespace

// imageBgr must be a non-empty CV_8UC3 image. Keypoints must be ordered as
// left eye, right eye, nose, left mouth corner, right mouth corner.
// The returned vector is float32 data in [1, 3, 80, 80] RGB NCHW order.
std::vector<float> prepareLivenessInput(
    const cv::Mat& imageBgr,
    const std::array<cv::Point2d, 5>& keypoints) {
    if (imageBgr.empty() || imageBgr.type() != CV_8UC3) {
        throw std::invalid_argument("imageBgr must be a non-empty CV_8UC3 image");
    }

    cv::Point2d sourceMean(0.0, 0.0);
    cv::Point2d destinationMean(0.0, 0.0);
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        if (!std::isfinite(keypoints[i].x) || !std::isfinite(keypoints[i].y)) {
            throw std::invalid_argument("Keypoints must be finite");
        }
        sourceMean += keypoints[i];
        destinationMean += kDestinationKeypoints[i];
    }
    sourceMean *= 1.0 / keypoints.size();
    destinationMean *= 1.0 / keypoints.size();

    // Fit x' = a*x - b*y + tx and y' = b*x + a*y + ty, a
    // least-squares similarity transform with uniform scale and rotation.
    double numeratorA = 0.0;
    double numeratorB = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        const cv::Point2d source = keypoints[i] - sourceMean;
        const cv::Point2d destination = kDestinationKeypoints[i] - destinationMean;
        numeratorA += source.x * destination.x + source.y * destination.y;
        numeratorB += source.x * destination.y - source.y * destination.x;
        denominator += source.x * source.x + source.y * source.y;
    }
    if (denominator <= 0.0) {
        throw std::runtime_error("Keypoints do not define a face alignment");
    }

    const double a = numeratorA / denominator;
    const double b = numeratorB / denominator;
    if (!std::isfinite(a) || !std::isfinite(b) || a * a + b * b <= 0.0) {
        throw std::runtime_error("Face alignment produced an invalid transform");
    }
    const cv::Matx23d transform(
        a, -b, destinationMean.x - a * sourceMean.x + b * sourceMean.y,
        b,  a, destinationMean.y - b * sourceMean.x - a * sourceMean.y);

    // Measure how much of the aligned square falls outside the source image.
    cv::Mat sourceMask(imageBgr.size(), CV_32FC1, cv::Scalar(1.0f));
    cv::Mat alignedMask;
    cv::warpAffine(
        sourceMask, alignedMask, transform, cv::Size(kInputSize, kInputSize),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0f));
    const double outOfBoundsRatio =
        std::max(0.0, std::min(1.0, 1.0 - cv::mean(alignedMask)[0]));
    if (outOfBoundsRatio > kMaxOutOfBoundsRatio) {
        throw std::runtime_error(
            "Too much of the aligned face is outside the input image");
    }

    // output the out-of-bounds ratio for debugging purposes
    std::cout << "Out-of-bounds ratio: " << outOfBoundsRatio << std::endl;


    cv::Mat alignedBgr;
    cv::warpAffine(
        imageBgr, alignedBgr, transform, cv::Size(kInputSize, kInputSize),
        cv::INTER_LINEAR, cv::BORDER_REPLICATE);


    // save the aligned face for debugging purposes
    if (!cv::imwrite("aligned_face.png", alignedBgr)) {
        throw std::runtime_error("Unable to save aligned face to aligned_face.png");
    }   

    // OpenCV stores pixels as BGR; the model expects RGB float values in [0, 1].
    constexpr std::size_t kPlaneSize = kInputSize * kInputSize;
    std::vector<float> input(3 * kPlaneSize);
    for (int y = 0; y < kInputSize; ++y) {
        for (int x = 0; x < kInputSize; ++x) {
            const cv::Vec3b pixel = alignedBgr.at<cv::Vec3b>(y, x);
            const std::size_t offset =
                static_cast<std::size_t>(y) * kInputSize + x;
            input[offset] = static_cast<float>(pixel[2]) / 255.0f;
            input[kPlaneSize + offset] =
                static_cast<float>(pixel[1]) / 255.0f;
            input[2 * kPlaneSize + offset] =
                static_cast<float>(pixel[0]) / 255.0f;
        }
    }
    return input;
}

int main(int argc, char** argv) {
    if (argc != 13) {
        std::cerr
            << "Usage: " << argv[0]
            << " <model.onnx> <image> <left-eye-x> <left-eye-y>"
               " <right-eye-x> <right-eye-y> <nose-x> <nose-y>"
               " <left-mouth-x> <left-mouth-y>"
               " <right-mouth-x> <right-mouth-y>\n";
        return 2;
    }

    try {
        const cv::Mat imageBgr = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (imageBgr.empty()) {
            throw std::runtime_error(std::string("Could not read image: ") + argv[2]);
        }

        std::array<cv::Point2d, 5> keypoints;
        for (std::size_t i = 0; i < keypoints.size(); ++i) {
            keypoints[i] = cv::Point2d(
                std::stod(argv[3 + 2 * i]),
                std::stod(argv[4 + 2 * i]));
        }

        std::vector<float> input =
            prepareLivenessInput(imageBgr, keypoints);
        const int inputShape[] = {1, 3, kInputSize, kInputSize};
        const cv::Mat inputTensor(
            4, inputShape, CV_32F, input.data());

        cv::dnn::Net network = cv::dnn::readNetFromONNX(argv[1]);
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        network.setInput(inputTensor);
        const cv::Mat output = network.forward();

        if (output.empty() || output.type() != CV_32FC1 || output.total() != 1) {
            throw std::runtime_error(
                "Expected the ONNX model to return one float live probability");
        }
        const float score = output.ptr<float>()[0];
        if (!std::isfinite(score) || score < 0.0f || score > 1.0f) {
            throw std::runtime_error(
                "The ONNX model returned an invalid live probability");
        }

        std::cout << "Live score: " << score << '\n'
                  << "Is live: "
                  << (score >= kLivenessThreshold ? "yes" : "no") << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Liveness inference failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}