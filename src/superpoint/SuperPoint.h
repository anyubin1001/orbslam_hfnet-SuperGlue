#ifndef SUPERPOINT_H
#define SUPERPOINT_H


#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <vector>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif


namespace ORB_SLAM2
{

class SPDetector {
public:
    SPDetector(std::shared_ptr<torch::jit::script::Module> _model);
    void detect(cv::Mat &image, bool cuda);
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms,cv::Mat mask);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    std::shared_ptr<torch::jit::script::Module> model;
    torch::Tensor mProb;
    torch::Tensor mDesc;
    bool use_cuda;
};

}  // ORB_SLAM

#endif
