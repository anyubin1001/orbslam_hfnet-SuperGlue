/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <Eigen/Core>

#include <torch/torch.h>
#include <torch/script.h>

namespace ORB_SLAM2 {


class ORBextractor {
 public:

  ORBextractor(tensorflow::Session* session,torch::DeviceType& _device_type,int nfeatures, float scaleFactor, int nlevels,
               int iniThFAST, int minThFAST);

  ~ORBextractor() {}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint> &keypoints,
                  cv::Mat& descriptors,Eigen::Matrix<float, 1, 4096, Eigen::RowMajor>& global_desc,bool GerateGlobalDesc,
                  int*& keys_tensor,float*& score_tensor);

  int inline GetLevels() {
    return nlevels;
  }

  float inline GetScaleFactor() {
    return scaleFactor;
  }

  std::vector<float> inline GetScaleFactors() {
    return mvScaleFactor;
  }

  std::vector<float> inline GetInverseScaleFactors() {
    return mvInvScaleFactor;
  }

  std::vector<float> inline GetScaleSigmaSquares() {
    return mvLevelSigma2;
  }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return mvInvLevelSigma2;
  }


 protected:
  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
  tensorflow::Session* session;
  torch::Device device;
  
  void cvmat_to_tensor(cv::Mat& img,tensorflow::Tensor* tensor,int rows,int cols);
  cv::Mat tfTensor2cvMat(tensorflow::Tensor* inputTensor);
};
    //typedef SPextractor ORBextractor;
} //namespace ORB_SLAM

#endif

