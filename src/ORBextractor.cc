/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2 {

ORBextractor::ORBextractor(tensorflow::Session* _session,torch::DeviceType& _device_type,int _nfeatures, float _scaleFactor, int _nlevels,
                           int _iniThFAST, int _minThFAST) :
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST),session(_session),device(_device_type) {
  mvScaleFactor.resize(nlevels);
  mvLevelSigma2.resize(nlevels);
  mvScaleFactor[0] = 1.0f;
  mvLevelSigma2[0] = 1.0f;
  for (int i = 1; i < nlevels; i++) {
    mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
    mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
  }

  mvInvScaleFactor.resize(nlevels);
  mvInvLevelSigma2.resize(nlevels);
  for (int i = 0; i < nlevels; i++) {
    mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
    mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
  }
}

//for local descriptors
cv::Mat ORBextractor::tfTensor2cvMat(tensorflow::Tensor* inputTensor){            
    tensorflow::TensorShape inputTensorShape = inputTensor->shape();
    //std::cout<<inputTensorShape.dim_size(1)<<"  "<<inputTensorShape.dim_size(2)<<endl;
    cv::Mat output(inputTensorShape.dim_size(1),inputTensorShape.dim_size(2),CV_32FC1);    //定义的时候指定大小，不然动态分配会很慢
    tensorflow::StringPiece tmp_data = inputTensor->tensor_data();
    memcpy(output.data,const_cast<char*>(tmp_data.data()),inputTensorShape.dim_size(1)*inputTensorShape.dim_size(2)* sizeof(float));
    return output;
}

void ORBextractor::cvmat_to_tensor(cv::Mat& img,tensorflow::Tensor* tensor,int rows,int cols){
    cv::resize(img,img,cv::Size(cols,rows),cv::INTER_AREA);
    //img.convertTo(img,CV_32FC1);                  
    float* p=tensor->flat<float>().data();         
    cv::Mat imagePixels(rows,cols,CV_32FC1,p);    
    img.convertTo(imagePixels,CV_32FC1);
}

void ORBextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints,
                              cv::Mat& _descriptors,Eigen::Matrix<float, 1, 4096, Eigen::RowMajor>& global_desc,bool GerateGlobalDesc,
                              int* &keys_tensor,float*& score_tensor) {
  if (_image.empty())
    return;

  Mat image = _image.getMat().rowRange(0,480);
  assert(image.type() == CV_8UC1);

  int width=640,height=480;
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,height,width, 1}));
  auto  imput_tensor_data=input_tensor.tensor<float,4>();  
  cvmat_to_tensor(image,&input_tensor,height,width);

  vector<pair<string, tensorflow::Tensor>> inputs = {  
          { "image", input_tensor}
  };
  vector<tensorflow::Tensor> outputs;                


  tensorflow::Status status;
  if(GerateGlobalDesc)
      status= session->Run(inputs, {"keypoints","local_descriptors","scores","global_descriptor"}, {}, &outputs);
  else
      status= session->Run(inputs, {"keypoints","local_descriptors","scores"}, {}, &outputs);
  if (!status.ok()) {
       cout <<"Failed to run the model!!!"<<status.ToString() << "\n";
            return;
  }

  tensorflow::Tensor desc_tensor = outputs[1];
  _descriptors=tfTensor2cvMat(&desc_tensor);
  //cv::Mat  desc_trans=_descriptors.t();
  //desc_torch = torch::from_blob(desc_trans.data, {1,256,_descriptors.rows }, torch::kFloat32).to(device);

  //tensorflow::Tensor keys = outputs[0];
  keys_tensor=outputs[0].flat<int>().data();         //转为数组,比Eigen::Tensor会快很多
  //keys_torch = torch::from_blob(keys_tensor, {1,_descriptors.rows,2}, torch::kInt).to(device);

//  _keypoints.clear();
//  _keypoints.reserve(_descriptors.rows);
//  for(int j=0;j<_descriptors.rows;j++){
//        _keypoints.push_back(cv::KeyPoint(keys_tensor[j*2],keys_tensor[j*2+1],64));
//  }

  score_tensor =outputs[2].flat<float>().data();
  //score_torch = torch::from_blob(score, {1,_descriptors.rows }, torch::kFloat32).to(device);

  if(GerateGlobalDesc){
      float *descriptor_ptr =outputs[3].flat<float>().data();
      Eigen::Map<Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> > descriptor_map(descriptor_ptr, 4096);  //行优先,符合tensorflow输出格式
      global_desc = descriptor_map;
      //delete []descriptor_ptr;
  }
    //  int vIndicateIndex = 0;
//  int rowCounter = 0;
//  vector<cv::KeyPoint> vTempKeypoint;
//  vTempKeypoint.reserve(mvKeys.size());
//  cv::Mat vTempDesMat(mDescriptors.rows, mDescriptors.cols, mDescriptors.type());
//
//    for (auto it = mvKeys.begin(); it != mvKeys.end(); it++) {
//        if (mMask.at<uchar>((*it).pt.y, (*it).pt.x) != 255) {
//            rowCounter++;
//            continue;
//        }
//
//    vTempKeypoint.push_back((*it));
//    mDescriptors.row(rowCounter).copyTo(vTempDesMat.row(vIndicateIndex));
//
//    vIndicateIndex++;
//    rowCounter++;
//  }
//
//  mvKeys.clear();
//  vTempKeypoint.resize(vIndicateIndex);
//  mvKeys = vTempKeypoint;
//
//  mDescriptors = cv::Mat();
//  vTempDesMat.rowRange(0, vIndicateIndex).copyTo(mDescriptors);
}



} //namespace ORB_SLAM
