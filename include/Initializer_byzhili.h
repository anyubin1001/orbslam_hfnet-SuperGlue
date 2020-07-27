#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "Frame.h"
#include "Thirdparty/models/include/Camera.h"
#include "Thirdparty/opengv/include/opengv/types.hpp"

namespace ORB_SLAM2{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer{
  typedef pair<int,int> Match;

 public:

  // Fix the reference frame
  Initializer(const Frame &ReferenceFrame);

  // Computes in parallel a fundamental matrix and a homography
  // Selects a model and tries to recover the motion and the structure from motion
  bool InitializeWithRays(const Frame &CurrentFrame,
                  const vector<int> &vMatches12,
                  cv::Mat &R21,
                  cv::Mat &t21,
                  vector<cv::Point3f> &vP3D,
                  vector<bool> &vbTriangulated);

 private:
  int CheckRT(const cv::Mat &R,
              const cv::Mat &t,
              const vector<cv::KeyPoint> &vKeys1,
              const vector<cv::KeyPoint> &vKeys2,
              const vector<cv::Point3f> &vKeyRays1,
              const vector<cv::Point3f> &vKeyRays2,
              const vector<Match> &vMatches12,
              vector<bool> &vbMatchesInliers,
              OCAMCamera camera,
              vector<cv::Point3f> &vP3D,
              float th2,
              vector<bool> &vbGood,
              float &parallax);

  void Triangulate(const cv::Point3f &ray1,
                   const cv::Point3f &ray2,
                   const cv::Mat &P1,
                   const cv::Mat &P2,
                   cv::Mat &x3D);

  // Keypoints from Reference Frame (Frame 1)
  vector<cv::KeyPoint> mvKeys1;
  vector<cv::Point3f> mvRays1;

  // Keypoints from Current Frame (Frame 2)
  vector<cv::KeyPoint> mvKeys2;
  vector<cv::Point3f> mvRays2;

  // Current Matches from Reference to Current
  vector<Match> mvMatches12;
  vector<bool> mvbMatched1;

  // Calibration
  OCAMCamera mCamera;
};

} //namespace ORB_SLAM
#endif // INITIALIZER_H
