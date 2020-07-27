#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include "Frame.h"
using namespace std;
namespace ORB_SLAM2 {
class Initializer {
  typedef pair<int, int> Match;

 public:

  // Fix the reference frame
  explicit Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

  // Computes in parallel a fundamental matrix and a homography
  // Selects a model and tries to recover the motion and the structure from motion
  bool InitializeWithRays(const Frame &CurrentFrame, const vector<int> &vMatches12,
                          cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

  void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

  // Keypoints from Reference Frame (Frame 1)
  vector<cv::KeyPoint> mvKeys1;
  vector<cv::Point2f> mvKeyRays1;

  // Keypoints from Current Frame (Frame 2)
  vector<cv::KeyPoint> mvKeys2;
  vector<cv::Point2f> mvKeyRays2;

  // Current Matches from Reference to Current
  vector<Match> mvMatches12;
  vector<bool> mvbMatched1;

  OCAMCamera mCamera;

};

}
#endif // INITIALIZER_H
