#include <thread>

#include "Random.h"
#include "Initializer.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations) {
  mCamera = ReferenceFrame.mCamera;

  mvKeys1 = ReferenceFrame.mvKeys;
  mvKeyRays1 = ReferenceFrame.mvKeysForInitialize;
}

bool Initializer::InitializeWithRays(const Frame &CurrentFrame,
                                     const vector<int> &vMatches12,
                                     cv::Mat &R21,
                                     cv::Mat &t21,
                                     vector<cv::Point3f> &vP3D,
                                     vector<bool> &vbTriangulated) {
  // Fill structures with current keypoints and matches with reference frame
  // Reference Frame: 1, Current Frame: 2
  mvKeys2 = CurrentFrame.mvKeys;
  mvKeyRays2 = CurrentFrame.mvKeysForInitialize;

  vector<cv::Point2f> vCorrespondences1;
  vector<cv::Point2f> vCorrespondences2;

  mvMatches12.clear();
  mvMatches12.reserve(mvKeys2.size());
  mvbMatched1.resize(mvKeys1.size());
  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] >= 0) {
      mvMatches12.emplace_back(i, vMatches12[i]);
      mvbMatched1[i] = true;
      vCorrespondences1.push_back(mvKeyRays1[i]);
      vCorrespondences2.push_back(mvKeyRays2[vMatches12[i]]);
    } else
      mvbMatched1[i] = false;
  }

  cv::Mat record;
  cv::Mat E = cv::findEssentialMat(vCorrespondences1, vCorrespondences2, 1, cv::Point2f(0, 0), cv::RANSAC, 0.99, 0.006, record);

  vector<Match> currentMatches;
  vector<cv::Point2f> inlier1;
  vector<cv::Point2f> inlier2;

  for(size_t i = 0; i < record.rows; i++){
    if(record.at<uchar>(0, i) != 0){
      currentMatches.push_back(mvMatches12[i]);
      inlier1.push_back(vCorrespondences1[i]);
      inlier2.push_back(vCorrespondences2[i]);
    }
  }

  cv::Mat R, t, mask2;
  cv::recoverPose(E, inlier1, inlier2, R, t, 1, cv::Point2f(0, 0), mask2);

  cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
  cv::Mat Id = cv::Mat::eye(3, 3, CV_32F);
  Id.copyTo(P1.rowRange(0, 3).colRange(0, 3));

  cv::Mat P2(3, 4, CV_32F);
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));

  vP3D.resize(mvKeys1.size());
  vbTriangulated.resize(mvKeys1.size(), false);

  int inliners_num = 0;
  for(size_t i = 0; i < mask2.rows; i++){
    if(mask2.at<uchar>(0, i) != 0){
      cv::Mat temp;
      Triangulate(inlier1[i], inlier2[i], P1, P2, temp);
      vP3D[currentMatches[i].first].x = temp.at<float>(0);
      vP3D[currentMatches[i].first].y = temp.at<float>(1);
      vP3D[currentMatches[i].first].z = temp.at<float>(2);
      vbTriangulated[currentMatches[i].first] = true;
      inliners_num++;
    }
  }

  R21 = R;
  t21 = t;
   //cout<<"initial_ransc nums: "<<inliners_num<<endl;
  if(inliners_num > 100) return true;
  else return false;
}

void Initializer::Triangulate(const cv::Point2f &kp1,
                              const cv::Point2f &kp2,
                              const cv::Mat &P1,
                              const cv::Mat &P2,
                              cv::Mat &x3D) {
  cv::Mat A(4, 4, CV_32F);

  A.row(0) = kp1.x * P1.row(2) - P1.row(0);
  A.row(1) = kp1.y * P1.row(2) - P1.row(1);
  A.row(2) = kp2.x * P2.row(2) - P2.row(0);
  A.row(3) = kp2.y * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

}
