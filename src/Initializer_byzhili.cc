#include "Thirdparty/DBoW2/DUtils/Random.h"
#include "Initializer.h"
#include "ORBmatcher.h"

#include "Thirdparty/opengv/include/opengv/types.hpp"
#include "Thirdparty/opengv/include/opengv/sac/Ransac.hpp"
#include "Thirdparty/opengv/include/opengv/relative_pose/CentralRelativeAdapter.hpp"
#include "Thirdparty/opengv/include/opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp"

#include <cmath>

#define M_PID   3.1415926535897932384626433832795028841971693993
const double RHOd = 180.0 / M_PID;

namespace ORB_SLAM2 {
Initializer::Initializer(const Frame &ReferenceFrame) {
  mCamera = ReferenceFrame.mCamera;

  mvKeys1 = ReferenceFrame.mvKeys;
  mvRays1 = ReferenceFrame.mvBearingVector;
}

bool Initializer::InitializeWithRays(const Frame &CurrentFrame,
                                     const vector<int> &vMatches12,
                                     cv::Mat &R21,
                                     cv::Mat &t21,
                                     vector<cv::Point3f> &vP3D,
                                     vector<bool> &vbTriangulated) {
  mvKeys2 = CurrentFrame.mvKeys;
  mvRays2 = CurrentFrame.mvBearingVector;

  mvMatches12.clear();
  mvMatches12.reserve(mvKeys2.size());
  mvbMatched1.resize(mvKeys1.size());
  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] >= 0) {
      mvMatches12.emplace_back(i, vMatches12[i]);
      mvbMatched1[i] = true;
    } else
      mvbMatched1[i] = false;
  }

  {
    opengv::bearingVectors_t correspondence_bearing1;
    opengv::bearingVectors_t correspondence_bearing2;

    int initCorrespondence = 0;
    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
      if (vMatches12[i] >= 0) {
        correspondence_bearing1.push_back(opengv::bearingVector_t(mvRays1[i].x,
                                                                  mvRays1[i].y,
                                                                  mvRays1[i].z));
        correspondence_bearing2.push_back(opengv::bearingVector_t(mvRays2[vMatches12[i]].x,
                                                                  mvRays2[vMatches12[i]].y,
                                                                  mvRays2[vMatches12[i]].z));
        initCorrespondence++;
      }
    }

    opengv::relative_pose::CentralRelativeAdapter adapter(correspondence_bearing1, correspondence_bearing2);

    opengv::sac::Ransac<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;
    std::shared_ptr<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> relposeproblem_ptr(
        new opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem(
            adapter,
            opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::STEWENIUS));

    ransac.sac_model_ = relposeproblem_ptr;
    ransac.threshold_ = 0.0001;
    ransac.max_iterations_ = 200;
    ransac.computeModel();

    Eigen::Matrix3d opengvR = ransac.model_coefficients_.block<3, 3>(0, 0);
    Eigen::Vector3d opengvt = ransac.model_coefficients_.block<3, 1>(0, 3);
    Eigen::Vector3d opengvt21 = -opengvR * opengvt;

    cv::Mat R_cv, t_cv;
    R_cv = (cv::Mat_<float>(3, 3) <<
                                  opengvR(0, 0), opengvR(0, 1), opengvR(0, 2),
        opengvR(1, 0), opengvR(1, 1), opengvR(1, 2),
        opengvR(2, 0), opengvR(2, 1), opengvR(2, 2));
    t_cv = (cv::Mat_<float>(3, 1) << opengvt21(0), opengvt21(1), opengvt21(2));

    vector<bool> vInliersMatches(mvKeys1.size(), false);
    for (size_t i = 0; i < ransac.inliers_.size(); i++) {
      int idx = ransac.inliers_[i];
      vInliersMatches[idx] = true;
    }

    vbTriangulated.resize(mvKeys1.size(), false);
    vP3D.resize(mvKeys1.size());

    float parallax = cos(1.0 / RHOd);
    int vGood = CheckRT(R_cv,
                        t_cv,
                        mvKeys1,
                        mvKeys2,
                        mvRays1,
                        mvRays2,
                        mvMatches12,
                        vInliersMatches,
                        mCamera,
                        vP3D,
                        5,
                        vbTriangulated,
                        parallax);

    R21 = R_cv;
    t21 = t_cv;
cout<<"good:"<<vGood<<endl;
    if (vGood >= 130) {
      return true;
    } else return false;
  }
}

int Initializer::CheckRT(const cv::Mat &R,
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
                         float &parallax) {
  vector<float> vCosParallax;
  vCosParallax.reserve(vKeyRays1.size());

  // Camera 1 Projection Matrix K[I|0]
  // [I|0] K is not needed
  cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
  cv::Mat Id = cv::Mat::eye(3, 3, CV_32F);
  Id.copyTo(P1.rowRange(0, 3).colRange(0, 3));

  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

  // Camera 2 Projection Matrix K[R|t]
  // [R|t] K is not needed
  cv::Mat P2(3, 4, CV_32F);
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));

  cv::Mat O2 = -R.t() * t;

  int nGood = 0;

  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (!vbMatchesInliers[vMatches12[i].first])
      continue;

    const cv::Point3f &ray1 = vKeyRays1[vMatches12[i].first];
    const cv::Point3f &ray2 = vKeyRays2[vMatches12[i].second];

    const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
    const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];

    cv::Mat p3dC1;

    Triangulate(ray1, ray2, P1, P2, p3dC1);

    if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2))) {
      vbGood[vMatches12[i].first] = false;
      continue;
    }

    // Check parallax
    cv::Mat normal1 = p3dC1 - O1;
    float dist1 = cv::norm(normal1);

    cv::Mat normal2 = p3dC1 - O2;
    float dist2 = cv::norm(normal2);

    float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

    // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
    if ((p3dC1.at<float>(2)) <= 0 && cosParallax > parallax)
      continue;

    // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
    cv::Mat p3dC2 = R * p3dC1 + t;
    if ((p3dC2.at<float>(2)) <= 0 && cosParallax > parallax)
      continue;

    float im1x, im1y;
    Eigen::Vector2d ptInImage1;
    camera.spaceToPlane(Eigen::Vector3d(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2)), ptInImage1);
    im1x = ptInImage1[0];
    im1y = ptInImage1[1];

    float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

    if (squareError1 > th2)
      continue;

    float im2x, im2y;
    Eigen::Vector2d ptInImage2;
    camera.spaceToPlane(Eigen::Vector3d(p3dC2.at<float>(0), p3dC2.at<float>(1), p3dC2.at<float>(2)), ptInImage2);
    im2x = ptInImage2[0];
    im2y = ptInImage2[1];

    float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

    if (squareError2 > th2)
      continue;

    vCosParallax.push_back(cosParallax);

    vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
    vbGood[vMatches12[i].first] = true;
    nGood++;
  }

  float medianParallax = 0;
  if (nGood > 0) {
    sort(vCosParallax.begin(), vCosParallax.end());
    size_t idx = min(50, int(vCosParallax.size() - 1));
    medianParallax = vCosParallax[idx];
  }

  if (medianParallax > parallax) {
    return nGood;
  } else {
    return 0;
  }
}

void Initializer::Triangulate(const cv::Point3f &ray1,
                              const cv::Point3f &ray2,
                              const cv::Mat &P1,
                              const cv::Mat &P2,
                              cv::Mat &x3D) {
  //Adapted vector-based triangulation method
  cv::Mat A(4, 4, CV_32F);
  const float &x1 = ray1.x, &y1 = ray1.y, &z1 = ray1.z;
  const float &x2 = ray2.x, &y2 = ray2.y, &z2 = ray2.z;

  A.row(0) = x1 * P1.row(2) - z1 * P1.row(0);
  A.row(1) = y1 * P1.row(2) - z1 * P1.row(1);
  A.row(2) = x2 * P2.row(2) - z2 * P2.row(0);
  A.row(3) = y2 * P2.row(2) - z2 * P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

}
