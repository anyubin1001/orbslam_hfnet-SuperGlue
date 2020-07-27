#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW3/src/DBoW3.h"
#include "Thirdparty/models/include/Camera.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include <torch/torch.h>
#include <torch/script.h>

namespace ORB_SLAM2 {
    using namespace std;
#define FRAME_GRID_ROWS 36
#define FRAME_GRID_COLS 64

class MapPoint;

class KeyFrame;

class Frame {
 public:
  Frame();

  // Copy constructor.
  Frame(const Frame &frame);

  // Constructor for Monocular cameras.
  Frame(const cv::Mat &imGray,
        const double &timeStamp,
        torch::DeviceType& _device_type,
        ORBextractor *extractor,
        ORBVocabulary *voc,
        OCAMCamera& camera,
        cv::Mat mask,
        bool GerateGlobalDesc);

  // Extract ORB on the image. 0 for left image and 1 for right image.
  void ExtractORB(int flag, const cv::Mat &im,bool GerateGlobalDesc);

  // Compute Bag of Words representation.
  void ComputeBoW();
  Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> global_desc;
//  at::Tensor keys_torch;
//  at::Tensor desc_torch;
//  at::Tensor score_torch;
  int* keys_tensor;
  float* score_tensor;
  vector<float> keys_for_torch,score_for_torch;


  // Set the camera pose.
  void SetPose(cv::Mat Tcw);

  // Computes rotation, translation and camera center matrices from the camera pose.
  void UpdatePoseMatrices();

  // Returns the camera center.
  inline cv::Mat GetCameraCenter() {
    return mOw.clone();
  }

  // Returns inverse of rotation
  inline cv::Mat GetRotationInverse() {
    return mRwc.clone();
  }

  // Check if a MapPoint is in the frustum of the camera
  // and fill variables of the MapPoint to be used by the tracking
  bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

  // Compute the cell of a keypoint (return false if outside the grid)
  bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

  vector<size_t> GetFeaturesInArea(const float &x,
                                   const float &y,
                                   const float &r,
                                   const int minLevel = -1,
                                   const int maxLevel = -1) const;

 public:
  // Vocabulary used for relocalization.
  ORBVocabulary *mpORBvocabulary;

  // Feature extractor. The right is used only in the stereo case.
  ORBextractor *mpORBextractorLeft;

  // Frame timestamp.
  double mTimeStamp;

  // Calibration matrix and OpenCV distortion parameters.
  OCAMCamera mCamera;
  cv::Mat mMask;

  // Number of KeyPoints.
  int N;

  // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
  // In the stereo case, mvKeysUn is redundant as images must be rectified.
  // In the RGB-D case, RGB images can be distorted.
  std::vector<cv::KeyPoint> mvKeys;
  std::vector<cv::Point3f> mvBearingVector;
  //std::vector<cv::Point2f> mvKeysForInitialize;

  // Bag of Words Vector structures.
  DBoW3::BowVector mBowVec;
  DBoW3::FeatureVector mFeatVec;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat mDescriptors;

  // MapPoints associated to keypoints, NULL pointer if no association.
  std::vector<MapPoint *> mvpMapPoints;

  // Flag to identify outlier associations.
  std::vector<bool> mvbOutlier;

  // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
  static float mfGridElementWidthInv;
  static float mfGridElementHeightInv;
  std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

  // Camera pose.
  cv::Mat mTcw;

  // Current and Next Frame id.
  static long unsigned int nNextId;
  long unsigned int mnId;

  // Reference Keyframe.
  KeyFrame *mpReferenceKF;

  // Scale pyramid info.
  int mnScaleLevels;
  float mfScaleFactor;
  float mfLogScaleFactor;
  vector<float> mvScaleFactors;
  vector<float> mvInvScaleFactors;
  vector<float> mvLevelSigma2;
  vector<float> mvInvLevelSigma2;

  // Undistorted Image Bounds (computed once).
  static float mnMinX;
  static float mnMaxX;
  static float mnMinY;
  static float mnMaxY;

  static bool mbInitialComputations;

 private:

  // Undistort keypoints given OpenCV distortion parameters.
  // Only for the RGB-D case. Stereo must be already rectified!
  // (called in the constructor).
  void UndistortKeyPoints(torch::DeviceType& _device_type);

  // Computes image bounds for the undistorted image (called in the constructor).
  void ComputeImageBounds(const cv::Mat &imLeft);

  // Assign keypoints to the grid for speed up feature matching (called in the constructor).
  void AssignFeaturesToGrid();

  // Rotation, translation and camera center
  cv::Mat mRcw;
  cv::Mat mtcw;
  cv::Mat mRwc;
  cv::Mat mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
