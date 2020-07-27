#include "Frame.h"
#include "Converter.h"
#include <thread>

namespace ORB_SLAM2 {

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame() {}

//Copy Constructor
Frame::Frame(const Frame &frame)
    : mpORBvocabulary(frame.mpORBvocabulary),
      mpORBextractorLeft(frame.mpORBextractorLeft),
      mTimeStamp(frame.mTimeStamp),
      mCamera(frame.mCamera),
      mMask(frame.mMask),
      N(frame.N),
      mvKeys(frame.mvKeys),
      mvBearingVector(frame.mvBearingVector),
      mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()),
      mvpMapPoints(frame.mvpMapPoints),
      mvbOutlier(frame.mvbOutlier),
      mnId(frame.mnId),
      mpReferenceKF(frame.mpReferenceKF),
      mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2),
      //mvKeysForInitialize(frame.mvKeysForInitialize),
      global_desc(frame.global_desc),
      keys_for_torch(frame.keys_for_torch),
      score_for_torch(frame.score_for_torch){
  for (int i = 0; i < FRAME_GRID_COLS; i++)
    for (int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j] = frame.mGrid[i][j];

  if (!frame.mTcw.empty())
    SetPose(frame.mTcw);
}

Frame::Frame(const cv::Mat &imGray,
             const double &timeStamp,
             torch::DeviceType& _device_type,
             ORBextractor *extractor,
             ORBVocabulary *voc,
             OCAMCamera& camera,
             cv::Mat mask,
             bool GerateGlobalDesc)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractor),
      mTimeStamp(timeStamp),
      mCamera(camera),
      mMask(mask){
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0, imGray,GerateGlobalDesc);

  UndistortKeyPoints(_device_type);

  N = mvKeys.size();
  if (mvKeys.empty())
        return;

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(nullptr));
  mvbOutlier = vector<bool>(N, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

    mbInitialComputations = false;
  }

  AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j].reserve(nReserve);

  for (int i = 0; i < N; i++) {
    const cv::KeyPoint &kp = mvKeys[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

void Frame::ExtractORB(int flag, const cv::Mat &im,bool GerateGlobalDesc) {
  (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors,global_desc,GerateGlobalDesc,keys_tensor,score_tensor);

  //std::cout<<"have key nums: "<<mvKeys.size()<<std::endl;

}

void Frame::SetPose(cv::Mat Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() {
  mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
  mRwc = mRcw.t();
  mtcw = mTcw.rowRange(0, 3).col(3);
  mOw = -mRcw.t() * mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
  pMP->mbTrackInView = false;

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  if (PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  Eigen::Vector2d imPt;
  mCamera.spaceToPlane(Eigen::Vector3d(PcX, PcY, PcZ), imPt);
  const float u = imPt[0];
  const float v = imPt[1];

  if (u < mnMinX || u > mnMaxX)
    return false;
  if (v < mnMinY || v > mnMaxY)
    return false;

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);

  if (dist < minDistance || dist > maxDistance)
    return false;

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit)
    return false;

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                        const int maxLevel) const {
  vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= FRAME_GRID_COLS)
    return vIndices;

  const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS)
    return vIndices;

  const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0)
    return vIndices;

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty())
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
        if (bCheckLevels) {
          if (kpUn.octave < minLevel)
            continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel)
              continue;
        }

        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r)
          vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  return !(posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS);
}

void Frame::ComputeBoW() {
  if (mBowVec.empty()) {
    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void Frame::UndistortKeyPoints(torch::DeviceType& _device_type ) {

    mvKeys.clear();
    keys_for_torch.clear();
    score_for_torch.clear();
    mvBearingVector.clear();
    mvKeys.reserve(mDescriptors.rows);
    mvBearingVector.reserve(mDescriptors.rows);
    keys_for_torch.reserve(mDescriptors.rows);
    score_for_torch.reserve(mDescriptors.rows);
    cv::Mat vTempDesMat(mDescriptors.rows, mDescriptors.cols, mDescriptors.type());

    //double avgx=0.0,avgy=0.0;
    //std::cout<<"ori  key nums:"<<mDescriptors.rows<<std::endl;

    for(int j=0;j<mDescriptors.rows;j++){
        int x=keys_tensor[j*2];
        int y=keys_tensor[j*2+1];
        if(((y>400)||(y<200&&(x<60||x>580)))&&(mMask.at<uchar>(y,x)!=255))
            continue;
        int idx=mvKeys.size();
        mDescriptors.row(j).copyTo(vTempDesMat.row(idx));

        score_for_torch.push_back(score_tensor[j]);
        keys_for_torch.push_back((float)x);
        keys_for_torch.push_back((float)y);

        Eigen::Vector3d spacePt;
        mCamera.liftSphere(Eigen::Vector2d(x, y), spacePt);
        mvBearingVector.push_back(cv::Point3f(spacePt[0],spacePt[1],spacePt[2]));

        mvKeys.push_back(cv::KeyPoint(x,y,64));

        //std::cout<<j<<" x: "<<(mvKeys[j].pt.x-320)*1.0/(spacePt[0]/spacePt[2])<<"    y: "<<(mvKeys[j].pt.y-240)*1.0/(spacePt[1]/spacePt[2])<<std::endl;
        //avgx+=abs((mvKeys[j].pt.x-320)*1.0/(spacePt[0]/spacePt[2]));
        //avgy+=abs((mvKeys[j].pt.y-240)*1.0/(spacePt[1]/spacePt[2]));
    }
    mDescriptors = cv::Mat();
    vTempDesMat.rowRange(0, mvKeys.size()).copyTo(mDescriptors);

    std::cout<<"keys nums: "<<mvKeys.size()<<std::endl;     //nms 12 ->750   //nms 16 480    nms 20 320
    //std::cout<<"-----------------------------------"<<std::endl<<"avgx: "<<avgx/mvKeys.size()<<endl<<"avgy: "<<avgy/mvKeys.size()<<endl<<"-----------------------------------"<<endl;
    //delete []score_tensor;
    //delete []keys_tensor;
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
}

} //namespace ORB_SLAM
