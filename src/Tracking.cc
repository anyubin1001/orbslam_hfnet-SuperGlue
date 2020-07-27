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


#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include<iostream>

#include<mutex>
#include "nabo/nabo.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
using namespace std;

namespace ORB_SLAM2 {

#ifdef FUNC_MAP_SAVE_LOAD
Tracking::Tracking(System *pSys,
                   ORBVocabulary *pVoc,
                   tensorflow::Session* session,
                   torch::DeviceType& _device_type,
                   std::shared_ptr<torch::jit::script::Module>& _model,
                   FrameDrawer *pFrameDrawer,
                   MapDrawer *pMapDrawer,
                   Map *pMap,
                   KeyFrameDatabase *pKFDB,
                   const string &strSettingPath,
                   const int sensor,
                   cv::Mat mask,
                   bool bReuseMap) :
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0),model(_model),device_type(_device_type)
#else
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, tensorflow::Session* session,torch::DeviceType& _device_type,
        std::shared_ptr<torch::jit::script::Module>& _model,FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
        KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0),model(_model),device_type(_device_type)
#endif
{
  // Load camera parameters from settings file

  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  OCAMCamera camera;
  camera.readFromYamlFile(strSettingPath);
  mCamera = camera;

  mMask = mask;

  float fps = fSettings["Camera.fps"];
  if (fps == 0)
    fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  cout << endl << "Camera Parameters: " << endl;
  cout << camera << endl;
  cout << "- fps: " << fps << endl;

  int nRGB = fSettings["Camera.RGB"];
  mbRGB = nRGB;

  if (mbRGB)
    cout << "- color order: RGB (ignored if grayscale)" << endl;
  else
    cout << "- color order: BGR (ignored if grayscale)" << endl;

  // Load ORB parameters

  int nFeatures = fSettings["ORBextractor.nFeatures"];
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  int nLevels = fSettings["ORBextractor.nLevels"];
  int fIniThFAST = 0;
  int fMinThFAST = 0;

  mpORBextractorLeft = new ORBextractor(session,device_type,nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

  /*if (sensor == System::MONOCULAR)
    mpIniORBextractor = new ORBextractor(session,2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);*/

  cout << endl << "ORB Extractor Parameters: " << endl;
  cout << "- Number of Features: " << nFeatures << endl;
  cout << "- Scale Levels: " << nLevels << endl;
  cout << "- Scale Factor: " << fScaleFactor << endl;
  //cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
  //cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

#ifdef FUNC_MAP_SAVE_LOAD
  if (bReuseMap)
    mState = LOST;
#endif

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer) {
  mpViewer = pViewer;
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
  mImGray = im;
  cv::resize(mImGray,mImGray,cv::Size(640,720),cv::INTER_AREA);

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  /*if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mCamera, mMask);
  else*/     //((mbOnlyTracking && (mState == LOST || mbVO)) || (!mbOnlyTracking && mState == LOST) )
    mCurrentFrame = Frame(mImGray, timestamp, device_type,mpORBextractorLeft, mpORBVocabulary, mCamera, mMask,mState == LOST ||(mbOnlyTracking && mbVO));

  Track();

  return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  mLastProcessedState = mState;

  // Get Map Mutex -> Map cannot be changed
  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

  if (mState == NOT_INITIALIZED) {

    MonocularInitialization();

    mpFrameDrawer->Update(this);

    if (mState != OK)
      return;
  } else {
    // System is initialized. Track Frame.
    bool bOK;

    // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
    if (!mbOnlyTracking) {
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.

      if (mState == OK) {
          lost_for_the_first_time=1;
        // Local Mapping might have changed some MapPoints tracked in last frame
        CheckReplacedInLastFrame();

        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
          bOK = TrackReferenceKeyFrame();
        } else {
          bOK = TrackWithMotionModel();
          if (!bOK)
            bOK = TrackReferenceKeyFrame();
        }
      } else {
        bOK = Relocalization();
      }
    } else {
      // Localization Mode: Local Mapping is deactivated

      if (mState == LOST) {
        bOK = Relocalization();
      } else {
        if (!mbVO) {
          // In last frame we tracked enough MapPoints in the map

          if (!mVelocity.empty()) {
            bOK = TrackWithMotionModel();
          } else {
            bOK = TrackReferenceKeyFrame();
          }
        } else {
          // In last frame we tracked mainly "visual odometry" points.

          // We compute two camera poses, one from motion model and one doing relocalization.
          // If relocalization is sucessfull we choose that solution, otherwise we retain
          // the "visual odometry" solution.

          bool bOKMM = false;
          bool bOKReloc = false;
          vector<MapPoint *> vpMPsMM;
          vector<bool> vbOutMM;
          cv::Mat TcwMM;
          if (!mVelocity.empty()) {
            bOKMM = TrackWithMotionModel();
            vpMPsMM = mCurrentFrame.mvpMapPoints;
            vbOutMM = mCurrentFrame.mvbOutlier;
            TcwMM = mCurrentFrame.mTcw.clone();
          }
          bOKReloc = Relocalization();

          if (bOKMM && !bOKReloc) {
            mCurrentFrame.SetPose(TcwMM);
            mCurrentFrame.mvpMapPoints = vpMPsMM;
            mCurrentFrame.mvbOutlier = vbOutMM;

            if (mbVO) {
              for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                  mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                }
              }
            }
          } else if (bOKReloc) {
            mbVO = false;
          }

          bOK = bOKReloc || bOKMM;
        }
      }
    }

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // If we have an initial estimation of the camera pose and matching. Track the local map.
    if (!mbOnlyTracking) {
      if (bOK)
        bOK = TrackLocalMap();
    } else {
      // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
      // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
      // the camera we will use the local map again.
      if (bOK && !mbVO)
        bOK = TrackLocalMap();
    }

    if (bOK)
      mState = OK;
    else
      mState = LOST;

    // Update drawer
    mpFrameDrawer->Update(this);

    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc;
      } else
        mVelocity = cv::Mat();

      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

      // Clean VO matches
      /*for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
          if (pMP->Observations() < 1) {
              cout<<endl<<"first-----------------------------"<<endl;
            mCurrentFrame.mvbOutlier[i] = false;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
          }
      }

      // Delete temporal MapPoints
      for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend;
           lit++) {
          cout<<"second***********"<<endl;
        MapPoint *pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();*/

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame())
        CreateNewKeyFrame();

      // We allow points with high innovation (considererd outliers by the Huber Function)
      // pass to the new keyframe, so that bundle adjustment will finally decide
      // if they are outliers or not. We don't want next frame to estimate its position
      // with those points so we discard them in the frame.
      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
      }
    }

    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        cout << "Track lost soon after initialisation, reseting..." << endl;
        mpSystem->Reset();
        return;
      }
    }

    if (!mCurrentFrame.mpReferenceKF)
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

    mLastFrame = Frame(mCurrentFrame);
  }

// Store frame pose information to retrieve the complete camera trajectory afterwards.
  if(mState == OK) {
    if (!mCurrentFrame.mTcw.empty()) {
      cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
      mlRelativeFramePoses.push_back(Tcr);
      mlpReferences.push_back(mpReferenceKF);
      mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
      mlbLost.push_back(mState == LOST);
    } else {
      // This can happen if tracking is lost
      if (!mlRelativeFramePoses.empty())
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
      mlpReferences.push_back(mlpReferences.back());
      mlFrameTimes.push_back(mlFrameTimes.back());
      mlbLost.push_back(mState == LOST);
    }
  }
}

void Tracking::MonocularInitialization() {
  if (!mpInitializer) {
    // Set Reference Frame
    if (mCurrentFrame.mvKeys.size() > 100) {
      mInitialFrame = Frame(mCurrentFrame);
      mInit_img=new cv::Mat(mImGray);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeys.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeys.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeys[i].pt;

      if (mpInitializer)
        delete mpInitializer;

      //mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);
        mpInitializer = new Initializer(mCurrentFrame);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      return;
    }
  } else {
    // Try to initialize
    if ((int) mCurrentFrame.mvKeys.size() <= 100) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer *>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }
    // Find correspondences
    ORBmatcher matcher(0.9, false,model);
    int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);
    //cout<<"SearchForInitialization:"<<nmatches<<endl;
    // Check if there are enough correspondences
    if (nmatches < 100) {       //nms12:100    nms16:80
      delete mpInitializer;
      mpInitializer = static_cast<Initializer *>(NULL);
      return;
    }

    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) {
      cout << "-- initial matrix has been calculated. " << endl;
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
  pKFini->img=new cv::Mat(*mInit_img);
  delete mInit_img;
  KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
  pKFcur->img=new cv::Mat(mImGray);

  //pKFini->ComputeBoW();
  //pKFcur->ComputeBoW();

  // Insert KFs in the map
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0)
      continue;

    //Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);
    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    //Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    //Add to Map
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "CreateInitialMapMonocular New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

  Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
    cout << "Wrong initialization, reseting..." << endl;
    cout << "Observed map points: " << pKFcur->TrackedMapPoints(1) << endl;
    Reset();
    return;
  }

  // Scale initial baseline
  cv::Mat Tc2w = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint *pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  pKFini->isforInit=true;
  pKFcur->isforInit=true;

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

  mpMap->mvpKeyFrameOrigins.push_back(pKFini);

  mState = OK;
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint *pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint *pRep = pMP->GetReplaced();
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracking::TrackReferenceKeyFrame() {
  // Compute Bag of Words vector
  //std::cout<<"Track ReferenceKeyframe"<<std::endl;
  //mCurrentFrame.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  float rationum=0.7;
  if(mbOnlyTracking)
      rationum=0.9;
  ORBmatcher matcher(rationum, mbOnlyTracking,model);
  vector<MapPoint *> vpMapPointMatches;

  int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  if (nmatches < 15)
    return false;

  mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);

  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame *pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();

  mLastFrame.SetPose(Tlr * pRef->GetPose());
}

bool Tracking::TrackWithMotionModel() {
  ORBmatcher matcher(0.9, mbOnlyTracking);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

  // Project points seen in previous frame
  int th = 15;

  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
  }

  if (nmatches < 20)
    return false;

  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > 20;
  }

  return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the frame.
  // We retrieve the local map and try to find matches to points in the local map.

  UpdateLocalMap();

  SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&mCurrentFrame);
  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      }
    }
  }

  int thresh1=mbOnlyTracking?20:50;
  int thresh2=mbOnlyTracking?15:30;
  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < thresh1)
    return false;

  if (mnMatchesInliers < thresh2)
    return false;
  else
    return true;
}

bool Tracking::NeedNewKeyFrame() {
  if (mbOnlyTracking)
    return false;

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    return false;

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    return false;

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2)
    nMinObs = 2;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;

  bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

  // Thresholds
  float thRefRatio = 0.9f;

  // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
  //Condition 1c: tracking is weak
  const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
  const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      return false;
    }
  } else
    return false;
}

void Tracking::CreateNewKeyFrame() {
  if (!mpLocalMapper->SetNotStop(true))
    return;

  KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
  pKF->img=new cv::Mat(mImGray);

  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  mpLocalMapper->InsertKeyFrame(pKF);

  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint *>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;
      }
    }
  }

  int nToMatch = 0;

  // Project points in frame and check its visibility
  for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend;
       vit++) {
    MapPoint *pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
      continue;
    if (pMP->isBad())
      continue;
    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    float rationum=0.8;
    if(mbOnlyTracking)
        rationum=0.9;
    ORBmatcher matcher(rationum,mbOnlyTracking);
    int th = 1;
    // If the camera has been relocalised recently, perform a coarser search
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
      th = 5;
    matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
  }
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    KeyFrame *pKF = *itKF;
    const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

    for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {
      MapPoint *pMP = *itMP;
      if (!pMP)
        continue;
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
        continue;
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  map<KeyFrame *, int> keyframeCounter;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend;
             it++)
          keyframeCounter[it->first]++;
      } else {
        mCurrentFrame.mvpMapPoints[i] = NULL;
      }
    }
  }

  if (keyframeCounter.empty())
    return;

  int max = 0;
  KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
  for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd;
       it++) {
    KeyFrame *pKF = it->first;

    if (pKF->isBad())
      continue;

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  // Include also some not-already-included keyframes that are neighbors to already-included keyframes
  for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80)
      break;

    KeyFrame *pKF = *itKF;

    const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame *pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    const set<KeyFrame *> spChilds = pKF->GetChilds();
    for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
      KeyFrame *pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    KeyFrame *pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }

  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}
bool Tracking::Relocalization2(){
    if(!mbOnlyTracking && lost_for_the_first_time){
        mpKeyFrameDB->CreatKDTreeAndPCA();
        lost_for_the_first_time=0;
    }
    vector<vector<KeyFrame *> > vpCandidateKFVectors = mpKeyFrameDB->DetectRelocalizationCandidatesVector(&mCurrentFrame);

      //一行一个描述子
      cv::Mat CurrentDesc=mCurrentFrame.mDescriptors;
      Eigen::Matrix<float, Eigen::Dynamic,256> CurrentFrame_local_descriptors(CurrentDesc.rows, 256);
      cv::cv2eigen(CurrentDesc,CurrentFrame_local_descriptors);

//    cv::Mat CurrentDesc=mCurrentFrame.mDescriptors.t();
//    //一列一个描述子
//    Eigen::Matrix<float, 256, Eigen::Dynamic> CurrentFrame_local_descriptors(256, CurrentDesc.cols);
//    cv::cv2eigen(CurrentDesc,CurrentFrame_local_descriptors);
    bool bMatch=false;
    for(std::vector<KeyFrame *>& component :vpCandidateKFVectors){
        //当时使用nms=12，key nums>700时需要这一步，当key nums<400 时 可省
        //所以前面必须使用BFS，不能使用DFS
        if (component.size() > 5) {
            component.resize(5);
        }
        else if(component.size()==0)
            continue;
        vector<MapPoint *>  mappoint_indices;
        mappoint_indices.clear();
        mappoint_indices.reserve(5000);
        //一列一个描述子
        //Eigen::MatrixXf db_local_descriptors(256, 5000);
        //一行一个描述子
        cv::Mat db_local_descriptors_Mat(5000,256,CV_32FC1);
        vector<float> keys_for_torch,score_for_torch;
        keys_for_torch.clear();
        score_for_torch.clear();
        keys_for_torch.reserve(5000);
        score_for_torch.reserve(5000);
        vector<MapPoint *> vpMapPointMatches;
        vpMapPointMatches.clear();
        unordered_set<int> visited;
        visited.clear();
        vpMapPointMatches.reserve(mCurrentFrame.N);
        //int mappointnum=0;
         for (KeyFrame * CandidateKF: component){
             const vector<MapPoint *> vpMapPointsKF = CandidateKF->GetMapPointMatches();
             cv::Mat vpMapPointsKFDesc=CandidateKF->mDescriptors;
//             vector<float> keys_for_torch_all=CandidateKF->keys_for_torch;
//             vector<float> score_for_torch_all=CandidateKF->score_for_torch;
             for(int i=0;i<vpMapPointsKF.size();i++){
                 MapPoint *pMP = vpMapPointsKF[i];
                 if(!pMP || pMP->isBad())
                     continue;
                 if(!visited.insert(pMP->mnId).second)
                     continue;
                 mappoint_indices.push_back(pMP);
                 db_local_descriptors_Mat.push_back(vpMapPointsKFDesc.row(i));
//                 score_for_torch.push_back(score_for_torch_all[i]);
//                 keys_for_torch.push_back(keys_for_torch_all[2*i]);
//                 keys_for_torch.push_back(keys_for_torch_all[2*i+1]);
             }
         }
        //db_local_descriptors.conservativeResize(Eigen::NoChange,mappointnum);
        db_local_descriptors_Mat.resize(mappoint_indices.size());
        cv::Mat db_local_descriptors_Mat_trans=db_local_descriptors_Mat.t();
        //一列一个描述子
        Eigen::MatrixXf db_local_descriptors(256, db_local_descriptors_Mat_trans.cols);
        cv::cv2eigen(db_local_descriptors_Mat_trans,db_local_descriptors);
        Nabo::NNSearchF* local_nns = Nabo::NNSearchF::createKDTreeTreeHeap(db_local_descriptors);
        constexpr int kLocalNumNeighbors = 2;
        Eigen::MatrixXi indices(kLocalNumNeighbors, CurrentFrame_local_descriptors.cols());
        Eigen::MatrixXf dists2(kLocalNumNeighbors, CurrentFrame_local_descriptors.cols());
        local_nns->knn(CurrentFrame_local_descriptors, indices, dists2, kLocalNumNeighbors, 0,
                Nabo::NNSearchF::SORT_RESULTS | Nabo::NNSearchF::ALLOW_SELF_MATCH);
        delete local_nns;
        const double kRatioTestValue = 0.9;
        int nmatches=0;
        for (int i = 0; i < indices.cols(); ++i) {
            MapPoint * map1=mappoint_indices[indices(0, i)];
            MapPoint * map2=mappoint_indices[indices(1, i)];
            //std::cout<<"dist: "<<dists2(0, i)<<endl;   //1.0
            if (/*dists2(0, i)<0.7&&*/ (map1->mnId == map2->mnId || dists2(0, i) < kRatioTestValue * dists2(1, i))) {
                vpMapPointMatches[i]=map1;
                nmatches++;
            }
        }
        std::cout<<"namtches:: "<<nmatches<<endl;
        if(nmatches>=15){
            PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vpMapPointMatches);
            pSolver->SetRansacParameters(0.99, 10, 500, 4, 0.5, 5.991);
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;
            cv::Mat Tcw = pSolver->iterate(10, bNoMore, vbInliers, nInliers);
            if (bNoMore) {
                continue;
            }
            bMatch = true;
            break;
//            if (!Tcw.empty() && nInliers > 20) {
//                Tcw.copyTo(mCurrentFrame.mTcw);
//
//                set<MapPoint *> sFound;
//                const int np = vbInliers.size();
//                for (int j = 0; j < np; j++) {
//                    if (vbInliers[j]) {
//                        mCurrentFrame.mvpMapPoints[j] = vpMapPointMatches[j];
//                        sFound.insert(vpMapPointMatches[j]);
//                    } else
//                        mCurrentFrame.mvpMapPoints[j] = NULL;
//                }
//
//                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
//                std::cout<<"ngoods: "<<nGood<<endl;
//                if (nGood < 10) {
//                    continue;
//                }
//                for (int io = 0; io < mCurrentFrame.N; io++) {
//                    if (mCurrentFrame.mvbOutlier[io]) {
//                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
//                    }
//                }
//
//                //todo project the mappoint to current tcw and then get more matches
//                if (nGood >= 10) {  //30
//                    bMatch = true;
//                    break;
//                }
//
//
//            }
        }
    }

    if (!bMatch) {
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        //std::cout<<"return true"<<endl;
        return true;
    }
}

bool Tracking::Relocalization() {
    // Compute Bag of Words Vector
    //mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    if (!mbOnlyTracking && lost_for_the_first_time) {
        mpKeyFrameDB->CreatKDTreeAndPCA();
        lost_for_the_first_time = 0;
    }
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    //cout << "-- The Number of Candidate Relocalized Keyframe Using BOW: " << vpCandidateKFs.size() << endl;
    const int nKFs = vpCandidateKFs.size();
    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, mbOnlyTracking, model);

    //vector<PnPsolver *> vpPnPsolvers;
    //vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);
    //vector<bool> vbDiscarded(nKFs, false);
    bool bMatch = false;
    //ORBmatcher matcher2(0.9, mbOnlyTracking);
    for (int i = 0; i < nKFs; i++) {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            //vbDiscarded[i] = true;
            continue;
        else {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            //std::cout<<"matches:"<< nmatches;
            if (nmatches < 30) {
                //vbDiscarded[i] = true;
                continue;
            } else {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 800, 4, 0.5, 5.991);
                //vpPnPsolvers[i] = pSolver;
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;
                cv::Mat Tcw = pSolver->iterate(10, bNoMore, vbInliers, nInliers);
                if (bNoMore) {
                    continue;
                }
                //std::cout<<"   inlies: "<<nInliers;
                if (!Tcw.empty() && nInliers > 20) {     //todo  nInliers > 20 is added after
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    // filter all useless map points after ransac
                    //------------------------------------------------
                    set<MapPoint *> sFound;
                    const int np = vbInliers.size();
                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }
                    //------------------------------------------------
                    // filter all useless map points after ransac
                    //进行PoseOptimization 必须要有一个较好的初值
                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    //cout << "-- Check ransac optimizer good map points(>50?): " << nGood << endl;
                    //std::cout<<"     nGood:   "<<nGood<<endl;
                    if (nGood < 15) {
                        continue;
                    }
                    // filter out all useless map points after pose optimize.
                    for (int io = 0; io < mCurrentFrame.N; io++) {
                        if (mCurrentFrame.mvbOutlier[io]) {
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
                        }
                    }
                    if(nGood<50) {
                        // filter out all useless map points after pose optimize.
                        vector<KeyFrame *> vpNeighs = pKF->GetBestCovisibilityKeyFrames(10);
                        KeyFrame* pKF1=NULL,*pKF2=NULL;
                        double score1=-1,score2=-1;
                        for (int k=0;k<vpNeighs.size();k++)
                            if(vpNeighs[k]->mnRelocQuery==mCurrentFrame.mnId) {
                                if(vpNeighs[k]->mRelocScore>score1) {
                                    score2=score1;
                                    pKF2=pKF1;
                                    score1=vpNeighs[k]->mRelocScore;
                                    pKF1=vpNeighs[k];
                                }
                                else if(vpNeighs[k]->mRelocScore>score2){
                                    score2=vpNeighs[k]->mRelocScore;
                                    pKF2=vpNeighs[k];
                                }
                            }
                        if(score1>-1){
                            //std::cout<<"run 1"<<endl;
                            int nadditional1 = matcher.GetMoreMatches(mCurrentFrame, pKF1,sFound);
                            if(nadditional1>15){
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io]){
                                        sFound.erase(mCurrentFrame.mvpMapPoints[io]);
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                                    }
                            }
                        }
                        if(score2>-1){
                            //std::cout<<"run 2"<<endl;
                            matcher.GetMoreMatches(mCurrentFrame, pKF2,sFound);
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                            for (int io = 0; io < mCurrentFrame.N; io++)
                                if (mCurrentFrame.mvbOutlier[io])
                                    mCurrentFrame.mvpMapPoints[io] = NULL;
                        }
                    }
                    if (nGood >= 30) {
                        bMatch = true;
                        break;
                    }
                } // optimize pnp result

            }
        }
    }
    if (!bMatch) {
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset() {

  cout << "System Reseting" << endl;
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) {
      std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
  }

  // Reset Local Mapping
  cout << "Reseting Local Mapper...";
  mpLocalMapper->RequestReset();
  cout << " done" << endl;

  // Reset Loop Closing
  cout << "Reseting Loop Closing...";
  mpLoopClosing->RequestReset();
  cout << " done" << endl;

  // Clear BoW Database
  cout << "Reseting Database...";
  mpKeyFrameDB->clear();
  cout << " done" << endl;

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  mState = NO_IMAGES_YET;

  if (mpInitializer) {
    delete mpInitializer;
    mpInitializer = static_cast<Initializer *>(NULL);
  }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();

  if (mpViewer)
    mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag) {
  mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
