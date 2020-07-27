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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2 {

LocalMapping::LocalMapping(Map *pMap, const float bMonocular,std::shared_ptr<torch::jit::script::Module>& _model) :
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),model(_model) {
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) {
  mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker) {
  mpTracker = pTracker;
}

void LocalMapping::Run() {

  mbFinished = false;

  while (1) {
    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // BoW conversion and insertion in Map
      ProcessNewKeyFrame();

      // Check recent MapPoints
      MapPointCulling();

      // Triangulate new MapPoints
      CreateNewMapPoints();

      if (!CheckNewKeyFrames()) {
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
      }

      mbAbortBA = false;

      if (!CheckNewKeyFrames() && !stopRequested()) {
        // Local BA
        if (mpMap->KeyFramesInMap() > 2)
          Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

        // Check redundant local Keyframes
        KeyFrameCulling();
      }

      mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
    } else if (Stop()) {
      // Safe area to stop
      while (isStopped() && !CheckFinish()) {
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
      }
      if (CheckFinish())
        break;
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(true);

    if (CheckFinish())
      break;

    std::this_thread::sleep_for(std::chrono::microseconds(3000));
  }

  SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexNewKFs);
  mlNewKeyFrames.push_back(pKF);
  mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mMutexNewKFs);
  return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame() {
  {
    unique_lock<mutex> lock(mMutexNewKFs);
    mpCurrentKeyFrame = mlNewKeyFrames.front();
    mlNewKeyFrames.pop_front();
  }

  // Compute Bags of Words structures
  //mpCurrentKeyFrame->ComputeBoW();

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

  for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        if (/*!pMP->IsInKeyFrame(mpCurrentKeyFrame)*/ !mpCurrentKeyFrame->isforInit) {
          pMP->AddObservation(mpCurrentKeyFrame, i);
          pMP->UpdateNormalAndDepth();
          pMP->ComputeDistinctiveDescriptors();
        }
        else // this can only happen for new stereo points inserted by the Tracking
        {
            mlpRecentAddedMapPoints.push_back(pMP);//??????????when to here  and then how to map
        }
      }
    }
  }

    if(!mpCurrentKeyFrame->isforInit) {
        // Update links in the Covisibility Graph
        mpCurrentKeyFrame->UpdateConnections();

        // Insert Keyframe in Map
        mpMap->AddKeyFrame(mpCurrentKeyFrame);
    }
}

void LocalMapping::MapPointCulling() {
  // Check Recent Added MapPoints
  list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
  const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

  int nThObs;
  if (mbMonocular)
    nThObs = 2;
  else
    nThObs = 3;
  const int cnThObs = nThObs;

  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint *pMP = *lit;
    if (pMP->isBad()) {
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (pMP->GetFoundRatio() < 0.25f) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 3)
      lit = mlpRecentAddedMapPoints.erase(lit);
    else
      lit++;
  }
}


void LocalMapping::CreateNewMapPoints(){
  // Retrieve neighbor keyframes in covisibility graph
  int nn=20;
  //int nn=10;                //20
  const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

  ORBmatcher matcher(0.8 /*0.6*/,false);

  cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
  cv::Mat Tcw1(3,4,CV_32F);
  Rcw1.copyTo(Tcw1.colRange(0,3));
  tcw1.copyTo(Tcw1.col(3));
  cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

  const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

//#include<chrono>
//    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    cv::FlannBasedMatcher Flannmatcher=matcher.CreatKDtree(mpCurrentKeyFrame);
    //cv::FlannBasedMatcher Flannmatcher=matcher.CreatKDtreeforMap(mpCurrentKeyFrame);
//    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
//    chrono::duration<double> match_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//    double d1=0,d2=match_time.count(),d3=match_time.count();

  // Search matches with epipolar restriction and triangulate
  for(size_t i=0; i<vpNeighKFs.size(); i++)
  {
    if(i>0 && CheckNewKeyFrames())
      return;

    KeyFrame* pKF2 = vpNeighKFs[i];

    // Check first that baseline is not too short
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2-Ow1;
    const float baseline = cv::norm(vBaseline);

    const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
    const float ratioBaselineDepth = baseline/medianDepthKF2;

    if(ratioBaselineDepth<0.01)
      continue;

    // Compute Essential Matrix
    cv::Mat E12 = ComputeF12(mpCurrentKeyFrame,pKF2);

    // Search matches that fullfil epipolar constraint
    vector<pair<size_t,size_t> > vMatchedIndices;
    //ORB bow-based impl
    matcher.SearchForTriangulation_ByKDtree(mpCurrentKeyFrame, pKF2, E12, vMatchedIndices, false,Flannmatcher);

    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = pKF2->GetTranslation();
    cv::Mat Tcw2(3,4,CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    for(int ikp=0; ikp<nmatches; ikp++)
    {
      const int &idx1 = vMatchedIndices[ikp].first;
      const int &idx2 = vMatchedIndices[ikp].second;

      // Get keypoints
      const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeys[idx1];
      const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];
      //cout<<"localmapping  "<<mpCurrentKeyFrame->mnId <<"   "<<mpCurrentKeyFrame->mvKeysForInitialize.size()<<endl;
      //cout<<pKF2->mnId<<"   "<<pKF2->mvKeysForInitialize.size()<<endl;
      // Get bearing vectors
      /*const cv::Point3f &kpRay1 = mpCurrentKeyFrame->mvBearingVector[idx1];
      const cv::Point3f &kpRay2 = pKF2->mvBearingVector[idx2];

      // Check parallax between rays
      const float &rx1 = kpRay1.x, &ry1 = kpRay1.y, &rz1 = kpRay1.z;
      const float &rx2 = kpRay2.x, &ry2 = kpRay2.y, &rz2 = kpRay2.z;

      cv::Mat ray1 = Rwc1 * cv::Mat(kpRay1);
      cv::Mat ray2 = Rwc2 * cv::Mat(kpRay2);*/
        cv::Point3f xn1_point=mpCurrentKeyFrame->mvBearingVector[idx1];
        cv::Point3f xn2_point=pKF2->mvBearingVector[idx2];

        cv::Mat xn1 = (cv::Mat_<float>(3,1) << xn1_point.x/xn1_point.z, xn1_point.y/xn1_point.z, 1.0);
        cv::Mat xn2 = (cv::Mat_<float>(3,1) << xn2_point.x/xn2_point.z, xn2_point.y/xn2_point.z, 1.0);
     // const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
     // float cosParallaxStereo = cosParallaxRays+1;

        cv::Mat ray1 = Rwc1*xn1;
        cv::Mat ray2 = Rwc2*xn2;
        const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

      cv::Mat x3D;
      if(cosParallaxRays>0 && cosParallaxRays<0.9998)
      {
        // Linear Triangulation Method
          /*cv::Mat A(4,4,CV_32F);
          A.row(0) = rx1*(Tcw1.row(1)+Tcw1.row(2)) - (ry1+rz1)*Tcw1.row(0);
          A.row(1) = ry1*(Tcw1.row(0)+Tcw1.row(2)) - (rx1+rz1)*Tcw1.row(1);
          A.row(2) = rx2*(Tcw2.row(1)+Tcw2.row(2)) - (ry2+rz2)*Tcw2.row(0);
          A.row(3) = ry2*(Tcw2.row(0)+Tcw2.row(2)) - (rx2+rz2)*Tcw2.row(1);

          cv::Mat w,u,vt;
          cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

          x3D = vt.row(3).t();*/
          cv::Mat A(4,4,CV_32F);
          A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
          A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
          A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
          A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);
          cv::Mat w,u,vt;
          cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
          x3D = vt.row(3).t();

          /*cv::Mat A(4, 4, CV_32F);
          A.row(0) = rx1 * P1.row(2) - P1.row(0);
          A.row(1) = ry1* P1.row(2) - P1.row(1);
          A.row(2) = rx2 * P2.row(2) - P2.row(0);
          A.row(3) = ry2 * P2.row(2) - P2.row(1);

          cv::Mat u, w, vt;
          cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
          x3D = vt.row(3).t();*/


        if(x3D.at<float>(3)==0)
          continue;

        // Euclidean coordinates
        x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

      }
      else
        continue; //No stereo and very low parallax

      cv::Mat x3Dt = x3D.t();

      //Not necessary in fisheye case
      //Check triangulation in front of cameras
      cv::Mat x3Dc1 = Rcw1 * x3D+tcw1;
      float d1 = x3Dc1.at<float>(2);
      if(d1 <= 0)
        continue;

      cv::Mat x3Dc2 = Rcw2 * x3D + tcw2;
      float d2 = x3Dc2.at<float>(2);
      if(d2 <= 0)
        continue;

      const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
      const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
      const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
      const float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
      float u1, v1;
      Eigen::Vector2d ptInImage1;
      mpCurrentKeyFrame->mCamera.spaceToPlane(Eigen::Vector3d(x1, y1, z1), ptInImage1);
      u1 = ptInImage1[0];
      v1 = ptInImage1[1];

      float errX1 = u1 - kp1.pt.x;
      float errY1 = v1 - kp1.pt.y;

      if((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
        continue;

      const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
      const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
      const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
      const float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
      float u2, v2;
      Eigen::Vector2d ptInImage2;
      mpCurrentKeyFrame->mCamera.spaceToPlane(Eigen::Vector3d(x2, y2, z2), ptInImage2);
      u2 = ptInImage2[0];
      v2 = ptInImage2[1];

      float errX2 = u2 - kp2.pt.x;
      float errY2 = v2 - kp2.pt.y;
      if((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
        continue;

      //Check scale consistency
      cv::Mat normal1 = x3D-Ow1;
      float dist1 = cv::norm(normal1);

      cv::Mat normal2 = x3D-Ow2;
      float dist2 = cv::norm(normal2);

      if(dist1==0 || dist2==0)
        continue;

      const float ratioDist = dist2/dist1;
      const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

      if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
        continue;

      // Triangulation is succesfull
      MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

      pMP->AddObservation(mpCurrentKeyFrame,idx1);
      pMP->AddObservation(pKF2,idx2);

      mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
      pKF2->AddMapPoint(pMP,idx2);

      pMP->ComputeDistinctiveDescriptors();

      pMP->UpdateNormalAndDepth();

      mpMap->AddMapPoint(pMP);
      mlpRecentAddedMapPoints.push_back(pMP);
    }
  }
}

void LocalMapping::SearchInNeighbors() {
  // Retrieve neighbor keyframes
  int nn = 10;
  if (mbMonocular)
    nn = 20;
  const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
  vector<KeyFrame *> vpTargetKFs;
  for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++) {
    KeyFrame *pKFi = *vit;
    if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
      continue;
    vpTargetKFs.push_back(pKFi);
    pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

    // Extend to some second neighbors
    const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
    for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end();
         vit2 != vend2; vit2++) {
      KeyFrame *pKFi2 = *vit2;
      if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId
          || pKFi2->mnId == mpCurrentKeyFrame->mnId)
        continue;
      vpTargetKFs.push_back(pKFi2);
    }
  }


  // Search matches by projection from current KF in target KFs
  ORBmatcher matcher;
  vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++) {
    KeyFrame *pKFi = *vit;

    matcher.Fuse(pKFi, vpMapPointMatches);
  }

  // Search matches by projection from target KFs in current KF
  vector<MapPoint *> vpFuseCandidates;
  vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

  for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++) {
    KeyFrame *pKFi = *vitKF;

    vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

    for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP;
         vitMP++) {
      MapPoint *pMP = *vitMP;
      if (!pMP)
        continue;
      if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
        continue;
      pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
      vpFuseCandidates.push_back(pMP);
    }
  }

  matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);


  // Update points
  vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
      }
    }
  }

  // Update connections in covisibility graph
  mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  cv::Mat R12 = R1w * R2w.t();
  cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

  cv::Mat t12x = SkewSymmetricMatrix(t12);

  return t12x * R12;
}

void LocalMapping::RequestStop() {
  unique_lock<mutex> lock(mMutexStop);
  mbStopRequested = true;
  unique_lock<mutex> lock2(mMutexNewKFs);
  mbAbortBA = true;
}

bool LocalMapping::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  if (mbStopRequested && !mbNotStop) {
    mbStopped = true;
    cout << "Local Mapping STOP" << endl;
    return true;
  }

  return false;
}

bool LocalMapping::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

bool LocalMapping::stopRequested() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopRequested;
}

void LocalMapping::Release() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);
  if (mbFinished)
    return;
  mbStopped = false;
  mbStopRequested = false;
  for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
    delete *lit;
  mlNewKeyFrames.clear();

  cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames() {
  unique_lock<mutex> lock(mMutexAccept);
  return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
  unique_lock<mutex> lock(mMutexAccept);
  mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag) {
  unique_lock<mutex> lock(mMutexStop);

  if (flag && mbStopped)
    return false;

  mbNotStop = flag;

  return true;
}

void LocalMapping::InterruptBA() {
  mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
  // in at least other 3 keyframes (in the same or finer scale)
  // We only consider close stereo points
  vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

  for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++) {
    KeyFrame *pKF = *vit;
    if (pKF->mnId == 0)
      continue;
    const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

    int nObs = 3;
    const int thObs = nObs;
    int nRedundantObservations = 0;
    int nMPs = 0;
    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad()) {
          nMPs++;
          if (pMP->Observations() > thObs) {
            const int &scaleLevel = pKF->mvKeys[i].octave;
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            int nObs = 0;
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
                 mit != mend; mit++) {
              KeyFrame *pKFi = mit->first;
              if (pKFi == pKF)
                continue;
              const int &scaleLeveli = pKFi->mvKeys[mit->second].octave;

              if (scaleLeveli <= scaleLevel + 1) {
                nObs++;
                if (nObs >= thObs)
                  break;
              }
            }
            if (nObs >= thObs) {
              nRedundantObservations++;
            }
          }
        }
      }
    }

    if (nRedundantObservations > 0.9 * nMPs)
      pKF->SetBadFlag();
  }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
  return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
      v.at<float>(2), 0, -v.at<float>(0),
      -v.at<float>(1), v.at<float>(0), 0);
}

void LocalMapping::RequestReset() {
  {
    unique_lock<mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  while (1) {
    {
      unique_lock<mutex> lock2(mMutexReset);
      if (!mbResetRequested)
        break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(3000));
  }
}

void LocalMapping::ResetIfRequested() {
  unique_lock<mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlNewKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();
    mbResetRequested = false;
  }
}

void LocalMapping::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool LocalMapping::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void LocalMapping::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
  unique_lock<mutex> lock2(mMutexStop);
  mbStopped = true;
}

bool LocalMapping::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

} //namespace ORB_SLAM
