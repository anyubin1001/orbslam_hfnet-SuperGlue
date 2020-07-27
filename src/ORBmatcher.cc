#include "ORBmatcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <stdint-gcc.h>
#include <chrono>
using namespace std;

namespace ORB_SLAM2 {
ORBmatcher::ORBmatcher(float nnratio, bool _isforreloc,std::shared_ptr<torch::jit::script::Module> _model,
        torch::Device _device) : mfNNratio(nnratio),model(_model),device(_device),isforreloc(_isforreloc) {
    if(isforreloc){
        TH_HIGH = 0.81;   //100
        TH_LOW = 0.55;    //50
    }
    else{
        TH_HIGH = 0.49;   //100
        TH_LOW = 0.25;    //50
    }
    if(!torch::cuda::is_available())
        device=torch::Device(torch::kCPU);
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th) {
  int nmatches = 0;

  const bool bFactor = th != 1.0;

  for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    MapPoint *pMP = vpMapPoints[iMP];
    if (!pMP->mbTrackInView)
      continue;

    if (pMP->isBad())
      continue;

    const int &nPredictedLevel = pMP->mnTrackScaleLevel;

    // The size of the window will depend on the viewing direction
    float r = RadiusByViewingCos(pMP->mTrackViewCos);

    if (bFactor)
      r *= th;

    const vector<size_t> vIndices =
        F.GetFeaturesInArea(pMP->mTrackProjX,
                            pMP->mTrackProjY,
                            r * F.mvScaleFactors[nPredictedLevel],
                            nPredictedLevel - 1,
                            nPredictedLevel);

    if (vIndices.empty())
      continue;

    const cv::Mat MPdescriptor = pMP->GetDescriptor();

    float bestDist = 256;
    int bestLevel = -1;
    float bestDist2 = 256;
    int bestLevel2 = -1;
    int bestIdx = -1;

    // Get best and second matches with near keypoints
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      if (F.mvpMapPoints[idx])
        if (F.mvpMapPoints[idx]->Observations() > 0)
          continue;

      const cv::Mat &d = F.mDescriptors.row(idx);

      const float dist = DescriptorDistance(MPdescriptor, d);

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestLevel2 = bestLevel;
        bestLevel = F.mvKeys[idx].octave;
        bestIdx = idx;
      } else if (dist < bestDist2) {
        bestLevel2 = F.mvKeys[idx].octave;
        bestDist2 = dist;
      }
    }

    // Apply ratio to second match (only if best and second are in the same scale level)
    if (bestDist <= TH_HIGH) {
      if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
        continue;

      F.mvpMapPoints[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
  if (viewCos > 0.998)
    return 2.5;
  else
    return 4.0;
}

bool ORBmatcher::CheckDistEpipolarLine(const cv::Point3f &P3M1,     //因为球上模型内参矩阵*相机坐标后还要进入g函数所以得不到基础矩阵F，只有本质矩阵E
                                       const cv::Point3f &P3M2,     //强行把相机坐标转为归一化平面坐标的话可能使得误差不准确，即图像坐标系上其实差了一点点而归一化平面上差得很多，使得误差人为增大
                                       const cv::Mat &E12,           //multipul P314+之前的索引章节 sampson error
                                       const float thresh) {
    const cv::Mat bearing_1 = (cv::Mat_<float>(3, 1) << P3M1.x, P3M1.y, P3M1.z);
    const cv::Mat bearing_2 = (cv::Mat_<float>(3, 1) << P3M2.x, P3M2.y, P3M2.z);

    cv::Mat nom = bearing_1.t() * E12 * bearing_2;
    cv::Mat Ex1 = E12 * bearing_2;
    cv::Mat Etx2 = E12.t() * bearing_1;

    const double den = Ex1.at<float>(0) * Ex1.at<float>(0) +
                       Ex1.at<float>(1) * Ex1.at<float>(1) +
                       Ex1.at<float>(2) * Ex1.at<float>(2) +
                       Etx2.at<float>(0) * Etx2.at<float>(0) +
                       Etx2.at<float>(1) * Etx2.at<float>(1) +
                       Etx2.at<float>(2) * Etx2.at<float>(2);

    if (den == 0.0)
        return false;

    const double dsqr = (nom.at<float>(0) * nom.at<float>(0)) / den;

    return dsqr < thresh;
    //归一化平面上
    /*float x1=P3M1.x*1.0/P3M1.z,y1=P3M1.y*1.0/P3M1.z,x2=P3M2.x*1.0/P3M2.z,y2=P3M2.y*1.0/P3M2.z;
    const float a = x1*F12.at<float>(0,0)+y1*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = x1*F12.at<float>(0,1)+y1*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = x1*F12.at<float>(0,2)+y1*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*x2+b*y2+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];*/
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches) {

  const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();
  vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

    int nmatches = 0;
    std::vector<torch::jit::IValue> inputs;
//    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    cv::Mat desc_trans1=F.mDescriptors.t();
    at::Tensor desc_torch1 = torch::from_blob(desc_trans1.data, {1,256,desc_trans1.cols }, torch::kFloat32).to(device);
    at::Tensor keys_torch1=torch::from_blob(F.keys_for_torch.data(), {1,desc_trans1.cols,2}, torch::kFloat32).to(device);
    at::Tensor score_torch1 = torch::from_blob(F.score_for_torch.data(), {1,desc_trans1.cols }, torch::kFloat32).to(device);

//    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
//    chrono::duration<double> match_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//    cout << "-- match tooks: " << 2*1000 * match_time.count() << "ms"<< endl;

    cv::Mat desc_trans2=pKF->mDescriptors.t();
    at::Tensor desc_torch2 = torch::from_blob(desc_trans2.data, {1,256,desc_trans2.cols }, torch::kFloat32).to(device);
    at::Tensor keys_torch2=torch::from_blob(pKF->keys_for_torch.data(), {1,desc_trans2.cols,2}, torch::kFloat32).to(device);
    at::Tensor score_torch2 = torch::from_blob(pKF->score_for_torch.data(), {1,desc_trans2.cols }, torch::kFloat32).to(device);

    inputs.push_back(keys_torch1);
    inputs.push_back(desc_torch1);
    inputs.push_back(score_torch1);
    inputs.push_back(keys_torch2);
    inputs.push_back(desc_torch2);
    inputs.push_back(score_torch2);

    auto out = model->forward(inputs).toTensor();
    out = out.to(torch::kCPU);
    long* indices=out.data<long>();
    for(int i=0;i<F.mvKeys.size();i++){
        int idx2=indices[i];
        if(idx2>=0){
            MapPoint *pMP = vpMapPointsKF[idx2];
            if(!pMP || pMP->isBad())
                continue;
            vpMapPointMatches[i] = pMP;
            nmatches++;
        }
    }
    //delete []indices;
    return nmatches;

   /* vector<cv::DMatch> good_matches;
    cv::BFMatcher matcher(cv::NORM_L2,true);    //todo modify SuperGlue or do the nabo knn ratio test
    matcher.match(pKF->mDescriptors, F.mDescriptors, good_matches);
//        //cv::FlannBasedMatcher matcher;
//        std::vector<std::vector<cv::DMatch> > good_matches;
//        cv::BFMatcher matcher(cv::NORM_L2);
//        matcher.knnMatch(pKF1->mDescriptors, pKF2->mDescriptors, good_matches,2);

    for(int i=0;i<good_matches.size();i++) {
        //std::cout<<"good:"<<good_matches[i].distance<<"\t";
        int idx1 = good_matches[i].queryIdx;
        MapPoint *pMP = vpMapPointsKF[idx1];
        int idx2 = good_matches[i].trainIdx;
        if (!pMP || pMP->isBad() || vpMapPointMatches[idx2] || good_matches[i].distance>sqrt(TH_LOW))
            continue;
        vpMapPointMatches[idx2] = pMP;     //none use ratio test can be done by nabo KD tree
        nmatches++;
    }
  return nmatches;*/
}

int ORBmatcher::SearchByProjection(KeyFrame *pKF,
                                   cv::Mat Scw,
                                   const vector<MapPoint *> &vpPoints,
                                   vector<MapPoint *> &vpMatched,
                                   int th) {
  // Decompose Scw
  cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  cv::Mat Rcw = sRcw / scw;
  cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
  cv::Mat Ow = -Rcw.t() * tcw;

  // Set of MapPoints already found in the KeyFrame
  set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
  spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

  int nmatches = 0;

  // For each Candidate MapPoint Project and Match
  for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++) {
    MapPoint *pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP))
      continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0)
      continue;

    // Project into Image
    Eigen::Vector2d ptInImage;
    pKF->mCamera.spaceToPlane(Eigen::Vector3d(p3Dc.at<float>(0), p3Dc.at<float>(1), p3Dc.at<float>(2)), ptInImage);
    const float u = ptInImage[0];
    const float v = ptInImage[1];

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    // Depth must be inside the scale invariance region of the point
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = 256;
    float bestDist2= 256;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;
      if (vpMatched[idx])
        continue;

      const int &kpLevel = pKF->mvKeys[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist2=bestDist;
        bestDist = dist;
        bestIdx = idx;
      }
      else if(dist<bestDist2){
          bestDist2=dist;
      }
    }

    if (bestDist <= TH_LOW && bestDist<0.9*bestDist2) {
      vpMatched[bestIdx] = pMP;
      nmatches++;
    }

  }

  return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched,
                                        vector<int> &vnMatches12, int windowSize) {
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeys.size(), -1);
    std::vector<torch::jit::IValue> inputs;
    cv::Mat desc_trans1=F1.mDescriptors.t();
    at::Tensor desc_torch1 = torch::from_blob(desc_trans1.data, {1,256,desc_trans1.cols }, torch::kFloat32).to(device);
    at::Tensor keys_torch1=torch::from_blob(F1.keys_for_torch.data(), {1,desc_trans1.cols,2}, torch::kFloat32).to(device);
    at::Tensor score_torch1 = torch::from_blob(F1.score_for_torch.data(), {1,desc_trans1.cols }, torch::kFloat32).to(device);

    cv::Mat desc_trans2=F2.mDescriptors.t();
    at::Tensor desc_torch2 = torch::from_blob(desc_trans2.data, {1,256,desc_trans2.cols }, torch::kFloat32).to(device);
    at::Tensor keys_torch2=torch::from_blob(F2.keys_for_torch.data(), {1,desc_trans2.cols,2}, torch::kFloat32).to(device);
    at::Tensor score_torch2 = torch::from_blob(F2.score_for_torch.data(), {1,desc_trans2.cols }, torch::kFloat32).to(device);

    inputs.push_back(keys_torch1);
    inputs.push_back(desc_torch1);
    inputs.push_back(score_torch1);
    inputs.push_back(keys_torch2);
    inputs.push_back(desc_torch2);
    inputs.push_back(score_torch2);

    auto out = model->forward(inputs).toTensor();
    out = out.to(torch::kCPU);
    long* indices=out.data<long>();
    for(int i=0;i<F1.mvKeys.size();i++){
        if(indices[i]>=0){
            vnMatches12[i]=indices[i];
            nmatches++;
        }
    }
    //std::cout<<"init matches:"<<nmatches<<std::endl;
    return nmatches;
  /*vnMatches12 = vector<int>(F1.mvKeys.size(), -1);

  vector<float> vMatchedDistance(F2.mvKeys.size(), INT_MAX);
  vector<int> vnMatches21(F2.mvKeys.size(), -1);

  for (size_t i1 = 0, iend1 = F1.mvKeys.size(); i1 < iend1; i1++) {
    cv::KeyPoint kp1 = F1.mvKeys[i1];
    int level1 = kp1.octave;
    if (level1 > 0)
      continue;

    vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize,
                                                    level1, level1);

    if (vIndices2.empty())
      continue;

    cv::Mat d1 = F1.mDescriptors.row(i1);

    float bestDist = INT_MAX;
    float bestDist2 = INT_MAX;
    int bestIdx2 = -1;

    for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
      size_t i2 = *vit;

      cv::Mat d2 = F2.mDescriptors.row(i2);

      float dist = DescriptorDistance(d1, d2);

      if (vMatchedDistance[i2] <= dist)
        continue;

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = i2;
      } else if (dist < bestDist2) {
        bestDist2 = dist;
      }
    }

    if (bestDist <= TH_LOW) {
      if (bestDist < (float) bestDist2 * mfNNratio) {
        if (vnMatches21[bestIdx2] >= 0) {
          vnMatches12[vnMatches21[bestIdx2]] = -1;
          nmatches--;
        }
        vnMatches12[i1] = bestIdx2;
        vnMatches21[bestIdx2] = i1;
        vMatchedDistance[bestIdx2] = bestDist;
        nmatches++;

      }
    }

  }

  //Update prev matched
  for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
    if (vnMatches12[i1] >= 0)
      vbPrevMatched[i1] = F2.mvKeys[vnMatches12[i1]].pt;

  return nmatches;*/
}
cv::FlannBasedMatcher ORBmatcher::CreatKDtree(KeyFrame *pKF1){
    cv::FlannBasedMatcher Flannmatcher(new cv::flann::KMeansIndexParams(32,11,cvflann::FLANN_CENTERS_KMEANSPP,0.2f),
            new cv::flann::SearchParams());
    vector<cv::Mat> vecDesc(1,pKF1->mDescriptors);
    Flannmatcher.add(vecDesc);
    Flannmatcher.train();
    //std::shared_ptr<cv::FlannBasedMatcher> returnvalue=std::make_shared<cv::FlannBasedMatcher>(Flannmatcher);
    return Flannmatcher;
}
cv::FlannBasedMatcher ORBmatcher::CreatKDtreeforLoop(KeyFrame *pKF1){
    cv::FlannBasedMatcher Flannmatcher(new cv::flann::KMeansIndexParams(32,11,cvflann::FLANN_CENTERS_KMEANSPP,0.2f),
                                       new cv::flann::SearchParams());
    cv::Mat temp=pKF1->mDescriptors;
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<cv::Mat> vecDesc;
    int N=temp.rows;
    vecDesc.reserve(N);
    for(int i=0;i<N;i++){
        MapPoint *pMP1 = vpMapPoints1[i];
        if(!pMP1 || pMP1->isBad())
            continue;
        vecDesc.push_back(temp.row(i));
    }
    //vector<cv::Mat> vecDesc(1,pKF1->mDescriptors);
    Flannmatcher.add(vecDesc);
    Flannmatcher.train();
    //std::shared_ptr<cv::FlannBasedMatcher> returnvalue=std::make_shared<cv::FlannBasedMatcher>(Flannmatcher);
    return Flannmatcher;
}
cv::FlannBasedMatcher ORBmatcher::CreatKDtreeforMap(KeyFrame *pKF1){
    cv::FlannBasedMatcher Flannmatcher(new cv::flann::KMeansIndexParams(10,21,cvflann::FLANN_CENTERS_KMEANSPP,0.2f),
                                       new cv::flann::SearchParams());
    cv::Mat temp=pKF1->mDescriptors;
    int N=temp.rows;
    cv::Mat input(temp.rows, temp.cols, temp.type());
    int num=0;
    for(int i=0;i<N;i++){
        MapPoint *pMP1 = pKF1->GetMapPoint(i);
        if(pMP1)
            continue;
        //input.push_back(temp.row(i));
        temp.row(i).copyTo(input.row(num));
        num++;
    }
    input.resize(num);
    vector<cv::Mat> vecDesc(1,input);
    Flannmatcher.add(vecDesc);
    Flannmatcher.train();
    //std::shared_ptr<cv::FlannBasedMatcher> returnvalue=std::make_shared<cv::FlannBasedMatcher>(Flannmatcher);
    return Flannmatcher;
}
int ORBmatcher::SearchLoop(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,cv::FlannBasedMatcher& Flannmatcher){
    int nmatches = 0;
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();

    vpMatches12 = vector<MapPoint *>(pKF1->N, static_cast<MapPoint *>(NULL));

    vector<int> vbMatched1(pKF1->N, -1);

    std::vector<std::vector<cv::DMatch> > good_matches;
    int N=pKF2->N;
    for(int idx2=0;idx2<N;idx2++){
        MapPoint *pMP2 = vpMapPoints2[idx2];
        if(!pMP2 || pMP2->isBad())
            continue;
        good_matches.clear();
        Flannmatcher.knnMatch(pKF2->mDescriptors.row(idx2), good_matches,2);   //query
        if(good_matches[0][0].distance > sqrt(TH_LOW) || (good_matches[0][0].distance>good_matches[0][1].distance*mfNNratio
                                   /*&& good_matches[0][0].trainIdx !=good_matches[0][1].trainIdx*/) )   //ratio:0.75
            continue;
        int idx1 = good_matches[0][0].trainIdx;
        MapPoint *pMP1 = vpMapPoints1[idx1];
        if(!pMP1 || pMP1->isBad())
            continue;
        if(vbMatched1[idx1]>=0){
            vpMatches12[idx1] =static_cast<MapPoint *>(NULL);
            nmatches--;
            continue;
        }
        vpMatches12[idx1] = pMP2;
        vbMatched1[idx1]=idx2;
        nmatches++;
    }
    return nmatches;

//    Flannmatcher.knnMatch(pKF2->mDescriptors, good_matches,2);   //query
//
//    for(int i=0;i<good_matches.size();i++) {
//        if(good_matches[i][0].distance > sqrt(TH_LOW) || good_matches[i][0].distance>good_matches[i][1].distance*mfNNratio)   //ratio:0.8
//            continue;
//        //std::cout<<"good:"<<good_matches[i].distance<<"\t";
//        int idx1 = good_matches[i][0].trainIdx;
//        MapPoint *pMP1 = vpMapPoints1[idx1];
//        int idx2 = good_matches[i][0].queryIdx;
//        MapPoint *pMP2 = vpMapPoints2[idx2];
//        // If there is already a MapPoint skip
//        if (!pMP1 || !pMP2 )    //none use ratio test can use nabo kd tree
//            continue;
//        if(pMP1->isBad()||pMP2->isBad())
//            continue;
//
//        vpMatches12[idx1] = pMP2;
//        nmatches++;
//    }
//    return nmatches;
}
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12) {

    int nmatches = 0;
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();

    vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));

//    vector<cv::DMatch> good_matches;
//    cv::BFMatcher matcher(cv::NORM_L2,true);
//    //cv::BFMatcher matcher(cv::NORM_L2);

        std::vector<std::vector<cv::DMatch> > good_matches;
        cv::FlannBasedMatcher matcher;        //默认使用随机k-d tree，效果一般

//        #include<chrono>
//        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

            //original dbow 2-4 ms and this function at max can endure 30ms so can not use SuperGlue
//      matcher.match(pKF1->mDescriptors, pKF2->mDescriptors, good_matches);               //4ms -40 ms
        matcher.knnMatch(pKF1->mDescriptors, pKF2->mDescriptors, good_matches,2);       //7ms -20ms
        for(int i=0;i<good_matches.size();i++) {
            if(good_matches[i][0].distance > sqrt(TH_LOW) || good_matches[i][0].distance>good_matches[i][1].distance*mfNNratio)   //ratio:0.8
                continue;
            //std::cout<<"good:"<<good_matches[i].distance<<"\t";
            int idx1 = good_matches[i][0].queryIdx;
            MapPoint *pMP1 = vpMapPoints1[idx1];
            int idx2 = good_matches[i][0].trainIdx;
            MapPoint *pMP2 = vpMapPoints2[idx2];
            // If there is already a MapPoint skip
            if (!pMP1 || !pMP2 )    //none use ratio test can use nabo kd tree
                continue;
            if(pMP1->isBad()||pMP2->isBad())
                continue;

            vpMatches12[idx1] = pMP2;
            nmatches++;
        }

//        todo try to match by myself!
//        double min=0, max=0;
//        cv::Point minLoc, maxLoc;
//
//        cv::Mat dotsim=pKF1->mDescriptors * (pKF2->mDescriptors.t());
//        cv::Mat eachrow;
//        cv::Mat eachcol;
//        for(int i=0;i<dotsim.rows;i++){
//            MapPoint *pMP1 = vpMapPoints1[i];
//            if(!pMP1 || pMP1->isBad())
//                continue;
//            eachrow=dotsim.row(i);
//            minMaxLoc(eachrow, &min, &max, &minLoc, &maxLoc);
//            if(max<1-0.5*sqrt(TH_LOW))
//                continue;
//            int idx2=maxLoc.x;
//            MapPoint *pMP2 = vpMapPoints2[idx2];
//            if(!pMP2 || pMP2->isBad())
//                continue;
//            eachcol=dotsim.col(idx2);
//            minMaxLoc(eachcol, &min, &max, &minLoc, &maxLoc);
//            if(maxLoc.y!=i)
//                continue;
//            vpMatches12[i] = pMP2;
//            nmatches++;
//        }

//    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
//    chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
//    cout<<"_______________________:"<<pKF1->mnId<<"   "<<pKF2->mnId<<"     " <<time_used.count()<<"-----------------------------"<<endl;
    return nmatches;
}

int ORBmatcher::SearchForTriangulation_ByKDtree(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo,cv::FlannBasedMatcher& Flannmatcher){
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w * Cw + t2w;

    float x1 = C2.at<float>(0);
    float y1 = C2.at<float>(1);
    float z1 = C2.at<float>(2);

    Eigen::Vector2d imPt1;
    pKF2->mCamera.spaceToPlane(Eigen::Vector3d(x1, y1, z1), imPt1);

    const float ex = imPt1[0];
    const float ey = imPt1[1];
    int nmatches = 0;
    vMatchedPairs.clear();
    vMatchedPairs.reserve(1000);
    std::vector<std::vector<cv::DMatch> > good_matches;
    //Flannmatcher.knnMatch(pKF2->mDescriptors, good_matches,2);   //query
    //matcher.knnMatch(pKF1->mDescriptors, pKF2->mDescriptors, good_matches,2);

    int N=pKF2->N;
    for(int idx2=0;idx2<N;idx2++){
        MapPoint *pMP2 = pKF2->GetMapPoint(idx2);
        if(pMP2)
            continue;
        good_matches.clear();
        Flannmatcher.knnMatch(pKF2->mDescriptors.row(idx2), good_matches,2);   //query
        if((good_matches[0][0].distance>mfNNratio*good_matches[0][1].distance /*&& good_matches[0][0].trainIdx !=good_matches[0][1].trainIdx*/)
               || good_matches[0][0].distance>sqrt(TH_LOW))   //ratio test 0.8
            continue;
        int idx1=good_matches[0][0].trainIdx;
        MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
        if(pMP1)
            continue;
        const cv::Point3f &kp3d1 = cv::Point3f(pKF1->mvBearingVector[idx1].x, pKF1->mvBearingVector[idx1].y, pKF1->mvBearingVector[idx1].z);

        const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];
        const cv::Point3f &kp3d2 = cv::Point3f(pKF2->mvBearingVector[idx2].x, pKF2->mvBearingVector[idx2].y,
                                               pKF2->mvBearingVector[idx2].z);
        const float distex = ex - kp2.pt.x;
        const float distey = ey - kp2.pt.y;
        if ((distex * distex + distey * distey) < 100 * pKF2->mvScaleFactors[kp2.octave])
            continue;

        //if (CheckDistEpipolarLine(kp3d1, kp3d2, F12, pKF2, pKF1->mvScaleFactors[kp1.octave])) {
        if(CheckDistEpipolarLine(kp3d1,kp3d2,F12,1e-2)){
            vMatchedPairs.push_back(make_pair(idx1, idx2));
            nmatches++;
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo) {

//        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
//        int temp=0;
        //Compute epipole in second image
        cv::Mat Cw = pKF1->GetCameraCenter();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        cv::Mat C2 = R2w * Cw + t2w;

        float x1 = C2.at<float>(0);
        float y1 = C2.at<float>(1);
        float z1 = C2.at<float>(2);

        Eigen::Vector2d imPt1;
        pKF2->mCamera.spaceToPlane(Eigen::Vector3d(x1, y1, z1), imPt1);

        const float ex = imPt1[0];
        const float ey = imPt1[1];

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches = 0;
        //vector<int> vMatches12(pKF1->N, -1);
        vMatchedPairs.clear();
        vMatchedPairs.reserve(1000);

        //vector<cv::DMatch> good_matches;
        //cv::BFMatcher matcher(cv::NORM_L2,true);
        //matcher.match(pKF1->mDescriptors, pKF2->mDescriptors, good_matches);
         cv::FlannBasedMatcher matcher;             //todo:transform to nabo   just create the tree for one time ,and then just do the query,do not need to trian!!
         std::vector<std::vector<cv::DMatch> > good_matches;
//       cv::BFMatcher matcher(cv::NORM_L2);
         matcher.knnMatch(pKF1->mDescriptors, pKF2->mDescriptors, good_matches,2);

        //todo done by SuperGlue
//        std::vector<torch::jit::IValue> inputs;
//        cv::Mat desc_trans1=pKF1->mDescriptors.t();
//        at::Tensor desc_torch1 = torch::from_blob(desc_trans1.data, {1,256,desc_trans1.cols }, torch::kFloat32).to(device);
//        at::Tensor keys_torch1=torch::from_blob(pKF1->keys_for_torch.data(), {1,desc_trans1.cols,2}, torch::kFloat32).to(device);
//        at::Tensor score_torch1 = torch::from_blob(pKF1->score_for_torch.data(), {1,desc_trans1.cols }, torch::kFloat32).to(device);
//
//        cv::Mat desc_trans2=pKF2->mDescriptors.t();
//        at::Tensor desc_torch2 = torch::from_blob(desc_trans2.data, {1,256,desc_trans2.cols }, torch::kFloat32).to(device);
//        at::Tensor keys_torch2=torch::from_blob(pKF2->keys_for_torch.data(), {1,desc_trans2.cols,2}, torch::kFloat32).to(device);
//        at::Tensor score_torch2 = torch::from_blob(pKF2->score_for_torch.data(), {1,desc_trans2.cols }, torch::kFloat32).to(device);
//
//        inputs.push_back(keys_torch1);
//        inputs.push_back(desc_torch1);
//        inputs.push_back(score_torch1);
//        inputs.push_back(keys_torch2);
//        inputs.push_back(desc_torch2);
//        inputs.push_back(score_torch2);
//        auto out = model->forward(inputs).toTensor();
//        out = out.to(torch::kCPU);
//        long* indices=out.data<long>();
//        for(int idx1=0;idx1<pKF1->mvKeys.size();idx1++){
//            int idx2=indices[idx1];
//            if(idx2>=0){
//                MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
//                MapPoint *pMP2 = pKF2->GetMapPoint(idx2);
//                if (pMP1||pMP2)
//                    continue;
//                //temp++;
//                const cv::Point3f &kp3d1 = cv::Point3f(pKF1->mvBearingVector[idx1].x, pKF1->mvBearingVector[idx1].y, pKF1->mvBearingVector[idx1].z);
//
//                const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];
//                const cv::Point3f &kp3d2 = cv::Point3f(pKF2->mvBearingVector[idx2].x, pKF2->mvBearingVector[idx2].y,
//                                                       pKF2->mvBearingVector[idx2].z);
//                const float distex = ex - kp2.pt.x;
//                const float distey = ey - kp2.pt.y;
//                if ((distex * distex + distey * distey) < 100 * pKF2->mvScaleFactors[kp2.octave])
//                    continue;
//
//                //if (CheckDistEpipolarLine(kp3d1, kp3d2, F12, pKF2, pKF1->mvScaleFactors[kp1.octave])) {
//                if(CheckDistEpipolarLine(kp3d1,kp3d2,F12,1e-2)){
//                    vMatchedPairs.push_back(make_pair(idx1, idx2));
//                    nmatches++;
//                }
//
//            }
//        }

//         chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
//         chrono::duration<double> match_time = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
//         chrono::duration<double> all_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//        cout << "-- match tooks: " << 1000 * match_time.count() << "ms"<<" ------all time: "<<1000*all_time.count()<<"ms" << endl;     //20
//         std::cout<<"nmatches: "<<nmatches<<"  ori: "<<temp<<std::endl;

        //return nmatches;

        for(int i=0;i<good_matches.size();i++){
            if(good_matches[i][0].distance>mfNNratio*good_matches[i][1].distance|| good_matches[i][0].distance>sqrt(TH_LOW))   //ratio test 0.8
                continue;
            //std::cout<<"good:"<<good_matches[i].distance<<"\t";
            int idx1=good_matches[i][0].queryIdx;
            MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
            int idx2=good_matches[i][0].trainIdx;
            MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

            // If there is already a MapPoint skip
            if (pMP1||pMP2)
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeys[idx1];
            const cv::Point3f &kp3d1 = cv::Point3f(pKF1->mvBearingVector[idx1].x, pKF1->mvBearingVector[idx1].y, pKF1->mvBearingVector[idx1].z);

            const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];
            const cv::Point3f &kp3d2 = cv::Point3f(pKF2->mvBearingVector[idx2].x, pKF2->mvBearingVector[idx2].y,
                                                   pKF2->mvBearingVector[idx2].z);
            const float distex = ex - kp2.pt.x;
            const float distey = ey - kp2.pt.y;
            if ((distex * distex + distey * distey) < 100 * pKF2->mvScaleFactors[kp2.octave])
                continue;

            //if (CheckDistEpipolarLine(kp3d1, kp3d2, F12, pKF2, pKF1->mvScaleFactors[kp1.octave])) {
            if(CheckDistEpipolarLine(kp3d1,kp3d2,F12,1e-2)){
                vMatchedPairs.push_back(make_pair(idx1, idx2));
                nmatches++;
            }
        }
        return nmatches;
}



/*
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo) {
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w * Cw + t2w;

    float x1 = C2.at<float>(0);
    float y1 = C2.at<float>(1);
    float z1 = C2.at<float>(2);

    Eigen::Vector2d imPt1;
    pKF2->mCamera.spaceToPlane(Eigen::Vector3d(x1, y1, z1), imPt1);

    const float ex = imPt1[0];
    const float ey = imPt1[1];

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches = 0;
    vector<int> vMatches12(pKF1->N, -1);
    vector<int> vMatches21(pKF2->N, -1);

    for(int i=0;i<pKF1->mDescriptors.rows;i++)
    {
        float bestdist=TH_LOW;
        float bestdist2=TH_LOW;
        int bestid=-1;
        MapPoint *pMP1 = pKF1->GetMapPoint(i);
        if (pMP1)
            continue;
        const cv::KeyPoint &kp1 = pKF1->mvKeys[i];

        const cv::Point3f &kp3d1 = cv::Point3f(pKF1->mvBearingVector[i].x, pKF1->mvBearingVector[i].y, pKF1->mvBearingVector[i].z);
        const cv::Mat &a=pKF1->mDescriptors.row(i);

        for(int j=0;j<pKF2->mDescriptors.rows;j++){
            MapPoint *pMP2 = pKF2->GetMapPoint(j);
            if (pMP2)
                continue;
            const cv::Mat &b = pKF2->mDescriptors.row(j);
            float dist=DescriptorDistance(a,b);
            if (dist > TH_LOW )
                continue;

            const cv::KeyPoint &kp2 = pKF2->mvKeys[j];

            const cv::Point3f &kp3d2 = cv::Point3f(pKF2->mvBearingVector[j].x, pKF2->mvBearingVector[j].y,
                                                   pKF2->mvBearingVector[j].z);

            const float distex = ex - kp2.pt.x;
            const float distey = ey - kp2.pt.y;
            if ((distex * distex + distey * distey) < 100 * pKF2->mvScaleFactors[kp2.octave])
                continue;

            //if (CheckDistEpipolarLine(kp3d1, kp3d2, F12, pKF2, pKF1->mvScaleFactors[kp1.octave])) {
            if(CheckDistEpipolarLine(kp3d1,kp3d2,F12,1e-2)){
                if(dist>bestdist){
                    if(dist<bestdist2)
                        bestdist2=dist;
                    continue;
                }
                bestid = j;
                bestdist = dist;
            }
        }
        if(bestid>-1 && (bestdist<0.9*bestdist2||bestdist2==TH_LOW)){
            vMatches12[i] = bestid;
            nmatches++;
        }
    }

    cv::Mat Cwother = pKF2->GetCameraCenter();
    cv::Mat R2wother = pKF1->GetRotation();
    cv::Mat t2wother = pKF1->GetTranslation();
    cv::Mat C2other = R2wother * Cwother + t2wother;

    float x1other = C2other.at<float>(0);
    float y1other = C2other.at<float>(1);
    float z1other = C2other.at<float>(2);

    Eigen::Vector2d imPt1other;
    pKF1->mCamera.spaceToPlane(Eigen::Vector3d(x1other, y1other, z1other), imPt1other);

    const float exother = imPt1other[0];
    const float eyother = imPt1other[1];

    for(int i=0;i<pKF2->mDescriptors.rows;i++)
    {
        float bestdist=TH_LOW;
        float bestdist2=TH_LOW;
        int bestid=-1;
        MapPoint *pMP2 = pKF2->GetMapPoint(i);
        if (pMP2)
            continue;
        const cv::KeyPoint &kp2 = pKF2->mvKeys[i];

        const cv::Point3f &kp3d2 = cv::Point3f(pKF2->mvBearingVector[i].x, pKF2->mvBearingVector[i].y, pKF2->mvBearingVector[i].z);
        const cv::Mat &a=pKF2->mDescriptors.row(i);

        for(int j=0;j<pKF1->mDescriptors.rows;j++){
            MapPoint *pMP1 = pKF1->GetMapPoint(j);
            if (pMP1)
                continue;
            const cv::Mat &b = pKF1->mDescriptors.row(j);
            float dist=DescriptorDistance(a,b);
            if (dist > TH_LOW )
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeys[j];

            const cv::Point3f &kp3d1 = cv::Point3f(pKF1->mvBearingVector[j].x, pKF1->mvBearingVector[j].y,
                                                   pKF1->mvBearingVector[j].z);

            const float distex = exother - kp1.pt.x;
            const float distey = eyother - kp1.pt.y;
            if ((distex * distex + distey * distey) < 100 * pKF1->mvScaleFactors[kp1.octave])
                continue;

            //if (CheckDistEpipolarLine(kp3d1, kp3d2, F12, pKF2, pKF1->mvScaleFactors[kp1.octave])) {
            if(CheckDistEpipolarLine(kp3d1,kp3d2,F12,1e-2)){
                if(dist>bestdist){
                    if(dist<bestdist2)
                        bestdist2=dist;
                    continue;
                }
                bestid = j;
                bestdist = dist;
            }
        }
        if(bestid>-1 && (bestdist<0.9*bestdist2||bestdist2==TH_LOW)){
            vMatches21[i] = bestid;
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);
    nmatches=0;
    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
        if (vMatches12[i] < 0 )
            continue;
        if(vMatches21[vMatches12[i]]==i){
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
            nmatches++;
        }

    }

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> gms_match_time = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "-- match tooks " << 1000 * gms_match_time.count() << "ms" << endl;
    std::cout<<"nmatches: "<<nmatches<<std::endl;
    return nmatches;
}*/

int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th) {
  cv::Mat Rcw = pKF->GetRotation();
  cv::Mat tcw = pKF->GetTranslation();

  cv::Mat Ow = pKF->GetCameraCenter();

  int nFused = 0;

  const int nMPs = vpMapPoints.size();

  for (int i = 0; i < nMPs; i++) {
    MapPoint *pMP = vpMapPoints[i];

    if (!pMP)
      continue;

    if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    Eigen::Vector2d ptInImage;
    pKF->mCamera.spaceToPlane(Eigen::Vector3d(p3Dc.at<float>(0), p3Dc.at<float>(1), p3Dc.at<float>(2)), ptInImage);
    const float u = ptInImage[0];
    const float v = ptInImage[1];

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = 256;
    float bestDist2=256;

    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF->mvKeys[idx];

      const int &kpLevel = kp.octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const float &kpx = kp.pt.x;
      const float &kpy = kp.pt.y;
      const float ex = u - kpx;
      const float ey = v - kpy;
      const float e2 = ex * ex + ey * ey;

      if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
        continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist2=bestDist;
        bestDist = dist;
        bestIdx = idx;
      }
      else if(dist<bestDist2)
          bestDist2=dist;
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW /*&& bestDist<bestDist2*0.9*/) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad()) {
          if (pMPinKF->Observations() > pMP->Observations())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF,
                     cv::Mat Scw,
                     const vector<MapPoint *> &vpPoints,
                     float th,
                     vector<MapPoint *> &vpReplacePoint) {
  // Decompose Scw
  cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  cv::Mat Rcw = sRcw / scw;
  cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
  cv::Mat Ow = -Rcw.t() * tcw;

  // Set of MapPoints already found in the KeyFrame
  const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

  int nFused = 0;

  const int nPoints = vpPoints.size();

  // For each candidate MapPoint project and match
  for (int iMP = 0; iMP < nPoints; iMP++) {
    MapPoint *pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP))
      continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    // Project into Image
    Eigen::Vector2d ptInImage;
    pKF->mCamera.spaceToPlane(Eigen::Vector3d(p3Dc.at<float>(0), p3Dc.at<float>(1), p3Dc.at<float>(2)), ptInImage);
    const float u = ptInImage[0];
    const float v = ptInImage[1];

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    // Depth must be inside the scale pyramid of the image
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);

    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    // Compute predicted scale level
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = INT_MAX;
    float bestDist2 = INT_MAX;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++) {
      const size_t idx = *vit;
      const int &kpLevel = pKF->mvKeys[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist2=bestDist;
        bestDist = dist;
        bestIdx = idx;
      }
      else if(dist<bestDist2)
          bestDist2=dist;
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW /*&& bestDist<0.9*bestDist2*/) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad())
          vpReplacePoint[iMP] = pMPinKF;
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th) {
  // Camera 1 from world
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();

  //Camera 2 from world
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  //Transformation between cameras
  cv::Mat sR12 = s12 * R12;
  cv::Mat sR21 = (1.0 / s12) * R12.t();
  cv::Mat t21 = -sR21 * t12;

  const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
  const int N1 = vpMapPoints1.size();

  const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
  const int N2 = vpMapPoints2.size();

  vector<bool> vbAlreadyMatched1(N1, false);
  vector<bool> vbAlreadyMatched2(N2, false);

  for (int i = 0; i < N1; i++) {
    MapPoint *pMP = vpMatches12[i];
    if (pMP) {
      vbAlreadyMatched1[i] = true;
      int idx2 = pMP->GetIndexInKeyFrame(pKF2);
      if (idx2 >= 0 && idx2 < N2)
        vbAlreadyMatched2[idx2] = true;
    }
  }

  vector<int> vnMatch1(N1, -1);
  vector<int> vnMatch2(N2, -1);

  // Transform from KF1 to KF2 and search
  for (int i1 = 0; i1 < N1; i1++) {
    MapPoint *pMP = vpMapPoints1[i1];

    if (!pMP || vbAlreadyMatched1[i1])
      continue;

    if (pMP->isBad())
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc1 = R1w * p3Dw + t1w;
    cv::Mat p3Dc2 = sR21 * p3Dc1 + t21;

    // Depth must be positive
    if (p3Dc2.at<float>(2) < 0.0)
      continue;

    Eigen::Vector2d ptInImage2;
    pKF2->mCamera.spaceToPlane(Eigen::Vector3d(p3Dc2.at<float>(0), p3Dc2.at<float>(1), p3Dc2.at<float>(2)), ptInImage2);
    const float u = ptInImage2[0];
    const float v = ptInImage2[1];
    // Point must be inside the image
    if (!pKF2->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = cv::norm(p3Dc2);

    // Depth must be inside the scale invariance region
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

    // Search in a radius
    const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = INT_MAX;
    float bestDist2= INT_MAX;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF2->mvKeys[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
          bestDist2=bestDist;
        bestDist = dist;
        bestIdx = idx;
      }
      else if( dist <bestDist2)
          bestDist2=dist;
    }

    if (bestDist <= TH_HIGH && bestDist<0.9*bestDist2) {
      vnMatch1[i1] = bestIdx;
    }
  }

  // Transform from KF2 to KF2 and search
  for (int i2 = 0; i2 < N2; i2++) {
    MapPoint *pMP = vpMapPoints2[i2];

    if (!pMP || vbAlreadyMatched2[i2])
      continue;

    if (pMP->isBad())
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc2 = R2w * p3Dw + t2w;
    cv::Mat p3Dc1 = sR12 * p3Dc2 + t12;

    // Depth must be positive
    if (p3Dc1.at<float>(2) < 0.0)
      continue;

    Eigen::Vector2d ptInImage1;
    pKF1->mCamera.spaceToPlane(Eigen::Vector3d(p3Dc1.at<float>(0), p3Dc1.at<float>(1), p3Dc1.at<float>(2)), ptInImage1);
    const float u = ptInImage1[0];
    const float v = ptInImage1[1];

    // Point must be inside the image
    if (!pKF1->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = cv::norm(p3Dc1);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

    // Search in a radius of 2.5*sigma(ScaleLevel)
    const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    float bestDist = INT_MAX;
    float bestDist2 = INT_MAX;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF1->mvKeys[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

      const float dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
          bestDist2=bestDist;
        bestDist = dist;
        bestIdx = idx;
      }
      else if(dist<bestDist2)
          bestDist2=dist;
    }

    if (bestDist <= TH_HIGH && bestDist<0.9*bestDist2) {
      vnMatch2[i2] = bestIdx;
    }
  }

  // Check agreement
  int nFound = 0;

  for (int i1 = 0; i1 < N1; i1++) {
    int idx2 = vnMatch1[i1];

    if (idx2 >= 0) {
      int idx1 = vnMatch2[idx2];
      if (idx1 == i1) {
        vpMatches12[i1] = vpMapPoints2[idx2];
        nFound++;
      }
    }
  }

  return nFound;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) {
//    int nmatches=0;
//    std::vector<torch::jit::IValue> inputs;
//    cv::Mat desc_trans1=CurrentFrame.mDescriptors.t();
//    at::Tensor desc_torch1 = torch::from_blob(desc_trans1.data, {1,256,desc_trans1.cols }, torch::kFloat32).to(device);
//    at::Tensor keys_torch1=torch::from_blob(CurrentFrame.keys_for_torch.data(), {1,desc_trans1.cols,2}, torch::kFloat32).to(device);
//    at::Tensor score_torch1 = torch::from_blob(CurrentFrame.score_for_torch.data(), {1,desc_trans1.cols }, torch::kFloat32).to(device);
//
//    cv::Mat desc_trans2=LastFrame.mDescriptors.t();
//    at::Tensor desc_torch2 = torch::from_blob(desc_trans2.data, {1,256,desc_trans2.cols }, torch::kFloat32).to(device);
//    vector<float > temp1(LastFrame.keys_for_torch.begin(),LastFrame.keys_for_torch.end());
//    vector<float > temp2(LastFrame.score_for_torch.begin(),LastFrame.score_for_torch.end());
//    at::Tensor keys_torch2=torch::from_blob(temp1.data(), {1,desc_trans2.cols,2}, torch::kFloat32).to(device);
//    at::Tensor score_torch2 = torch::from_blob(temp2.data(), {1,desc_trans2.cols }, torch::kFloat32).to(device);
//
//    inputs.push_back(keys_torch1);
//    inputs.push_back(desc_torch1);
//    inputs.push_back(score_torch1);
//    inputs.push_back(keys_torch2);
//    inputs.push_back(desc_torch2);
//    inputs.push_back(score_torch2);
//
//    auto out = model->forward(inputs).toTensor();
//    out = out.to(torch::kCPU);
//    long* indices=out.data<long>();
//    for(int i=0;i<CurrentFrame.mvKeys.size();i++){
//        if(indices[i]>=0){
//            MapPoint *pMP = LastFrame.mvpMapPoints[indices[i]];
//            CurrentFrame.mvpMapPoints[i] = pMP;
//            nmatches++;
//        }
//    }
//    return nmatches;

  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

  const cv::Mat twc = -Rcw.t() * tcw;

  const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

  const cv::Mat tlc = Rlw * twc + tlw;

  for (int i = 0; i < LastFrame.N; i++) {
    MapPoint *pMP = LastFrame.mvpMapPoints[i];

    if (pMP) {
      if (!LastFrame.mvbOutlier[i]) {
        // Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        if (invzc < 0)
          continue;

        Eigen::Vector2d ptInImage;
        CurrentFrame.mCamera.spaceToPlane(Eigen::Vector3d(xc, yc, x3Dc.at<float>(2)), ptInImage);

        float u = ptInImage[0];
        float v = ptInImage[1];

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        int nLastOctave = LastFrame.mvKeys[i].octave;

        // Search in a window. Size depends on scale
        float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

        vector<size_t> vIndices2;

        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 256;
        float bestDist2= 256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
              continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const float dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
              bestDist2=bestDist;
            bestDist = dist;
            bestIdx2 = i2;
          }
          else if(dist<bestDist2)
              bestDist2=dist;
        }
        float ratio=isforreloc?1.1:0.9;
        if (bestDist <= TH_HIGH && bestDist<ratio*bestDist2) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;

        }
      }
    }
  }

  return nmatches;
}
int ORBmatcher::GetMoreMatches(Frame &CurrentFrame,KeyFrame *pKF,set<MapPoint *> &sAlreadyFound){

    const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();
    vector<MapPoint *> vpMapPointsAlreadyMatched = CurrentFrame.mvpMapPoints;

    int nmatches = 0;
    std::vector<torch::jit::IValue> inputs;
    cv::Mat desc_trans1=CurrentFrame.mDescriptors.t();
    at::Tensor desc_torch1 = torch::from_blob(desc_trans1.data, {1,256,desc_trans1.cols }, torch::kFloat32).to(device);
    at::Tensor keys_torch1=torch::from_blob(CurrentFrame.keys_for_torch.data(), {1,desc_trans1.cols,2}, torch::kFloat32).to(device);
    at::Tensor score_torch1 = torch::from_blob(CurrentFrame.score_for_torch.data(), {1,desc_trans1.cols }, torch::kFloat32).to(device);

    cv::Mat desc_trans2=pKF->mDescriptors.t();
    at::Tensor desc_torch2 = torch::from_blob(desc_trans2.data, {1,256,desc_trans2.cols }, torch::kFloat32).to(device);
    at::Tensor keys_torch2=torch::from_blob(pKF->keys_for_torch.data(), {1,desc_trans2.cols,2}, torch::kFloat32).to(device);
    at::Tensor score_torch2 = torch::from_blob(pKF->score_for_torch.data(), {1,desc_trans2.cols }, torch::kFloat32).to(device);

    inputs.push_back(keys_torch1);
    inputs.push_back(desc_torch1);
    inputs.push_back(score_torch1);
    inputs.push_back(keys_torch2);
    inputs.push_back(desc_torch2);
    inputs.push_back(score_torch2);

    auto out = model->forward(inputs).toTensor();
    out = out.to(torch::kCPU);
    long* indices=out.data<long>();
    for(int i=0;i<CurrentFrame.mvKeys.size();i++){
        if(vpMapPointsAlreadyMatched[i])
            continue;
        int idx2=indices[i];
        if(idx2>=0){
            MapPoint *pMP = vpMapPointsKF[idx2];
            if(!pMP || pMP->isBad()|| sAlreadyFound.count(pMP))
                continue;
            vpMapPointsAlreadyMatched[i] = pMP;
            sAlreadyFound.insert(pMP);
            nmatches++;
        }
    }
    //std::cout<<"more matches: "<<nmatches<<endl;
    return nmatches;
}


int ORBmatcher::SearchByProjection(Frame &CurrentFrame,           //todo:mCurrentFrame<->vpCandidateKFs modify match NN or SuperGlue
                                   KeyFrame *pKF,
                                   const set<MapPoint *> &sAlreadyFound,
                                   const float th,
                                   const float ORBdist) {
  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);
  const cv::Mat Ow = -Rcw.t() * tcw;

  const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint *pMP = vpMPs[i];

    if (pMP) {
      if (!pMP->isBad() && !sAlreadyFound.count(pMP)) {
        //Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        Eigen::Vector2d ptInImage;
        CurrentFrame.mCamera.spaceToPlane(Eigen::Vector3d(xc, yc, x3Dc.at<float>(2)), ptInImage);
        const float u = ptInImage[0];
        const float v = ptInImage[1];

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        // Compute predicted scale level
        cv::Mat PO = x3Dw - Ow;
        float dist3D = cv::norm(PO);

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
          continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        // Search in a window
        const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

        const vector<size_t>
            vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 256;
        float bestDist2=256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const float dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
              bestDist2=bestDist;
            bestDist = dist;
            bestIdx2 = i2;
          }
          else if(dist<bestDist2)
              bestDist2=dist;
        }

        if (bestDist <= ORBdist && bestDist<0.9*bestDist2) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;
        }

      }
    }
  }

  return nmatches;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
float ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    double t=a.dot(b);
    //float dist=sqrt(2-2*t);
    return 2-2*t;
}

} //namespace ORB_SLAM
