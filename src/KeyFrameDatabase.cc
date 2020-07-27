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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW3/src/DBoW3.h"
#include <algorithm>
#include "nabo/nabo.h"
#include <set>
#include <mutex>

using namespace std;

namespace ORB_SLAM2 {

KeyFrameDatabase::KeyFrameDatabase(ORBVocabulary *voc) :
    mpVoc(voc) {
    new_image_index=0;
    //resize操作会执行析构函数：元素的值会被改变，conservativeResize则不会
    image_descriptors_.resize(4096, 0);   //转为一列一个描述子
    //image_descriptors_=cv::Mat();
    image_descriptors_keyframe_indices.clear();
    nns_=NULL;
    //Nabo::NNSearchF* nns = Nabo::NNSearchF::createKDTreeLinearHeap(M);

  mvInvertedFile.resize(voc->size());
}
bool cmp(std::pair<float,ORB_SLAM2::KeyFrame *> a ,std::pair<float, ORB_SLAM2::KeyFrame *> b)
{
    return a.first > b.first;

}
void KeyFrameDatabase::CreatKDTreeAndPCA()
{
    unique_lock<mutex> lock(mMutex);
    //delete nns_;   //注意回收内存
    nns_ = Nabo::NNSearchF::createKDTreeLinearHeap(image_descriptors_);
}

void KeyFrameDatabase::add(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutex);

  /*for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
    mvInvertedFile[vit->first].push_back(pKF);*/
     image_descriptors_.conservativeResize(Eigen::NoChange,new_image_index+1);
     image_descriptors_.col(new_image_index) = pKF->global_desc;
    //CHECK_EQ(image_descriptors_.rows, new_image_index);
    // image_descriptors_keyframe_indices.insert({pKF,new_image_index});
    image_descriptors_keyframe_indices.push_back(pKF);
    //image_descriptors_.push_back(pKF->global_desc);
    new_image_index++;
}

void KeyFrameDatabase::erase(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutex);

  // Erase elements in the Inverse File for the entry
  /*for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++) {
    // List of keyframes that share the word
    list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

    for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
      if (pKF == *lit) {
        lKFs.erase(lit);
        break;
      }
    }
  }*/
    auto erase_pos=find(image_descriptors_keyframe_indices.begin(),image_descriptors_keyframe_indices.end(),pKF);
    int erase_indice=erase_pos-image_descriptors_keyframe_indices.begin();
    //image_descriptors_keyframe_indices[erase_indice]=static_cast<KeyFrame *>(NULL);
    image_descriptors_keyframe_indices.erase(erase_pos);
   // int erase_indice=image_descriptors_keyframe_indices[pKF];
   if(erase_indice<new_image_index-1)
         image_descriptors_.block(0,erase_indice,4096,new_image_index-erase_indice-1) =
            image_descriptors_.block(0,erase_indice+1,4096,new_image_index-erase_indice-1);
    image_descriptors_.conservativeResize(Eigen::NoChange,--new_image_index);
    //image_descriptors_keyframe_indices.erase(pKF);

}

void KeyFrameDatabase::clear() {
    new_image_index=0;
    image_descriptors_keyframe_indices.clear();
    image_descriptors_.resize(4096,0);
    //image_descriptors_=cv::Mat();

  mvInvertedFile.clear();
  mvInvertedFile.resize(mpVoc->size());
}

float KeyFrameDatabase::GlobalDistance(Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> a,Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> b)   //similarity
{
    float dist=a.dot(b);
    //return 2-2*dist;
    return dist;
}

vector<KeyFrame *> KeyFrameDatabase::DetectLoopCandidates(KeyFrame *pKF, float minScore) {
  set<KeyFrame *> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
  //list<KeyFrame *> lKFsSharingWords;
    list<pair<float, KeyFrame *> > lScoreAndMatch;

    Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> Current_Globaldesc= pKF->global_desc;
    Eigen::MatrixXf  distance;
    float best_score=0;
  // Search all keyframes that share a word with current keyframes
  // Discard keyframes connected to the query keyframe
  {
    unique_lock<mutex> lock(mMutex);

    /*for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++) {
      list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

      for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
        KeyFrame *pKFi = *lit;
        if (pKFi->mnLoopQuery != pKF->mnId) {
          pKFi->mnLoopWords = 0;
          if (!spConnectedKeyFrames.count(pKFi)) {
            pKFi->mnLoopQuery = pKF->mnId;
            lKFsSharingWords.push_back(pKFi);
          }
        }
        pKFi->mnLoopWords++;
      }
    }*/
     distance=Current_Globaldesc*image_descriptors_;

      for(int i=0;i<distance.cols();i++){
          float score=distance(0,i);
          KeyFrame *pKFi =image_descriptors_keyframe_indices[i];
          pKFi->mLoopScore=score;
          pKFi->mnLoopQuery = pKF->mnId;
          if(score>best_score)  best_score=score;
          if(score>=minScore && !spConnectedKeyFrames.count(pKFi) )     //time cost ,try to just do for have connected map points
              lScoreAndMatch.push_back(make_pair(score,pKFi ));

      }
      //std::cout<<"lScoreAndMatch.size():"<<lScoreAndMatch.size()<<std::endl;
  }

//
//  if (lKFsSharingWords.empty())
//    return vector<KeyFrame *>();
//
//  // Only compare against those keyframes that share enough words
//  int maxCommonWords = 0;
//  for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++) {
//    if ((*lit)->mnLoopWords > maxCommonWords)
//      maxCommonWords = (*lit)->mnLoopWords;
//  }
//
//  int minCommonWords = maxCommonWords * 0.8f;
//
//  int nscores = 0;
//
//  // Compute similarity score. Retain the matches whose score is higher than minScore
//  for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++) {
//    KeyFrame *pKFi = *lit;
//
//    if (pKFi->mnLoopWords > minCommonWords) {
//      nscores++;
//
//      float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
//
//      pKFi->mLoopScore = si;
//      if (si >= minScore)
//        lScoreAndMatch.push_back(make_pair(si, pKFi));
//    }
//  }

  if (lScoreAndMatch.empty())
    return vector<KeyFrame *>();

  list<pair<float, KeyFrame *> > lAccScoreAndMatch;
  float bestAccScore = 0.0;
  //std::sort(lScoreAndMatch.begin(),lScoreAndMatch.end());
    lScoreAndMatch.sort(cmp);
  //float ration =lScoreAndMatch.size()>10? 0.9:0.8;
  // Lets now accumulate score by covisibility
  int cmpnum=0;
  for (list<pair<float, KeyFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend && cmpnum<10;
       it++,cmpnum++) {
      if(it->first<best_score*0.8)  continue;
    KeyFrame *pKFi = it->second;
    vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

    float k=1.0/(vpNeighs.size()+1);
    float bestScore = it->first;
    float accScore = k*it->first;
    KeyFrame *pBestKF = pKFi;
    for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
      KeyFrame *pKF2 = *vit;
      if (pKF2->mnLoopQuery == pKF->mnId /*&& pKF2->mnLoopWords > minCommonWords*/) {
        accScore += k*pKF2->mLoopScore;
        if (pKF2->mLoopScore > bestScore) {
          pBestKF = pKF2;
          bestScore = pKF2->mLoopScore;
        }
      }
    }

    lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
    if (accScore > bestAccScore)
      bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;

  set<KeyFrame *> spAlreadyAddedKF;     //for one
  vector<KeyFrame *> vpLoopCandidates;
  vpLoopCandidates.reserve(lAccScoreAndMatch.size());

  for (list<pair<float, KeyFrame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    if (it->first > minScoreToRetain) {
      KeyFrame *pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpLoopCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }
  //std::cout<<"after all vpLoopCandidates.size():"<<vpLoopCandidates.size()<<std::endl;
  return vpLoopCandidates;
}


vector<vector<KeyFrame *> > KeyFrameDatabase::DetectRelocalizationCandidatesVector(Frame *F) {
    Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> Current_Globaldesc= F->global_desc;
    Eigen::MatrixXf  distance;
    constexpr int kNumNeighbors = 10;
    Eigen::VectorXi indices(kNumNeighbors);
    Eigen::VectorXf dists2(kNumNeighbors);
    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);
        nns_->knn(Current_Globaldesc, indices, dists2, kNumNeighbors, 0, Nabo::NNSearchF::SORT_RESULTS | Nabo::NNSearchF::ALLOW_SELF_MATCH);
    }
    vector<vector<KeyFrame* > > components;
    std::unordered_set<int> visited;

    //这样写只是取出了与第一个存在共视关系的
//    for(int i=0;i<indices.size();i++) {
//        KeyFrame* pKFi=image_descriptors_keyframe_indices[indices(i)];
//        if (!visited.insert(pKFi->mnId).second)
//            continue;
//        components.resize(components.size() + 1);
//        components.back().push_back(pKFi);
//        for(int j=i+1;j<indices.size();j++){
//            KeyFrame* pKFj=image_descriptors_keyframe_indices[indices(j)];
//            if(visited.count(pKFj->mnId))
//                continue;
//            vector<KeyFrame *> vpNeighs = pKFi->GetVectorCovisibleKeyFrames();
//            if(find(vpNeighs.begin(),vpNeighs.end(),pKFj)!=vpNeighs.end()){
//                components.back().push_back(pKFj);
//                visited.insert(pKFj->mnId);
//            }
//        }
//    }
    //使用BFS 找到存在相互共视关系的
    std::unordered_set<int> frame_ids;
    for (int i = 0; i < indices.size(); ++i) {
        frame_ids.insert(image_descriptors_keyframe_indices[indices(i)]->mnId);
    }
    for (int i = 0; i < indices.size(); ++i) {
        KeyFrame* pKFi=image_descriptors_keyframe_indices[indices(i)];
        if (visited.count(pKFi->mnId) > 0u) {
            continue;
        }

        components.resize(components.size() + 1);

        std::queue<KeyFrame *> queue;
        queue.push(pKFi);
        while (!queue.empty()) {
            KeyFrame* pKF_query = queue.front();
            queue.pop();

            if (!visited.insert(pKF_query->mnId).second) {
                continue;
            }

            components.back().push_back(pKF_query);
            //vector<KeyFrame *> vpNeighs = pKF_query->GetVectorCovisibleKeyFrames();
            vector<KeyFrame *> vpNeighs = pKF_query->GetBestCovisibilityKeyFrames(15);
            for (KeyFrame * connected_frame: vpNeighs) {
                if (frame_ids.count(connected_frame->mnId) > 0u && visited.count(connected_frame->mnId) == 0u) {
                    queue.push(connected_frame);
                }
            }
        }
    }

    return components;
}

vector<KeyFrame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F) {
    list<pair<float, KeyFrame *> > lScoreAndMatch;

    Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> Current_Globaldesc= F->global_desc;
    Eigen::MatrixXf  distance;

  // Search all keyframes that share a word with current frame
  {
    unique_lock<mutex> lock(mMutex);

//    for (DBoW3::BowVector::const_iterator vit = F->mBowVec.begin(), vend = F->mBowVec.end(); vit != vend; vit++) {
//      list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];
//
//      for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
//        KeyFrame *pKFi = *lit;
//        if (pKFi->mnRelocQuery != F->mnId) {
//          pKFi->mnRelocWords = 0;
//          pKFi->mnRelocQuery = F->mnId;
//          lKFsSharingWords.push_back(pKFi);
//        }
//        pKFi->mnRelocWords++;
//      }
//    }
        constexpr int kNumNeighbors = 10;
        Eigen::VectorXi indices(kNumNeighbors);
        Eigen::VectorXf dists2(kNumNeighbors);

        nns_->knn(Current_Globaldesc, indices, dists2, kNumNeighbors, 0, Nabo::NNSearchF::SORT_RESULTS | Nabo::NNSearchF::ALLOW_SELF_MATCH);

        for(int i=0;i<indices.size();i++) {
            KeyFrame* pKFi=image_descriptors_keyframe_indices[indices(i)];
            float score=1.0-0.5*dists2(i);
            //std::cout<<"ori dist2: "<<dists2(i)<<"     score:"<<score;
            pKFi->mRelocScore = score;
            pKFi->mnRelocQuery = F->mnId;
            lScoreAndMatch.push_back(make_pair(score, pKFi));
        }
  }
//  if (lKFsSharingWords.empty())
//    return vector<KeyFrame *>();
//
//  // Only compare against those keyframes that share enough words
//  int maxCommonWords = 0;
//  for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++) {
//    if ((*lit)->mnRelocWords > maxCommonWords)
//      maxCommonWords = (*lit)->mnRelocWords;
//  }
//
//  int minCommonWords = maxCommonWords * 0.8f;
//
//
//  int nscores = 0;
//
//  // Compute similarity score.
//  for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++) {
//    KeyFrame *pKFi = *lit;
//
//    if (pKFi->mnRelocWords > minCommonWords) {
//      nscores++;
//      float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
//      pKFi->mRelocScore = si;
//      lScoreAndMatch.push_back(make_pair(si, pKFi));
//    }
//  }

  if (lScoreAndMatch.empty())
    return vector<KeyFrame *>();

  list<pair<float, KeyFrame *> > lAccScoreAndMatch;
  float bestAccScore = 0.0;

  // Lets now accumulate score by covisibility
  // 每一组选出一个代表即可，从而去除共视的keyframe
  for (list<pair<float, KeyFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend;
       it++) {
    KeyFrame *pKFi = it->second;
    vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
    float k=1.0/(vpNeighs.size()+1);

    float bestScore = it->first;
    float accScore = k*bestScore;
    KeyFrame *pBestKF = pKFi;
    for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
      KeyFrame *pKF2 = *vit;
      if (pKF2->mnRelocQuery != F->mnId){
          Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> pKF2_global_desc=pKF2->global_desc;
          float scoretemp=GlobalDistance(Current_Globaldesc,pKF2_global_desc);
          //std::cout<<"   myself similar score: "<<scoretemp;
          pKF2->mRelocScore=scoretemp;
          pKF2->mnRelocQuery=F->mnId;
      }

      accScore += k*pKF2->mRelocScore;
      if (pKF2->mRelocScore > bestScore) {
        pBestKF = pKF2;
        bestScore = pKF2->mRelocScore;
      }

    }
    lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
    if (accScore > bestAccScore)
      bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;
  set<KeyFrame *> spAlreadyAddedKF;
  vector<KeyFrame *> vpRelocCandidates;
  vpRelocCandidates.reserve(lAccScoreAndMatch.size());
  for (list<pair<float, KeyFrame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    const float &si = it->first;
    if (si > minScoreToRetain) {
      KeyFrame *pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpRelocCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }
  //std::cout<<"vpRelocCandidates.size: "<<vpRelocCandidates.size()<<endl;
  return vpRelocCandidates;
}

#ifdef FUNC_MAP_SAVE_LOAD
template<class Archive>
void KeyFrameDatabase::serialize(Archive &ar, const unsigned int version) {
  // don't save associated vocabulary, KFDB restore by created explicitly from a new ORBvocabulary instance
  // inverted file
  ar & mvInvertedFile;
  ar & image_descriptors_;
  ar & image_descriptors_keyframe_indices;
  // don't save mutex
}
template void KeyFrameDatabase::serialize(boost::archive::binary_iarchive &, const unsigned int);
template void KeyFrameDatabase::serialize(boost::archive::binary_oarchive &, const unsigned int);
#endif
} //namespace ORB_SLAM
