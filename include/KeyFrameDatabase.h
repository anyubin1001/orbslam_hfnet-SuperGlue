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

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>
#include <map>
#include <Eigen/Core>
#include "nabo/nabo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"

#include <mutex>
#ifdef FUNC_MAP_SAVE_LOAD
#include "BoostArchiver.h"
#endif

namespace ORB_SLAM2 {
    using namespace std;
class KeyFrame;
class Frame;

class KeyFrameDatabase {
 public:

  KeyFrameDatabase(ORBVocabulary *voc);

  void add(KeyFrame *pKF);

  void erase(KeyFrame *pKF);

  void clear();
  //bool  cmp(pair<float, KeyFrame *> a,pair<float, KeyFrame *> b);

  int new_image_index;
  // Loop Detection
  std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame *pKF, float minScore);

  // Relocalization
  std::vector<KeyFrame *> DetectRelocalizationCandidates(Frame *F);
  std::vector<std::vector<KeyFrame *> > DetectRelocalizationCandidatesVector(Frame *F);

    Eigen::MatrixXf image_descriptors_;
    Nabo::NNSearchF* nns_;
    //cv::Mat image_descriptors_;
    vector<KeyFrame*> image_descriptors_keyframe_indices;
    void CreatKDTreeAndPCA();
    //unordered_map<KeyFrame*,int> image_descriptors_keyframe_indices;
    float GlobalDistance(Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> a,Eigen::Matrix<float, 1, 4096, Eigen::RowMajor> b);

#ifdef FUNC_MAP_SAVE_LOAD
 public:
  // for serialization
  KeyFrameDatabase() {}
  void SetORBvocabulary(ORBVocabulary *porbv) { mpVoc = porbv; }

 private:
  // serialize is recommended to be private
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version);
#endif
 protected:

  // Associated vocabulary
  ORBVocabulary *mpVoc;

  // Inverted file
  std::vector<list<KeyFrame *> > mvInvertedFile;

  // Mutex
  std::mutex mMutex;
};

} //namespace ORB_SLAM

#endif
