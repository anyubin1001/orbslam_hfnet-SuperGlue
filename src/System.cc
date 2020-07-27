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

#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>


static bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}

namespace ORB_SLAM2 {
#ifdef FUNC_MAP_SAVE_LOAD
System::System(const string & model_path,const string & SuperGlue_model_path,const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer, cv::Mat mask, bool is_save_map_)
    : mSensor(sensor), mpViewer(static_cast<Viewer *>(NULL)), mbReset(false), mbActivateLocalizationMode(false),
      mbDeactivateLocalizationMode(false), is_save_map(is_save_map_)
#else
System::System(const string & model_path,const string & SuperGlue_model_path,const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
#endif
{
  // Output welcome message
  cout << "Wide-angle Camera Version" << endl;

  cout << "Input sensor was set to: ";

  if (mSensor == MONOCULAR)
    cout << "Monocular" << endl;

  //Check settings file
  cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "Failed to open settings file at: " << strSettingsFile << endl;
    exit(-1);
  }
#ifdef FUNC_MAP_SAVE_LOAD
  cv::FileNode mapfilen = fsSettings["Map.mapfile"];

  bool bReuseMap = false;
  if (!mapfilen.empty()) {
    mapfile = (string) mapfilen;
  }
#endif

    cout<<"find SuperGlue model form "<<SuperGlue_model_path<<endl;
    cout << endl << "Loading SuperGlue model. This could take a while..." << endl;
    std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(torch::jit::load(SuperGlue_model_path));

    bool use_cuda =torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda){
        device_type = torch::kCUDA;
        std::cout<<"run the SuperGlue in cuda!"<<std::endl;
    }
    else{
        device_type = torch::kCPU;
        std::cout<<"run the SuperGlue in CPU!"<<std::endl;
    }
    torch::Device device(device_type);

    model->eval();
    model->to(device);

    cout<<"find hfnet model form "<<model_path<<endl;
    cout << "Loading hfnet model. This could take a while..." << endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(at::randn({1,100, 2}, at::kFloat).to(device));
    inputs.push_back(at::randn({1,256, 100},at::kFloat).to(device));
    inputs.push_back(at::randn({1,100},at::kFloat).to(device));
    inputs.push_back(at::randn({1,100, 2}, at::kFloat).to(device));
    inputs.push_back(at::randn({1,256, 100},at::kFloat).to(device));
    inputs.push_back(at::randn({1,100},at::kFloat).to(device));
    auto out = model->forward(inputs).toTensor();
    cout << endl << "SuperGlue model loaded successfully!!" << endl;

    tensorflow::Session* session;
    auto options=tensorflow::SessionOptions();
    options.config.mutable_gpu_options()->set_allow_growth(true);
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.7);
    tensorflow::Status status = NewSession(options, &session);

    if (!status.ok()) {
        cout <<"Filed create session! "<<endl<<status.ToString() << "\n";
        return;
    }
    cout << "Session successfully created.\n";

    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(),model_path, &graph_def);
    if (!status.ok()) {
        cout <<"Failed ReadBinaryProto!! "<<endl<<status.ToString() << "\n";
        return;
    }
    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        cout <<"Failed add the graph to the session! "<<endl<< status.ToString() << "\n";
        return;
    }
    cout << "tensorflow loaded successfully!!" << "\n";


  //Load ORB Vocabulary
//  cout << endl << "Loading hfnet Vocabulary. This could take a while..." << endl;

  mpVocabulary = new ORBVocabulary();
//  cout<<"Voc found at: "<<strVocFile<<endl;
//
//  mpVocabulary->load(strVocFile);
//  cout << "Vocabulary loaded!" <<*mpVocabulary<< endl << endl;



  //Create KeyFrame Database
  //Create the Map
#ifdef FUNC_MAP_SAVE_LOAD
  if (!mapfile.empty() && LoadMap(mapfile)) {
    bReuseMap = true;
  } else
#endif
  {
    mpKeyFrameDatabase = new KeyFrameDatabase(mpVocabulary);
    mpMap = new Map();
  }

  //Create Drawers. These are used by the Viewer
#ifdef FUNC_MAP_SAVE_LOAD
  mpFrameDrawer = new FrameDrawer(mpMap, bReuseMap);
#else
  mpFrameDrawer = new FrameDrawer(mpMap);
#endif
  mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

  //Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this constructor)
  mpTracker = new Tracking(this, mpVocabulary,session,device_type,model, mpFrameDrawer, mpMapDrawer,
#ifdef FUNC_MAP_SAVE_LOAD
                           mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor, mask, bReuseMap);
#else
  mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);
#endif

  //Initialize the Local Mapping thread and launch
  mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR,model);
  mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

  //Initialize the Loop Closing thread and launch
  mpLoopCloser = new LoopClosing(session,mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
  mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

  //Initialize the Viewer thread and launch
  if (bUseViewer) {
#ifdef FUNC_MAP_SAVE_LOAD
    mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile, bReuseMap);
#else
    mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
#endif
    mptViewer = new thread(&Viewer::Run, mpViewer);
    mpTracker->SetViewer(mpViewer);
  }

  //Set pointers between threads
  mpTracker->SetLocalMapper(mpLocalMapper);
  mpTracker->SetLoopClosing(mpLoopCloser);

  mpLocalMapper->SetTracker(mpTracker);
  mpLocalMapper->SetLoopCloser(mpLoopCloser);

  mpLoopCloser->SetTracker(mpTracker);
  mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp) {
  if (mSensor != MONOCULAR) {
    cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
    exit(-1);
  }

  // Check mode change
  {
    unique_lock<mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    unique_lock<mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

  unique_lock<mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeys;

  return Tcw;
}

void System::ActivateLocalizationMode() {
  unique_lock<mutex> lock(mMutexMode);
  mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode() {
  unique_lock<mutex> lock(mMutexMode);
  mbDeactivateLocalizationMode = true;
}

bool System::MapChanged() {
  static int n = 0;
  int curn = mpMap->GetLastBigChangeIdx();
  if (n < curn) {
    n = curn;
    return true;
  } else
    return false;
}

void System::Reset() {
  unique_lock<mutex> lock(mMutexReset);
  mbReset = true;
}

void System::Shutdown() {
  mpLocalMapper->RequestFinish();
  mpLoopCloser->RequestFinish();
  if (mpViewer) {
    mpViewer->RequestFinish();
    while (!mpViewer->isFinished()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
  }

  // Wait until all thread have effectively stopped
  while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA()) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
  }
  //delete session;
 // if (mpViewer)
 //   pangolin::BindToContext("YiHang: Wide-angel Mapper");
#ifdef FUNC_MAP_SAVE_LOAD
  if (is_save_map)
    SaveMap(mapfile);
#endif
}

void System::SaveTrajectoryTUM(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
  if (mSensor == MONOCULAR) {
    cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
    return;
  }

  vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
  // We need to get first the keyframe pose and then concatenate the relative transformation.
  // Frames not localized (tracking failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
  // which is true when tracking failed (lbL).
  list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
  list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  list<bool>::iterator lbL = mpTracker->mlbLost.begin();
  for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
    if (*lbL)
      continue;

    KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
    while (pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    vector<float> q = Converter::toQuaternion(Rwc);

    f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " "
      << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
  }
  f.close();
  std::cout  << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
  cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

  vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  //cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame *pKF = vpKFs[i];

    // pKF->SetPose(pKF->GetPose()*Two);

    if (pKF->isBad())
      continue;

    cv::Mat R = pKF->GetRotation().t();
    vector<float> q = Converter::toQuaternion(R);
    cv::Mat t = pKF->GetCameraCenter();
    f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " "
      << t.at<float>(2)
      << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

  }

  f.close();
  cout  << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
  if (mSensor == MONOCULAR) {
    cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
    return;
  }

  vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
  // We need to get first the keyframe pose and then concatenate the relative transformation.
  // Frames not localized (tracking failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
  // which is true when tracking failed (lbL).
  list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
  list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
           lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++) {
    ORB_SLAM2::KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    while (pKF->isBad()) {
      //  cout << "bad parent" << endl;
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " "
      << twc.at<float>(0) << " " <<
      Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " "
      <<
      Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2)
      << endl;
  }
  f.close();
  cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState() {
  unique_lock<mutex> lock(mMutexState);
  return mTrackingState;
}

vector<MapPoint *> System::GetTrackedMapPoints() {
  unique_lock<mutex> lock(mMutexState);
  return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
  unique_lock<mutex> lock(mMutexState);
  return mTrackedKeyPointsUn;
}

#ifdef FUNC_MAP_SAVE_LOAD
void System::SaveMap(const string &filename) {
  std::ofstream out(filename, std::ios_base::binary);
  if (!out) {
    cerr << "Cannot Write to Mapfile: " << mapfile << std::endl;
    exit(-1);
  }
  cout << "Saving Mapfile: " << mapfile << std::flush;
  boost::archive::binary_oarchive oa(out, boost::archive::no_header);
  oa << mpMap;
  oa << mpKeyFrameDatabase;
  cout <<endl<< " ... done! The map have been saved!" << std::endl;
  out.close();
}

bool System::LoadMap(const string &filename) {
  std::ifstream in(filename, std::ios_base::binary);
  if (!in) {
    cerr << "Cannot Open Mapfile: " << mapfile << " , Create a new one" << std::endl;
    return false;
  }
  cout << "Loading Mapfile: " << mapfile << std::flush;
  boost::archive::binary_iarchive ia(in, boost::archive::no_header);
  ia >> mpMap;
  ia >> mpKeyFrameDatabase;
  mpKeyFrameDatabase->SetORBvocabulary(mpVocabulary);
  mpKeyFrameDatabase->CreatKDTreeAndPCA();
  cout << " ...done" << std::endl;
  cout << "Map Reconstructing" << flush;
//  vector<ORB_SLAM2::KeyFrame *> vpKFS = mpMap->GetAllKeyFrames();
//  for (auto it:vpKFS) {
//    it->SetORBvocabulary(mpVocabulary);
//    it->ComputeBoW();
//  }
  cout << " ...done" << endl;
  in.close();
  return true;
}

#endif
} //namespace ORB_SLAM
