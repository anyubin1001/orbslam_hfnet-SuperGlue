#include <iostream>
#include <algorithm>
#include <chrono>
#include <unistd.h>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

int main(int argc, char **argv) {
  if (argc != 6) {
    cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_mask [1|0](save map?)" << endl;
    return 1;
  }

  cv::Mat imMask = cv::imread(std::string(argv[4]), 0);

  // Retrieve paths to images
  vector<string> vstrImageFilenames;
  vector<double> vTimestamps;
  LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

  int nImages = vstrImageFilenames.size();

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true,  imMask, (bool) atoi(argv[5]));

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;

  // Main loop
  cv::Mat im;
  for (int ni = 0; ni < nImages; ni++) {
    // Read image from file
    im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[ni];

    if (im.empty()) {
      cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
      return 1;
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

    // Pass the image to the SLAM system
    SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

    double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimestamps[ni + 1] - tframe;
    else if (ni > 0)
      T = tframe - vTimestamps[ni - 1];

    if (ttrack < T)
      usleep((T - ttrack) * 1e6);
  }

  // Stop all threads
  SLAM.Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
  cout << "mean tracking time: " << totaltime / nImages << endl;

  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("VisualTrajectory.txt");

  return 0;
}

vector< string> split(string str, string pattern) {
  vector<string> ret;
  if (pattern.empty()) return ret;
  size_t start = 0, index = str.find_first_of(pattern, 0);
  while (index != str.npos) {
    if (start != index)
      ret.push_back(str.substr(start, index - start));
    start = index + 1;
    index = str.find_first_of(pattern, start);
  }
  if (!str.substr(start).empty())
    ret.push_back(str.substr(start));
  return ret;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/time.txt";

  string strPrefixLeft = strPathToSequence + "/data/";

  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      // vector<string> result = split(s, " ");

      stringstream ss;
      ss << s;

      vstrImageFilenames.push_back(strPrefixLeft + ss.str() + ".png");

      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }
}
