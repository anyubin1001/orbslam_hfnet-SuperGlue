#include <iostream>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <string>
#include <sstream>

// include sys
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// include Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "System.h"
#include "conversions.h"
#include <thread>
#include <mutex>
#include <stdio.h>
#include <termios.h>
using namespace std;
using namespace cv;
static int peek_character = -1;
static struct termios initial_settings, new_settings;
const std::string ROOT_Path = ros::package::getPath("omni_vslam");
std::string mask_image_path = ROOT_Path + "/config/mask.png";
std::string yaml_path= ROOT_Path + "/config/datong-right.yaml";
std::string visual_trajectory_path= ROOT_Path + "/config/VisualTrajectory.txt";
std::string vocabulary_path = ROOT_Path + "/Vocabulary/orb_fisheye_voc.bin";
std::string transform_matrix_path = ROOT_Path + "/config/transformation.txt";
std::string model_path = ROOT_Path + "/config/hf_net/myself_hfnetall_with_SuperGlue4_thresh0.001_nms12_1000keys.pb";
std::string SuperGlue_model_path = ROOT_Path + "/config/SuperGlue/myself_SuperGlue_640_480_outdoor.pt";

bool is_buildingup_map = false;

Eigen::Matrix4f Tcam2gps = Eigen::Matrix4f::Identity();
float scale = 1.0f;
Eigen::Matrix4f extrinsic_matrix = Eigen::Matrix4f::Identity();
Eigen::Matrix4f rotate_matrix = Eigen::Matrix4f::Identity();
Eigen::Matrix4f extrinsic_matrix_all = Eigen::Matrix4f::Identity();

ORB_SLAM2::System * pSLAM = nullptr;
ros::Subscriber image_sub;
ros::Publisher gps_pub;
ros::Publisher imu_pub;
ros::Publisher raw_imu_pub;
//std::mutex flag;
//bool stop_flag = false;
void init_keyboard()
{
    tcgetattr(0,&initial_settings);
    new_settings = initial_settings;
    new_settings.c_lflag |= ICANON;
    new_settings.c_lflag |= ECHO;
    new_settings.c_lflag |= ISIG;
    new_settings.c_cc[VMIN] = 1;
    new_settings.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
}
void close_keyboard()
{
    tcsetattr(0, TCSANOW, &initial_settings);
}
int kbhit()
{
    unsigned char ch;
    int nread;
 
    if (peek_character != -1) return 1;
    new_settings.c_cc[VMIN]=0;
    tcsetattr(0, TCSANOW, &new_settings);
    nread = read(0,&ch,1);
    new_settings.c_cc[VMIN]=1;
    tcsetattr(0, TCSANOW, &new_settings);
    if(nread == 1) 
    {
        peek_character = ch;
        return 1;
    }
    return 0;
}
int readch()
{
    char ch;
    if(peek_character != -1)
    {
        ch = peek_character;
        peek_character = -1;
        return ch;
    }
    read(0,&ch,1);
    return ch;
}
void getch() {
    init_keyboard();
    while (true) { 
        if (kbhit()) {
           //char ch=getchar();           //直接这样写这个进程会卡在这边，循环等待输入，不会执行sleep_for，耗费cpu资源
            char ch=readch();
            if(cin.peek()=='\n')
                getchar();
            //std::cout<<"havechar: "<<ch<<endl;
            if(ch =='\n') continue;
            else if(ch=='q'){
                /*{
                    std::unique_lock<std::mutex> lock(flag);
                    stop_flag = true;
                }*/
                pSLAM -> Shutdown();
                // Save camera trajectory
                pSLAM -> SaveKeyFrameTrajectoryTUM(visual_trajectory_path);
                if (pSLAM != nullptr) {
                    delete pSLAM;
                    pSLAM = nullptr;
                }
                //ros::requestShutdown();
                std::cout<<"Have done all the saves! Now you can press 'Ctrl+C' to shutdown the YiHang Slam!"<<endl;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
    close_keyboard();
}

void multiple_channel_process(const sensor_msgs::ImageConstPtr& front_image,
                  const sensor_msgs::ImageConstPtr& right_image,
                      const sensor_msgs::ImageConstPtr& back_image,
                           const sensor_msgs::ImageConstPtr& left_image)
 {
     cv_bridge::CvImagePtr cv_ptr[4];
        try
        {
            // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv_ptr[0] = cv_bridge::toCvCopy(front_image, sensor_msgs::image_encodings::TYPE_8UC1);
            cv_ptr[1] = cv_bridge::toCvCopy(right_image, sensor_msgs::image_encodings::TYPE_8UC1);
            cv_ptr[2] = cv_bridge::toCvCopy(back_image, sensor_msgs::image_encodings::TYPE_8UC1);
            cv_ptr[3] = cv_bridge::toCvCopy(left_image, sensor_msgs::image_encodings::TYPE_8UC1);
            //ROS_INFO("The image has been subscribed");
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("%s | %d", __FILE__,__LINE__);
        }

    cv::Mat image[4];
    unsigned char * pbuffer[4];
    for (int i=0;i<4;i++){
        image[i] = cv::imdecode(cv::Mat(cv_ptr[i]->image), CV_LOAD_IMAGE_COLOR);        
        pbuffer[i] = image[i].ptr<unsigned char>(0);
    }
 }

void single_channel_process(const sensor_msgs::ImageConstPtr& _image)
{
    /*{
       std::unique_lock<std::mutex> lock(flag);
       if(stop_flag){
          //ros::shutdown();
          return;
       }
    }*/
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(_image, sensor_msgs::image_encodings::TYPE_8UC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("%s | %d", __FILE__,__LINE__);
    }

    cv::Mat image;
    unsigned char * pbuffer;
    
    image= cv::imdecode(cv::Mat(cv_ptr->image), CV_LOAD_IMAGE_COLOR);

   // cout << "the time stamp is  " <<  ss.str() << endl;
    cv::Mat Tcw = pSLAM -> TrackMonocular(image, _image->header.stamp.toSec());
    // cout << "Tcw: " << endl << Tcw << endl;

    // cv2eigen(Tcw, Tcw_eigen);
    
    if(!is_buildingup_map){
      Eigen::Matrix4f Twc_eigen;

      // cout << "come in ===========" << endl;
      if(Tcw.rows != 4 || Tcw.cols != 4) return;

      cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
      cv::Mat twc = - Rwc * Tcw.rowRange(0, 3).col(3); 

      // cout << "Tcw.at<float>(0, 0) " << Tcw.at<float>(0, 0) << endl;
      
      Twc_eigen << Rwc.at<float>(0, 0), Rwc.at<float>(0, 1), Rwc.at<float>(0, 2), twc.at<float>(0), 
                   Rwc.at<float>(1, 0), Rwc.at<float>(1, 1), Rwc.at<float>(1, 2), twc.at<float>(1), 
                   Rwc.at<float>(2, 0), Rwc.at<float>(2, 1), Rwc.at<float>(2, 2), twc.at<float>(2), 
                   0, 0, 0, 1;

      // recover scale
      Twc_eigen.block<3, 1>(0, 3) = scale * Twc_eigen.block<3, 1>(0, 3);

      // transform to gps data
      Eigen::Matrix4f se3 = extrinsic_matrix_all * Twc_eigen * extrinsic_matrix_all.inverse();

      // convert from eigen matrix to cv mat
      // eigen2cv(se3, T_to_publish);

      // construct a gps message
      sensor_msgs::NavSatFix gps_msg;
      double utmE = se3(0, 3);   //x,y
      double utmN = se3(1, 3);
/*      double Lat, Long;
      gps_common::UTMtoLL(utmN, utmE, "50S", Lat, Long);*/
      gps_msg.header.stamp = ros::Time::now();
      //gps_msg.latitude = utmE;
      //gps_msg.longitude = utmN;
      gps_msg.latitude = utmN;
      gps_msg.longitude = utmE;

      gps_msg.altitude = 0;
      gps_pub.publish(gps_msg);

      // construct a imu message   //转为IMU格式
      sensor_msgs::Imu imu_msg;
      Eigen::Matrix3f orientation = se3.block<3, 3>(0, 0);
      Eigen::Vector3f euler = orientation.eulerAngles(2, 1, 0);

      euler[0] = - euler[0] + M_PI / 2;     //只考虑绕Z轴旋转，其他两个方向为0
      euler[1] = 0;
      euler[2] = 0;

      Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitX()));
      Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()));
      Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitZ()));

      Eigen::Quaterniond quaternion;
      quaternion = yawAngle * pitchAngle * rollAngle;

      imu_msg.header.stamp = ros::Time::now();
      imu_msg.orientation.w = quaternion.w();
      imu_msg.orientation.x = quaternion.x();
      imu_msg.orientation.y = quaternion.y();
      imu_msg.orientation.z = quaternion.z();    

      sensor_msgs::Imu raw_imu_msg;
      Eigen::Matrix3f raw_orientation = se3.block<3, 3>(0, 0);
      Eigen::Quaterniond raw_q(raw_orientation.cast<double>());
      raw_imu_msg.orientation.w = raw_q.w();
      raw_imu_msg.orientation.x = raw_q.x();
      raw_imu_msg.orientation.y = raw_q.y();
      raw_imu_msg.orientation.z = raw_q.z();
      
      imu_pub.publish(imu_msg);
      raw_imu_pub.publish(raw_imu_msg);          
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "omni_vslam");
  
    ros::NodeHandle nh_, private_nh_("~");

  /*  //the front, right, left, rear camera image using together ,later for mapping
    message_filters::Subscriber<sensor_msgs::Image>  image_sub[4];
    image_sub[0].subscribe(nh_, "/sensor/image/capturefrontview", 10);
    image_sub[1].subscribe(nh_, "/sensor/image/capturerightview", 10);
    image_sub[2].subscribe(nh_, "/sensor/image/capturebackview", 10);
    image_sub[3].subscribe(nh_, "/sensor/image/captureleftview", 10);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
          sensor_msgs::Image,
          sensor_msgs::Image,
          sensor_msgs::Image > Pickout_SyncPolicy;
    boost::shared_ptr< message_filters::Synchronizer<Pickout_SyncPolicy> > _pickout_sync;
	// Define Synchronizer
    _pickout_sync.reset( new message_filters::Synchronizer<Pickout_SyncPolicy>(
                            Pickout_SyncPolicy(10), image_sub[0], image_sub[1], image_sub[2], image_sub[3]) );
    _pickout_sync->registerCallback(boost::bind(&multiple_channel_process, _1, _2, _3, _4,));	
    ROS_INFO("after Synchronizer");
    //end the multiple channels image subscribe
    */

    private_nh_.param<std::string>("mask_image_path", mask_image_path, mask_image_path);    
    private_nh_.param<std::string>("yaml_path", yaml_path, yaml_path);
    private_nh_.param<std::string>("visual_trajectory_path", visual_trajectory_path, visual_trajectory_path); 
    private_nh_.param<std::string>("vocabulary_path", vocabulary_path, vocabulary_path); 
    private_nh_.param<std::string>("transform_matrix_path", transform_matrix_path, transform_matrix_path); 
    private_nh_.param<std::string>("hfnet_model_path", model_path, model_path);
    private_nh_.param<std::string>("SuperGlue_model_path", SuperGlue_model_path, SuperGlue_model_path);

    private_nh_.param<bool>("is_buildingup_map", is_buildingup_map, false);    

    //only the front camera or the right camera to localize  
    image_sub = nh_.subscribe("/sensor/image/capturerightview", 10, single_channel_process);

    // publish information of position and orientation  using for testing and debugging
    // gps_pub = nh_.advertise<sensor_msgs::NavSatFix>("vslam_gps/fix", 10);
    // imu_pub = nh_.advertise<sensor_msgs::Imu>("vslam_imu/data", 10);

    gps_pub = nh_.advertise<sensor_msgs::NavSatFix>("/gps/fix", 10);
    imu_pub = nh_.advertise<sensor_msgs::Imu>("/imu/data", 10);
    raw_imu_pub = nh_.advertise<sensor_msgs::Imu>("/vslam_raw_imu/raw_data", 10);

    if(!is_buildingup_map){
      ifstream transRead;
      transRead.open(transform_matrix_path.c_str());
      for(int i = 0; i < 4; i++){
        string s;
        getline(transRead, s);
        if(!s.empty()){
          stringstream ss;
          ss << s;
          ss >> Tcam2gps(i, 0) >> Tcam2gps(i, 1) >> Tcam2gps(i, 2) >> Tcam2gps(i, 3);
        }
      }
      scale = Tcam2gps(3, 3);
      Tcam2gps(3, 3) = 1.0f;

     /* extrinsic_matrix << -0.032384, -0.999270, 0.020263, 1994.368126,
                            0.499388, 0.001385, 0.866377, -981.334066,
                           -0.865773, 0.038176, 0.498979, 1151.138814,
                            0.000000, 0.000000, 0.000000, 1.000000;*/
                            
       extrinsic_matrix << -0.999550, -0.029185, -0.006867, 1998.497027 ,
                            -0.008653, 0.500128, -0.865908, -1099.645500, 
                            0.028706, -0.865460, -0.500156 ,1134.270856, 
                            0.000000, 0.000000, 0.000000, 1.000000 ;

      extrinsic_matrix.block<3, 1>(0, 3) = extrinsic_matrix.block<3, 1>(0, 3) / 1000;

      rotate_matrix <<  0,  1,  0,  0,
                        1,  0,  0,  0,
                        0,  0, -1,  0,
                        0,  0,  0,  1;

      //extrinsic_matrix_all = extrinsic_matrix * rotate_matrix;
      extrinsic_matrix_all = extrinsic_matrix;

    }

    cv::Mat imMask = cv::imread(mask_image_path, 0);
    cv::resize(imMask,imMask,cv::Size(640,720),cv::INTER_NEAREST);
//std::cout<<"******************"<<imMask.size()<<std::endl;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    pSLAM = new ORB_SLAM2::System(model_path,SuperGlue_model_path,vocabulary_path, yaml_path, ORB_SLAM2::System::MONOCULAR, true,  imMask, is_buildingup_map);
    
    std::thread waik_for_keybord(getch);
    waik_for_keybord.detach();

    ros::spin();

    //if(!stop_flag)
        std::cout<<endl<<"Please press 'q + Enter' next time to stop the YiHang slam and save the map"<<endl<<"just press 'Ctrl + C' will not save the map now!!"<<std::endl<<std::endl<<std::endl;

//    pSLAM -> Shutdown();
//    // Save camera trajectory
//    pSLAM -> SaveKeyFrameTrajectoryTUM(visual_trajectory_path);
//    if (pSLAM != nullptr) {
//        delete pSLAM;
//        pSLAM = nullptr;
//    }
    
    return 0;
}
