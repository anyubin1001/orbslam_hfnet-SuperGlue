#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <memory>

#include "SPextractor.h"
#include "SuperPoint.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         float _iniThFAST, float _minThFAST,const string &model_path,cv::Mat _mask):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST),mask(_mask)
{
    //model = make_shared<torch::jit::script::Module>();
    //std::shared_ptr<torch::jit::script::Module>
    model = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));

    //cout<<"-----------------"<<*model->attributes(1).begin()   /* attr("conv1a.weight")*/ <<endl ;

    //cout<<model_path<<endl;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    /*float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);*/
    mnFeaturesPerLevel[0] = nfeatures;
}


void SPextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints, cv::Mat &_desc)
{
    allKeypoints.resize(nlevels);

    vector<cv::Mat> vDesc;

    //cout<<"nlevels: "<<"    "<<nlevels<<"    mnFeaturesPerLevel[0]"<<mnFeaturesPerLevel[0]<<endl;
    for (int level = 0; level < nlevels; ++level)
    {
        SPDetector detector(model);
        detector.detect(mvImagePyramid[level], true);


        //之后可以把这一部分全都去掉，octave写到superpoint中去
        //mask增大  可以直接resize ，frame类中直接判断.at当前像素

        /*const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;*/

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures*5);

        detector.getKeyPoints(iniThFAST, 0, mvImagePyramid[level].cols, 0, mvImagePyramid[level].rows, keypoints, true,mask);

        //cout<<"after: "<<keypoints.size()<<endl; // 0.015 2100多

        /*if(keypoints.size()<100)
        {
            cout<<"before: "<<keypoints.size()<<endl;
            //cv::imshow("image"+to_string(keypoints.size()),imagetemp);
            //cv::waitKey(100);
        }*/

        //cout<<"before: "<<"    "<<keypoints.size()<<endl;
        //要不要换成*1  特征点是不是减少一点匹配u会更好？
        if(keypoints.size()>mnFeaturesPerLevel[level])
            KeyPointsFilter::retainBest(keypoints, mnFeaturesPerLevel[level]);


        //初始化的时候会经常到这边？？？
       /* else if(keypoints.size()<mnFeaturesPerLevel[level]*0.5)
        {
            keypoints.clear();
            detector.getKeyPoints(minThFAST, 0, mvImagePyramid[level].cols, 0, mvImagePyramid[level].rows, keypoints, true,mask);
            //cout<<"---------------------------before: "<<keypoints.size()<<endl;
        }*/


        //cout<<"after: "<<"    "<<keypoints.size()<<endl;

        //const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
       /* const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x=(keypoints[i].pt.x+minBorderX);
            keypoints[i].pt.y=(keypoints[i].pt.y+minBorderY);
            keypoints[i].octave=level;
            //keypoints[i].size = scaledPatchSize;
        }*/




        cv::Mat desc;
        detector.computeDescriptors(keypoints, desc);
        vDesc.push_back(desc);
    }

    cv::vconcat(vDesc, _desc);

    // // compute orientations
    // for (int level = 0; level < nlevels; ++level)
    //     computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}


void SPextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{


    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    //cv::Mat image=image0.rowRange(0,(int)(image0.rows/2));
    //cv::resize(image_temp,image,Size(/*640,360*/320,180));

    Mat descriptors;

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    vector < vector<KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints, descriptors);


    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        _descriptors.create(nkeypoints, 256, CV_32F);
        descriptors.copyTo(_descriptors.getMat());
    }


    _keypoints.clear();
    _keypoints.assign(allKeypoints[0].begin(),allKeypoints[0].end());

    /*_keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;


        // Scale keypoint coordinates
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
            keypoint->pt.x=   keypoint->pt.x*4;keypoint->pt.y=   keypoint->pt.y*4;
        }

        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }*/



}

//需要需要金字塔?? - > 已验证：不需要
void SPextractor::ComputePyramid(cv::Mat image)
{
    /*for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
    }*/
    mvImagePyramid[0]=image;
}

} //namespace ORB_SLAM
