#include "SuperPoint.h"

namespace ORB_SLAM2
{
void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint>& pts,
            int border, int dist_thresh, int img_width, int img_height);


SPDetector::SPDetector(std::shared_ptr<torch::jit::script::Module> _model) : model(_model)
{
   //printf("run the model in GPU? %d\n",torch::cuda::is_available());
}

void SPDetector::detect(cv::Mat &img, bool cuda)
{
   /* cv::Mat temp=img.clone();
    img.convertTo(temp,CV_32FC1,1.0f / 255.0f);
    auto x = torch::from_blob(temp.data, {1, 1, temp.rows, temp.cols}); */  //trans to tensor

    auto x = torch::from_blob(img.data, {1, 1, img.rows, img.cols}, torch::kByte);  //trans to tensor
    x = x.to(torch::kFloat) / 255.0;

    use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;

    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);

    model->to(device);
    model->eval();

    x = x.set_requires_grad(false);
    auto x1=x.to(device);
    //一个返回值这样写，这里有2个返回值，靠！！要先转为tensor
    //auto out = model->forward({x1}).toTensor();
    auto outputs = model->forward({x1}).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();  //[1, 65, H/8, W/8]
    torch::Tensor out2 = outputs->elements()[1].toTensor();  //[1, 256, H/8, W/8]

    auto dense = torch::softmax(out1, 1);  //对65这个第一维进行softmax

    //cv::Mat temp1(cv::Size(dense.size(0), dense.size(1)), CV_32FC1, prob.data<float>());
    /*double sum=0;
    for(int i=0;i<65;i++)
        sum+=dense[0][i][0][0].item<float>();
    std::cout<<"-------sum:   "<<sum<<std::endl;*/
    auto nodust = dense.slice(1, 0, 64);    //丢弃  slice(维度,行init，行last)
    auto nodust1 = nodust.permute({0, 2, 3, 1});  // [1, H/8, W/8, 64]   //将维度换位


    int Hc = nodust1.size(1);
    int Wc = nodust1.size(2);

    auto heatmap = nodust1.contiguous().view({-1, Hc, Wc, 8, 8}); //[1,H/8,W/8,8,8]
    auto heatmap1 = heatmap.permute({0, 1, 3, 2, 4});         //[1,H/8,8,W/8,8]
    auto heatmap2 = heatmap1.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]

    mProb = heatmap2.squeeze(0);  // [H, W]
    mDesc = out2 ;             // [1, 256, H/8, W/8]
}


void SPDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms,cv::Mat mask)
{
    auto prob = mProb; //.slice(0, iniY, maxY).slice(1, iniX, maxX);  // [h, w]

    auto kpts = (prob > threshold /*0.01*/);
    kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)   //获取不为0的坐标

    std::vector<cv::KeyPoint> keypoints_no_nms;
    for (int i = 0; i < kpts.size(0); i++) {
        //std::cout<<"y:   "<<kpts[i][0].item<float>()<<"    "<<kpts[i][1].item<float>()<<std::endl;
        if(mask.at<uchar>(kpts[i][0].item<float>(),kpts[i][1].item<float>())==255) {
            float response = prob[kpts[i][0]][kpts[i][1]].item<float>();   //获取tensor内容的方法!!!!
            keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1,
                                                    response));  //x(同at标准),ysize:8   angle:-1
            // std::cout<<response<<"  *****  ";
        }
    }


    /*std::cout<<"before: "<<keypoints_no_nms.size()<<std::endl;
    if(keypoints_no_nms.size()==0){
        cv::imwrite("/home/hawk/Documents/slam_by_zhili/Cnn/beifen/SuperPointPretrainedNetwork (1)/no_points/"+std::to_string(sum++)+".png",temp);
        std::cout<<"have done"<<std::endl;
    }*/


    if (nms) {
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
        for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
            int x = keypoints_no_nms[i].pt.x;
            int y = keypoints_no_nms[i].pt.y;
            conf.at<float>(i, 0) = prob[y][x].item<float>();
        }

        // cv::Mat descriptors;

        int border = 0;
        int dist_thresh = 4;   //4;
        int height = maxY - iniY;
        int width = maxX - iniX;

        NMS2(keypoints_no_nms, conf, keypoints, border, dist_thresh, width, height);
    }
    else {
        keypoints = keypoints_no_nms;
    }
}


void SPDetector::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)   //将特征点重新分配到指定的grid中，从而找到grid的描述子
{
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);  // [n_keypoints, 2]   (y, x)  一行一个keypoint

    for (size_t i = 0; i < keypoints.size(); i++) {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
    }

    auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);

    auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x   将坐标归一化到[-1,1]   //不写slice(dim0)即相当于dim0全部都取，即取n个keypoints都
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y

    //torch::Tensor mDesc1=mDesc.to(torch::kCPU);
    //可以加一个if use_cuda控制
    auto grid1=grid.to(use_cuda?torch::kCUDA:torch::kCPU);
    auto desc = torch::grid_sampler(mDesc, grid1, 0, 0,1);  // [1, 256, 1, n_keypoints] //双线性插值
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints]   /一列一个描述子

    //std::cout<<"before:  "<<desc.size(0)<<std::endl;   //256 hang
    // normalize to 1
    auto dn = torch::norm(desc, 2, 0);
    desc = desc.div(torch::unsqueeze(dn, 0));
    //std::cout<<dn<<std::endl;


    /*double sum0=0;
        for(int i=0;i<256;i++) {
            sum0 += desc.at<float>(0, i)*desc.at<float>(0, i);
            // sum1 +=desc.
        }
    std::cout<<"size:   "<<desc.size<<"   sum: "<<sum0<<std::endl;*/



    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]   //一行一个描述子

    if(use_cuda)
        desc = desc.to(torch::kCPU);

    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>()); //一行一个描述子

    /*double sum0=0;
        for(int i=0;i<256;i++) {
            sum0 += desc_mat.at<float>(0, i)*desc_mat.at<float>(0, i);
            // sum1 +=desc.
        }
    std::cout<<"size:   "<<desc_mat.size<<"   sum: "<<sum0<<std::endl;*/

    descriptors = desc_mat.clone();
}


void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint>& pts,
            int border, int dist_thresh, int img_width, int img_height)
{

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.size(); i++){

        int u = (int) det[i].pt.x;
        int v = (int) det[i].pt.y;

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }
    
    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                if ( confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu) )
                    grid.at<char>(vv + k, uu + j) = 0;
                
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                cv::Point2f p = pts_raw[select_ind];
                float response = conf.at<float>(select_ind, 0);
                pts.push_back(cv::KeyPoint(p, 8.0f, -1, response));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }

}


} //namespace ORB_SLAM
