/*
 * map save/load extension for ORB_SLAM2
 * This header contains boost headers needed by serialization
 *
 * object to save:
 *   - KeyFrame
 *   - KeyFrameDatabase
 *   - Map
 *   - MapPoint
 */
#ifndef BOOST_ARCHIVER_H
#define BOOST_ARCHIVER_H
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
// set serialization needed by KeyFrame::mspChildrens ...
#include <boost/serialization/map.hpp>
// map serialization needed by KeyFrame::mConnectedKeyFrameWeights ...
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/base_object.hpp>
// base object needed by DBoW3::BowVector and DBoW3::FeatureVector

// for eigen3
#include <boost/serialization/array.hpp>
#include <Eigen/Core>

#include <opencv2/core/core.hpp>

#include "Thirdparty/DBoW3/src/DBoW3.h"
#include "Thirdparty/models/include/Camera.h"

BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
namespace boost {
namespace serialization {

/* serialization for DBoW3 BowVector */
template<class Archive>
void serialize(Archive &ar, DBoW3::BowVector &BowVec, const unsigned int file_version) {
  ar & boost::serialization::base_object<DBoW3::BowVector::super>(BowVec);
}
/* serialization for DBoW3 FeatureVector */
template<class Archive>
void serialize(Archive &ar, DBoW3::FeatureVector &FeatVec, const unsigned int file_version) {
  ar & boost::serialization::base_object<DBoW3::FeatureVector::super>(FeatVec);
}

/* serialization for CV KeyPoint */
template<class Archive>
void serialize(Archive &ar, ::cv::KeyPoint &kf, const unsigned int file_version) {
  ar & kf.angle;
  ar & kf.class_id;
  ar & kf.octave;
  ar & kf.response;
  ar & kf.response;
  ar & kf.pt.x;
  ar & kf.pt.y;
}
/* serialization for CV Point3f */
template<class Archive>
void serialize(Archive &ar, ::cv::Point3f &p3f, const unsigned int file_version) {
  ar & p3f.x;
  ar & p3f.y;
  ar & p3f.z;
}

//TODO: serialization for Ocam model
template<class Archive>
void serialize(Archive &ar, ::ORB_SLAM2::OCAMCamera &camera, const unsigned int file_version) {
  ar & camera.mWidth;
  ar & camera.mHeight;
  ar & camera.m_C;
  ar & camera.m_D;
  ar & camera.m_E;
  ar & camera.m_center_x;
  ar & camera.m_center_y;
  ar & camera.m_inv_scale;
  ar & camera.errorMultiplier2;

  ar & boost::serialization::make_array<double>(camera.m_poly, SCARAMUZZA_POLY_SIZE);
  ar & boost::serialization::make_array<double>(camera.m_inv_poly, SCARAMUZZA_INV_POLY_SIZE);
}

//save for eigen
template< class Archive,class S,int Rows_,int Cols_,int Ops_,int MaxRows_,int MaxCols_>
inline void serialize(Archive & ar,Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix,const unsigned int version)
{
    int rows = matrix.rows();
    int cols = matrix.cols();
    ar & make_nvp("rows", rows);
    ar & make_nvp("cols", cols);
    matrix.resize(rows, cols); // no-op if size does not change!

    // always save/load row-major
    for(int r = 0; r < rows; ++r)
        for(int c = 0; c < cols; ++c)
            ar & make_nvp("val", matrix(r,c));
}

template< class Archive,class S,int Dim_,int Mode_,int Options_>
inline void serialize(Archive & ar,Eigen::Transform<S, Dim_, Mode_, Options_> & transform,const unsigned int version)
{
    serialize(ar, transform.matrix(), version);
}


/* serialization for CV Mat */
template<class Archive>
void save(Archive &ar, const ::cv::Mat &m, const unsigned int file_version) {
  cv::Mat m_ = m;
  if (!m.isContinuous())
    m_ = m.clone();
  size_t elem_size = m_.elemSize();
  size_t elem_type = m_.type();
  ar & m_.cols;
  ar & m_.rows;
  ar & elem_size;
  ar & elem_type;

  const size_t data_size = m_.cols * m_.rows * elem_size;

  ar & boost::serialization::make_array(m_.ptr(), data_size);
}
template<class Archive>
void load(Archive &ar, ::cv::Mat &m, const unsigned int version) {
  int cols, rows;
  size_t elem_size, elem_type;

  ar & cols;
  ar & rows;
  ar & elem_size;
  ar & elem_type;

  m.create(rows, cols, elem_type);
  size_t data_size = m.cols * m.rows * elem_size;

  ar & boost::serialization::make_array(m.ptr(), data_size);
}
}
}
// TODO: boost::iostream zlib compressed binary format
#endif // BOOST_ARCHIVER_H
