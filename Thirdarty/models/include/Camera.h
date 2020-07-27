#ifndef CAMERA_H
#define CAMERA_H

#include <string>

// include opencv
#include <opencv2/core/core.hpp>

// include boost
#include <boost/shared_ptr.hpp>

// include eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace ORB_SLAM2 {

#define SCARAMUZZA_POLY_SIZE 5
#define SCARAMUZZA_INV_POLY_SIZE 12

class OCAMCamera{
 public:
  OCAMCamera();

  // construct camera from config file.
  bool readFromYamlFile(const std::string &filename);

  // construct camera from other camera.
  OCAMCamera(const OCAMCamera &other);
  OCAMCamera &operator=(const OCAMCamera &other);

  friend std::ostream& operator<<(std::ostream &out, const OCAMCamera& camera);

  void liftSphere(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;

  void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;

  void spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const;

  double computeErrorMultiplier();

  // camera calibration parameters
  int mWidth;
  int mHeight;

  double m_poly[SCARAMUZZA_POLY_SIZE];
  double m_inv_poly[SCARAMUZZA_INV_POLY_SIZE];
  double m_C;
  double m_D;
  double m_E;
  double m_center_x;
  double m_center_y;

  double m_inv_scale;
  double errorMultiplier2;
};

typedef boost::shared_ptr<OCAMCamera> OCAMCameraPtr;
typedef boost::shared_ptr<const OCAMCamera> OCAMCameraConstPtr;

}

#endif
