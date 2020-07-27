#include "../include/Camera.h"

#include <cmath>
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <boost/lexical_cast.hpp>

namespace ORB_SLAM2 {
OCAMCamera::OCAMCamera(){
  memset(m_poly, 0, sizeof(double) * SCARAMUZZA_POLY_SIZE);
  memset(m_inv_poly, 0, sizeof(double) * SCARAMUZZA_INV_POLY_SIZE);
}

OCAMCamera::OCAMCamera(const OCAMCamera &other){
  mWidth = other.mWidth;
  mHeight = other.mHeight;
  m_C = other.m_C;
  m_D = other.m_D;
  m_E = other.m_E;
  m_center_x = other.m_center_x;
  m_center_y = other.m_center_y;

  memcpy(m_poly, other.m_poly, sizeof(double) * SCARAMUZZA_POLY_SIZE);
  memcpy(m_inv_poly, other.m_inv_poly, sizeof(double) * SCARAMUZZA_INV_POLY_SIZE);

  m_inv_scale = other.m_inv_scale;
  errorMultiplier2 = other.errorMultiplier2;
}

bool OCAMCamera::readFromYamlFile(const std::string &filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  mWidth = static_cast<int>(fs["image_width"]);
  mHeight = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["poly_parameters"];
  for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++)
    m_poly[i] = static_cast<double>(n[std::string("p") + boost::lexical_cast<std::string>(i)]);

  n = fs["inv_poly_parameters"];
  for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
    m_inv_poly[i] = static_cast<double>(n[std::string("p") + boost::lexical_cast<std::string>(i)]);

  n = fs["affine_parameters"];
  m_C = static_cast<double>(n["ac"]);
  m_D = static_cast<double>(n["ad"]);
  m_E = static_cast<double>(n["ae"]);

  m_center_x = static_cast<double>(n["cx"]);
  m_center_y = static_cast<double>(n["cy"]);

  m_inv_scale = 1.0 / (m_C - m_D * m_E);
  double errorMultiplier = computeErrorMultiplier();
  errorMultiplier2 = sqrt(errorMultiplier) / 2;

  return true;
}

OCAMCamera& OCAMCamera::operator=(const OCAMCamera &other) {
  if (this != &other) {
    mWidth = other.mWidth;
    mHeight = other.mHeight;
    m_C = other.m_C;
    m_D = other.m_D;
    m_E = other.m_E;
    m_center_x = other.m_center_x;
    m_center_y = other.m_center_y;

    memcpy(m_poly, other.m_poly, sizeof(double) * SCARAMUZZA_POLY_SIZE);
    memcpy(m_inv_poly, other.m_inv_poly, sizeof(double) * SCARAMUZZA_INV_POLY_SIZE);

    m_inv_scale = other.m_inv_scale;
    errorMultiplier2 = other.errorMultiplier2;
  }

  return *this;
}

std::ostream& operator<<(std::ostream &out, const OCAMCamera& camera) {
  out << "- image_width: " << camera.mWidth << std::endl;
  out << "- image_height: " << camera.mHeight << std::endl;

  out << std::fixed << std::setprecision(10);

  out << "Poly Parameters" << std::endl;
  for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++)
    out << std::string("- p") + boost::lexical_cast<std::string>(i) << ": " << camera.m_poly[i] << std::endl;

  out << "Inverse Poly Parameters" << std::endl;
  for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
    out << std::string("- p") + boost::lexical_cast<std::string>(i) << ": " << camera.m_inv_poly[i] << std::endl;

  out << "Affine Parameters" << std::endl;
  out << "- ac: " << camera.m_C << std::endl
      << "- ad: " << camera.m_D << std::endl
      << "- ae: " << camera.m_E << std::endl;
  out << "- cx: " << camera.m_center_x << std::endl
      << "- cy: " << camera.m_center_y << std::endl;

  return out;
}

void OCAMCamera::liftSphere(const Eigen::Vector2d &p, Eigen::Vector3d &P) const {
    Eigen::Vector2d pnew(p[0]*2.0,p[1]);
    liftProjective(pnew, P);
    P.normalize();
}

void OCAMCamera::liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const {
  // Relative to Center
  Eigen::Vector2d xc(p[0] - m_center_x, p[1] - m_center_y);

  // Affine Transformation
  // xc_a = inv(A) * xc;
  Eigen::Vector2d xc_a(
      m_inv_scale * (xc[0] - m_D * xc[1]),
      m_inv_scale * (-m_E * xc[0] + m_C * xc[1])
  );

  double phi = std::sqrt(xc_a[0] * xc_a[0] + xc_a[1] * xc_a[1]);
  double phi_i = 1.0;
  double z = 0.0;

  for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++) {
    z += phi_i * m_poly[i];
    phi_i *= phi;
  }

  P << xc[0], xc[1], -z;
}

void OCAMCamera::spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const {
  double norm = std::sqrt(P[0] * P[0] + P[1] * P[1]);
  double theta = std::atan2(-P[2], norm);
  double rho = 0.0;
  double theta_i = 1.0;

  for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++) {
    rho += theta_i * m_inv_poly[i];
    theta_i *= theta;
  }

  double invNorm = 1.0 / norm;
  Eigen::Vector2d xn(
      P[0] * invNorm * rho,
      P[1] * invNorm * rho
  );

  p << (xn[0] * m_C + xn[1] * m_D + m_center_x)/2.0,
          (xn[0] * m_E + xn[1] + m_center_y);
}

double OCAMCamera::computeErrorMultiplier(){
  Eigen::Vector3d vector1;
  liftSphere( Eigen::Vector2d(0.5 * mWidth, 0.5 * mHeight),  vector1);
  Eigen::Vector3d vector2;
  liftSphere( Eigen::Vector2d(0.5 * mWidth + 0.5, 0.5 * mHeight), vector2 );

  double factor1 = 0.5 / ( 1 - vector1.dot(vector2) );

  liftSphere(Eigen::Vector2d(mWidth, 0.5 * mHeight), vector1);
  liftSphere(Eigen::Vector2d(-0.5 + (double) mWidth , 0.5 * mHeight), vector2 );

  double factor2 = 0.5/( 1 - vector1.dot(vector2) );

  return ( factor2 + factor1 ) * 0.5;
}

}
