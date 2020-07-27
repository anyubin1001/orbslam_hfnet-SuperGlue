// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Modified by Raúl Mur Artal (2014)
// Added EdgeSE3ProjectXYZ (project using focal_length in x,y directions)
// Modified by Raúl Mur Artal (2016)
// Added EdgeStereoSE3ProjectXYZ (project using focal_length in x,y directions)
// Added EdgeSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)
// Added EdgeStereoSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include <iostream>
#include "../core/base_vertex.h"
#include "../core/base_binary_edge.h"
#include "../core/base_unary_edge.h"
#include "se3_ops.h"
#include "se3quat.h"
#include "types_sba.h"
#include <Eigen/Geometry>
#include "../../ceres/autodiff.h"
#include "Thirdparty/models/include/Camera.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace g2o {
namespace types_six_dof_expmap {
void init();
}

using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;

/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
class VertexSE3Expmap : public BaseVertex<6, SE3Quat> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }

  virtual void oplusImpl(const double *update_) {
    Eigen::Map<const Vector6d> update(update_);
    setEstimate(SE3Quat::exp(update) * estimate());
  }
};

class EdgeSE3ProjectXYZ : public BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZ();

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
    const VertexSBAPointXYZ *v2 = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
    const VertexSBAPointXYZ *v2 = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2) > 0.0;
  }

  template<typename T>
  bool operator()(const T *point3d, T *error) const {
    T p[3];
    for (int i = 0; i < 3; i++) {
      p[i] = point3d[i];
    }

    T c     = T(mCamera.m_C);
    T d     = T(mCamera.m_D);
    T e     = T(mCamera.m_E);
    T xc[2] = { T(mCamera.m_center_x), T(mCamera.m_center_y) };

    T inv_poly[SCARAMUZZA_INV_POLY_SIZE];
    for ( int i     = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++ )
      inv_poly[i] = T(mCamera.m_inv_poly[i]);

    T norm_sqr = p[0] * p[0] + p[1] * p[1];
    T norm     = T( 0.0 );
    if ( norm_sqr > T( 0.0 ) )
      norm = sqrt( norm_sqr );

    T theta   = atan2( -p[2], norm );
    T rho     = T( 0.0 );
    T theta_i = T( 1.0 );

    for ( int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++ )
    {
      rho += theta_i * inv_poly[i];
      theta_i *= theta;
    }

    T invNorm = T( 1.0 ) / norm;
    T xn[2]   = { p[0] * invNorm * rho, p[1] * invNorm * rho };

    error[0] = xn[0] * c + xn[1] * d + xc[0];
    error[1] = xn[0] * e + xn[1] + xc[1];

    return true;
  }

  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d &trans_xyz) const;

  ORB_SLAM2::OCAMCamera mCamera;
};

class EdgeSE3ProjectXYZOnlyPose : public BaseUnaryEdge<2, Vector2d, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyPose() {}

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    return (v1->estimate().map(Xw))(2) > 0.0;
  }

  template<typename T>
  bool operator()(const T *point3d, T *error) const {
    T p[3];
    for (int i = 0; i < 3; i++) {
      p[i] = point3d[i];
    }

    T c     = T(mCamera.m_C);
    T d     = T(mCamera.m_D);
    T e     = T(mCamera.m_E);
    T xc[2] = { T(mCamera.m_center_x), T(mCamera.m_center_y) };

    T inv_poly[SCARAMUZZA_INV_POLY_SIZE];
    for ( int i     = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++ )
      inv_poly[i] = T(mCamera.m_inv_poly[i]);

    T norm_sqr = p[0] * p[0] + p[1] * p[1];
    T norm     = T( 0.0 );
    if ( norm_sqr > T( 0.0 ) )
      norm = sqrt( norm_sqr );

    T theta   = atan2( -p[2], norm );
    T rho     = T( 0.0 );
    T theta_i = T( 1.0 );

    for ( int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++ )
    {
      rho += theta_i * inv_poly[i];
      theta_i *= theta;
    }

    T invNorm = T( 1.0 ) / norm;
    T xn[2]   = { p[0] * invNorm * rho, p[1] * invNorm * rho };

    error[0] = xn[0] * c + xn[1] * d + xc[0];
    error[1] = xn[0] * e + xn[1] + xc[1];

    return true;
  }

  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d &trans_xyz) const;

  Vector3d Xw;

  ORB_SLAM2::OCAMCamera mCamera;
};

} // end namespace

#endif
