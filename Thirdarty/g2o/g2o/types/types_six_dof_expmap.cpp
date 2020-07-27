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
#define _USE_MATH_DEFINES
#include <math.h>
#include "types_six_dof_expmap.h"
#include <iostream>

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

using namespace std;

VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {
}

bool VertexSE3Expmap::read(std::istream &is) {
  Vector7d est;
  for (int i = 0; i < 7; i++)
    is >> est[i];
  SE3Quat cam2world;
  cam2world.fromVector(est);
  setEstimate(cam2world.inverse());
  return true;
}

bool VertexSE3Expmap::write(std::ostream &os) const {
  SE3Quat cam2world(estimate().inverse());
  for (int i = 0; i < 7; i++)
    os << cam2world[i] << " ";
  return os.good();
}

EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeSE3ProjectXYZ::read(std::istream &is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream &os) const {

  for (int i = 0; i < 2; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  typedef ceres::internal::AutoDiff<EdgeSE3ProjectXYZ, double, 3> BalAutoDiff;
  Matrix<double, 2, 3, Eigen::RowMajor> dError_dPoint;
  dError_dPoint.setZero();

  double point_array[3];
  for (int i = 0; i < 3; i++) {
    point_array[i] = xyz_trans[i];
  }

  double *parameters[] = {point_array};
  double *jacobians[] = {dError_dPoint.data()};
  double value[2];
  bool diffState = BalAutoDiff::Differentiate(*this, parameters, 2, value, jacobians);

  Matrix<double, 2, 3, ColMajor> Acolmajor = dError_dPoint;
  Matrix<double, 3, 6, ColMajor> dPdXi;
  dPdXi << 0, xyz_trans[2], -xyz_trans[1], 1, 0, 0,
      -xyz_trans[2], 0, xyz_trans[0], 0, 1, 0,
      xyz_trans[1], -xyz_trans[0], 0, 0, 0, 1;

  if (diffState) {
    _jacobianOplusXi = -Acolmajor * T.rotation().toRotationMatrix();
    _jacobianOplusXj = -Acolmajor * dPdXi;
  } else {
    _jacobianOplusXi.setZero();
    _jacobianOplusXj.setZero();
  }
}

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d &trans_xyz) const {
  Vector2d res;
  mCamera.spaceToPlane(trans_xyz, res);
  return res;
}
//Only Pose

bool EdgeSE3ProjectXYZOnlyPose::read(std::istream &is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream &os) const {

  for (int i = 0; i < 2; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);

  Vector3d xyz_trans = vi->estimate().map(Xw);

  typedef ceres::internal::AutoDiff<EdgeSE3ProjectXYZOnlyPose, double, 3> BalAutoDiff;
  Matrix<double, 2, 3, Eigen::RowMajor> dError_dPoint;
  dError_dPoint.setZero();

  double point_array[3];
  for (int i = 0; i < 3; i++) {
    point_array[i] = xyz_trans[i];
  }

  double *parameters[] = {point_array};
  double *jacobians[] = {dError_dPoint.data()};
  double value[2];
  bool diffState = BalAutoDiff::Differentiate(*this, parameters, 2, value, jacobians);

  Matrix<double, 2, 3, ColMajor> Acolmajor = dError_dPoint;
  Matrix<double, 3, 6, ColMajor> dPdXi;
  dPdXi << 0, xyz_trans[2], -xyz_trans[1], 1, 0, 0,
      -xyz_trans[2], 0, xyz_trans[0], 0, 1, 0,
      xyz_trans[1], -xyz_trans[0], 0, 0, 0, 1;

  if (diffState) {
    _jacobianOplusXi = -Acolmajor * dPdXi;
  } else {
    _jacobianOplusXi.setZero();
  }
}

// camera coordinate to image coordinate
Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d &trans_xyz) const {
  Vector2d res;
  mCamera.spaceToPlane(trans_xyz, res);
  return res;
}

} // end namespace
