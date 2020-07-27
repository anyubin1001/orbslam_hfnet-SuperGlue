/******************************************************************************
 * Author:   Laurent Kneip                                                    *
 * Contact:  kneip.laurent@gmail.com                                          *
 * License:  Copyright (c) 2013 Laurent Kneip, ANU. All rights reserved.      *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 * * Redistributions of source code must retain the above copyright           *
 *   notice, this list of conditions and the following disclaimer.            *
 * * Redistributions in binary form must reproduce the above copyright        *
 *   notice, this list of conditions and the following disclaimer in the      *
 *   documentation and/or other materials provided with the distribution.     *
 * * Neither the name of ANU nor the names of its contributors may be         *
 *   used to endorse or promote products derived from this software without   *
 *   specific prior written permission.                                       *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"*
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE  *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE *
 * ARE DISCLAIMED. IN NO EVENT SHALL ANU OR THE CONTRIBUTORS BE LIABLE        *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL *
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT         *
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY  *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF     *
 * SUCH DAMAGE.                                                               *
 ******************************************************************************/

/**
 * \file MANoncentralRelativeMulti.hpp
 * \brief Adapter-class for passing bearing-vector correspondences to the
 *        non-central relative-pose algorithms. Maps Matlab types
 *        to opengv types. Manages multiple match-lists for pairs of cameras.
 *        This allows to draw samples homogeneously over the cameras.
 */

#ifndef OPENGV_RELATIVE_POSE_MANONCENTRALRELATIVEMULTI_HPP_
#define OPENGV_RELATIVE_POSE_MANONCENTRALRELATIVEMULTI_HPP_

#include <stdlib.h>
#include <vector>
#include <opengv/types.hpp>
#include <opengv/relative_pose/RelativeMultiAdapterBase.hpp>

/**
 * \brief The namespace of this library.
 */
namespace opengv {
/**
 * \brief The namespace for the relative pose methods.
 */
namespace relative_pose {

/**
 * Check the documentation of the parent-class to understand the meaning of
 * a RelativeMultiAdapter. This child-class is for the relative non-central case
 * and holds data in form of pointers to matlab-types. It is meant to be used
 * for problems involving two non-central viewpoints, but in the special case
 * where correspondences result from two cameras with equal transformation
 * to their two viewpoints.
 */
class MANoncentralRelativeMulti : public RelativeMultiAdapterBase {
 protected:
  using RelativeMultiAdapterBase::_t12;
  using RelativeMultiAdapterBase::_R12;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * \brief Constructor. See protected class-members to understand parameters
   */
  MANoncentralRelativeMulti(
      const std::vector<double *> &bearingVectors1,
      const std::vector<double *> &bearingVectors2,
      const double *camOffsets,
      const std::vector<int> &numberBearingVectors);
  /**
   * \brief Destructor.
   */
  virtual ~MANoncentralRelativeMulti();

  //camera-pair-wise access of correspondences

  /** See parent-class */
  virtual bearingVector_t getBearingVector1(
      size_t pairIndex, size_t correspondenceIndex) const;
  /** See parent-class */
  virtual bearingVector_t getBearingVector2(
      size_t pairIndex, size_t correspondenceIndex) const;
  /** See parent-class */
  virtual double getWeight(size_t camIndex, size_t correspondenceIndex) const;
  /** See parent-class */
  virtual translation_t getCamOffset(size_t pairIndex) const;
  /** See parent-class */
  virtual rotation_t getCamRotation(size_t pairIndex) const;
  /** See parent-class */
  virtual size_t getNumberCorrespondences(size_t pairIndex) const;
  /** See parent-class */
  virtual size_t getNumberPairs() const;

  //Conversion to and from serialized indices

  /** See parent-class */
  virtual std::vector<int> convertMultiIndices(
      const std::vector<std::vector<int> > &multiIndices) const;
  /** See parent-class */
  virtual int convertMultiIndex(
      size_t camIndex, size_t correspondenceIndex) const;
  /** See parent-class */
  virtual int multiPairIndex(size_t index) const;
  /** See parent-class */
  virtual int multiCorrespondenceIndex(size_t index) const;

 protected:
  /** A pointer to the bearing-vectors in viewpoint 1 */
  std::vector<double *> _bearingVectors1;

  /** A pointer to the bearing-vectors in viewpoint 2 */
  std::vector<double *> _bearingVectors2;

  /** The offset from the viewpoint origin of each vector */
  const double *_camOffsets;

  /** The number of bearing-vectors in each camera */
  std::vector<int> _numberBearingVectors;

  /** Initialized in constructor, used for (de)-serialiaztion of indices */
  std::vector<int> multiPairIndices;
  /** Initialized in constructor, used for (de)-serialiaztion of indices */
  std::vector<int> multiKeypointIndices;
  /** Initialized in constructor, used for (de)-serialiaztion of indices */
  std::vector<int> singleIndexOffsets;
};

}
}

#endif /* OPENGV_RELATIVE_POSE_MANONCENTRALRELATIVEMULTI_HPP_ */
