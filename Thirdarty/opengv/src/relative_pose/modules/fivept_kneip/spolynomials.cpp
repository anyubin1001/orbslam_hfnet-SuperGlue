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


#include <opengv/relative_pose/modules/fivept_kneip/modules.hpp>

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial30(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(30, 1) =
      (groebnerMatrix(0, 1) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 1) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 2) =
      (groebnerMatrix(0, 2) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 2) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 3) =
      (groebnerMatrix(0, 3) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 3) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 4) =
      (groebnerMatrix(0, 4) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 4) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 5) =
      (groebnerMatrix(0, 5) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 5) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 6) =
      (groebnerMatrix(0, 6) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 6) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 8) =
      (groebnerMatrix(0, 8) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 8) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 10) =
      (groebnerMatrix(0, 10) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 10) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 11) =
      (groebnerMatrix(0, 11) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 11) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 12) =
      (groebnerMatrix(0, 12) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 12) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 13) =
      (groebnerMatrix(0, 13) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 13) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 14) =
      (groebnerMatrix(0, 14) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 14) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 15) =
      (groebnerMatrix(0, 15) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 15) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 16) =
      (groebnerMatrix(0, 16) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 16) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 18) =
      (groebnerMatrix(0, 18) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 18) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 19) =
      (groebnerMatrix(0, 19) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 19) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 20) =
      (groebnerMatrix(0, 20) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 20) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 21) =
      (groebnerMatrix(0, 21) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 21) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 22) =
      (groebnerMatrix(0, 22) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 22) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 23) =
      (groebnerMatrix(0, 23) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 23) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 25) =
      (groebnerMatrix(0, 25) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 25) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 26) =
      (groebnerMatrix(0, 26) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 26) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 27) =
      (groebnerMatrix(0, 27) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 27) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 28) =
      (groebnerMatrix(0, 28) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 28) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 29) =
      (groebnerMatrix(0, 29) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 29) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 30) =
      (groebnerMatrix(0, 30) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 30) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 33) =
      (groebnerMatrix(0, 33) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 33) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 34) =
      (groebnerMatrix(0, 34) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 34) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 39) =
      (groebnerMatrix(0, 39) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 39) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 40) =
      (groebnerMatrix(0, 40) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 40) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 41) =
      (groebnerMatrix(0, 41) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 41) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 42) =
      (groebnerMatrix(0, 42) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 42) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 43) =
      (groebnerMatrix(0, 43) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 43) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 44) =
      (groebnerMatrix(0, 44) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 44) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 45) =
      (groebnerMatrix(0, 45) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 45) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 46) =
      (groebnerMatrix(0, 46) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 46) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 47) =
      (groebnerMatrix(0, 47) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 47) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 48) =
      (groebnerMatrix(0, 48) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 48) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 49) =
      (groebnerMatrix(0, 49) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 49) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 50) =
      (groebnerMatrix(0, 50) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 50) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 51) =
      (groebnerMatrix(0, 51) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 51) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 52) =
      (groebnerMatrix(0, 52) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 52) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 53) =
      (groebnerMatrix(0, 53) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 53) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 54) =
      (groebnerMatrix(0, 54) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 54) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 55) =
      (groebnerMatrix(0, 55) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 55) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 56) =
      (groebnerMatrix(0, 56) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 56) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 57) =
      (groebnerMatrix(0, 57) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 57) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 58) =
      (groebnerMatrix(0, 58) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 58) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 59) =
      (groebnerMatrix(0, 59) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 59) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 60) =
      (groebnerMatrix(0, 60) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 60) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 61) =
      (groebnerMatrix(0, 61) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 61) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 62) =
      (groebnerMatrix(0, 62) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 62) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 64) =
      (groebnerMatrix(0, 64) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 64) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 65) =
      (groebnerMatrix(0, 65) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 65) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 66) =
      (groebnerMatrix(0, 66) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 66) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 67) =
      (groebnerMatrix(0, 67) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 67) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 68) =
      (groebnerMatrix(0, 68) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 68) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 69) =
      (groebnerMatrix(0, 69) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 69) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 70) =
      (groebnerMatrix(0, 70) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 70) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 71) =
      (groebnerMatrix(0, 71) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 71) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 73) =
      (groebnerMatrix(0, 73) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 73) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 74) =
      (groebnerMatrix(0, 74) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 74) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 76) =
      (groebnerMatrix(0, 76) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 76) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 77) =
      (groebnerMatrix(0, 77) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 77) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 78) =
      (groebnerMatrix(0, 78) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 78) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 79) =
      (groebnerMatrix(0, 79) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 79) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 80) =
      (groebnerMatrix(0, 80) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 80) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 81) =
      (groebnerMatrix(0, 81) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 81) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 82) =
      (groebnerMatrix(0, 82) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 82) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 83) =
      (groebnerMatrix(0, 83) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 83) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 84) =
      (groebnerMatrix(0, 84) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 84) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 85) =
      (groebnerMatrix(0, 85) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 85) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 86) =
      (groebnerMatrix(0, 86) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 86) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 87) =
      (groebnerMatrix(0, 87) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 87) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 89) =
      (groebnerMatrix(0, 89) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 89) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 91) =
      (groebnerMatrix(0, 91) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 91) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 92) =
      (groebnerMatrix(0, 92) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 92) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 94) =
      (groebnerMatrix(0, 94) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 94) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 97) =
      (groebnerMatrix(0, 97) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 97) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 98) =
      (groebnerMatrix(0, 98) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 98) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 99) =
      (groebnerMatrix(0, 99) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 99) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 100) =
      (groebnerMatrix(0, 100) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 100) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 101) =
      (groebnerMatrix(0, 101) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 101) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 103) =
      (groebnerMatrix(0, 103) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 103) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 104) =
      (groebnerMatrix(0, 104) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 104) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 105) =
      (groebnerMatrix(0, 105) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 105) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 106) =
      (groebnerMatrix(0, 106) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 106) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 107) =
      (groebnerMatrix(0, 107) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 107) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 108) =
      (groebnerMatrix(0, 108) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 108) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 109) =
      (groebnerMatrix(0, 109) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 109) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 110) =
      (groebnerMatrix(0, 110) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 110) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 111) =
      (groebnerMatrix(0, 111) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 111) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 112) =
      (groebnerMatrix(0, 112) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 112) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 113) =
      (groebnerMatrix(0, 113) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 113) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 115) =
      (groebnerMatrix(0, 115) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 115) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 116) =
      (groebnerMatrix(0, 116) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 116) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 118) =
      (groebnerMatrix(0, 118) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 118) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 119) =
      (groebnerMatrix(0, 119) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 119) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 120) =
      (groebnerMatrix(0, 120) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 120) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 121) =
      (groebnerMatrix(0, 121) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 121) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 122) =
      (groebnerMatrix(0, 122) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 122) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 123) =
      (groebnerMatrix(0, 123) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 123) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 125) =
      (groebnerMatrix(0, 125) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 125) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 126) =
      (groebnerMatrix(0, 126) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 126) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 127) =
      (groebnerMatrix(0, 127) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 127) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 128) =
      (groebnerMatrix(0, 128) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 128) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 129) =
      (groebnerMatrix(0, 129) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 129) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 130) =
      (groebnerMatrix(0, 130) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 130) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 133) =
      (groebnerMatrix(0, 133) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 133) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 134) =
      (groebnerMatrix(0, 134) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 134) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 136) =
      (groebnerMatrix(0, 136) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 136) / (groebnerMatrix(1, 0)));
  groebnerMatrix(30, 137) =
      (groebnerMatrix(0, 137) / (groebnerMatrix(0, 0)) - groebnerMatrix(1, 137) / (groebnerMatrix(1, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial31(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(31, 1) =
      (groebnerMatrix(1, 1) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 1) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 2) =
      (groebnerMatrix(1, 2) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 2) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 3) =
      (groebnerMatrix(1, 3) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 3) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 4) =
      (groebnerMatrix(1, 4) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 4) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 5) =
      (groebnerMatrix(1, 5) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 5) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 6) =
      (groebnerMatrix(1, 6) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 6) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 8) =
      (groebnerMatrix(1, 8) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 8) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 10) =
      (groebnerMatrix(1, 10) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 10) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 11) =
      (groebnerMatrix(1, 11) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 11) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 12) =
      (groebnerMatrix(1, 12) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 12) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 13) =
      (groebnerMatrix(1, 13) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 13) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 14) =
      (groebnerMatrix(1, 14) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 14) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 15) =
      (groebnerMatrix(1, 15) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 15) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 16) =
      (groebnerMatrix(1, 16) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 16) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 18) =
      (groebnerMatrix(1, 18) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 18) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 19) =
      (groebnerMatrix(1, 19) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 19) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 20) =
      (groebnerMatrix(1, 20) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 20) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 21) =
      (groebnerMatrix(1, 21) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 21) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 22) =
      (groebnerMatrix(1, 22) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 22) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 23) =
      (groebnerMatrix(1, 23) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 23) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 25) =
      (groebnerMatrix(1, 25) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 25) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 26) =
      (groebnerMatrix(1, 26) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 26) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 27) =
      (groebnerMatrix(1, 27) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 27) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 28) =
      (groebnerMatrix(1, 28) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 28) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 29) =
      (groebnerMatrix(1, 29) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 29) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 30) =
      (groebnerMatrix(1, 30) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 30) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 33) =
      (groebnerMatrix(1, 33) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 33) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 34) =
      (groebnerMatrix(1, 34) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 34) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 39) =
      (groebnerMatrix(1, 39) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 39) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 40) =
      (groebnerMatrix(1, 40) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 40) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 41) =
      (groebnerMatrix(1, 41) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 41) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 42) =
      (groebnerMatrix(1, 42) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 42) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 43) =
      (groebnerMatrix(1, 43) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 43) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 44) =
      (groebnerMatrix(1, 44) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 44) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 45) =
      (groebnerMatrix(1, 45) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 45) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 46) =
      (groebnerMatrix(1, 46) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 46) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 47) =
      (groebnerMatrix(1, 47) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 47) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 48) =
      (groebnerMatrix(1, 48) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 48) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 49) =
      (groebnerMatrix(1, 49) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 49) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 50) =
      (groebnerMatrix(1, 50) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 50) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 51) =
      (groebnerMatrix(1, 51) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 51) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 52) =
      (groebnerMatrix(1, 52) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 52) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 53) =
      (groebnerMatrix(1, 53) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 53) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 54) =
      (groebnerMatrix(1, 54) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 54) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 55) =
      (groebnerMatrix(1, 55) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 55) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 56) =
      (groebnerMatrix(1, 56) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 56) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 57) =
      (groebnerMatrix(1, 57) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 57) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 58) =
      (groebnerMatrix(1, 58) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 58) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 59) =
      (groebnerMatrix(1, 59) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 59) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 60) =
      (groebnerMatrix(1, 60) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 60) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 61) =
      (groebnerMatrix(1, 61) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 61) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 62) =
      (groebnerMatrix(1, 62) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 62) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 64) =
      (groebnerMatrix(1, 64) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 64) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 65) =
      (groebnerMatrix(1, 65) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 65) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 66) =
      (groebnerMatrix(1, 66) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 66) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 67) =
      (groebnerMatrix(1, 67) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 67) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 68) =
      (groebnerMatrix(1, 68) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 68) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 69) =
      (groebnerMatrix(1, 69) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 69) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 70) =
      (groebnerMatrix(1, 70) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 70) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 71) =
      (groebnerMatrix(1, 71) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 71) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 73) =
      (groebnerMatrix(1, 73) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 73) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 74) =
      (groebnerMatrix(1, 74) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 74) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 76) =
      (groebnerMatrix(1, 76) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 76) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 77) =
      (groebnerMatrix(1, 77) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 77) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 78) =
      (groebnerMatrix(1, 78) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 78) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 79) =
      (groebnerMatrix(1, 79) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 79) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 80) =
      (groebnerMatrix(1, 80) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 80) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 81) =
      (groebnerMatrix(1, 81) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 81) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 82) =
      (groebnerMatrix(1, 82) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 82) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 83) =
      (groebnerMatrix(1, 83) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 83) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 84) =
      (groebnerMatrix(1, 84) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 84) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 85) =
      (groebnerMatrix(1, 85) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 85) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 86) =
      (groebnerMatrix(1, 86) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 86) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 87) =
      (groebnerMatrix(1, 87) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 87) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 89) =
      (groebnerMatrix(1, 89) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 89) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 91) =
      (groebnerMatrix(1, 91) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 91) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 92) =
      (groebnerMatrix(1, 92) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 92) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 94) =
      (groebnerMatrix(1, 94) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 94) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 97) =
      (groebnerMatrix(1, 97) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 97) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 98) =
      (groebnerMatrix(1, 98) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 98) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 99) =
      (groebnerMatrix(1, 99) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 99) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 100) =
      (groebnerMatrix(1, 100) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 100) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 101) =
      (groebnerMatrix(1, 101) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 101) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 103) =
      (groebnerMatrix(1, 103) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 103) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 104) =
      (groebnerMatrix(1, 104) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 104) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 105) =
      (groebnerMatrix(1, 105) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 105) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 106) =
      (groebnerMatrix(1, 106) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 106) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 107) =
      (groebnerMatrix(1, 107) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 107) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 108) =
      (groebnerMatrix(1, 108) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 108) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 109) =
      (groebnerMatrix(1, 109) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 109) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 110) =
      (groebnerMatrix(1, 110) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 110) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 111) =
      (groebnerMatrix(1, 111) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 111) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 112) =
      (groebnerMatrix(1, 112) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 112) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 113) =
      (groebnerMatrix(1, 113) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 113) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 115) =
      (groebnerMatrix(1, 115) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 115) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 116) =
      (groebnerMatrix(1, 116) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 116) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 118) =
      (groebnerMatrix(1, 118) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 118) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 119) =
      (groebnerMatrix(1, 119) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 119) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 120) =
      (groebnerMatrix(1, 120) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 120) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 121) =
      (groebnerMatrix(1, 121) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 121) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 122) =
      (groebnerMatrix(1, 122) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 122) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 123) =
      (groebnerMatrix(1, 123) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 123) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 125) =
      (groebnerMatrix(1, 125) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 125) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 126) =
      (groebnerMatrix(1, 126) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 126) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 127) =
      (groebnerMatrix(1, 127) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 127) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 128) =
      (groebnerMatrix(1, 128) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 128) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 129) =
      (groebnerMatrix(1, 129) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 129) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 130) =
      (groebnerMatrix(1, 130) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 130) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 133) =
      (groebnerMatrix(1, 133) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 133) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 134) =
      (groebnerMatrix(1, 134) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 134) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 136) =
      (groebnerMatrix(1, 136) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 136) / (groebnerMatrix(2, 0)));
  groebnerMatrix(31, 137) =
      (groebnerMatrix(1, 137) / (groebnerMatrix(1, 0)) - groebnerMatrix(2, 137) / (groebnerMatrix(2, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial32(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(32, 1) =
      (groebnerMatrix(2, 1) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 1) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 2) =
      (groebnerMatrix(2, 2) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 2) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 3) =
      (groebnerMatrix(2, 3) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 3) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 4) =
      (groebnerMatrix(2, 4) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 4) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 5) =
      (groebnerMatrix(2, 5) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 5) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 6) =
      (groebnerMatrix(2, 6) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 6) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 8) =
      (groebnerMatrix(2, 8) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 8) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 10) =
      (groebnerMatrix(2, 10) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 10) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 11) =
      (groebnerMatrix(2, 11) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 11) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 12) =
      (groebnerMatrix(2, 12) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 12) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 13) =
      (groebnerMatrix(2, 13) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 13) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 14) =
      (groebnerMatrix(2, 14) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 14) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 15) =
      (groebnerMatrix(2, 15) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 15) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 16) =
      (groebnerMatrix(2, 16) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 16) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 18) =
      (groebnerMatrix(2, 18) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 18) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 19) =
      (groebnerMatrix(2, 19) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 19) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 20) =
      (groebnerMatrix(2, 20) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 20) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 21) =
      (groebnerMatrix(2, 21) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 21) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 22) =
      (groebnerMatrix(2, 22) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 22) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 23) =
      (groebnerMatrix(2, 23) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 23) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 25) =
      (groebnerMatrix(2, 25) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 25) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 26) =
      (groebnerMatrix(2, 26) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 26) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 27) =
      (groebnerMatrix(2, 27) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 27) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 28) =
      (groebnerMatrix(2, 28) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 28) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 29) =
      (groebnerMatrix(2, 29) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 29) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 30) =
      (groebnerMatrix(2, 30) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 30) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 33) =
      (groebnerMatrix(2, 33) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 33) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 34) =
      (groebnerMatrix(2, 34) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 34) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 39) =
      (groebnerMatrix(2, 39) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 39) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 40) =
      (groebnerMatrix(2, 40) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 40) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 41) =
      (groebnerMatrix(2, 41) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 41) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 42) =
      (groebnerMatrix(2, 42) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 42) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 43) =
      (groebnerMatrix(2, 43) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 43) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 44) =
      (groebnerMatrix(2, 44) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 44) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 45) =
      (groebnerMatrix(2, 45) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 45) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 46) =
      (groebnerMatrix(2, 46) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 46) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 47) =
      (groebnerMatrix(2, 47) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 47) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 48) =
      (groebnerMatrix(2, 48) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 48) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 49) =
      (groebnerMatrix(2, 49) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 49) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 50) =
      (groebnerMatrix(2, 50) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 50) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 51) =
      (groebnerMatrix(2, 51) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 51) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 52) =
      (groebnerMatrix(2, 52) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 52) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 53) =
      (groebnerMatrix(2, 53) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 53) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 54) =
      (groebnerMatrix(2, 54) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 54) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 55) =
      (groebnerMatrix(2, 55) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 55) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 56) =
      (groebnerMatrix(2, 56) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 56) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 57) =
      (groebnerMatrix(2, 57) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 57) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 58) =
      (groebnerMatrix(2, 58) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 58) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 59) =
      (groebnerMatrix(2, 59) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 59) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 60) =
      (groebnerMatrix(2, 60) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 60) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 61) =
      (groebnerMatrix(2, 61) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 61) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 62) =
      (groebnerMatrix(2, 62) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 62) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 64) =
      (groebnerMatrix(2, 64) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 64) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 65) =
      (groebnerMatrix(2, 65) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 65) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 66) =
      (groebnerMatrix(2, 66) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 66) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 67) =
      (groebnerMatrix(2, 67) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 67) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 68) =
      (groebnerMatrix(2, 68) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 68) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 69) =
      (groebnerMatrix(2, 69) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 69) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 70) =
      (groebnerMatrix(2, 70) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 70) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 71) =
      (groebnerMatrix(2, 71) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 71) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 73) =
      (groebnerMatrix(2, 73) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 73) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 74) =
      (groebnerMatrix(2, 74) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 74) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 76) =
      (groebnerMatrix(2, 76) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 76) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 77) =
      (groebnerMatrix(2, 77) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 77) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 78) =
      (groebnerMatrix(2, 78) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 78) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 79) =
      (groebnerMatrix(2, 79) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 79) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 80) =
      (groebnerMatrix(2, 80) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 80) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 81) =
      (groebnerMatrix(2, 81) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 81) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 82) =
      (groebnerMatrix(2, 82) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 82) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 83) =
      (groebnerMatrix(2, 83) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 83) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 84) =
      (groebnerMatrix(2, 84) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 84) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 85) =
      (groebnerMatrix(2, 85) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 85) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 86) =
      (groebnerMatrix(2, 86) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 86) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 87) =
      (groebnerMatrix(2, 87) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 87) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 89) =
      (groebnerMatrix(2, 89) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 89) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 91) =
      (groebnerMatrix(2, 91) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 91) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 92) =
      (groebnerMatrix(2, 92) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 92) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 94) =
      (groebnerMatrix(2, 94) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 94) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 97) =
      (groebnerMatrix(2, 97) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 97) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 98) =
      (groebnerMatrix(2, 98) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 98) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 99) =
      (groebnerMatrix(2, 99) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 99) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 100) =
      (groebnerMatrix(2, 100) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 100) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 101) =
      (groebnerMatrix(2, 101) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 101) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 103) =
      (groebnerMatrix(2, 103) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 103) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 104) =
      (groebnerMatrix(2, 104) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 104) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 105) =
      (groebnerMatrix(2, 105) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 105) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 106) =
      (groebnerMatrix(2, 106) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 106) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 107) =
      (groebnerMatrix(2, 107) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 107) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 108) =
      (groebnerMatrix(2, 108) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 108) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 109) =
      (groebnerMatrix(2, 109) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 109) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 110) =
      (groebnerMatrix(2, 110) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 110) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 111) =
      (groebnerMatrix(2, 111) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 111) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 112) =
      (groebnerMatrix(2, 112) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 112) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 113) =
      (groebnerMatrix(2, 113) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 113) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 115) =
      (groebnerMatrix(2, 115) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 115) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 116) =
      (groebnerMatrix(2, 116) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 116) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 118) =
      (groebnerMatrix(2, 118) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 118) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 119) =
      (groebnerMatrix(2, 119) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 119) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 120) =
      (groebnerMatrix(2, 120) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 120) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 121) =
      (groebnerMatrix(2, 121) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 121) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 122) =
      (groebnerMatrix(2, 122) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 122) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 123) =
      (groebnerMatrix(2, 123) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 123) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 125) =
      (groebnerMatrix(2, 125) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 125) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 126) =
      (groebnerMatrix(2, 126) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 126) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 127) =
      (groebnerMatrix(2, 127) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 127) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 128) =
      (groebnerMatrix(2, 128) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 128) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 129) =
      (groebnerMatrix(2, 129) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 129) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 130) =
      (groebnerMatrix(2, 130) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 130) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 133) =
      (groebnerMatrix(2, 133) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 133) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 134) =
      (groebnerMatrix(2, 134) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 134) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 136) =
      (groebnerMatrix(2, 136) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 136) / (groebnerMatrix(3, 0)));
  groebnerMatrix(32, 137) =
      (groebnerMatrix(2, 137) / (groebnerMatrix(2, 0)) - groebnerMatrix(3, 137) / (groebnerMatrix(3, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial33(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(33, 1) =
      (groebnerMatrix(3, 1) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 1) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 2) =
      (groebnerMatrix(3, 2) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 2) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 3) =
      (groebnerMatrix(3, 3) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 3) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 4) =
      (groebnerMatrix(3, 4) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 4) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 5) =
      (groebnerMatrix(3, 5) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 5) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 6) =
      (groebnerMatrix(3, 6) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 6) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 8) =
      (groebnerMatrix(3, 8) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 8) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 10) =
      (groebnerMatrix(3, 10) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 10) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 11) =
      (groebnerMatrix(3, 11) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 11) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 12) =
      (groebnerMatrix(3, 12) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 12) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 13) =
      (groebnerMatrix(3, 13) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 13) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 14) =
      (groebnerMatrix(3, 14) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 14) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 15) =
      (groebnerMatrix(3, 15) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 15) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 16) =
      (groebnerMatrix(3, 16) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 16) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 18) =
      (groebnerMatrix(3, 18) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 18) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 19) =
      (groebnerMatrix(3, 19) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 19) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 20) =
      (groebnerMatrix(3, 20) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 20) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 21) =
      (groebnerMatrix(3, 21) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 21) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 22) =
      (groebnerMatrix(3, 22) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 22) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 23) =
      (groebnerMatrix(3, 23) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 23) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 25) =
      (groebnerMatrix(3, 25) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 25) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 26) =
      (groebnerMatrix(3, 26) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 26) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 27) =
      (groebnerMatrix(3, 27) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 27) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 28) =
      (groebnerMatrix(3, 28) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 28) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 29) =
      (groebnerMatrix(3, 29) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 29) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 30) =
      (groebnerMatrix(3, 30) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 30) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 33) =
      (groebnerMatrix(3, 33) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 33) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 34) =
      (groebnerMatrix(3, 34) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 34) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 39) =
      (groebnerMatrix(3, 39) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 39) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 40) =
      (groebnerMatrix(3, 40) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 40) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 41) =
      (groebnerMatrix(3, 41) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 41) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 42) =
      (groebnerMatrix(3, 42) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 42) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 43) =
      (groebnerMatrix(3, 43) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 43) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 44) =
      (groebnerMatrix(3, 44) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 44) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 45) =
      (groebnerMatrix(3, 45) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 45) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 46) =
      (groebnerMatrix(3, 46) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 46) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 47) =
      (groebnerMatrix(3, 47) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 47) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 48) =
      (groebnerMatrix(3, 48) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 48) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 49) =
      (groebnerMatrix(3, 49) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 49) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 50) =
      (groebnerMatrix(3, 50) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 50) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 51) =
      (groebnerMatrix(3, 51) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 51) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 52) =
      (groebnerMatrix(3, 52) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 52) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 53) =
      (groebnerMatrix(3, 53) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 53) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 54) =
      (groebnerMatrix(3, 54) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 54) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 55) =
      (groebnerMatrix(3, 55) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 55) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 56) =
      (groebnerMatrix(3, 56) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 56) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 57) =
      (groebnerMatrix(3, 57) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 57) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 58) =
      (groebnerMatrix(3, 58) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 58) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 59) =
      (groebnerMatrix(3, 59) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 59) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 60) =
      (groebnerMatrix(3, 60) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 60) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 61) =
      (groebnerMatrix(3, 61) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 61) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 62) =
      (groebnerMatrix(3, 62) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 62) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 64) =
      (groebnerMatrix(3, 64) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 64) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 65) =
      (groebnerMatrix(3, 65) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 65) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 66) =
      (groebnerMatrix(3, 66) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 66) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 67) =
      (groebnerMatrix(3, 67) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 67) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 68) =
      (groebnerMatrix(3, 68) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 68) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 69) =
      (groebnerMatrix(3, 69) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 69) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 70) =
      (groebnerMatrix(3, 70) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 70) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 71) =
      (groebnerMatrix(3, 71) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 71) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 73) =
      (groebnerMatrix(3, 73) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 73) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 74) =
      (groebnerMatrix(3, 74) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 74) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 76) =
      (groebnerMatrix(3, 76) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 76) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 77) =
      (groebnerMatrix(3, 77) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 77) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 78) =
      (groebnerMatrix(3, 78) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 78) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 79) =
      (groebnerMatrix(3, 79) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 79) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 80) =
      (groebnerMatrix(3, 80) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 80) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 81) =
      (groebnerMatrix(3, 81) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 81) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 82) =
      (groebnerMatrix(3, 82) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 82) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 83) =
      (groebnerMatrix(3, 83) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 83) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 84) =
      (groebnerMatrix(3, 84) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 84) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 85) =
      (groebnerMatrix(3, 85) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 85) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 86) =
      (groebnerMatrix(3, 86) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 86) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 87) =
      (groebnerMatrix(3, 87) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 87) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 89) =
      (groebnerMatrix(3, 89) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 89) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 91) =
      (groebnerMatrix(3, 91) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 91) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 92) =
      (groebnerMatrix(3, 92) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 92) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 94) =
      (groebnerMatrix(3, 94) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 94) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 97) =
      (groebnerMatrix(3, 97) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 97) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 98) =
      (groebnerMatrix(3, 98) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 98) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 99) =
      (groebnerMatrix(3, 99) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 99) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 100) =
      (groebnerMatrix(3, 100) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 100) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 101) =
      (groebnerMatrix(3, 101) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 101) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 103) =
      (groebnerMatrix(3, 103) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 103) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 104) =
      (groebnerMatrix(3, 104) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 104) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 105) =
      (groebnerMatrix(3, 105) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 105) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 106) =
      (groebnerMatrix(3, 106) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 106) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 107) =
      (groebnerMatrix(3, 107) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 107) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 108) =
      (groebnerMatrix(3, 108) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 108) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 109) =
      (groebnerMatrix(3, 109) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 109) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 110) =
      (groebnerMatrix(3, 110) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 110) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 111) =
      (groebnerMatrix(3, 111) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 111) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 112) =
      (groebnerMatrix(3, 112) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 112) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 113) =
      (groebnerMatrix(3, 113) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 113) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 115) =
      (groebnerMatrix(3, 115) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 115) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 116) =
      (groebnerMatrix(3, 116) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 116) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 118) =
      (groebnerMatrix(3, 118) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 118) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 119) =
      (groebnerMatrix(3, 119) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 119) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 120) =
      (groebnerMatrix(3, 120) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 120) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 121) =
      (groebnerMatrix(3, 121) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 121) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 122) =
      (groebnerMatrix(3, 122) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 122) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 123) =
      (groebnerMatrix(3, 123) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 123) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 125) =
      (groebnerMatrix(3, 125) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 125) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 126) =
      (groebnerMatrix(3, 126) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 126) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 127) =
      (groebnerMatrix(3, 127) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 127) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 128) =
      (groebnerMatrix(3, 128) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 128) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 129) =
      (groebnerMatrix(3, 129) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 129) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 130) =
      (groebnerMatrix(3, 130) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 130) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 133) =
      (groebnerMatrix(3, 133) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 133) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 134) =
      (groebnerMatrix(3, 134) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 134) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 136) =
      (groebnerMatrix(3, 136) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 136) / (groebnerMatrix(4, 0)));
  groebnerMatrix(33, 137) =
      (groebnerMatrix(3, 137) / (groebnerMatrix(3, 0)) - groebnerMatrix(4, 137) / (groebnerMatrix(4, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial34(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(34, 1) =
      (groebnerMatrix(4, 1) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 1) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 2) =
      (groebnerMatrix(4, 2) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 2) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 3) =
      (groebnerMatrix(4, 3) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 3) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 4) =
      (groebnerMatrix(4, 4) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 4) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 5) =
      (groebnerMatrix(4, 5) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 5) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 6) =
      (groebnerMatrix(4, 6) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 6) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 8) =
      (groebnerMatrix(4, 8) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 8) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 10) =
      (groebnerMatrix(4, 10) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 10) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 11) =
      (groebnerMatrix(4, 11) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 11) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 12) =
      (groebnerMatrix(4, 12) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 12) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 13) =
      (groebnerMatrix(4, 13) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 13) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 14) =
      (groebnerMatrix(4, 14) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 14) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 15) =
      (groebnerMatrix(4, 15) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 15) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 16) =
      (groebnerMatrix(4, 16) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 16) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 18) =
      (groebnerMatrix(4, 18) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 18) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 19) =
      (groebnerMatrix(4, 19) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 19) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 20) =
      (groebnerMatrix(4, 20) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 20) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 21) =
      (groebnerMatrix(4, 21) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 21) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 22) =
      (groebnerMatrix(4, 22) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 22) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 23) =
      (groebnerMatrix(4, 23) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 23) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 25) =
      (groebnerMatrix(4, 25) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 25) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 26) =
      (groebnerMatrix(4, 26) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 26) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 27) =
      (groebnerMatrix(4, 27) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 27) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 28) =
      (groebnerMatrix(4, 28) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 28) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 29) =
      (groebnerMatrix(4, 29) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 29) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 30) =
      (groebnerMatrix(4, 30) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 30) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 33) =
      (groebnerMatrix(4, 33) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 33) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 34) =
      (groebnerMatrix(4, 34) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 34) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 39) =
      (groebnerMatrix(4, 39) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 39) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 40) =
      (groebnerMatrix(4, 40) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 40) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 41) =
      (groebnerMatrix(4, 41) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 41) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 42) =
      (groebnerMatrix(4, 42) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 42) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 43) =
      (groebnerMatrix(4, 43) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 43) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 44) =
      (groebnerMatrix(4, 44) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 44) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 45) =
      (groebnerMatrix(4, 45) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 45) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 46) =
      (groebnerMatrix(4, 46) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 46) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 47) =
      (groebnerMatrix(4, 47) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 47) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 48) =
      (groebnerMatrix(4, 48) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 48) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 49) =
      (groebnerMatrix(4, 49) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 49) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 50) =
      (groebnerMatrix(4, 50) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 50) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 51) =
      (groebnerMatrix(4, 51) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 51) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 52) =
      (groebnerMatrix(4, 52) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 52) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 53) =
      (groebnerMatrix(4, 53) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 53) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 54) =
      (groebnerMatrix(4, 54) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 54) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 55) =
      (groebnerMatrix(4, 55) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 55) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 56) =
      (groebnerMatrix(4, 56) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 56) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 57) =
      (groebnerMatrix(4, 57) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 57) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 58) =
      (groebnerMatrix(4, 58) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 58) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 59) =
      (groebnerMatrix(4, 59) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 59) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 60) =
      (groebnerMatrix(4, 60) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 60) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 61) =
      (groebnerMatrix(4, 61) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 61) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 62) =
      (groebnerMatrix(4, 62) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 62) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 64) =
      (groebnerMatrix(4, 64) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 64) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 65) =
      (groebnerMatrix(4, 65) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 65) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 66) =
      (groebnerMatrix(4, 66) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 66) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 67) =
      (groebnerMatrix(4, 67) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 67) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 68) =
      (groebnerMatrix(4, 68) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 68) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 69) =
      (groebnerMatrix(4, 69) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 69) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 70) =
      (groebnerMatrix(4, 70) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 70) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 71) =
      (groebnerMatrix(4, 71) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 71) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 73) =
      (groebnerMatrix(4, 73) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 73) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 74) =
      (groebnerMatrix(4, 74) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 74) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 76) =
      (groebnerMatrix(4, 76) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 76) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 77) =
      (groebnerMatrix(4, 77) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 77) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 78) =
      (groebnerMatrix(4, 78) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 78) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 79) =
      (groebnerMatrix(4, 79) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 79) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 80) =
      (groebnerMatrix(4, 80) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 80) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 81) =
      (groebnerMatrix(4, 81) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 81) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 82) =
      (groebnerMatrix(4, 82) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 82) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 83) =
      (groebnerMatrix(4, 83) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 83) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 84) =
      (groebnerMatrix(4, 84) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 84) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 85) =
      (groebnerMatrix(4, 85) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 85) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 86) =
      (groebnerMatrix(4, 86) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 86) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 87) =
      (groebnerMatrix(4, 87) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 87) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 89) =
      (groebnerMatrix(4, 89) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 89) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 91) =
      (groebnerMatrix(4, 91) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 91) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 92) =
      (groebnerMatrix(4, 92) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 92) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 94) =
      (groebnerMatrix(4, 94) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 94) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 97) =
      (groebnerMatrix(4, 97) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 97) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 98) =
      (groebnerMatrix(4, 98) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 98) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 99) =
      (groebnerMatrix(4, 99) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 99) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 100) =
      (groebnerMatrix(4, 100) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 100) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 101) =
      (groebnerMatrix(4, 101) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 101) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 103) =
      (groebnerMatrix(4, 103) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 103) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 104) =
      (groebnerMatrix(4, 104) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 104) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 105) =
      (groebnerMatrix(4, 105) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 105) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 106) =
      (groebnerMatrix(4, 106) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 106) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 107) =
      (groebnerMatrix(4, 107) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 107) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 108) =
      (groebnerMatrix(4, 108) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 108) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 109) =
      (groebnerMatrix(4, 109) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 109) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 110) =
      (groebnerMatrix(4, 110) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 110) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 111) =
      (groebnerMatrix(4, 111) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 111) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 112) =
      (groebnerMatrix(4, 112) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 112) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 113) =
      (groebnerMatrix(4, 113) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 113) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 115) =
      (groebnerMatrix(4, 115) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 115) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 116) =
      (groebnerMatrix(4, 116) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 116) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 118) =
      (groebnerMatrix(4, 118) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 118) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 119) =
      (groebnerMatrix(4, 119) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 119) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 120) =
      (groebnerMatrix(4, 120) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 120) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 121) =
      (groebnerMatrix(4, 121) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 121) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 122) =
      (groebnerMatrix(4, 122) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 122) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 123) =
      (groebnerMatrix(4, 123) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 123) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 125) =
      (groebnerMatrix(4, 125) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 125) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 126) =
      (groebnerMatrix(4, 126) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 126) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 127) =
      (groebnerMatrix(4, 127) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 127) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 128) =
      (groebnerMatrix(4, 128) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 128) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 129) =
      (groebnerMatrix(4, 129) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 129) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 130) =
      (groebnerMatrix(4, 130) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 130) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 133) =
      (groebnerMatrix(4, 133) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 133) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 134) =
      (groebnerMatrix(4, 134) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 134) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 136) =
      (groebnerMatrix(4, 136) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 136) / (groebnerMatrix(5, 0)));
  groebnerMatrix(34, 137) =
      (groebnerMatrix(4, 137) / (groebnerMatrix(4, 0)) - groebnerMatrix(5, 137) / (groebnerMatrix(5, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial35(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(35, 1) =
      (groebnerMatrix(5, 1) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 1) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 2) =
      (groebnerMatrix(5, 2) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 2) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 3) =
      (groebnerMatrix(5, 3) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 3) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 4) =
      (groebnerMatrix(5, 4) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 4) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 5) =
      (groebnerMatrix(5, 5) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 5) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 6) =
      (groebnerMatrix(5, 6) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 6) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 8) =
      (groebnerMatrix(5, 8) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 8) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 10) =
      (groebnerMatrix(5, 10) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 10) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 11) =
      (groebnerMatrix(5, 11) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 11) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 12) =
      (groebnerMatrix(5, 12) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 12) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 13) =
      (groebnerMatrix(5, 13) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 13) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 14) =
      (groebnerMatrix(5, 14) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 14) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 15) =
      (groebnerMatrix(5, 15) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 15) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 16) =
      (groebnerMatrix(5, 16) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 16) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 18) =
      (groebnerMatrix(5, 18) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 18) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 19) =
      (groebnerMatrix(5, 19) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 19) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 20) =
      (groebnerMatrix(5, 20) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 20) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 21) =
      (groebnerMatrix(5, 21) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 21) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 22) =
      (groebnerMatrix(5, 22) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 22) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 23) =
      (groebnerMatrix(5, 23) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 23) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 25) =
      (groebnerMatrix(5, 25) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 25) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 26) =
      (groebnerMatrix(5, 26) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 26) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 27) =
      (groebnerMatrix(5, 27) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 27) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 28) =
      (groebnerMatrix(5, 28) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 28) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 29) =
      (groebnerMatrix(5, 29) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 29) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 30) =
      (groebnerMatrix(5, 30) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 30) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 33) =
      (groebnerMatrix(5, 33) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 33) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 34) =
      (groebnerMatrix(5, 34) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 34) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 39) =
      (groebnerMatrix(5, 39) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 39) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 40) =
      (groebnerMatrix(5, 40) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 40) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 41) =
      (groebnerMatrix(5, 41) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 41) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 42) =
      (groebnerMatrix(5, 42) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 42) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 43) =
      (groebnerMatrix(5, 43) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 43) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 44) =
      (groebnerMatrix(5, 44) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 44) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 45) =
      (groebnerMatrix(5, 45) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 45) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 46) =
      (groebnerMatrix(5, 46) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 46) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 47) =
      (groebnerMatrix(5, 47) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 47) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 48) =
      (groebnerMatrix(5, 48) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 48) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 49) =
      (groebnerMatrix(5, 49) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 49) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 50) =
      (groebnerMatrix(5, 50) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 50) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 51) =
      (groebnerMatrix(5, 51) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 51) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 52) =
      (groebnerMatrix(5, 52) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 52) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 53) =
      (groebnerMatrix(5, 53) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 53) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 54) =
      (groebnerMatrix(5, 54) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 54) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 55) =
      (groebnerMatrix(5, 55) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 55) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 56) =
      (groebnerMatrix(5, 56) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 56) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 57) =
      (groebnerMatrix(5, 57) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 57) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 58) =
      (groebnerMatrix(5, 58) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 58) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 59) =
      (groebnerMatrix(5, 59) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 59) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 60) =
      (groebnerMatrix(5, 60) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 60) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 61) =
      (groebnerMatrix(5, 61) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 61) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 62) =
      (groebnerMatrix(5, 62) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 62) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 64) =
      (groebnerMatrix(5, 64) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 64) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 65) =
      (groebnerMatrix(5, 65) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 65) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 66) =
      (groebnerMatrix(5, 66) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 66) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 67) =
      (groebnerMatrix(5, 67) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 67) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 68) =
      (groebnerMatrix(5, 68) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 68) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 69) =
      (groebnerMatrix(5, 69) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 69) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 70) =
      (groebnerMatrix(5, 70) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 70) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 71) =
      (groebnerMatrix(5, 71) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 71) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 73) =
      (groebnerMatrix(5, 73) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 73) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 74) =
      (groebnerMatrix(5, 74) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 74) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 76) =
      (groebnerMatrix(5, 76) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 76) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 77) =
      (groebnerMatrix(5, 77) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 77) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 78) =
      (groebnerMatrix(5, 78) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 78) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 79) =
      (groebnerMatrix(5, 79) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 79) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 80) =
      (groebnerMatrix(5, 80) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 80) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 81) =
      (groebnerMatrix(5, 81) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 81) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 82) =
      (groebnerMatrix(5, 82) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 82) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 83) =
      (groebnerMatrix(5, 83) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 83) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 84) =
      (groebnerMatrix(5, 84) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 84) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 85) =
      (groebnerMatrix(5, 85) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 85) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 86) =
      (groebnerMatrix(5, 86) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 86) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 87) =
      (groebnerMatrix(5, 87) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 87) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 89) =
      (groebnerMatrix(5, 89) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 89) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 91) =
      (groebnerMatrix(5, 91) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 91) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 92) =
      (groebnerMatrix(5, 92) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 92) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 94) =
      (groebnerMatrix(5, 94) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 94) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 97) =
      (groebnerMatrix(5, 97) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 97) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 98) =
      (groebnerMatrix(5, 98) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 98) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 99) =
      (groebnerMatrix(5, 99) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 99) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 100) =
      (groebnerMatrix(5, 100) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 100) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 101) =
      (groebnerMatrix(5, 101) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 101) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 103) =
      (groebnerMatrix(5, 103) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 103) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 104) =
      (groebnerMatrix(5, 104) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 104) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 105) =
      (groebnerMatrix(5, 105) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 105) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 106) =
      (groebnerMatrix(5, 106) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 106) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 107) =
      (groebnerMatrix(5, 107) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 107) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 108) =
      (groebnerMatrix(5, 108) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 108) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 109) =
      (groebnerMatrix(5, 109) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 109) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 110) =
      (groebnerMatrix(5, 110) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 110) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 111) =
      (groebnerMatrix(5, 111) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 111) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 112) =
      (groebnerMatrix(5, 112) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 112) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 113) =
      (groebnerMatrix(5, 113) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 113) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 115) =
      (groebnerMatrix(5, 115) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 115) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 116) =
      (groebnerMatrix(5, 116) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 116) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 118) =
      (groebnerMatrix(5, 118) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 118) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 119) =
      (groebnerMatrix(5, 119) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 119) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 120) =
      (groebnerMatrix(5, 120) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 120) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 121) =
      (groebnerMatrix(5, 121) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 121) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 122) =
      (groebnerMatrix(5, 122) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 122) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 123) =
      (groebnerMatrix(5, 123) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 123) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 125) =
      (groebnerMatrix(5, 125) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 125) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 126) =
      (groebnerMatrix(5, 126) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 126) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 127) =
      (groebnerMatrix(5, 127) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 127) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 128) =
      (groebnerMatrix(5, 128) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 128) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 129) =
      (groebnerMatrix(5, 129) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 129) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 130) =
      (groebnerMatrix(5, 130) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 130) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 133) =
      (groebnerMatrix(5, 133) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 133) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 134) =
      (groebnerMatrix(5, 134) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 134) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 136) =
      (groebnerMatrix(5, 136) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 136) / (groebnerMatrix(6, 0)));
  groebnerMatrix(35, 137) =
      (groebnerMatrix(5, 137) / (groebnerMatrix(5, 0)) - groebnerMatrix(6, 137) / (groebnerMatrix(6, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial36(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(36, 1) =
      (groebnerMatrix(6, 1) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 1) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 2) =
      (groebnerMatrix(6, 2) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 2) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 3) =
      (groebnerMatrix(6, 3) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 3) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 4) =
      (groebnerMatrix(6, 4) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 4) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 5) =
      (groebnerMatrix(6, 5) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 5) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 6) =
      (groebnerMatrix(6, 6) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 6) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 8) =
      (groebnerMatrix(6, 8) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 8) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 10) =
      (groebnerMatrix(6, 10) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 10) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 11) =
      (groebnerMatrix(6, 11) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 11) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 12) =
      (groebnerMatrix(6, 12) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 12) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 13) =
      (groebnerMatrix(6, 13) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 13) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 14) =
      (groebnerMatrix(6, 14) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 14) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 15) =
      (groebnerMatrix(6, 15) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 15) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 16) =
      (groebnerMatrix(6, 16) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 16) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 18) =
      (groebnerMatrix(6, 18) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 18) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 19) =
      (groebnerMatrix(6, 19) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 19) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 20) =
      (groebnerMatrix(6, 20) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 20) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 21) =
      (groebnerMatrix(6, 21) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 21) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 22) =
      (groebnerMatrix(6, 22) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 22) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 23) =
      (groebnerMatrix(6, 23) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 23) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 25) =
      (groebnerMatrix(6, 25) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 25) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 26) =
      (groebnerMatrix(6, 26) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 26) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 27) =
      (groebnerMatrix(6, 27) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 27) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 28) =
      (groebnerMatrix(6, 28) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 28) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 29) =
      (groebnerMatrix(6, 29) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 29) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 30) =
      (groebnerMatrix(6, 30) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 30) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 33) =
      (groebnerMatrix(6, 33) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 33) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 34) =
      (groebnerMatrix(6, 34) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 34) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 39) =
      (groebnerMatrix(6, 39) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 39) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 40) =
      (groebnerMatrix(6, 40) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 40) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 41) =
      (groebnerMatrix(6, 41) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 41) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 42) =
      (groebnerMatrix(6, 42) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 42) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 43) =
      (groebnerMatrix(6, 43) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 43) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 44) =
      (groebnerMatrix(6, 44) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 44) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 45) =
      (groebnerMatrix(6, 45) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 45) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 46) =
      (groebnerMatrix(6, 46) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 46) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 47) =
      (groebnerMatrix(6, 47) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 47) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 48) =
      (groebnerMatrix(6, 48) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 48) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 49) =
      (groebnerMatrix(6, 49) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 49) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 50) =
      (groebnerMatrix(6, 50) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 50) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 51) =
      (groebnerMatrix(6, 51) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 51) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 52) =
      (groebnerMatrix(6, 52) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 52) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 53) =
      (groebnerMatrix(6, 53) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 53) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 54) =
      (groebnerMatrix(6, 54) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 54) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 55) =
      (groebnerMatrix(6, 55) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 55) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 56) =
      (groebnerMatrix(6, 56) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 56) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 57) =
      (groebnerMatrix(6, 57) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 57) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 58) =
      (groebnerMatrix(6, 58) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 58) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 59) =
      (groebnerMatrix(6, 59) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 59) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 60) =
      (groebnerMatrix(6, 60) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 60) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 61) =
      (groebnerMatrix(6, 61) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 61) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 62) =
      (groebnerMatrix(6, 62) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 62) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 64) =
      (groebnerMatrix(6, 64) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 64) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 65) =
      (groebnerMatrix(6, 65) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 65) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 66) =
      (groebnerMatrix(6, 66) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 66) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 67) =
      (groebnerMatrix(6, 67) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 67) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 68) =
      (groebnerMatrix(6, 68) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 68) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 69) =
      (groebnerMatrix(6, 69) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 69) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 70) =
      (groebnerMatrix(6, 70) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 70) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 71) =
      (groebnerMatrix(6, 71) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 71) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 73) =
      (groebnerMatrix(6, 73) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 73) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 74) =
      (groebnerMatrix(6, 74) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 74) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 76) =
      (groebnerMatrix(6, 76) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 76) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 77) =
      (groebnerMatrix(6, 77) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 77) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 78) =
      (groebnerMatrix(6, 78) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 78) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 79) =
      (groebnerMatrix(6, 79) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 79) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 80) =
      (groebnerMatrix(6, 80) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 80) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 81) =
      (groebnerMatrix(6, 81) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 81) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 82) =
      (groebnerMatrix(6, 82) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 82) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 83) =
      (groebnerMatrix(6, 83) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 83) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 84) =
      (groebnerMatrix(6, 84) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 84) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 85) =
      (groebnerMatrix(6, 85) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 85) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 86) =
      (groebnerMatrix(6, 86) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 86) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 87) =
      (groebnerMatrix(6, 87) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 87) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 89) =
      (groebnerMatrix(6, 89) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 89) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 91) =
      (groebnerMatrix(6, 91) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 91) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 92) =
      (groebnerMatrix(6, 92) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 92) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 94) =
      (groebnerMatrix(6, 94) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 94) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 97) =
      (groebnerMatrix(6, 97) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 97) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 98) =
      (groebnerMatrix(6, 98) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 98) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 99) =
      (groebnerMatrix(6, 99) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 99) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 100) =
      (groebnerMatrix(6, 100) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 100) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 101) =
      (groebnerMatrix(6, 101) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 101) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 103) =
      (groebnerMatrix(6, 103) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 103) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 104) =
      (groebnerMatrix(6, 104) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 104) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 105) =
      (groebnerMatrix(6, 105) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 105) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 106) =
      (groebnerMatrix(6, 106) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 106) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 107) =
      (groebnerMatrix(6, 107) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 107) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 108) =
      (groebnerMatrix(6, 108) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 108) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 109) =
      (groebnerMatrix(6, 109) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 109) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 110) =
      (groebnerMatrix(6, 110) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 110) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 111) =
      (groebnerMatrix(6, 111) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 111) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 112) =
      (groebnerMatrix(6, 112) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 112) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 113) =
      (groebnerMatrix(6, 113) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 113) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 115) =
      (groebnerMatrix(6, 115) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 115) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 116) =
      (groebnerMatrix(6, 116) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 116) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 118) =
      (groebnerMatrix(6, 118) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 118) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 119) =
      (groebnerMatrix(6, 119) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 119) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 120) =
      (groebnerMatrix(6, 120) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 120) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 121) =
      (groebnerMatrix(6, 121) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 121) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 122) =
      (groebnerMatrix(6, 122) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 122) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 123) =
      (groebnerMatrix(6, 123) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 123) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 125) =
      (groebnerMatrix(6, 125) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 125) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 126) =
      (groebnerMatrix(6, 126) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 126) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 127) =
      (groebnerMatrix(6, 127) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 127) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 128) =
      (groebnerMatrix(6, 128) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 128) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 129) =
      (groebnerMatrix(6, 129) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 129) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 130) =
      (groebnerMatrix(6, 130) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 130) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 133) =
      (groebnerMatrix(6, 133) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 133) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 134) =
      (groebnerMatrix(6, 134) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 134) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 136) =
      (groebnerMatrix(6, 136) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 136) / (groebnerMatrix(7, 0)));
  groebnerMatrix(36, 137) =
      (groebnerMatrix(6, 137) / (groebnerMatrix(6, 0)) - groebnerMatrix(7, 137) / (groebnerMatrix(7, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial37(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(37, 1) =
      (groebnerMatrix(7, 1) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 1) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 2) =
      (groebnerMatrix(7, 2) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 2) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 3) =
      (groebnerMatrix(7, 3) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 3) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 4) =
      (groebnerMatrix(7, 4) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 4) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 5) =
      (groebnerMatrix(7, 5) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 5) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 6) =
      (groebnerMatrix(7, 6) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 6) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 8) =
      (groebnerMatrix(7, 8) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 8) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 10) =
      (groebnerMatrix(7, 10) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 10) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 11) =
      (groebnerMatrix(7, 11) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 11) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 12) =
      (groebnerMatrix(7, 12) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 12) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 13) =
      (groebnerMatrix(7, 13) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 13) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 14) =
      (groebnerMatrix(7, 14) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 14) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 15) =
      (groebnerMatrix(7, 15) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 15) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 16) =
      (groebnerMatrix(7, 16) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 16) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 18) =
      (groebnerMatrix(7, 18) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 18) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 19) =
      (groebnerMatrix(7, 19) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 19) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 20) =
      (groebnerMatrix(7, 20) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 20) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 21) =
      (groebnerMatrix(7, 21) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 21) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 22) =
      (groebnerMatrix(7, 22) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 22) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 23) =
      (groebnerMatrix(7, 23) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 23) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 25) =
      (groebnerMatrix(7, 25) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 25) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 26) =
      (groebnerMatrix(7, 26) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 26) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 27) =
      (groebnerMatrix(7, 27) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 27) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 28) =
      (groebnerMatrix(7, 28) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 28) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 29) =
      (groebnerMatrix(7, 29) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 29) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 30) =
      (groebnerMatrix(7, 30) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 30) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 33) =
      (groebnerMatrix(7, 33) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 33) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 34) =
      (groebnerMatrix(7, 34) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 34) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 39) =
      (groebnerMatrix(7, 39) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 39) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 40) =
      (groebnerMatrix(7, 40) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 40) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 41) =
      (groebnerMatrix(7, 41) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 41) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 42) =
      (groebnerMatrix(7, 42) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 42) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 43) =
      (groebnerMatrix(7, 43) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 43) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 44) =
      (groebnerMatrix(7, 44) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 44) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 45) =
      (groebnerMatrix(7, 45) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 45) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 46) =
      (groebnerMatrix(7, 46) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 46) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 47) =
      (groebnerMatrix(7, 47) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 47) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 48) =
      (groebnerMatrix(7, 48) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 48) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 49) =
      (groebnerMatrix(7, 49) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 49) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 50) =
      (groebnerMatrix(7, 50) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 50) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 51) =
      (groebnerMatrix(7, 51) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 51) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 52) =
      (groebnerMatrix(7, 52) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 52) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 53) =
      (groebnerMatrix(7, 53) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 53) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 54) =
      (groebnerMatrix(7, 54) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 54) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 55) =
      (groebnerMatrix(7, 55) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 55) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 56) =
      (groebnerMatrix(7, 56) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 56) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 57) =
      (groebnerMatrix(7, 57) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 57) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 58) =
      (groebnerMatrix(7, 58) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 58) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 59) =
      (groebnerMatrix(7, 59) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 59) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 60) =
      (groebnerMatrix(7, 60) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 60) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 61) =
      (groebnerMatrix(7, 61) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 61) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 62) =
      (groebnerMatrix(7, 62) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 62) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 64) =
      (groebnerMatrix(7, 64) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 64) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 65) =
      (groebnerMatrix(7, 65) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 65) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 66) =
      (groebnerMatrix(7, 66) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 66) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 67) =
      (groebnerMatrix(7, 67) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 67) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 68) =
      (groebnerMatrix(7, 68) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 68) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 69) =
      (groebnerMatrix(7, 69) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 69) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 70) =
      (groebnerMatrix(7, 70) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 70) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 71) =
      (groebnerMatrix(7, 71) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 71) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 73) =
      (groebnerMatrix(7, 73) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 73) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 74) =
      (groebnerMatrix(7, 74) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 74) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 76) =
      (groebnerMatrix(7, 76) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 76) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 77) =
      (groebnerMatrix(7, 77) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 77) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 78) =
      (groebnerMatrix(7, 78) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 78) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 79) =
      (groebnerMatrix(7, 79) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 79) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 80) =
      (groebnerMatrix(7, 80) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 80) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 81) =
      (groebnerMatrix(7, 81) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 81) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 82) =
      (groebnerMatrix(7, 82) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 82) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 83) =
      (groebnerMatrix(7, 83) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 83) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 84) =
      (groebnerMatrix(7, 84) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 84) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 85) =
      (groebnerMatrix(7, 85) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 85) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 86) =
      (groebnerMatrix(7, 86) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 86) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 87) =
      (groebnerMatrix(7, 87) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 87) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 89) =
      (groebnerMatrix(7, 89) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 89) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 91) =
      (groebnerMatrix(7, 91) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 91) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 92) =
      (groebnerMatrix(7, 92) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 92) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 94) =
      (groebnerMatrix(7, 94) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 94) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 97) =
      (groebnerMatrix(7, 97) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 97) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 98) =
      (groebnerMatrix(7, 98) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 98) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 99) =
      (groebnerMatrix(7, 99) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 99) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 100) =
      (groebnerMatrix(7, 100) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 100) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 101) =
      (groebnerMatrix(7, 101) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 101) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 103) =
      (groebnerMatrix(7, 103) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 103) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 104) =
      (groebnerMatrix(7, 104) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 104) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 105) =
      (groebnerMatrix(7, 105) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 105) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 106) =
      (groebnerMatrix(7, 106) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 106) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 107) =
      (groebnerMatrix(7, 107) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 107) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 108) =
      (groebnerMatrix(7, 108) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 108) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 109) =
      (groebnerMatrix(7, 109) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 109) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 110) =
      (groebnerMatrix(7, 110) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 110) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 111) =
      (groebnerMatrix(7, 111) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 111) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 112) =
      (groebnerMatrix(7, 112) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 112) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 113) =
      (groebnerMatrix(7, 113) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 113) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 115) =
      (groebnerMatrix(7, 115) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 115) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 116) =
      (groebnerMatrix(7, 116) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 116) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 118) =
      (groebnerMatrix(7, 118) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 118) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 119) =
      (groebnerMatrix(7, 119) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 119) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 120) =
      (groebnerMatrix(7, 120) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 120) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 121) =
      (groebnerMatrix(7, 121) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 121) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 122) =
      (groebnerMatrix(7, 122) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 122) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 123) =
      (groebnerMatrix(7, 123) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 123) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 125) =
      (groebnerMatrix(7, 125) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 125) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 126) =
      (groebnerMatrix(7, 126) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 126) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 127) =
      (groebnerMatrix(7, 127) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 127) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 128) =
      (groebnerMatrix(7, 128) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 128) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 129) =
      (groebnerMatrix(7, 129) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 129) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 130) =
      (groebnerMatrix(7, 130) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 130) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 133) =
      (groebnerMatrix(7, 133) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 133) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 134) =
      (groebnerMatrix(7, 134) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 134) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 136) =
      (groebnerMatrix(7, 136) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 136) / (groebnerMatrix(8, 0)));
  groebnerMatrix(37, 137) =
      (groebnerMatrix(7, 137) / (groebnerMatrix(7, 0)) - groebnerMatrix(8, 137) / (groebnerMatrix(8, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial38(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(38, 1) =
      (groebnerMatrix(8, 1) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 1) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 2) =
      (groebnerMatrix(8, 2) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 2) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 3) =
      (groebnerMatrix(8, 3) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 3) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 4) =
      (groebnerMatrix(8, 4) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 4) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 5) =
      (groebnerMatrix(8, 5) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 5) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 6) =
      (groebnerMatrix(8, 6) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 6) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 8) =
      (groebnerMatrix(8, 8) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 8) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 10) =
      (groebnerMatrix(8, 10) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 10) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 11) =
      (groebnerMatrix(8, 11) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 11) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 12) =
      (groebnerMatrix(8, 12) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 12) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 13) =
      (groebnerMatrix(8, 13) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 13) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 14) =
      (groebnerMatrix(8, 14) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 14) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 15) =
      (groebnerMatrix(8, 15) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 15) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 16) =
      (groebnerMatrix(8, 16) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 16) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 18) =
      (groebnerMatrix(8, 18) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 18) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 19) =
      (groebnerMatrix(8, 19) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 19) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 20) =
      (groebnerMatrix(8, 20) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 20) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 21) =
      (groebnerMatrix(8, 21) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 21) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 22) =
      (groebnerMatrix(8, 22) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 22) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 23) =
      (groebnerMatrix(8, 23) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 23) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 25) =
      (groebnerMatrix(8, 25) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 25) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 26) =
      (groebnerMatrix(8, 26) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 26) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 27) =
      (groebnerMatrix(8, 27) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 27) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 28) =
      (groebnerMatrix(8, 28) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 28) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 29) =
      (groebnerMatrix(8, 29) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 29) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 30) =
      (groebnerMatrix(8, 30) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 30) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 33) =
      (groebnerMatrix(8, 33) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 33) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 34) =
      (groebnerMatrix(8, 34) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 34) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 39) =
      (groebnerMatrix(8, 39) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 39) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 40) =
      (groebnerMatrix(8, 40) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 40) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 41) =
      (groebnerMatrix(8, 41) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 41) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 42) =
      (groebnerMatrix(8, 42) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 42) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 43) =
      (groebnerMatrix(8, 43) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 43) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 44) =
      (groebnerMatrix(8, 44) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 44) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 45) =
      (groebnerMatrix(8, 45) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 45) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 46) =
      (groebnerMatrix(8, 46) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 46) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 47) =
      (groebnerMatrix(8, 47) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 47) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 48) =
      (groebnerMatrix(8, 48) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 48) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 49) =
      (groebnerMatrix(8, 49) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 49) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 50) =
      (groebnerMatrix(8, 50) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 50) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 51) =
      (groebnerMatrix(8, 51) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 51) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 52) =
      (groebnerMatrix(8, 52) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 52) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 53) =
      (groebnerMatrix(8, 53) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 53) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 54) =
      (groebnerMatrix(8, 54) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 54) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 55) =
      (groebnerMatrix(8, 55) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 55) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 56) =
      (groebnerMatrix(8, 56) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 56) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 57) =
      (groebnerMatrix(8, 57) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 57) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 58) =
      (groebnerMatrix(8, 58) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 58) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 59) =
      (groebnerMatrix(8, 59) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 59) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 60) =
      (groebnerMatrix(8, 60) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 60) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 61) =
      (groebnerMatrix(8, 61) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 61) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 62) =
      (groebnerMatrix(8, 62) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 62) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 64) =
      (groebnerMatrix(8, 64) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 64) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 65) =
      (groebnerMatrix(8, 65) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 65) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 66) =
      (groebnerMatrix(8, 66) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 66) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 67) =
      (groebnerMatrix(8, 67) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 67) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 68) =
      (groebnerMatrix(8, 68) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 68) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 69) =
      (groebnerMatrix(8, 69) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 69) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 70) =
      (groebnerMatrix(8, 70) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 70) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 71) =
      (groebnerMatrix(8, 71) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 71) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 73) =
      (groebnerMatrix(8, 73) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 73) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 74) =
      (groebnerMatrix(8, 74) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 74) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 76) =
      (groebnerMatrix(8, 76) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 76) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 77) =
      (groebnerMatrix(8, 77) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 77) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 78) =
      (groebnerMatrix(8, 78) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 78) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 79) =
      (groebnerMatrix(8, 79) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 79) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 80) =
      (groebnerMatrix(8, 80) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 80) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 81) =
      (groebnerMatrix(8, 81) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 81) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 82) =
      (groebnerMatrix(8, 82) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 82) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 83) =
      (groebnerMatrix(8, 83) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 83) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 84) =
      (groebnerMatrix(8, 84) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 84) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 85) =
      (groebnerMatrix(8, 85) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 85) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 86) =
      (groebnerMatrix(8, 86) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 86) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 87) =
      (groebnerMatrix(8, 87) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 87) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 89) =
      (groebnerMatrix(8, 89) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 89) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 91) =
      (groebnerMatrix(8, 91) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 91) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 92) =
      (groebnerMatrix(8, 92) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 92) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 94) =
      (groebnerMatrix(8, 94) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 94) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 97) =
      (groebnerMatrix(8, 97) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 97) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 98) =
      (groebnerMatrix(8, 98) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 98) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 99) =
      (groebnerMatrix(8, 99) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 99) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 100) =
      (groebnerMatrix(8, 100) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 100) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 101) =
      (groebnerMatrix(8, 101) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 101) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 103) =
      (groebnerMatrix(8, 103) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 103) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 104) =
      (groebnerMatrix(8, 104) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 104) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 105) =
      (groebnerMatrix(8, 105) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 105) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 106) =
      (groebnerMatrix(8, 106) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 106) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 107) =
      (groebnerMatrix(8, 107) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 107) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 108) =
      (groebnerMatrix(8, 108) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 108) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 109) =
      (groebnerMatrix(8, 109) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 109) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 110) =
      (groebnerMatrix(8, 110) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 110) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 111) =
      (groebnerMatrix(8, 111) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 111) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 112) =
      (groebnerMatrix(8, 112) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 112) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 113) =
      (groebnerMatrix(8, 113) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 113) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 115) =
      (groebnerMatrix(8, 115) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 115) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 116) =
      (groebnerMatrix(8, 116) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 116) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 118) =
      (groebnerMatrix(8, 118) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 118) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 119) =
      (groebnerMatrix(8, 119) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 119) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 120) =
      (groebnerMatrix(8, 120) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 120) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 121) =
      (groebnerMatrix(8, 121) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 121) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 122) =
      (groebnerMatrix(8, 122) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 122) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 123) =
      (groebnerMatrix(8, 123) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 123) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 125) =
      (groebnerMatrix(8, 125) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 125) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 126) =
      (groebnerMatrix(8, 126) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 126) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 127) =
      (groebnerMatrix(8, 127) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 127) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 128) =
      (groebnerMatrix(8, 128) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 128) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 129) =
      (groebnerMatrix(8, 129) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 129) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 130) =
      (groebnerMatrix(8, 130) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 130) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 133) =
      (groebnerMatrix(8, 133) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 133) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 134) =
      (groebnerMatrix(8, 134) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 134) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 136) =
      (groebnerMatrix(8, 136) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 136) / (groebnerMatrix(9, 0)));
  groebnerMatrix(38, 137) =
      (groebnerMatrix(8, 137) / (groebnerMatrix(8, 0)) - groebnerMatrix(9, 137) / (groebnerMatrix(9, 0)));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial39(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(39, 1) = groebnerMatrix(9, 1) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 2) = groebnerMatrix(9, 2) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 3) = groebnerMatrix(9, 3) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 4) = groebnerMatrix(9, 4) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 5) = groebnerMatrix(9, 5) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 6) = groebnerMatrix(9, 6) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 7) = groebnerMatrix(9, 7) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 8) = groebnerMatrix(9, 8) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 9) = -groebnerMatrix(14, 153) / (groebnerMatrix(14, 148));
  groebnerMatrix(39, 10) = groebnerMatrix(9, 10) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 11) = groebnerMatrix(9, 11) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 12) = groebnerMatrix(9, 12) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 13) = groebnerMatrix(9, 13) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 14) = groebnerMatrix(9, 14) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 15) = groebnerMatrix(9, 15) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 16) = groebnerMatrix(9, 16) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 18) = groebnerMatrix(9, 18) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 19) = groebnerMatrix(9, 19) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 20) = groebnerMatrix(9, 20) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 21) = groebnerMatrix(9, 21) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 22) = groebnerMatrix(9, 22) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 23) =
      (groebnerMatrix(9, 23) / (groebnerMatrix(9, 0)) - groebnerMatrix(14, 159) / (groebnerMatrix(14, 148)));
  groebnerMatrix(39, 25) = groebnerMatrix(9, 25) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 26) = groebnerMatrix(9, 26) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 27) = groebnerMatrix(9, 27) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 28) = groebnerMatrix(9, 28) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 29) = groebnerMatrix(9, 29) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 30) = groebnerMatrix(9, 30) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 33) = groebnerMatrix(9, 33) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 34) = groebnerMatrix(9, 34) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 39) = groebnerMatrix(9, 39) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 40) = groebnerMatrix(9, 40) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 41) = groebnerMatrix(9, 41) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 42) = groebnerMatrix(9, 42) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 43) = groebnerMatrix(9, 43) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 44) = groebnerMatrix(9, 44) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 45) = groebnerMatrix(9, 45) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 46) = groebnerMatrix(9, 46) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 47) = groebnerMatrix(9, 47) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 48) = groebnerMatrix(9, 48) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 49) = groebnerMatrix(9, 49) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 50) = groebnerMatrix(9, 50) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 51) = groebnerMatrix(9, 51) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 52) = groebnerMatrix(9, 52) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 53) = groebnerMatrix(9, 53) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 54) = groebnerMatrix(9, 54) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 55) = groebnerMatrix(9, 55) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 56) = groebnerMatrix(9, 56) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 57) = groebnerMatrix(9, 57) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 58) = groebnerMatrix(9, 58) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 59) = groebnerMatrix(9, 59) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 60) = groebnerMatrix(9, 60) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 61) = groebnerMatrix(9, 61) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 62) = groebnerMatrix(9, 62) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 64) = groebnerMatrix(9, 64) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 65) = groebnerMatrix(9, 65) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 66) = groebnerMatrix(9, 66) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 67) = groebnerMatrix(9, 67) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 68) = groebnerMatrix(9, 68) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 69) = groebnerMatrix(9, 69) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 70) = groebnerMatrix(9, 70) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 71) = groebnerMatrix(9, 71) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 73) = groebnerMatrix(9, 73) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 74) = groebnerMatrix(9, 74) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 76) = groebnerMatrix(9, 76) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 77) = groebnerMatrix(9, 77) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 78) = groebnerMatrix(9, 78) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 79) = groebnerMatrix(9, 79) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 80) = groebnerMatrix(9, 80) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 81) = groebnerMatrix(9, 81) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 82) = groebnerMatrix(9, 82) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 83) = groebnerMatrix(9, 83) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 84) = groebnerMatrix(9, 84) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 85) = groebnerMatrix(9, 85) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 86) = groebnerMatrix(9, 86) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 87) = groebnerMatrix(9, 87) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 89) = groebnerMatrix(9, 89) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 91) = groebnerMatrix(9, 91) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 92) = groebnerMatrix(9, 92) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 94) = groebnerMatrix(9, 94) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 97) = groebnerMatrix(9, 97) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 98) = groebnerMatrix(9, 98) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 99) = groebnerMatrix(9, 99) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 100) = groebnerMatrix(9, 100) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 101) = groebnerMatrix(9, 101) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 103) = groebnerMatrix(9, 103) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 104) = groebnerMatrix(9, 104) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 105) = groebnerMatrix(9, 105) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 106) = groebnerMatrix(9, 106) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 107) = groebnerMatrix(9, 107) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 108) = groebnerMatrix(9, 108) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 109) = groebnerMatrix(9, 109) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 110) = groebnerMatrix(9, 110) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 111) = groebnerMatrix(9, 111) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 112) = groebnerMatrix(9, 112) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 113) = groebnerMatrix(9, 113) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 115) = groebnerMatrix(9, 115) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 116) = groebnerMatrix(9, 116) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 118) = groebnerMatrix(9, 118) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 119) = groebnerMatrix(9, 119) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 120) = groebnerMatrix(9, 120) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 121) = groebnerMatrix(9, 121) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 122) = groebnerMatrix(9, 122) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 123) = groebnerMatrix(9, 123) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 125) = groebnerMatrix(9, 125) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 126) = groebnerMatrix(9, 126) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 127) = groebnerMatrix(9, 127) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 128) = groebnerMatrix(9, 128) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 129) = groebnerMatrix(9, 129) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 130) = groebnerMatrix(9, 130) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 133) = groebnerMatrix(9, 133) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 134) = groebnerMatrix(9, 134) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 136) = groebnerMatrix(9, 136) / (groebnerMatrix(9, 0));
  groebnerMatrix(39, 137) = groebnerMatrix(9, 137) / (groebnerMatrix(9, 0));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial40(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(40, 89) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 90) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 92) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 93) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 95) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 96) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 116) = groebnerMatrix(18, 182) / (groebnerMatrix(18, 175));
  groebnerMatrix(40, 125) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 126) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 127) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 128) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 129) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 130) = -groebnerMatrix(39, 183) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 131) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 132) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 140) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 157) = groebnerMatrix(18, 187) / (groebnerMatrix(18, 175));
  groebnerMatrix(40, 170) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 171) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 172) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 173) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 174) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 175) = -groebnerMatrix(39, 192) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 176) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 177) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 185) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(40, 194) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial41(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(41, 81) = -groebnerMatrix(38, 162) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 89) = -groebnerMatrix(38, 170) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 90) = -groebnerMatrix(38, 171) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 92) = -groebnerMatrix(38, 173) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 93) = -groebnerMatrix(38, 174) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 95) = -groebnerMatrix(38, 176) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 96) = -groebnerMatrix(38, 177) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 111) = groebnerMatrix(18, 182) / (groebnerMatrix(18, 175));
  groebnerMatrix(41, 125) = -groebnerMatrix(38, 178) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 126) = -groebnerMatrix(38, 179) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 127) = -groebnerMatrix(38, 180) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 128) = -groebnerMatrix(38, 181) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 129) = -groebnerMatrix(38, 182) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 130) = -groebnerMatrix(38, 183) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 131) = -groebnerMatrix(38, 184) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 132) = -groebnerMatrix(38, 185) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 140) = -groebnerMatrix(38, 186) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 152) = groebnerMatrix(18, 187) / (groebnerMatrix(18, 175));
  groebnerMatrix(41, 170) = -groebnerMatrix(38, 187) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 171) = -groebnerMatrix(38, 188) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 172) = -groebnerMatrix(38, 189) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 173) = -groebnerMatrix(38, 190) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 174) = -groebnerMatrix(38, 191) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 175) = -groebnerMatrix(38, 192) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 176) = -groebnerMatrix(38, 193) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 177) = -groebnerMatrix(38, 194) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 185) = -groebnerMatrix(38, 195) / (groebnerMatrix(38, 161));
  groebnerMatrix(41, 194) = -groebnerMatrix(38, 196) / (groebnerMatrix(38, 161));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial42(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(42, 80) = -groebnerMatrix(37, 161) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 81) = -groebnerMatrix(37, 162) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 89) = -groebnerMatrix(37, 170) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 90) = -groebnerMatrix(37, 171) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 92) = -groebnerMatrix(37, 173) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 93) = -groebnerMatrix(37, 174) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 95) = -groebnerMatrix(37, 176) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 96) = -groebnerMatrix(37, 177) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 110) = groebnerMatrix(18, 182) / (groebnerMatrix(18, 175));
  groebnerMatrix(42, 125) = -groebnerMatrix(37, 178) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 126) = -groebnerMatrix(37, 179) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 127) = -groebnerMatrix(37, 180) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 128) = -groebnerMatrix(37, 181) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 129) = -groebnerMatrix(37, 182) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 130) = -groebnerMatrix(37, 183) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 131) = -groebnerMatrix(37, 184) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 132) = -groebnerMatrix(37, 185) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 140) = -groebnerMatrix(37, 186) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 148) = groebnerMatrix(18, 187) / (groebnerMatrix(18, 175));
  groebnerMatrix(42, 170) = -groebnerMatrix(37, 187) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 171) = -groebnerMatrix(37, 188) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 172) = -groebnerMatrix(37, 189) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 173) = -groebnerMatrix(37, 190) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 174) = -groebnerMatrix(37, 191) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 175) = -groebnerMatrix(37, 192) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 176) = -groebnerMatrix(37, 193) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 177) = -groebnerMatrix(37, 194) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 185) = -groebnerMatrix(37, 195) / (groebnerMatrix(37, 160));
  groebnerMatrix(42, 194) = -groebnerMatrix(37, 196) / (groebnerMatrix(37, 160));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial43(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(43, 79) = -groebnerMatrix(36, 160) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 80) = -groebnerMatrix(36, 161) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 81) = -groebnerMatrix(36, 162) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 89) = -groebnerMatrix(36, 170) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 90) = -groebnerMatrix(36, 171) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 92) = -groebnerMatrix(36, 173) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 93) = -groebnerMatrix(36, 174) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 95) = -groebnerMatrix(36, 176) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 96) = -groebnerMatrix(36, 177) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 113) = groebnerMatrix(17, 179) / (groebnerMatrix(17, 172));
  groebnerMatrix(43, 125) = -groebnerMatrix(36, 178) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 126) = -groebnerMatrix(36, 179) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 127) = -groebnerMatrix(36, 180) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 128) = -groebnerMatrix(36, 181) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 129) = -groebnerMatrix(36, 182) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 130) = -groebnerMatrix(36, 183) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 131) = -groebnerMatrix(36, 184) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 132) = -groebnerMatrix(36, 185) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 140) = -groebnerMatrix(36, 186) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 160) = groebnerMatrix(17, 190) / (groebnerMatrix(17, 172));
  groebnerMatrix(43, 170) = -groebnerMatrix(36, 187) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 171) = -groebnerMatrix(36, 188) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 172) = -groebnerMatrix(36, 189) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 173) = -groebnerMatrix(36, 190) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 174) = -groebnerMatrix(36, 191) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 175) = -groebnerMatrix(36, 192) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 176) = -groebnerMatrix(36, 193) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 177) = -groebnerMatrix(36, 194) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 185) = -groebnerMatrix(36, 195) / (groebnerMatrix(36, 159));
  groebnerMatrix(43, 194) = -groebnerMatrix(36, 196) / (groebnerMatrix(36, 159));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial44(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(44, 78) = -groebnerMatrix(35, 159) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 79) = -groebnerMatrix(35, 160) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 80) = -groebnerMatrix(35, 161) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 81) = -groebnerMatrix(35, 162) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 89) = -groebnerMatrix(35, 170) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 90) = -groebnerMatrix(35, 171) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 92) = -groebnerMatrix(35, 173) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 93) = -groebnerMatrix(35, 174) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 95) = -groebnerMatrix(35, 176) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 96) = -groebnerMatrix(35, 177) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 108) = groebnerMatrix(18, 182) / (groebnerMatrix(18, 175));
  groebnerMatrix(44, 125) = -groebnerMatrix(35, 178) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 126) = -groebnerMatrix(35, 179) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 127) = -groebnerMatrix(35, 180) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 128) = -groebnerMatrix(35, 181) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 129) = -groebnerMatrix(35, 182) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 130) = -groebnerMatrix(35, 183) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 131) = -groebnerMatrix(35, 184) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 132) = -groebnerMatrix(35, 185) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 140) = -groebnerMatrix(35, 186) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 143) = groebnerMatrix(18, 187) / (groebnerMatrix(18, 175));
  groebnerMatrix(44, 170) = -groebnerMatrix(35, 187) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 171) = -groebnerMatrix(35, 188) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 172) = -groebnerMatrix(35, 189) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 173) = -groebnerMatrix(35, 190) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 174) = -groebnerMatrix(35, 191) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 175) = -groebnerMatrix(35, 192) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 176) = -groebnerMatrix(35, 193) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 177) = -groebnerMatrix(35, 194) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 185) = -groebnerMatrix(35, 195) / (groebnerMatrix(35, 158));
  groebnerMatrix(44, 194) = -groebnerMatrix(35, 196) / (groebnerMatrix(35, 158));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial45(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(45, 77) = -groebnerMatrix(34, 158) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 78) = -groebnerMatrix(34, 159) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 79) = -groebnerMatrix(34, 160) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 80) = -groebnerMatrix(34, 161) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 81) = -groebnerMatrix(34, 162) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 89) = -groebnerMatrix(34, 170) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 90) = -groebnerMatrix(34, 171) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 92) = -groebnerMatrix(34, 173) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 93) = -groebnerMatrix(34, 174) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 95) = -groebnerMatrix(34, 176) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 96) = -groebnerMatrix(34, 177) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 107) = groebnerMatrix(18, 182) / (groebnerMatrix(18, 175));
  groebnerMatrix(45, 125) = -groebnerMatrix(34, 178) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 126) = -groebnerMatrix(34, 179) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 127) = -groebnerMatrix(34, 180) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 128) = -groebnerMatrix(34, 181) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 129) = -groebnerMatrix(34, 182) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 130) = -groebnerMatrix(34, 183) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 131) = -groebnerMatrix(34, 184) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 132) = -groebnerMatrix(34, 185) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 140) = -groebnerMatrix(34, 186) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 142) = groebnerMatrix(18, 187) / (groebnerMatrix(18, 175));
  groebnerMatrix(45, 170) = -groebnerMatrix(34, 187) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 171) = -groebnerMatrix(34, 188) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 172) = -groebnerMatrix(34, 189) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 173) = -groebnerMatrix(34, 190) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 174) = -groebnerMatrix(34, 191) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 175) = -groebnerMatrix(34, 192) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 176) = -groebnerMatrix(34, 193) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 177) = -groebnerMatrix(34, 194) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 185) = -groebnerMatrix(34, 195) / (groebnerMatrix(34, 157));
  groebnerMatrix(45, 194) = -groebnerMatrix(34, 196) / (groebnerMatrix(34, 157));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial46(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(46, 82) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 83) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 85) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 86) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 88) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 95) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 115) = groebnerMatrix(27, 181) / (groebnerMatrix(27, 168));
  groebnerMatrix(46, 118) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 119) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 120) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 121) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 122) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 123) = -groebnerMatrix(39, 183) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 124) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 131) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 139) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 158) = groebnerMatrix(27, 188) / (groebnerMatrix(27, 168));
  groebnerMatrix(46, 163) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 164) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 165) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 166) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 167) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 168) = -groebnerMatrix(39, 192) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 169) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 176) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 184) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(46, 193) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial47(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(47, 56) = -groebnerMatrix(38, 162) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 79) = groebnerMatrix(21, 173) / (groebnerMatrix(21, 167));
  groebnerMatrix(47, 82) = -groebnerMatrix(38, 170) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 83) = -groebnerMatrix(38, 171) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 85) = -groebnerMatrix(38, 173) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 86) = -groebnerMatrix(38, 174) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 88) = -groebnerMatrix(38, 176) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 95) = -groebnerMatrix(38, 177) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 118) = -groebnerMatrix(38, 178) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 119) = -groebnerMatrix(38, 179) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 120) = -groebnerMatrix(38, 180) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 121) = -groebnerMatrix(38, 181) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 122) = -groebnerMatrix(38, 182) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 123) = -groebnerMatrix(38, 183) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 124) = -groebnerMatrix(38, 184) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 131) = -groebnerMatrix(38, 185) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 139) = -groebnerMatrix(38, 186) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 159) = groebnerMatrix(21, 189) / (groebnerMatrix(21, 167));
  groebnerMatrix(47, 163) = -groebnerMatrix(38, 187) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 164) = -groebnerMatrix(38, 188) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 165) = -groebnerMatrix(38, 189) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 166) = -groebnerMatrix(38, 190) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 167) = -groebnerMatrix(38, 191) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 168) = -groebnerMatrix(38, 192) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 169) = -groebnerMatrix(38, 193) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 176) = -groebnerMatrix(38, 194) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 184) = -groebnerMatrix(38, 195) / (groebnerMatrix(38, 161));
  groebnerMatrix(47, 193) = -groebnerMatrix(38, 196) / (groebnerMatrix(38, 161));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial48(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(48, 55) = -groebnerMatrix(37, 161) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 56) = -groebnerMatrix(37, 162) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 80) = groebnerMatrix(11, 174) / (groebnerMatrix(11, 166));
  groebnerMatrix(48, 82) = -groebnerMatrix(37, 170) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 83) = -groebnerMatrix(37, 171) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 85) = -groebnerMatrix(37, 173) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 86) = -groebnerMatrix(37, 174) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 88) = -groebnerMatrix(37, 176) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 95) = -groebnerMatrix(37, 177) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 117) = groebnerMatrix(11, 183) / (groebnerMatrix(11, 166));
  groebnerMatrix(48, 118) = -groebnerMatrix(37, 178) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 119) = -groebnerMatrix(37, 179) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 120) = -groebnerMatrix(37, 180) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 121) = -groebnerMatrix(37, 181) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 122) = -groebnerMatrix(37, 182) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 123) = -groebnerMatrix(37, 183) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 124) = -groebnerMatrix(37, 184) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 131) = -groebnerMatrix(37, 185) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 139) = -groebnerMatrix(37, 186) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 163) = -groebnerMatrix(37, 187) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 164) = -groebnerMatrix(37, 188) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 165) = -groebnerMatrix(37, 189) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 166) = -groebnerMatrix(37, 190) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 167) = -groebnerMatrix(37, 191) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 168) = -groebnerMatrix(37, 192) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 169) = -groebnerMatrix(37, 193) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 176) = -groebnerMatrix(37, 194) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 184) = -groebnerMatrix(37, 195) / (groebnerMatrix(37, 160));
  groebnerMatrix(48, 193) = -groebnerMatrix(37, 196) / (groebnerMatrix(37, 160));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial49(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(49, 54) = -groebnerMatrix(36, 160) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 55) = -groebnerMatrix(36, 161) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 56) = -groebnerMatrix(36, 162) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 82) = -groebnerMatrix(36, 170) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 83) = -groebnerMatrix(36, 171) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 85) = -groebnerMatrix(36, 173) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 86) = -groebnerMatrix(36, 174) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 88) = -groebnerMatrix(36, 176) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 95) = -groebnerMatrix(36, 177) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 105) = groebnerMatrix(27, 181) / (groebnerMatrix(27, 168));
  groebnerMatrix(49, 118) = -groebnerMatrix(36, 178) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 119) = -groebnerMatrix(36, 179) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 120) = -groebnerMatrix(36, 180) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 121) = -groebnerMatrix(36, 181) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 122) = -groebnerMatrix(36, 182) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 123) = -groebnerMatrix(36, 183) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 124) = -groebnerMatrix(36, 184) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 131) = -groebnerMatrix(36, 185) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 139) = -groebnerMatrix(36, 186) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 146) = groebnerMatrix(27, 188) / (groebnerMatrix(27, 168));
  groebnerMatrix(49, 163) = -groebnerMatrix(36, 187) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 164) = -groebnerMatrix(36, 188) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 165) = -groebnerMatrix(36, 189) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 166) = -groebnerMatrix(36, 190) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 167) = -groebnerMatrix(36, 191) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 168) = -groebnerMatrix(36, 192) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 169) = -groebnerMatrix(36, 193) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 176) = -groebnerMatrix(36, 194) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 184) = -groebnerMatrix(36, 195) / (groebnerMatrix(36, 159));
  groebnerMatrix(49, 193) = -groebnerMatrix(36, 196) / (groebnerMatrix(36, 159));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial50(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(50, 53) = -groebnerMatrix(35, 159) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 54) = -groebnerMatrix(35, 160) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 55) = -groebnerMatrix(35, 161) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 56) = -groebnerMatrix(35, 162) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 76) = groebnerMatrix(20, 170) / (groebnerMatrix(20, 164));
  groebnerMatrix(50, 82) = -groebnerMatrix(35, 170) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 83) = -groebnerMatrix(35, 171) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 85) = -groebnerMatrix(35, 173) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 86) = -groebnerMatrix(35, 174) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 88) = -groebnerMatrix(35, 176) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 95) = -groebnerMatrix(35, 177) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 118) = -groebnerMatrix(35, 178) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 119) = -groebnerMatrix(35, 179) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 120) = -groebnerMatrix(35, 180) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 121) = -groebnerMatrix(35, 181) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 122) = -groebnerMatrix(35, 182) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 123) = -groebnerMatrix(35, 183) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 124) = -groebnerMatrix(35, 184) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 131) = -groebnerMatrix(35, 185) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 139) = -groebnerMatrix(35, 186) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 162) = groebnerMatrix(20, 192) / (groebnerMatrix(20, 164));
  groebnerMatrix(50, 163) = -groebnerMatrix(35, 187) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 164) = -groebnerMatrix(35, 188) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 165) = -groebnerMatrix(35, 189) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 166) = -groebnerMatrix(35, 190) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 167) = -groebnerMatrix(35, 191) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 168) = -groebnerMatrix(35, 192) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 169) = -groebnerMatrix(35, 193) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 176) = -groebnerMatrix(35, 194) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 184) = -groebnerMatrix(35, 195) / (groebnerMatrix(35, 158));
  groebnerMatrix(50, 193) = -groebnerMatrix(35, 196) / (groebnerMatrix(35, 158));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial51(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(51, 52) = -groebnerMatrix(34, 158) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 53) = -groebnerMatrix(34, 159) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 54) = -groebnerMatrix(34, 160) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 55) = -groebnerMatrix(34, 161) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 56) = -groebnerMatrix(34, 162) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 77) = groebnerMatrix(12, 171) / (groebnerMatrix(12, 163));
  groebnerMatrix(51, 82) = -groebnerMatrix(34, 170) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 83) = -groebnerMatrix(34, 171) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 85) = -groebnerMatrix(34, 173) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 86) = -groebnerMatrix(34, 174) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 88) = -groebnerMatrix(34, 176) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 95) = -groebnerMatrix(34, 177) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 114) = groebnerMatrix(12, 180) / (groebnerMatrix(12, 163));
  groebnerMatrix(51, 118) = -groebnerMatrix(34, 178) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 119) = -groebnerMatrix(34, 179) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 120) = -groebnerMatrix(34, 180) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 121) = -groebnerMatrix(34, 181) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 122) = -groebnerMatrix(34, 182) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 123) = -groebnerMatrix(34, 183) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 124) = -groebnerMatrix(34, 184) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 131) = -groebnerMatrix(34, 185) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 139) = -groebnerMatrix(34, 186) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 163) = -groebnerMatrix(34, 187) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 164) = -groebnerMatrix(34, 188) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 165) = -groebnerMatrix(34, 189) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 166) = -groebnerMatrix(34, 190) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 167) = -groebnerMatrix(34, 191) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 168) = -groebnerMatrix(34, 192) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 169) = -groebnerMatrix(34, 193) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 176) = -groebnerMatrix(34, 194) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 184) = -groebnerMatrix(34, 195) / (groebnerMatrix(34, 157));
  groebnerMatrix(51, 193) = -groebnerMatrix(34, 196) / (groebnerMatrix(34, 157));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial52(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(52, 51) = -groebnerMatrix(33, 157) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 52) = -groebnerMatrix(33, 158) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 53) = -groebnerMatrix(33, 159) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 54) = -groebnerMatrix(33, 160) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 55) = -groebnerMatrix(33, 161) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 56) = -groebnerMatrix(33, 162) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 74) = groebnerMatrix(21, 173) / (groebnerMatrix(21, 167));
  groebnerMatrix(52, 82) = -groebnerMatrix(33, 170) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 83) = -groebnerMatrix(33, 171) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 85) = -groebnerMatrix(33, 173) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 86) = -groebnerMatrix(33, 174) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 88) = -groebnerMatrix(33, 176) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 95) = -groebnerMatrix(33, 177) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 118) = -groebnerMatrix(33, 178) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 119) = -groebnerMatrix(33, 179) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 120) = -groebnerMatrix(33, 180) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 121) = -groebnerMatrix(33, 181) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 122) = -groebnerMatrix(33, 182) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 123) = -groebnerMatrix(33, 183) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 124) = -groebnerMatrix(33, 184) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 131) = -groebnerMatrix(33, 185) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 139) = -groebnerMatrix(33, 186) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 154) = groebnerMatrix(21, 189) / (groebnerMatrix(21, 167));
  groebnerMatrix(52, 163) = -groebnerMatrix(33, 187) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 164) = -groebnerMatrix(33, 188) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 165) = -groebnerMatrix(33, 189) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 166) = -groebnerMatrix(33, 190) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 167) = -groebnerMatrix(33, 191) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 168) = -groebnerMatrix(33, 192) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 169) = -groebnerMatrix(33, 193) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 176) = -groebnerMatrix(33, 194) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 184) = -groebnerMatrix(33, 195) / (groebnerMatrix(33, 156));
  groebnerMatrix(52, 193) = -groebnerMatrix(33, 196) / (groebnerMatrix(33, 156));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial53(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(53, 50) = -groebnerMatrix(32, 156) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 51) = -groebnerMatrix(32, 157) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 52) = -groebnerMatrix(32, 158) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 53) = -groebnerMatrix(32, 159) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 54) = -groebnerMatrix(32, 160) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 55) = -groebnerMatrix(32, 161) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 56) = -groebnerMatrix(32, 162) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 75) = groebnerMatrix(11, 174) / (groebnerMatrix(11, 166));
  groebnerMatrix(53, 82) = -groebnerMatrix(32, 170) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 83) = -groebnerMatrix(32, 171) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 85) = -groebnerMatrix(32, 173) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 86) = -groebnerMatrix(32, 174) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 88) = -groebnerMatrix(32, 176) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 95) = -groebnerMatrix(32, 177) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 116) = groebnerMatrix(11, 183) / (groebnerMatrix(11, 166));
  groebnerMatrix(53, 118) = -groebnerMatrix(32, 178) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 119) = -groebnerMatrix(32, 179) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 120) = -groebnerMatrix(32, 180) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 121) = -groebnerMatrix(32, 181) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 122) = -groebnerMatrix(32, 182) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 123) = -groebnerMatrix(32, 183) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 124) = -groebnerMatrix(32, 184) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 131) = -groebnerMatrix(32, 185) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 139) = -groebnerMatrix(32, 186) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 163) = -groebnerMatrix(32, 187) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 164) = -groebnerMatrix(32, 188) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 165) = -groebnerMatrix(32, 189) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 166) = -groebnerMatrix(32, 190) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 167) = -groebnerMatrix(32, 191) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 168) = -groebnerMatrix(32, 192) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 169) = -groebnerMatrix(32, 193) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 176) = -groebnerMatrix(32, 194) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 184) = -groebnerMatrix(32, 195) / (groebnerMatrix(32, 155));
  groebnerMatrix(53, 193) = -groebnerMatrix(32, 196) / (groebnerMatrix(32, 155));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial54(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(54, 49) = -groebnerMatrix(31, 155) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 50) = -groebnerMatrix(31, 156) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 51) = -groebnerMatrix(31, 157) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 52) = -groebnerMatrix(31, 158) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 53) = -groebnerMatrix(31, 159) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 54) = -groebnerMatrix(31, 160) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 55) = -groebnerMatrix(31, 161) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 56) = -groebnerMatrix(31, 162) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 71) = groebnerMatrix(20, 170) / (groebnerMatrix(20, 164));
  groebnerMatrix(54, 82) = -groebnerMatrix(31, 170) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 83) = -groebnerMatrix(31, 171) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 85) = -groebnerMatrix(31, 173) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 86) = -groebnerMatrix(31, 174) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 88) = -groebnerMatrix(31, 176) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 95) = -groebnerMatrix(31, 177) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 118) = -groebnerMatrix(31, 178) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 119) = -groebnerMatrix(31, 179) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 120) = -groebnerMatrix(31, 180) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 121) = -groebnerMatrix(31, 181) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 122) = -groebnerMatrix(31, 182) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 123) = -groebnerMatrix(31, 183) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 124) = -groebnerMatrix(31, 184) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 131) = -groebnerMatrix(31, 185) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 139) = -groebnerMatrix(31, 186) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 161) = groebnerMatrix(20, 192) / (groebnerMatrix(20, 164));
  groebnerMatrix(54, 163) = -groebnerMatrix(31, 187) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 164) = -groebnerMatrix(31, 188) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 165) = -groebnerMatrix(31, 189) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 166) = -groebnerMatrix(31, 190) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 167) = -groebnerMatrix(31, 191) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 168) = -groebnerMatrix(31, 192) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 169) = -groebnerMatrix(31, 193) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 176) = -groebnerMatrix(31, 194) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 184) = -groebnerMatrix(31, 195) / (groebnerMatrix(31, 153));
  groebnerMatrix(54, 193) = -groebnerMatrix(31, 196) / (groebnerMatrix(31, 153));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial55(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(55, 38) = groebnerMatrix(38, 162) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 71) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 72) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 74) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 75) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 76) = groebnerMatrix(38, 170) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 77) = groebnerMatrix(38, 171) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 79) = groebnerMatrix(38, 173) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 80) = groebnerMatrix(38, 174) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 86) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 87) = groebnerMatrix(38, 176) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 93) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 94) = groebnerMatrix(38, 177) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 107) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 108) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 109) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 110) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 111) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 112) = groebnerMatrix(38, 178) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 113) = groebnerMatrix(38, 179) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 114) = groebnerMatrix(38, 180) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 115) = groebnerMatrix(38, 181) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 116) =
      (groebnerMatrix(38, 182) / (groebnerMatrix(38, 161)) - groebnerMatrix(39, 183) / (groebnerMatrix(39, 162)));
  groebnerMatrix(55, 117) = groebnerMatrix(38, 183) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 122) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 123) = groebnerMatrix(38, 184) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 129) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 130) = groebnerMatrix(38, 185) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 137) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 138) = groebnerMatrix(38, 186) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 152) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 153) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 154) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 155) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 156) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 157) = groebnerMatrix(38, 187) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 158) = groebnerMatrix(38, 188) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 159) = groebnerMatrix(38, 189) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 160) = groebnerMatrix(38, 190) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 161) =
      (groebnerMatrix(38, 191) / (groebnerMatrix(38, 161)) - groebnerMatrix(39, 192) / (groebnerMatrix(39, 162)));
  groebnerMatrix(55, 162) = groebnerMatrix(38, 192) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 167) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 168) = groebnerMatrix(38, 193) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 174) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 175) = groebnerMatrix(38, 194) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 182) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 183) = groebnerMatrix(38, 195) / (groebnerMatrix(38, 161));
  groebnerMatrix(55, 191) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
  groebnerMatrix(55, 192) = groebnerMatrix(38, 196) / (groebnerMatrix(38, 161));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial56(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(56, 37) = groebnerMatrix(37, 161) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 38) = groebnerMatrix(37, 162) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 67) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 68) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 70) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 74) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 76) = groebnerMatrix(37, 170) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 77) = groebnerMatrix(37, 171) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 79) = groebnerMatrix(37, 173) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 80) = groebnerMatrix(37, 174) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 85) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 87) = groebnerMatrix(37, 176) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 92) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 94) = groebnerMatrix(37, 177) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 103) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 104) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 105) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 106) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 110) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 112) = groebnerMatrix(37, 178) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 113) = groebnerMatrix(37, 179) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 114) = groebnerMatrix(37, 180) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 115) =
      (groebnerMatrix(37, 181) / (groebnerMatrix(37, 160)) - groebnerMatrix(39, 183) / (groebnerMatrix(39, 162)));
  groebnerMatrix(56, 116) = groebnerMatrix(37, 182) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 117) = groebnerMatrix(37, 183) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 121) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 123) = groebnerMatrix(37, 184) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 128) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 130) = groebnerMatrix(37, 185) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 136) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 138) = groebnerMatrix(37, 186) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 148) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 149) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 150) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 151) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 155) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 157) = groebnerMatrix(37, 187) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 158) = groebnerMatrix(37, 188) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 159) = groebnerMatrix(37, 189) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 160) =
      (groebnerMatrix(37, 190) / (groebnerMatrix(37, 160)) - groebnerMatrix(39, 192) / (groebnerMatrix(39, 162)));
  groebnerMatrix(56, 161) = groebnerMatrix(37, 191) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 162) = groebnerMatrix(37, 192) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 166) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 168) = groebnerMatrix(37, 193) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 173) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 175) = groebnerMatrix(37, 194) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 181) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 183) = groebnerMatrix(37, 195) / (groebnerMatrix(37, 160));
  groebnerMatrix(56, 190) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
  groebnerMatrix(56, 192) = groebnerMatrix(37, 196) / (groebnerMatrix(37, 160));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial57(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(57, 36) = groebnerMatrix(36, 160) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 37) = groebnerMatrix(36, 161) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 38) = groebnerMatrix(36, 162) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 64) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 65) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 69) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 73) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 76) = groebnerMatrix(36, 170) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 77) = groebnerMatrix(36, 171) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 79) = groebnerMatrix(36, 173) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 80) = groebnerMatrix(36, 174) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 84) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 87) = groebnerMatrix(36, 176) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 91) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 94) = groebnerMatrix(36, 177) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 100) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 101) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 102) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 105) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 109) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 112) = groebnerMatrix(36, 178) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 113) = groebnerMatrix(36, 179) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 114) =
      (groebnerMatrix(36, 180) / (groebnerMatrix(36, 159)) - groebnerMatrix(39, 183) / (groebnerMatrix(39, 162)));
  groebnerMatrix(57, 115) = groebnerMatrix(36, 181) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 116) = groebnerMatrix(36, 182) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 117) = groebnerMatrix(36, 183) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 120) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 123) = groebnerMatrix(36, 184) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 127) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 130) = groebnerMatrix(36, 185) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 135) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 138) = groebnerMatrix(36, 186) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 145) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 146) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 147) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 150) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 154) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 157) = groebnerMatrix(36, 187) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 158) = groebnerMatrix(36, 188) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 159) =
      (groebnerMatrix(36, 189) / (groebnerMatrix(36, 159)) - groebnerMatrix(39, 192) / (groebnerMatrix(39, 162)));
  groebnerMatrix(57, 160) = groebnerMatrix(36, 190) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 161) = groebnerMatrix(36, 191) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 162) = groebnerMatrix(36, 192) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 165) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 168) = groebnerMatrix(36, 193) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 172) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 175) = groebnerMatrix(36, 194) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 180) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 183) = groebnerMatrix(36, 195) / (groebnerMatrix(36, 159));
  groebnerMatrix(57, 189) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
  groebnerMatrix(57, 192) = groebnerMatrix(36, 196) / (groebnerMatrix(36, 159));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial58(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(58, 35) = groebnerMatrix(35, 159) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 36) = groebnerMatrix(35, 160) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 37) = groebnerMatrix(35, 161) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 38) = groebnerMatrix(35, 162) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 62) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 63) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 68) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 72) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 76) = groebnerMatrix(35, 170) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 77) = groebnerMatrix(35, 171) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 79) = groebnerMatrix(35, 173) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 80) = groebnerMatrix(35, 174) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 83) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 87) = groebnerMatrix(35, 176) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 90) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 94) = groebnerMatrix(35, 177) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 98) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 99) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 101) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 104) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 108) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 112) = groebnerMatrix(35, 178) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 113) =
      (groebnerMatrix(35, 179) / (groebnerMatrix(35, 158)) - groebnerMatrix(39, 183) / (groebnerMatrix(39, 162)));
  groebnerMatrix(58, 114) = groebnerMatrix(35, 180) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 115) = groebnerMatrix(35, 181) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 116) = groebnerMatrix(35, 182) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 117) = groebnerMatrix(35, 183) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 119) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 123) = groebnerMatrix(35, 184) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 126) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 130) = groebnerMatrix(35, 185) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 134) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 138) = groebnerMatrix(35, 186) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 143) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 144) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 146) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 149) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 153) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 157) = groebnerMatrix(35, 187) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 158) =
      (groebnerMatrix(35, 188) / (groebnerMatrix(35, 158)) - groebnerMatrix(39, 192) / (groebnerMatrix(39, 162)));
  groebnerMatrix(58, 159) = groebnerMatrix(35, 189) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 160) = groebnerMatrix(35, 190) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 161) = groebnerMatrix(35, 191) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 162) = groebnerMatrix(35, 192) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 164) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 168) = groebnerMatrix(35, 193) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 171) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 175) = groebnerMatrix(35, 194) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 179) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 183) = groebnerMatrix(35, 195) / (groebnerMatrix(35, 158));
  groebnerMatrix(58, 188) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
  groebnerMatrix(58, 192) = groebnerMatrix(35, 196) / (groebnerMatrix(35, 158));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial59(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(59, 34) = groebnerMatrix(34, 158) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 35) = groebnerMatrix(34, 159) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 36) = groebnerMatrix(34, 160) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 37) = groebnerMatrix(34, 161) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 38) = groebnerMatrix(34, 162) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 61) = -groebnerMatrix(39, 170) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 62) = -groebnerMatrix(39, 171) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 67) = -groebnerMatrix(39, 173) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 71) = -groebnerMatrix(39, 174) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 76) = groebnerMatrix(34, 170) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 77) = groebnerMatrix(34, 171) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 79) = groebnerMatrix(34, 173) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 80) = groebnerMatrix(34, 174) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 82) = -groebnerMatrix(39, 176) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 87) = groebnerMatrix(34, 176) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 89) = -groebnerMatrix(39, 177) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 94) = groebnerMatrix(34, 177) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 97) = -groebnerMatrix(39, 178) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 98) = -groebnerMatrix(39, 179) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 100) = -groebnerMatrix(39, 180) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 103) = -groebnerMatrix(39, 181) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 107) = -groebnerMatrix(39, 182) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 112) =
      (groebnerMatrix(34, 178) / (groebnerMatrix(34, 157)) - groebnerMatrix(39, 183) / (groebnerMatrix(39, 162)));
  groebnerMatrix(59, 113) = groebnerMatrix(34, 179) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 114) = groebnerMatrix(34, 180) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 115) = groebnerMatrix(34, 181) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 116) = groebnerMatrix(34, 182) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 117) = groebnerMatrix(34, 183) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 118) = -groebnerMatrix(39, 184) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 123) = groebnerMatrix(34, 184) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 125) = -groebnerMatrix(39, 185) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 130) = groebnerMatrix(34, 185) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 133) = -groebnerMatrix(39, 186) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 138) = groebnerMatrix(34, 186) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 142) = -groebnerMatrix(39, 187) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 143) = -groebnerMatrix(39, 188) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 145) = -groebnerMatrix(39, 189) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 148) = -groebnerMatrix(39, 190) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 152) = -groebnerMatrix(39, 191) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 157) =
      (groebnerMatrix(34, 187) / (groebnerMatrix(34, 157)) - groebnerMatrix(39, 192) / (groebnerMatrix(39, 162)));
  groebnerMatrix(59, 158) = groebnerMatrix(34, 188) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 159) = groebnerMatrix(34, 189) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 160) = groebnerMatrix(34, 190) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 161) = groebnerMatrix(34, 191) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 162) = groebnerMatrix(34, 192) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 163) = -groebnerMatrix(39, 193) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 168) = groebnerMatrix(34, 193) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 170) = -groebnerMatrix(39, 194) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 175) = groebnerMatrix(34, 194) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 178) = -groebnerMatrix(39, 195) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 183) = groebnerMatrix(34, 195) / (groebnerMatrix(34, 157));
  groebnerMatrix(59, 187) = -groebnerMatrix(39, 196) / (groebnerMatrix(39, 162));
  groebnerMatrix(59, 192) = groebnerMatrix(34, 196) / (groebnerMatrix(34, 157));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial60(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(60, 33) = groebnerMatrix(33, 157) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 34) = groebnerMatrix(33, 158) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 35) = groebnerMatrix(33, 159) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 36) = groebnerMatrix(33, 160) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 37) =
      (groebnerMatrix(33, 161) / (groebnerMatrix(33, 156)) - groebnerMatrix(38, 162) / (groebnerMatrix(38, 161)));
  groebnerMatrix(60, 38) = groebnerMatrix(33, 162) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 71) = -groebnerMatrix(38, 170) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 72) = -groebnerMatrix(38, 171) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 74) = -groebnerMatrix(38, 173) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 75) = -groebnerMatrix(38, 174) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 76) = groebnerMatrix(33, 170) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 77) = groebnerMatrix(33, 171) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 79) = groebnerMatrix(33, 173) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 80) = groebnerMatrix(33, 174) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 86) = -groebnerMatrix(38, 176) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 87) = groebnerMatrix(33, 176) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 93) = -groebnerMatrix(38, 177) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 94) = groebnerMatrix(33, 177) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 107) = -groebnerMatrix(38, 178) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 108) = -groebnerMatrix(38, 179) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 109) = -groebnerMatrix(38, 180) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 110) = -groebnerMatrix(38, 181) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 111) = -groebnerMatrix(38, 182) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 112) = groebnerMatrix(33, 178) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 113) = groebnerMatrix(33, 179) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 114) = groebnerMatrix(33, 180) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 115) = groebnerMatrix(33, 181) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 116) =
      (groebnerMatrix(33, 182) / (groebnerMatrix(33, 156)) - groebnerMatrix(38, 183) / (groebnerMatrix(38, 161)));
  groebnerMatrix(60, 117) = groebnerMatrix(33, 183) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 122) = -groebnerMatrix(38, 184) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 123) = groebnerMatrix(33, 184) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 129) = -groebnerMatrix(38, 185) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 130) = groebnerMatrix(33, 185) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 137) = -groebnerMatrix(38, 186) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 138) = groebnerMatrix(33, 186) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 152) = -groebnerMatrix(38, 187) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 153) = -groebnerMatrix(38, 188) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 154) = -groebnerMatrix(38, 189) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 155) = -groebnerMatrix(38, 190) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 156) = -groebnerMatrix(38, 191) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 157) = groebnerMatrix(33, 187) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 158) = groebnerMatrix(33, 188) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 159) = groebnerMatrix(33, 189) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 160) = groebnerMatrix(33, 190) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 161) =
      (groebnerMatrix(33, 191) / (groebnerMatrix(33, 156)) - groebnerMatrix(38, 192) / (groebnerMatrix(38, 161)));
  groebnerMatrix(60, 162) = groebnerMatrix(33, 192) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 167) = -groebnerMatrix(38, 193) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 168) = groebnerMatrix(33, 193) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 174) = -groebnerMatrix(38, 194) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 175) = groebnerMatrix(33, 194) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 182) = -groebnerMatrix(38, 195) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 183) = groebnerMatrix(33, 195) / (groebnerMatrix(33, 156));
  groebnerMatrix(60, 191) = -groebnerMatrix(38, 196) / (groebnerMatrix(38, 161));
  groebnerMatrix(60, 192) = groebnerMatrix(33, 196) / (groebnerMatrix(33, 156));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial61(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(61, 32) =
      (groebnerMatrix(32, 156) / (groebnerMatrix(32, 155)) - groebnerMatrix(37, 161) / (groebnerMatrix(37, 160)));
  groebnerMatrix(61, 33) = groebnerMatrix(32, 157) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 34) = groebnerMatrix(32, 158) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 35) = groebnerMatrix(32, 159) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 36) = groebnerMatrix(32, 160) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 37) =
      (groebnerMatrix(32, 161) / (groebnerMatrix(32, 155)) - groebnerMatrix(37, 162) / (groebnerMatrix(37, 160)));
  groebnerMatrix(61, 38) = groebnerMatrix(32, 162) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 71) = -groebnerMatrix(37, 170) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 72) = -groebnerMatrix(37, 171) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 74) = -groebnerMatrix(37, 173) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 75) = -groebnerMatrix(37, 174) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 76) = groebnerMatrix(32, 170) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 77) = groebnerMatrix(32, 171) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 79) = groebnerMatrix(32, 173) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 80) = groebnerMatrix(32, 174) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 86) = -groebnerMatrix(37, 176) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 87) = groebnerMatrix(32, 176) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 93) = -groebnerMatrix(37, 177) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 94) = groebnerMatrix(32, 177) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 107) = -groebnerMatrix(37, 178) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 108) = -groebnerMatrix(37, 179) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 109) = -groebnerMatrix(37, 180) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 110) = -groebnerMatrix(37, 181) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 111) = -groebnerMatrix(37, 182) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 112) = groebnerMatrix(32, 178) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 113) = groebnerMatrix(32, 179) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 114) = groebnerMatrix(32, 180) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 115) = groebnerMatrix(32, 181) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 116) =
      (groebnerMatrix(32, 182) / (groebnerMatrix(32, 155)) - groebnerMatrix(37, 183) / (groebnerMatrix(37, 160)));
  groebnerMatrix(61, 117) = groebnerMatrix(32, 183) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 122) = -groebnerMatrix(37, 184) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 123) = groebnerMatrix(32, 184) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 129) = -groebnerMatrix(37, 185) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 130) = groebnerMatrix(32, 185) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 137) = -groebnerMatrix(37, 186) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 138) = groebnerMatrix(32, 186) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 152) = -groebnerMatrix(37, 187) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 153) = -groebnerMatrix(37, 188) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 154) = -groebnerMatrix(37, 189) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 155) = -groebnerMatrix(37, 190) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 156) = -groebnerMatrix(37, 191) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 157) = groebnerMatrix(32, 187) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 158) = groebnerMatrix(32, 188) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 159) = groebnerMatrix(32, 189) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 160) = groebnerMatrix(32, 190) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 161) =
      (groebnerMatrix(32, 191) / (groebnerMatrix(32, 155)) - groebnerMatrix(37, 192) / (groebnerMatrix(37, 160)));
  groebnerMatrix(61, 162) = groebnerMatrix(32, 192) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 167) = -groebnerMatrix(37, 193) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 168) = groebnerMatrix(32, 193) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 174) = -groebnerMatrix(37, 194) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 175) = groebnerMatrix(32, 194) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 182) = -groebnerMatrix(37, 195) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 183) = groebnerMatrix(32, 195) / (groebnerMatrix(32, 155));
  groebnerMatrix(61, 191) = -groebnerMatrix(37, 196) / (groebnerMatrix(37, 160));
  groebnerMatrix(61, 192) = groebnerMatrix(32, 196) / (groebnerMatrix(32, 155));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial62(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(62, 32) = groebnerMatrix(32, 156) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 33) = groebnerMatrix(32, 157) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 34) = groebnerMatrix(32, 158) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 35) = groebnerMatrix(32, 159) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 36) =
      (groebnerMatrix(32, 160) / (groebnerMatrix(32, 155)) - groebnerMatrix(38, 162) / (groebnerMatrix(38, 161)));
  groebnerMatrix(62, 37) = groebnerMatrix(32, 161) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 38) = groebnerMatrix(32, 162) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 67) = -groebnerMatrix(38, 170) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 68) = -groebnerMatrix(38, 171) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 70) = -groebnerMatrix(38, 173) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 74) = -groebnerMatrix(38, 174) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 76) = groebnerMatrix(32, 170) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 77) = groebnerMatrix(32, 171) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 79) = groebnerMatrix(32, 173) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 80) = groebnerMatrix(32, 174) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 85) = -groebnerMatrix(38, 176) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 87) = groebnerMatrix(32, 176) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 92) = -groebnerMatrix(38, 177) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 94) = groebnerMatrix(32, 177) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 103) = -groebnerMatrix(38, 178) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 104) = -groebnerMatrix(38, 179) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 105) = -groebnerMatrix(38, 180) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 106) = -groebnerMatrix(38, 181) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 110) = -groebnerMatrix(38, 182) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 112) = groebnerMatrix(32, 178) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 113) = groebnerMatrix(32, 179) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 114) = groebnerMatrix(32, 180) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 115) =
      (groebnerMatrix(32, 181) / (groebnerMatrix(32, 155)) - groebnerMatrix(38, 183) / (groebnerMatrix(38, 161)));
  groebnerMatrix(62, 116) = groebnerMatrix(32, 182) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 117) = groebnerMatrix(32, 183) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 121) = -groebnerMatrix(38, 184) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 123) = groebnerMatrix(32, 184) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 128) = -groebnerMatrix(38, 185) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 130) = groebnerMatrix(32, 185) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 136) = -groebnerMatrix(38, 186) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 138) = groebnerMatrix(32, 186) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 148) = -groebnerMatrix(38, 187) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 149) = -groebnerMatrix(38, 188) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 150) = -groebnerMatrix(38, 189) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 151) = -groebnerMatrix(38, 190) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 155) = -groebnerMatrix(38, 191) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 157) = groebnerMatrix(32, 187) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 158) = groebnerMatrix(32, 188) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 159) = groebnerMatrix(32, 189) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 160) =
      (groebnerMatrix(32, 190) / (groebnerMatrix(32, 155)) - groebnerMatrix(38, 192) / (groebnerMatrix(38, 161)));
  groebnerMatrix(62, 161) = groebnerMatrix(32, 191) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 162) = groebnerMatrix(32, 192) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 166) = -groebnerMatrix(38, 193) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 168) = groebnerMatrix(32, 193) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 173) = -groebnerMatrix(38, 194) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 175) = groebnerMatrix(32, 194) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 181) = -groebnerMatrix(38, 195) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 183) = groebnerMatrix(32, 195) / (groebnerMatrix(32, 155));
  groebnerMatrix(62, 190) = -groebnerMatrix(38, 196) / (groebnerMatrix(38, 161));
  groebnerMatrix(62, 192) = groebnerMatrix(32, 196) / (groebnerMatrix(32, 155));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial63(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(63, 31) = -groebnerMatrix(36, 160) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 32) = -groebnerMatrix(36, 161) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 34) = groebnerMatrix(16, 158) / (groebnerMatrix(16, 154));
  groebnerMatrix(63, 37) = -groebnerMatrix(36, 162) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 71) = -groebnerMatrix(36, 170) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 72) = -groebnerMatrix(36, 171) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 74) = -groebnerMatrix(36, 173) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 75) = -groebnerMatrix(36, 174) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 86) = -groebnerMatrix(36, 176) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 93) = -groebnerMatrix(36, 177) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 107) = -groebnerMatrix(36, 178) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 108) = -groebnerMatrix(36, 179) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 109) = -groebnerMatrix(36, 180) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 110) = -groebnerMatrix(36, 181) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 111) = -groebnerMatrix(36, 182) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 116) = -groebnerMatrix(36, 183) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 122) = -groebnerMatrix(36, 184) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 129) = -groebnerMatrix(36, 185) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 137) = -groebnerMatrix(36, 186) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 152) = -groebnerMatrix(36, 187) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 153) = -groebnerMatrix(36, 188) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 154) = -groebnerMatrix(36, 189) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 155) = -groebnerMatrix(36, 190) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 156) = -groebnerMatrix(36, 191) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 161) = -groebnerMatrix(36, 192) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 167) = -groebnerMatrix(36, 193) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 168) = groebnerMatrix(16, 193) / (groebnerMatrix(16, 154));
  groebnerMatrix(63, 174) = -groebnerMatrix(36, 194) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 182) = -groebnerMatrix(36, 195) / (groebnerMatrix(36, 159));
  groebnerMatrix(63, 191) = -groebnerMatrix(36, 196) / (groebnerMatrix(36, 159));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial64(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(64, 34) = groebnerMatrix(16, 158) / (groebnerMatrix(16, 154));
  groebnerMatrix(64, 35) = -groebnerMatrix(38, 162) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 64) = -groebnerMatrix(38, 170) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 65) = -groebnerMatrix(38, 171) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 69) = -groebnerMatrix(38, 173) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 73) = -groebnerMatrix(38, 174) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 84) = -groebnerMatrix(38, 176) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 91) = -groebnerMatrix(38, 177) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 100) = -groebnerMatrix(38, 178) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 101) = -groebnerMatrix(38, 179) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 102) = -groebnerMatrix(38, 180) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 105) = -groebnerMatrix(38, 181) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 109) = -groebnerMatrix(38, 182) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 114) = -groebnerMatrix(38, 183) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 120) = -groebnerMatrix(38, 184) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 127) = -groebnerMatrix(38, 185) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 135) = -groebnerMatrix(38, 186) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 145) = -groebnerMatrix(38, 187) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 146) = -groebnerMatrix(38, 188) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 147) = -groebnerMatrix(38, 189) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 150) = -groebnerMatrix(38, 190) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 154) = -groebnerMatrix(38, 191) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 159) = -groebnerMatrix(38, 192) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 165) = -groebnerMatrix(38, 193) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 168) = groebnerMatrix(16, 193) / (groebnerMatrix(16, 154));
  groebnerMatrix(64, 172) = -groebnerMatrix(38, 194) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 180) = -groebnerMatrix(38, 195) / (groebnerMatrix(38, 161));
  groebnerMatrix(64, 189) = -groebnerMatrix(38, 196) / (groebnerMatrix(38, 161));
}

void
opengv::relative_pose::modules::fivept_kneip::sPolynomial65(Eigen::Matrix<double, 66, 197> &groebnerMatrix) {
  groebnerMatrix(65, 30) = -groebnerMatrix(35, 159) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 31) =
      (groebnerMatrix(31, 155) / (groebnerMatrix(31, 153)) - groebnerMatrix(35, 160) / (groebnerMatrix(35, 158)));
  groebnerMatrix(65, 32) =
      (groebnerMatrix(31, 156) / (groebnerMatrix(31, 153)) - groebnerMatrix(35, 161) / (groebnerMatrix(35, 158)));
  groebnerMatrix(65, 33) = groebnerMatrix(31, 157) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 34) = groebnerMatrix(31, 158) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 35) = groebnerMatrix(31, 159) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 36) = groebnerMatrix(31, 160) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 37) =
      (groebnerMatrix(31, 161) / (groebnerMatrix(31, 153)) - groebnerMatrix(35, 162) / (groebnerMatrix(35, 158)));
  groebnerMatrix(65, 38) = groebnerMatrix(31, 162) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 71) = -groebnerMatrix(35, 170) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 72) = -groebnerMatrix(35, 171) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 74) = -groebnerMatrix(35, 173) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 75) = -groebnerMatrix(35, 174) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 76) = groebnerMatrix(31, 170) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 77) = groebnerMatrix(31, 171) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 79) = groebnerMatrix(31, 173) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 80) = groebnerMatrix(31, 174) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 86) = -groebnerMatrix(35, 176) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 87) = groebnerMatrix(31, 176) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 93) = -groebnerMatrix(35, 177) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 94) = groebnerMatrix(31, 177) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 107) = -groebnerMatrix(35, 178) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 108) = -groebnerMatrix(35, 179) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 109) = -groebnerMatrix(35, 180) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 110) = -groebnerMatrix(35, 181) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 111) = -groebnerMatrix(35, 182) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 112) = groebnerMatrix(31, 178) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 113) = groebnerMatrix(31, 179) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 114) = groebnerMatrix(31, 180) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 115) = groebnerMatrix(31, 181) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 116) =
      (groebnerMatrix(31, 182) / (groebnerMatrix(31, 153)) - groebnerMatrix(35, 183) / (groebnerMatrix(35, 158)));
  groebnerMatrix(65, 117) = groebnerMatrix(31, 183) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 122) = -groebnerMatrix(35, 184) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 123) = groebnerMatrix(31, 184) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 129) = -groebnerMatrix(35, 185) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 130) = groebnerMatrix(31, 185) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 137) = -groebnerMatrix(35, 186) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 138) = groebnerMatrix(31, 186) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 152) = -groebnerMatrix(35, 187) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 153) = -groebnerMatrix(35, 188) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 154) = -groebnerMatrix(35, 189) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 155) = -groebnerMatrix(35, 190) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 156) = -groebnerMatrix(35, 191) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 157) = groebnerMatrix(31, 187) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 158) = groebnerMatrix(31, 188) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 159) = groebnerMatrix(31, 189) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 160) = groebnerMatrix(31, 190) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 161) =
      (groebnerMatrix(31, 191) / (groebnerMatrix(31, 153)) - groebnerMatrix(35, 192) / (groebnerMatrix(35, 158)));
  groebnerMatrix(65, 162) = groebnerMatrix(31, 192) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 167) = -groebnerMatrix(35, 193) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 168) = groebnerMatrix(31, 193) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 174) = -groebnerMatrix(35, 194) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 175) = groebnerMatrix(31, 194) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 182) = -groebnerMatrix(35, 195) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 183) = groebnerMatrix(31, 195) / (groebnerMatrix(31, 153));
  groebnerMatrix(65, 191) = -groebnerMatrix(35, 196) / (groebnerMatrix(35, 158));
  groebnerMatrix(65, 192) = groebnerMatrix(31, 196) / (groebnerMatrix(31, 153));
}

