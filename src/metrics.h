#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <immintrin.h>
#include <stdint.h>
#include "common.h"

namespace metrics {

	inline double sum_of_squared_deviations(double* p1, double* p2, double* weights, size_t n) {
		double sum = 0;

		for (size_t i = 0; i < n; i++) {
			double tmp = p1[i] - p2[i];
			tmp = tmp * tmp * weights[i];
			sum += tmp;
		}
		return sum;
	}



	inline double sum_of_squared_deviations_weightless(double* p1, double* p2, double* weights, size_t n) {
		double sum = 0;

		for (size_t i = 0; i < n; i++) {
			double tmp = p1[i] - p2[i];
			tmp = tmp * tmp;
			sum += tmp;
		}
		return sum;
	}


	inline double euclidean_distance(double* p1, double* p2, size_t n) {
		return sqrt(sum_of_squared_deviations_weightless(p1, p2, nullptr, n));
	}


	inline double deviation_squares_sum_weightless_avx(double* p1, double* p2, size_t n) {
		double sum = 0;

		size_t vec_loop_cnt = (n / common::AVX_VEC_LEN) * common::AVX_VEC_LEN;
		size_t v = 0;
		__m256d sum_vec = _mm256_set_pd(0, 0, 0, 0);

		for (; v < vec_loop_cnt; v += common::AVX_VEC_LEN) {
			__m256d v1 = _mm256_loadu_pd(p1 + v);
			__m256d v2 = _mm256_loadu_pd(p2 + v);
			__m256d delta = _mm256_sub_pd(v2, v1);
			__m256d delta_sqr = _mm256_mul_pd(delta, delta);
			sum_vec = _mm256_add_pd(sum_vec, delta_sqr);
		}

		sum = common::reduce_avx_vector_d(sum_vec);

		for (; v < n; v++) {
			double tmp = p1[v] - p2[v];
			tmp = tmp * tmp;
			sum += tmp;
		}
		return sum;
	}
}