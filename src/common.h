#pragma once

#include <immintrin.h>

namespace common {

const size_t AVX_VEC_LEN = 4;

	inline double reduce_avx_vector_d(__m256d v) {
		__m128d v_low = _mm256_castpd256_pd128(v);
		__m128d v_high = _mm256_extractf128_pd(v, 1);
		v_low = _mm_add_pd(v_low, v_high);

		__m128d high64 = _mm_unpackhi_pd(v_low, v_low);
		double result = _mm_cvtsd_f64(_mm_add_sd(v_low, high64));
		return result;
	}
}