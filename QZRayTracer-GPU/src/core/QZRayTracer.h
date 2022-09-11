#ifndef QZRT_CORE_QZRAYTRACER_H
#define QZRT_CORE_QZRAYTRACER_H



#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "../ext/logging.h"

#define QZRT_CONSTEXPR constexpr
#define GPUMODE
// #define PBRT_FLOAT_AS_DOUBLE
namespace raytracer {

	

#ifdef PBRT_FLOAT_AS_DOUBLE
	typedef double Float;
#else
	typedef float Float;
#endif  // PBRT_FLOAT_AS_DOUBLE
	template <typename T>
	class Vector2;
	template <typename T>
	class Vector3;
	template <typename T>
	class Point3;
	template <typename T>
	class Point2;
	template <typename T>
	class Normal3;
	class Ray;
	template <typename T>
	class Bounds2;
	template <typename T>
	class Bounds3;
	// class Shape;
	class Material;
	class ProgressBar;
	class ParamSet;
	template <typename T>
	struct ParamSetItem;
	struct RendererSet;


	// Global Constants
#ifdef _MSC_VER
	__device__ static QZRT_CONSTEXPR Float MaxFloat = std::numeric_limits<Float>::max();
	__device__ static QZRT_CONSTEXPR Float Infinity = std::numeric_limits<Float>::infinity();
	__device__ static QZRT_CONSTEXPR Float MinFloat = std::numeric_limits<Float>::lowest();
#else
	static QZRT_CONSTEXPR Float MaxFloat = std::numeric_limits<Float>::max();
	static QZRT_CONSTEXPR Float Infinity = std::numeric_limits<Float>::infinity();
#endif
#ifdef _MSC_VER
#define MachineEpsilon (std::numeric_limits<Float>::epsilon() * 0.5)
#else
	static QZRT_CONSTEXPR Float MachineEpsilon =
		std::numeric_limits<Float>::epsilon() * 0.5;
#endif
	__device__ static constexpr Float ShadowEpsilon = 0.001f;
	__device__ static QZRT_CONSTEXPR Float Pi = 3.14159265358979323846;
	__device__ static QZRT_CONSTEXPR Float InvPi = 0.31830988618379067154;
	__device__ static QZRT_CONSTEXPR Float Inv2Pi = 0.15915494309189533577;
	__device__ static QZRT_CONSTEXPR Float Inv4Pi = 0.07957747154594766788;
	__device__ static QZRT_CONSTEXPR Float PiOver2 = 1.57079632679489661923;
	__device__ static QZRT_CONSTEXPR Float PiOver4 = 0.78539816339744830961;
	__device__ static QZRT_CONSTEXPR Float Sqrt2 = 1.41421356237309504880;
	__device__ static QZRT_CONSTEXPR Float Rad2Degree = 57.29577951308232087680;
	__device__ static QZRT_CONSTEXPR Float Degree2Rad = 0.01745329251994329577;
	__device__ static QZRT_CONSTEXPR Float Gamma = 1.0 / 2.2;
#ifdef GPUMODE
#else
	
#endif // GPUMODE
	static std::default_random_engine seeds;
	static std::uniform_real_distribution<Float> randomNum(0, 1); // ×ó±ÕÓÒ±ÕÇø¼ä
	// Global Inline Functions

	// Global Inline Functions
	__device__ inline bool Quadratic(Float A, Float B, Float C, Float& t0, Float& t1) {
		// Find quadratic discriminant
		double discrim = (double)B * (double)B - 4.f * (double)A * (double)C;
		if (discrim < 0.0f) return false;
		double rootDiscrim = std::sqrt(discrim);
		// Compute quadratic _t_ values
		Float q;
		if ((float)B < 0.0f)
			q = -.5f * (B - rootDiscrim);
		else
			q = -.5f * (B + rootDiscrim);
		t0 = q / A;
		t1 = C / q;
		if ((float)t0 > (float)t1) {
			float temp = t0;
			t0 = t1;
			t1 = temp;
		}
		return true;
	}

	template <typename T>
	__device__ inline T Min(T a, T b) {
		return a < b ? a : b;
	}

	template <typename T>
	__device__ inline T Max(T a, T b) {
		return a < b ? b : a;
	}
	



	



}



#endif  // QZRT_CORE_QZRAYTRACER_H