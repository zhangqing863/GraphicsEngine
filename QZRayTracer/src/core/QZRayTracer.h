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
#include "../ext/logging.h"

#define QZRT_CONSTEXPR constexpr
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
	class Shape;
	class ShapeList;
	class Material;
	class ProgressBar;
	class ParamSet;
	template <typename T>
	struct ParamSetItem;
	struct RendererSet;


	// Global Constants
#ifdef _MSC_VER
#define MaxFloat std::numeric_limits<Float>::max()
#define Infinity std::numeric_limits<Float>::infinity()
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
	static constexpr Float ShadowEpsilon = 0.001f;
	static QZRT_CONSTEXPR Float Pi = 3.14159265358979323846;
	static QZRT_CONSTEXPR Float InvPi = 0.31830988618379067154;
	static QZRT_CONSTEXPR Float Inv2Pi = 0.15915494309189533577;
	static QZRT_CONSTEXPR Float Inv4Pi = 0.07957747154594766788;
	static QZRT_CONSTEXPR Float PiOver2 = 1.57079632679489661923;
	static QZRT_CONSTEXPR Float PiOver4 = 0.78539816339744830961;
	static QZRT_CONSTEXPR Float Sqrt2 = 1.41421356237309504880;
	static QZRT_CONSTEXPR Float Rad2Degree = 57.29577951308232087680;
	static QZRT_CONSTEXPR Float Degree2Rad = 0.01745329251994329577;
	static QZRT_CONSTEXPR Float Gamma = 1.0 / 2.2;
	static std::default_random_engine seeds;
	static std::uniform_real_distribution<Float> randomNum(0, 1); // ×ó±ÕÓÒ±ÕÇø¼ä
	// Global Inline Functions

	// Global Inline Functions
	inline bool Quadratic(Float A, Float B, Float C, Float& t0, Float& t1) {
		// Find quadratic discriminant
		double discrim = (double)B * (double)B - 4. * (double)A * (double)C;
		if (discrim < 0.) return false;
		double rootDiscrim = std::sqrt(discrim);
		// Compute quadratic _t_ values
		Float q;
		if ((float)B < 0)
			q = -.5 * (B - rootDiscrim);
		else
			q = -.5 * (B + rootDiscrim);
		t0 = q / A;
		t1 = C / q;
		if ((float)t0 > (float)t1) std::swap(t0, t1);
		return true;
	}
	
}



#endif  // QZRT_CORE_QZRAYTRACER_H