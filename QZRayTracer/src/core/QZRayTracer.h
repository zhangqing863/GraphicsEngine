#ifndef QZRT_CORE_QZRAYTRACER_H
#define QZRT_CORE_QZRAYTRACER_H



#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <random>
#include <vector>
#include "../ext/logging.h"

#define QZRT_CONSTEXPR constexpr

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
	static constexpr Float ShadowEpsilon = 0.0001f;
	static QZRT_CONSTEXPR Float Pi = 3.14159265358979323846;
	static QZRT_CONSTEXPR Float InvPi = 0.31830988618379067154;
	static QZRT_CONSTEXPR Float Inv2Pi = 0.15915494309189533577;
	static QZRT_CONSTEXPR Float Inv4Pi = 0.07957747154594766788;
	static QZRT_CONSTEXPR Float PiOver2 = 1.57079632679489661923;
	static QZRT_CONSTEXPR Float PiOver4 = 0.78539816339744830961;
	static QZRT_CONSTEXPR Float Sqrt2 = 1.41421356237309504880;

	// Global Inline Functions
	
}



#endif  // QZRT_CORE_QZRAYTRACER_H