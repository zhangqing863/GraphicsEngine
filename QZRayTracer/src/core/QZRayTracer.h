#ifndef QZRT_CORE_QZRAYTRACER_H
#define QZRT_CORE_QZRAYTRACER_H



#include <iostream>
#include "../ext/logging.h"

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


}



#endif  // QZRT_CORE_QZRAYTRACER_H