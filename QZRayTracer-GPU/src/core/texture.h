#ifndef QZRT_CORE_TEXTURE_H
#define QZRT_CORE_TEXTURE_H

#include "QZRayTracer.h"
#include "geometry.h"


namespace raytracer {
	class Texture {
	public:
		__device__ virtual Point3f value(float u, float v, const Point3f& p)const = 0;
	};
}

#endif // QZRT_CORE_TEXTURE_H