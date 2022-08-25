#ifndef QZRT_CORE_CAMERA_H
#define QZRT_CORE_CAMERA_H

#include "QZRayTracer.h"
#include "geometry.h"

namespace raytracer{
	class Camera {
	public:
		Camera() {
			lowerLeftCorner = Vector3f(-2.0, -1.0, -1.0);
			horizontal = Vector3f(4.0, 0.0, 0.0);
			vertical = Vector3f(0.0, 2.0, 0.0);
			origin = Point3f(0.0, 0.0, 0.0);
		}

		Ray GenerateRay(float u, float v) { return Ray(origin, lowerLeftCorner + u * horizontal + v * vertical - Vector3f(origin)); }

		Vector3f lowerLeftCorner;
		Vector3f horizontal;
		Vector3f vertical;
		Point3f origin;
	};
}


#endif // QZRT_CORE_CAMERA_H