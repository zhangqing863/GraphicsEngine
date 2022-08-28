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

		/// <summary>
		/// 
		/// </summary>
		/// <param name="vFov">垂直方向的FOV，单位：Degree</param>
		/// <param name="aspect">水平与垂直方向长度的比例</param>
		Camera(Float vFov, Float aspect) {
			Float theta = vFov * Degree2Rad;
			Float halfHeight = tan(theta / 2.0);
			Float halfWidth = aspect * halfHeight;
			lowerLeftCorner = Vector3f(-halfWidth, -halfHeight, -1.0);
			horizontal = Vector3f(2.0 * halfWidth, 0.0, 0.0);
			vertical = Vector3f(0.0, 2.0 * halfHeight, 0.0);
			origin = Point3f(0.0, 0.0, 0.0);
		}

		Camera(Point3f looFrom, Point3f lookAt, Vector3f Up, Float vFov, Float aspect) {
			Vector3f u, v, w;
			Float theta = vFov * Degree2Rad;
			Float halfHeight = tan(theta / 2.0);
			Float halfWidth = aspect * halfHeight;
			origin = looFrom;
			w = Normalize(looFrom - lookAt);
			u = Normalize(Cross(Up, w));
			v = Cross(w, u);
			lowerLeftCorner = Vector3f(origin) - halfWidth * u - halfHeight * v - w;
			horizontal = 2 * halfWidth * u;
			vertical = 2 * halfHeight * v;
		}

		Ray GenerateRay(float u, float v) { return Ray(origin, lowerLeftCorner + u * horizontal + v * vertical - Vector3f(origin)); }

		Vector3f lowerLeftCorner;
		Vector3f horizontal;
		Vector3f vertical;
		Point3f origin;
		
	};
}


#endif // QZRT_CORE_CAMERA_H