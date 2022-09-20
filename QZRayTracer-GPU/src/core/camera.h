#ifndef QZRT_CORE_CAMERA_H
#define QZRT_CORE_CAMERA_H

#include "QZRayTracer.h"
#include "geometry.h"

namespace raytracer{
	class Camera {
	public:
		__device__ Camera() {
			time0 = 0.f;
			time1 = 1.f;
			lowerLeftCorner = Vector3f(-2.0, -1.0, -1.0);
			horizontal = Vector3f(4.0, 0.0, 0.0);
			vertical = Vector3f(0.0, 2.0, 0.0);
			origin = Point3f(0.0, 0.0, 0.0);
			lensRadius = 0.f;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="vFov">垂直方向的FOV，单位：Degree</param>
		/// <param name="aspect">水平与垂直方向长度的比例</param>
		__device__ Camera(Float vFov, Float aspect) {
			time0 = 0.f;
			time1 = 1.f;
			lensRadius = 0.f;
			Float theta = vFov * Degree2Rad;
			Float halfHeight = tanf(theta / 2.0);
			Float halfWidth = aspect * halfHeight;
			lowerLeftCorner = Vector3f(-halfWidth, -halfHeight, -1.0);
			horizontal = Vector3f(2.0 * halfWidth, 0.0, 0.0);
			vertical = Vector3f(0.0, 2.0 * halfHeight, 0.0);
			origin = Point3f(0.0, 0.0, 0.0);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="looFrom">观测点位置</param>
		/// <param name="lookAt">观测目标的位置</param>
		/// <param name="Up">方向向上的向量</param>
		/// <param name="vFov">垂直方向的视场</param>
		/// <param name="aspect">图像比例</param>
		__device__ Camera(Point3f looFrom, Point3f lookAt, Vector3f Up, Float vFov, Float aspect) {
			time0 = 0.f;
			time1 = 1.f;
			lensRadius = 0.f;
			Float theta = vFov * Degree2Rad;
			Float halfHeight = tanf(theta / 2.0);
			Float halfWidth = aspect * halfHeight;
			origin = looFrom;
			w = Normalize(looFrom - lookAt);
			u = Normalize(Cross(Up, w));
			v = Cross(w, u);
			lowerLeftCorner = Vector3f(origin) - halfWidth * u - halfHeight * v - w;
			horizontal = 2 * halfWidth * u;
			vertical = 2 * halfHeight * v;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="looFrom">观测点位置</param>
		/// <param name="lookAt">观测目标的位置</param>
		/// <param name="Up">方向向上的向量</param>
		/// <param name="vFov">垂直方向的视场</param>
		/// <param name="aspect">图像比例</param>
		/// <param name="aperture">光圈大小</param>
		/// <param name="focusDis">焦距</param>
		__device__ Camera(Point3f looFrom, Point3f lookAt, Vector3f Up, Float vFov, Float aspect, Float aperture, Float focusDis) {
			time0 = 0.f;
			time1 = 1.f;
			lensRadius = aperture * 0.5f;
			Float theta = vFov * Degree2Rad;
			Float halfHeight = tan(theta / 2.0);
			Float halfWidth = aspect * halfHeight;
			origin = looFrom;
			w = Normalize(looFrom - lookAt);
			u = Normalize(Cross(Up, w));
			v = Cross(w, u);
			lowerLeftCorner = Vector3f(origin) - halfWidth * u * focusDis - halfHeight * v * focusDis - w * focusDis;
			horizontal = 2 * halfWidth * u * focusDis;
			vertical = 2 * halfHeight * v * focusDis;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="looFrom">观测点位置</param>
		/// <param name="lookAt">观测目标的位置</param>
		/// <param name="Up">方向向上的向量</param>
		/// <param name="vFov">垂直方向的视场</param>
		/// <param name="aspect">图像比例</param>
		/// <param name="aperture">光圈大小</param>
		/// <param name="focusDis">焦距</param>
		/// <param name="t0">快门开启时间</param>
		/// <param name="t1">快门关闭时间</param>
		/// <returns></returns>
		__device__ Camera(Point3f looFrom, Point3f lookAt, Vector3f Up, Float vFov, Float aspect, Float aperture, Float focusDis, Float t0, Float t1) {
			time0 = t0;
			time1 = t1;
			lensRadius = aperture * 0.5f;
			Float theta = vFov * Degree2Rad;
			Float halfHeight = tan(theta / 2.0f);
			Float halfWidth = aspect * halfHeight;
			origin = looFrom;
			w = Normalize(looFrom - lookAt);
			u = Normalize(Cross(Up, w));
			v = Cross(w, u);
			lowerLeftCorner = Vector3f(origin) - halfWidth * u * focusDis - halfHeight * v * focusDis - w * focusDis;
			horizontal = 2 * halfWidth * u * focusDis;
			vertical = 2 * halfHeight * v * focusDis;
		}

		__device__ Ray GenerateRay(Float s, Float t, curandState* local_rand_state) {
			Point3f randomLoc = lensRadius * RandomInUnitDisk(local_rand_state);
			Vector3f offset = u * randomLoc.x + v * randomLoc.y;
			Float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
			//printf("time0: %f, time1: %f and time %f \n", time0, time1, time);
			return Ray(origin + offset, Normalize(lowerLeftCorner + s * horizontal + t * vertical - Vector3f(origin) - offset), time);
		}

		Vector3f lowerLeftCorner;
		Vector3f horizontal;
		Vector3f vertical;
		Vector3f u, v, w; // 基向量
		Point3f origin;
		Float lensRadius; // 镜头半径
		Float time0, time1; // 快门开启时间和关闭时间，用来模拟运动模糊所需要的参数
	};
}


#endif // QZRT_CORE_CAMERA_H