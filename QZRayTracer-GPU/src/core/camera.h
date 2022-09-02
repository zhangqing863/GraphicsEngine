#ifndef QZRT_CORE_CAMERA_H
#define QZRT_CORE_CAMERA_H

#include "QZRayTracer.h"
#include "geometry.h"

namespace raytracer{
	class Camera {
	public:
		__device__ Camera() {
			lowerLeftCorner = Vector3f(-2.0, -1.0, -1.0);
			horizontal = Vector3f(4.0, 0.0, 0.0);
			vertical = Vector3f(0.0, 2.0, 0.0);
			origin = Point3f(0.0, 0.0, 0.0);
			lensRadius = 0.f;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="vFov">��ֱ�����FOV����λ��Degree</param>
		/// <param name="aspect">ˮƽ�봹ֱ���򳤶ȵı���</param>
		__device__ Camera(Float vFov, Float aspect) {
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
		/// <param name="looFrom">�۲��λ��</param>
		/// <param name="lookAt">�۲�Ŀ���λ��</param>
		/// <param name="Up">�������ϵ�����</param>
		/// <param name="vFov">��ֱ������ӳ�</param>
		/// <param name="aspect">ͼ�����</param>
		__device__ Camera(Point3f looFrom, Point3f lookAt, Vector3f Up, Float vFov, Float aspect) {
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
		/// <param name="looFrom">�۲��λ��</param>
		/// <param name="lookAt">�۲�Ŀ���λ��</param>
		/// <param name="Up">�������ϵ�����</param>
		/// <param name="vFov">��ֱ������ӳ�</param>
		/// <param name="aspect">ͼ�����</param>
		/// <param name="aperture">��Ȧ��С</param>
		/// <param name="focusDis">����</param>
		__device__ Camera(Point3f looFrom, Point3f lookAt, Vector3f Up, Float vFov, Float aspect, Float aperture, Float focusDis) {
			lensRadius = aperture * 0.5;
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

		__device__ Ray GenerateRay(Float s, Float t, curandState* local_rand_state) {
			Point3f randomLoc = lensRadius * RandomInUnitDisk(local_rand_state);
			Vector3f offset = u * randomLoc.x + v * randomLoc.y;
			return Ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - Vector3f(origin) - offset);
		}

		Vector3f lowerLeftCorner;
		Vector3f horizontal;
		Vector3f vertical;
		Vector3f u, v, w; // ������
		Point3f origin;
		Float lensRadius; // ��ͷ�뾶
		
	};
}


#endif // QZRT_CORE_CAMERA_H