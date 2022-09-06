#ifndef QZRT_SHAPE_DSPHERE_H
#define QZRT_SHAPE_DSPHERE_H
#include "../core/shape.h"

namespace raytracer {
	/// <summary>
	/// ��̬������
	/// </summary>
	class DSphere :public Shape {
	public:
		Point3f center0, center1;
		Float time0, time1;
		Float invOverTime;
		Float radius;
		Float invRadius;
		Material* material;
		__device__ DSphere() :center0(Point3f(0, 0, 0)), center1(Point3f(0, 0, 0)), radius(1.0f), invRadius(1.0f) { material = nullptr; invOverTime = 0; }
		__device__ DSphere(Point3f center0, Point3f center1, Float t0, Float t1, Float radius) : center0(center0), center1(center1), time0(t0), time1(t1), radius(radius), invRadius(1.0f) { material = nullptr; invOverTime = 1.0f / (time1 - time0);}
		__device__ DSphere(Point3f center0, Point3f center1, Float t0, Float t1, Float radius, Material* mat) : center0(center0), center1(center1), time0(t0), time1(t1), radius(radius), material(mat) {
			invRadius = 1.0f / radius;
			invOverTime = 1.0f / (time1 - time0);
		};
		// ͨ�� Shape �̳�
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		__device__ Point3f Center(Float time)const {
			return Lerp((time - time0) * invOverTime, center0, center1);
		}
	};

	

	__device__ inline bool DSphere::Hit(const Ray& ray, HitRecord& rec) const {
		Vector3f oc = ray.o - Center(ray.time);
		Float a = Dot(ray.d, ray.d);
		Float b = 2.0f * Dot(oc, ray.d);
		Float c = Dot(oc, oc) - radius * radius;
		Float discriminant = b * b - 4.0f * a * c;
		// �ж��и���������ȡС�ĸ���Ϊ���е�����Ҫ��ʱ��(���԰�t�����ʱ��)
		Float t0, t1;
		if (!Quadratic(a, b, c, t0, t1)) return false;

		if (t0 > ray.tMax || t1 <= ShadowEpsilon) return false;
		Float tShapeHit = t0 < ShadowEpsilon ? t1 : t0;
		rec.t = tShapeHit;
		rec.p = ray(tShapeHit);
		rec.normal = Normal3f((rec.p - Center(ray.time)) * invRadius);
		rec.mat = material;

		return true;
	}
}
#endif // QZRT_SHAPE_DSPHERE_H