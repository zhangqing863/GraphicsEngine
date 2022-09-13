#ifndef QZRT_SHAPE_SPHERE_H
#define QZRT_SHAPE_SPHERE_H
#include "../core/shape.h"

namespace raytracer {
	class Sphere :public Shape {
	public:
		Point3f center;
		Float radius;
		Float invRadius;
		//Material* material;
		__device__ Sphere() :center(Point3f(0, 0, 0)), radius(1.0f), invRadius(1.0f) { material = nullptr; }
		__device__ Sphere(Point3f center, Float radius) : center(center), radius(radius), invRadius(1.0f) { material = nullptr; }
		__device__ Sphere(Point3f center, Float radius, Material* mat) :center(center), radius(radius) {
			invRadius = 1.0f / radius;
			material = mat;
		};
		// ͨ�� Shape �̳�
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// ͨ�� Shape �̳�
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool Sphere::Hit(const Ray& ray, HitRecord& rec) const {
		Vector3f oc = ray.o - center;
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
		rec.normal = Normal3f((rec.p - center) * invRadius);
		rec.mat = material;

		Vector3f unit_p = Normalize(rec.p - center);

		Float phi = atan2f(unit_p.z, unit_p.x);
		Float theta = asinf(unit_p.y);

		rec.u = 1.f - (phi + Pi) * Inv2Pi;
		rec.v = (theta + Pi * 0.5f) * InvPi;

		return true;
	}
	__device__ inline bool Sphere::BoundingBox(Bounds3f& box) const {
		box = Bounds3f(center + Vector3f(-radius, -radius, -radius), center + Vector3f(radius, radius, radius));
		return true;
	}
	// Shape* CreateSphereShape(Point3f center, Float radius, Material* material);
}
#endif // QZRT_SHAPE_SPHERE_H