#ifndef QZRT_SHAPE_SPHERE_H
#define QZRT_SHAPE_SPHERE_H
#include "../core/shape.h"

namespace raytracer {
	class Sphere :public Shape {
	public:
		Point3f center;
		Float radius;
		Float invRadius;
		Material* material;
		__device__ Sphere() :center(Point3f(0, 0, 0)), radius(1.0f), invRadius(1.0f) { material = nullptr; }
		__device__ Sphere(Point3f center, Float radius) : center(center), radius(radius), invRadius(1.0f) { material = nullptr; }
		__device__ Sphere(Point3f center, Float radius, Material* mat) :center(center), radius(radius), material(mat) {
			invRadius = 1.0f / radius;
		};
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;
	};
	__device__ inline bool Sphere::Hit(const Ray& ray, HitRecord& rec) const {
		Vector3f oc = ray.o - center;
		Float a = Dot(ray.d, ray.d);
		Float b = 2.0f * Dot(oc, ray.d);
		Float c = Dot(oc, oc) - radius * radius;
		Float discriminant = b * b - 4.0f * a * c;
		// 判断有根与否并求根，取小的根作为击中点所需要的时间(可以把t抽象成时间)
		Float t0, t1;
		if (!Quadratic(a, b, c, t0, t1)) return false;

		if (t0 > ray.tMax || t1 <= ShadowEpsilon) return false;
		Float tShapeHit = t0 < ShadowEpsilon ? t1 : t0;
		rec.t = tShapeHit;
		rec.p = ray(tShapeHit);
		rec.normal = Normal3f((rec.p - center) * invRadius);
		rec.mat = material;

		return true;
	}
	// Shape* CreateSphereShape(Point3f center, Float radius, Material* material);
}
#endif // QZRT_SHAPE_SPHERE_H