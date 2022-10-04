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
		__device__ Sphere(Point3f center, Float radius, Material* mat, const  Transform& _trans = Transform()) :center(center), radius(radius) {
			transform = _trans;
			invRadius = 1.0f / radius;
			material = mat;
		};
		__device__ Sphere(Material* mat, const  Transform& _trans = Transform()) :center(center), radius(radius) {
			center = Point3f(0, 1, 0);
			radius = 1.f;
			transform = _trans;
			invRadius = 1.0f / radius;
			material = mat;
		};
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool Sphere::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTrans = Inverse(transform);

		Ray tansRay = Ray(invTrans(ray.o), invTrans(Normalize(ray.d)));
		Vector3f oc = tansRay.o - center;
		Float a = Dot(tansRay.d, tansRay.d);
		Float b = 2.0f * Dot(oc, tansRay.d);
		Float c = Dot(oc, oc) - radius * radius;
		Float discriminant = b * b - 4.0f * a * c;
		// 判断有根与否并求根，取小的根作为击中点所需要的时间(可以把t抽象成时间)
		Float t0, t1;
		if (!Quadratic(a, b, c, t0, t1)) return false;

		if (t0 > tansRay.tMax || t1 <= ShadowEpsilon) return false;
		Float dir = 1;
		Float tShapeHit = t0;
		if (t0 < ShadowEpsilon) {
			dir = -1;
			tShapeHit = t1;
		}
		rec.t = tShapeHit;
		//printf("t0:%f,t1:%f,t:%f\n", t0, t1, tShapeHit);
		rec.t0 = t0;
		rec.t1 = t1;
		Point3f hitP = tansRay(tShapeHit);


		Vector3f unit_p = Normalize(hitP - center);

		Float phi = atan2f(unit_p.z, unit_p.x);
		Float theta = asinf(unit_p.y);

		rec.u = 1.f - (phi + Pi) * Inv2Pi;
		rec.v = (theta + Pi * 0.5f) * InvPi;

		rec.p = transform(hitP);
		rec.normal = Normalize(transform(Normal3f((hitP - center) * invRadius)) * dir);
		rec.mat = material;


		return true;
	}
	__device__ inline bool Sphere::BoundingBox(Bounds3f& box) const {
		box = transform(Bounds3f(center + Vector3f(-radius, -radius, -radius), center + Vector3f(radius, radius, radius)));
		return true;
	}
	// Shape* CreateSphereShape(Point3f center, Float radius, Material* material);
}
#endif // QZRT_SHAPE_SPHERE_H