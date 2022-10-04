#ifndef QZRT_SHAPE_DSPHERE_H
#define QZRT_SHAPE_DSPHERE_H
#include "../core/shape.h"

namespace raytracer {
	/// <summary>
	/// 动态的球体
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
		__device__ DSphere(Point3f center0, Point3f center1, Float t0, Float t1, Float radius, Material* mat, const  Transform& _trans = Transform()) : center0(center0), center1(center1), time0(t0), time1(t1), radius(radius), material(mat) {
			transform = _trans;
			invRadius = 1.0f / radius;
			invOverTime = 1.0f / (time1 - time0);
			Bounds3f box0 = Bounds3f(center0 + Vector3f(-radius, -radius, -radius), center0 + Vector3f(radius, radius, radius));
			Bounds3f box1 = Bounds3f(center1 + Vector3f(-radius, -radius, -radius), center1 + Vector3f(radius, radius, radius));
			this->box = transform(Union(box0, box1));

			//printf("box: %f, %f, %f | %f, %f, %f\n", box.pMin.x, box.pMin.y, box.pMin.z, box.pMax.x, box.pMax.y, box.pMax.z);
		};
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;

		__device__ Point3f Center(Float time)const {
			return Lerp((time - time0) * invOverTime, center0, center1);
		}
	};

	

	__device__ inline bool DSphere::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTrans = Inverse(transform);

		Ray tansRay = Ray(invTrans(ray.o), invTrans(Normalize(ray.d)), ray.time);
		//Point3f transCenter = 

		Vector3f oc = tansRay.o - Center(tansRay.time);
		Float a = Dot(tansRay.d, tansRay.d);
		Float b = 2.0f * Dot(oc, tansRay.d);
		Float c = Dot(oc, oc) - radius * radius;
		Float discriminant = b * b - 4.0f * a * c;
		// 判断有根与否并求根，取小的根作为击中点所需要的时间(可以把t抽象成时间)
		Float t0, t1;
		if (!Quadratic(a, b, c, t0, t1)) return false;

		if (t0 > tansRay.tMax || t1 <= ShadowEpsilon) return false;
		//printf("Change Center: %f, %f, %f and time %f \n", Center(tansRay.time).x, Center(tansRay.time).y, Center(tansRay.time).z, tansRay.time);
		Float tShapeHit = t0 < ShadowEpsilon ? t1 : t0;
		rec.t = tShapeHit;
		rec.t0 = t0;
		rec.t1 = t1;
		Point3f hitP = tansRay(tShapeHit);
		rec.p = transform(hitP);
		rec.mat = material;


		Vector3f unit_p = Normalize(hitP - Center(tansRay.time));

		Float phi = atan2f(unit_p.z, unit_p.x);
		Float theta = asinf(unit_p.y);

		rec.u = 1.f - (phi + Pi) * Inv2Pi;
		rec.v = (theta + Pi * 0.5f) * InvPi;
		rec.normal = Normalize(transform(Normal3f((hitP - Center(tansRay.time)) * invRadius)));

		return true;
	}

	__device__ inline bool DSphere::BoundingBox(Bounds3f& box) const {
		box = this->box;
		return true;
	}
}
#endif // QZRT_SHAPE_DSPHERE_H