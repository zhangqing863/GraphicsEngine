#ifndef QZRT_SHAPE_XZ_RECT_H
#define QZRT_SHAPE_XZ_RECT_H
#include "../core/shape.h"

namespace raytracer {
	class XZRect :public Shape {
	public:
		Float x0, x1, z0, z1, k;
		__device__ XZRect() {}
		__device__ XZRect(Float _x0, Float _x1, Float _z0, Float _z1, Float _k, Material* mat, const  Transform& _trans = Transform()) : k(_k) {
			transform = _trans;
			if (_x0 < _x1) {
				x0 = _x0;
				x1 = _x1;
			}
			else {
				x0 = _x1;
				x1 = _x0;
			}
			if (_z0 < _z1) {
				z0 = _z0;
				z1 = _z1;
			}
			else {
				z0 = _z1;
				z1 = _z0;
			}
			material = mat;
		}
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool XZRect::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTrans = Inverse(transform);

		Ray tansRay = Ray(invTrans(ray.o), Normalize(invTrans(ray.d)));
		Float t = (k - tansRay.o.y) / tansRay.d.y;

		if (t > tansRay.tMax || t <= ShadowEpsilon) {
			return false;
		}
		Point3f hitP = tansRay(t);

		if (hitP.x < x0 || hitP.x > x1 || hitP.z < z0 || hitP.z > z1) {
			return false;
		}

		rec.u = (hitP.x - x0) / (x1 - x0);
		rec.v = (hitP.z - z0) / (z1 - z0);
		rec.t = t;
		rec.p = transform(hitP);
		Normal3f normal = Normal3f(0, 1, 0);
		if (ray.o.y < k) {
			normal = -normal;
			//printf("Flip Normal\n");
		}
		rec.normal = Normalize(transform(normal));
		rec.mat = material;
		return true;
	}

	__device__ inline bool XZRect::BoundingBox(Bounds3f& box) const {
		box = transform(Bounds3f(Point3f(x0, k - 0.001f, z0), Point3f(x1, k + 0.001f, z1)));
		return true;
	}
}
#endif // QZRT_SHAPE_XZ_RECT_H