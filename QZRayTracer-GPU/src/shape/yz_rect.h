#ifndef QZRT_SHAPE_YZ_RECT_H
#define QZRT_SHAPE_YZ_RECT_H
#include "../core/shape.h"

namespace raytracer {
	class YZRect :public Shape {
	public:
		Float y0, y1, z0, z1, k;
		__device__ YZRect() {}
		__device__ YZRect(Float _y0, Float _y1, Float _z0, Float _z1, Float _k, Material* mat, const  Transform& _trans = Transform()) : k(_k) {
			transform = _trans;
			if (_y0 < _y1) {
				y0 = _y0;
				y1 = _y1;
			}
			else {
				y0 = _y1;
				y1 = _y0;
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
	__device__ inline bool YZRect::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTrans = Inverse(transform);

		Ray tansRay = Ray(invTrans(ray.o), invTrans(Normalize(ray.d)));
		Float t = (k - tansRay.o.x) / tansRay.d.x;

		if (t > tansRay.tMax || t <= ShadowEpsilon) {
			return false;
		}
		Point3f hitP = tansRay(t);

		if (hitP.y < y0 || hitP.y > y1 || hitP.z < z0 || hitP.z > z1) {
			return false;
		}

		rec.u = (hitP.y - y0) / (y1 - y0);
		rec.v = (hitP.z - z0) / (z1 - z0);
		rec.t = t;
		rec.t0 = t;
		rec.t1 = t;
		rec.p = transform(hitP);
		rec.p = transform(Point3f(k, hitP.y, hitP.z));
		Normal3f normal = Normal3f(1, 0, 0);
		if (tansRay.o.x < k) {
			normal = -normal;
		}
		//printf("t:%f,rayo:%f,%f,%f,k:%f,normal:%f,%f,%f\n", t, ray.o.x, ray.o.y, ray.o.z, k, normal.x, normal.y, normal.z);
		rec.normal = Normalize(transform(normal));
		rec.mat = material;
		return true;
	}

	__device__ inline bool YZRect::BoundingBox(Bounds3f& box) const {
		box = transform(Bounds3f(Point3f(k - 0.001f, y0, z0), Point3f(k + 0.001f, y1, z1)));
		return true;
	}
}
#endif // QZRT_SHAPE_YZ_RECT_H