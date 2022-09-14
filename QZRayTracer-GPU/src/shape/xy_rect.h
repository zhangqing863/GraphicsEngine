#ifndef QZRT_SHAPE_XY_RECT_H
#define QZRT_SHAPE_XY_RECT_H
#include "../core/shape.h"

namespace raytracer {
	class XYRect :public Shape {
	public:
		Float x0, x1, y0, y1, k;
		__device__ XYRect() { }
		__device__ XYRect(Float _x0, Float _x1, Float _y0, Float _y1, Float _k, Material* mat) : k(_k) {
			if (_x0 < _x1) {
				x0 = _x0;
				x1 = _x1;
			}
			else {
				x0 = _x1;
				x1 = _x0;
			}
			if (_y0 < _y1) {
				y0 = _y0;
				y1 = _y1;
			}
			else {
				y0 = _y1;
				y1 = _y0;
			}
			material = mat;
		}
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool XYRect::Hit(const Ray& ray, HitRecord& rec) const {
		Float t = (k - ray.o.z) / ray.d.z;

		if (t > ray.tMax || t <= ShadowEpsilon) {
			return false;
		}
		Point3f hitP = ray(t);
		if (hitP.x < x0 || hitP.x > x1 || hitP.y < y0 || hitP.y > y1) {
			return false;
		}

		//printf("Hit XYRECT\n");
		rec.u = (hitP.x - x0) / (x1 - x0);
		rec.v = (hitP.y - y0) / (y1 - y0);
		rec.t = t;
		rec.p = hitP;
		Normal3f normal = Normal3f(0, 0, 1);
		if (ray.o.z < k) {
			normal = -normal;
		}
		rec.normal = normal;
		rec.mat = material;
		return true;
	}

	__device__ inline bool XYRect::BoundingBox(Bounds3f& box) const {
		box = Bounds3f(Point3f(x0, y0, k - 0.001f), Point3f(x1, y1, k + 0.001f));
		return true;
	}
}
#endif // QZRT_SHAPE_XY_RECT_H