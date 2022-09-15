#ifndef QZRT_SHAPE_BOX_H
#define QZRT_SHAPE_BOX_H
#include "../core/shape.h"
#include "shapeList.h"
#include "xy_rect.h"
#include "xz_rect.h"
#include "yz_rect.h"

namespace raytracer {
	class Box :public Shape {
	public:

		// Bounds3 Public Data
		Shape* cube = nullptr;
		__device__ Box() {}
		__device__ Box(const Point3f& p0, const Point3f& p1, Material* _mat, const  Transform& _trans = Transform()) {
			transform = _trans;
			box = Bounds3f(p0, p1);

			/*printf("p0:[%f,%f,%f],p1:[%f,%f,%f] \n After tranform:[%f,%f,%f], [%f,%f,%f]\n",
				p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, box.pMin.x, box.pMin.y, box.pMin.z, box.pMax.x, box.pMax.y, box.pMax.z);*/
			material = _mat;
			Shape** rects = new Shape * [6];
			rects[0] = new XYRect(box.pMin.x, box.pMax.x, box.pMin.y, box.pMax.y, box.pMax.z, material);
			rects[1] = new XYRect(box.pMin.x, box.pMax.x, box.pMin.y, box.pMax.y, box.pMin.z, material);
			rects[2] = new YZRect(box.pMin.y, box.pMax.y, box.pMin.z, box.pMax.z, box.pMax.x, material);
			rects[3] = new YZRect(box.pMin.y, box.pMax.y, box.pMin.z, box.pMax.z, box.pMin.x, material);
			rects[4] = new XZRect(box.pMin.x, box.pMax.x, box.pMin.z, box.pMax.z, box.pMax.y, material);
			rects[5] = new XZRect(box.pMin.x, box.pMax.x, box.pMin.z, box.pMax.z, box.pMin.y, material);
			cube = new ShapeList(rects, 6);
		}
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;


	};
	__device__ inline bool Box::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTransform = Inverse(transform);
		Ray tansRay = Ray(invTransform(ray.o), invTransform(ray.d));
		if (cube->Hit(tansRay, rec)) {
			rec.p = transform(rec.p);
			rec.normal = Normalize(transform(rec.normal));
			return true;
		}
		else {
			return false;
		}
		/*return cube->Hit(ray, rec);*/
	}

	__device__ inline bool Box::BoundingBox(Bounds3f& box) const {
		box = transform(this->box);
		return true;
	}
}
#endif // QZRT_SHAPE_BOX_H