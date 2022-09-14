#ifndef QZRT_SHAPE_FLIP_NORMALS_H
#define QZRT_SHAPE_FLIP_NORMALS_H
#include "../core/shape.h"

namespace raytracer {
	class FlipNormals :public Shape {
	public:
		Shape* ptr;
		__device__ FlipNormals(Shape* p) :ptr(p) {}
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool FlipNormals::Hit(const Ray& ray, HitRecord& rec) const {
		if (ptr->Hit(ray, rec)) {
			rec.normal = -rec.normal;
			return true;
		}
		return false;
	}

	__device__ inline bool FlipNormals::BoundingBox(Bounds3f& box) const {
		return ptr->BoundingBox(box);
	}
}
#endif // QZRT_SHAPE_FLIP_NORMALS_H