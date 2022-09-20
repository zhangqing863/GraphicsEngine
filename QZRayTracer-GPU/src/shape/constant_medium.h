#ifndef QZRT_SHAPE_CONSTANT_MEDIUM_H
#define QZRT_SHAPE_CONSTANT_MEDIUM_H
#include "../core/shape.h"

namespace raytracer {
	class ConstantMedium :public Shape {
	public:
		Float density;
		Float invDensity;
		Shape* boundary;
		curandState* rand_state;
		__device__ ConstantMedium() {}
		__device__ ConstantMedium(Shape* shape, Float dens, Material* a, curandState* rand_state, const  Transform& _trans = Transform()) : density(dens), boundary(shape) {
			transform = _trans;
			material = a;
			invDensity = 1.f / density;
			this->rand_state = rand_state;
		}
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool ConstantMedium::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTrans = Inverse(transform);
		Ray tansRay = Ray(invTrans(ray.o), Normalize(invTrans(ray.d)), ray.tMax, ray.tMin);

		HitRecord rec1, rec2;
		if (boundary->Hit(tansRay, rec)) {
			Float t0 = rec.t0;
			Float t1 = rec.t1;
			if (t0 > t1)return false;
			if (t0 < 0)t0 = 0;
			//printf("t0:%f, t1.t%f\n", t0, t1);
			Float distance_inside_boundary = (t1 - t0) * tansRay.d.Length();
			Float hit_distance = -invDensity * logf(curand_uniform(rand_state));
			if (hit_distance < distance_inside_boundary) {
				rec.t = t0 + hit_distance / tansRay.d.Length();
				rec.p = transform(tansRay(rec.t));
				rec.normal = transform(Normal3f(1, 0, 0));
				rec.mat = material;
				rec.u = 0;
				rec.v = 0;
				return true;
			}
		}
		return false;
	}

	__device__ inline bool ConstantMedium::BoundingBox(Bounds3f& box) const {
		return boundary->BoundingBox(box);
	}
}
#endif // QZRT_SHAPE_CONSTANT_MEDIUM_H