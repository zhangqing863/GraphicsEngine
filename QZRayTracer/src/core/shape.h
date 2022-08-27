#ifndef QZRT_CORE_SHAPE_H
#define QZRT_CORE_SHAPE_H

#include <memory>
#include "QZRayTracer.h"
#include "geometry.h"
namespace raytracer {
	class Material;
	class Lambertian;
	struct HitRecord {
		Float t; // time
		Point3f p; // ���е�
		Normal3f normal; // ����
		std::shared_ptr<Material> mat; // ����
	};

	class Shape {
	public:
		virtual bool Hit(const Ray& ray, HitRecord& rec)const = 0;
	};

}

#endif // QZRT_CORE_SHAPE_H

