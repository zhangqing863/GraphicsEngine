#ifndef QZRT_CORE_SHAPE_H
#define QZRT_CORE_SHAPE_H

#include <memory>
#include "QZRayTracer.h"
#include "geometry.h"
namespace raytracer {


	struct HitRecord {
		Float t; // time
		Point3f p; // 击中点
		Normal3f normal; // 法线
	};

	class Shape {
	public:
		virtual bool hit(const Ray& ray, HitRecord& rec)const = 0;
	};

}

#endif // QZRT_CORE_SHAPE_H

