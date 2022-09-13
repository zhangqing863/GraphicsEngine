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
		Material* mat; // 材质
		Float u, v;// u,v 坐标
	};

	class Shape {
	public:
		Material* material = nullptr;
		int left = -1;
		int right = -1;
		Bounds3f box;
		int numShapes = 0;
		int numNodes = 0;

		// flag={-1,0,1,2}; 
		// -1(表示普通的Shape，没有左右孩子)
		// 0(表示BVHNode，且左右孩子为空)
		// 1(表示BVHNode，只有左孩子)
		// 2(表示BVHNode，只有右孩子)
		int flag = -1;
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec)const = 0;
		__device__ virtual bool BoundingBox(Bounds3f& box)const = 0;
	};

}

#endif // QZRT_CORE_SHAPE_H

