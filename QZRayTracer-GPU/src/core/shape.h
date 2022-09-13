#ifndef QZRT_CORE_SHAPE_H
#define QZRT_CORE_SHAPE_H

#include <memory>
#include "QZRayTracer.h"
#include "geometry.h"


namespace raytracer {

	

	struct HitRecord {
		Float t; // time
		Point3f p; // ���е�
		Normal3f normal; // ����
		Material* mat; // ����
		Float u, v;// u,v ����
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
		// -1(��ʾ��ͨ��Shape��û�����Һ���)
		// 0(��ʾBVHNode�������Һ���Ϊ��)
		// 1(��ʾBVHNode��ֻ������)
		// 2(��ʾBVHNode��ֻ���Һ���)
		int flag = -1;
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec)const = 0;
		__device__ virtual bool BoundingBox(Bounds3f& box)const = 0;
	};

}

#endif // QZRT_CORE_SHAPE_H

