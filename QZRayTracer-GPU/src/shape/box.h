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
		//printf("Hiting Box......................\n");
		Transform invTransform = Inverse(transform);
		Ray tansRay = Ray(invTransform(ray.o), Normalize(invTransform(ray.d)));



		Float t0 = tansRay.tMin, t1 = tansRay.tMax;
		int axis1 = -1;
		int axis2 = -1;
		for (int i = 0; i < 3; ++i) {
			// Update interval for _i_th bounding box slab
			Float invRayDir = 1.f / tansRay.d[i];
			Float tNear = (box.pMin[i] - tansRay.o[i]) * invRayDir;
			Float tFar = (box.pMax[i] - tansRay.o[i]) * invRayDir;

			// Update parametric interval from slab intersection $t$ values
			// 做这步的原因就是因为光线方向分量为负，导致近的在远平面找到
			// 近远平面的定义主要是按照boundbox的轴，分量值越小，在该分量轴上就定义为近平面
			if (tNear > tFar) {
				Float temp = tNear;
				tNear = tFar;
				tFar = temp;
			}

			// Update _tFar_ to ensure robust ray--bounds intersection
			// tFar *= 1 + 2 * gamma(3);
			// 判断区间是否重叠，没重叠就返回false
			if (tNear > t0) {
				t0 = tNear;
				axis1 = i;
			}
			if (tFar < t1) {
				t1 = tFar;
				axis2 = i;
			}
			if (t0 > t1) return false;
		}

		if (t0 < 0 || t1 > tansRay.tMax)return false;

		Float axis = t0 > tansRay.tMin + ShadowEpsilon ? axis1 : axis2;
		Float tShapeHit = t0 > tansRay.tMin + ShadowEpsilon ? t0 : t1;

		Point3f hitP = tansRay(tShapeHit);
		rec.t = tShapeHit;
		rec.t0 = t0;
		rec.t1 = t1;

		rec.p = Point3f(hitP);
		if (tShapeHit == t0)
			rec.p[axis] = box.pMin[axis];
		else
			rec.p[axis] = box.pMax[axis];
		rec.p = transform(rec.p);
		if (axis == 0) {
			rec.u = (hitP.y - box.pMin.y) / (box.pMax.y - box.pMin.y);
			rec.v = (hitP.z - box.pMin.z) / (box.pMax.z - box.pMin.z);
		}
		else if (axis == 1) {
			rec.u = (hitP.x - box.pMin.x) / (box.pMax.x - box.pMin.x);
			rec.v = (hitP.z - box.pMin.z) / (box.pMax.z - box.pMin.z);
		}
		else {
			rec.u = (hitP.x - box.pMin.x) / (box.pMax.x - box.pMin.x);
			rec.v = (hitP.y - box.pMin.y) / (box.pMax.y - box.pMin.y);
		}
		rec.mat = material;
		hitP = Point3f(tansRay.o);
		hitP[axis] = tansRay(tShapeHit)[axis];
		//if (Normal3f(tansRay.o - hitP).LengthSquared() == 0) {
		//	printf("t0:%f,t1:%f\n", t0, t1);
		//}
		Normal3f normal = Normalize(Normal3f(tansRay.o - hitP));
		//printf("normal:%f, %f, %f\n", normal.x, normal.y, normal.z);
		rec.normal = Normalize(transform(normal));
		//printf("t0 : %f, t1 : %f\n", t0, t1);
		return true;


		/*if (cube->Hit(tansRay, rec)) {
			rec.p = transform(rec.p);
			rec.normal = Normalize(transform(rec.normal));
			return true;
		}
		else {
			return false;
		}*/
		/*return cube->Hit(ray, rec);*/
	}

	__device__ inline bool Box::BoundingBox(Bounds3f& box) const {
		box = transform(this->box);
		return true;
	}
}
#endif // QZRT_SHAPE_BOX_H