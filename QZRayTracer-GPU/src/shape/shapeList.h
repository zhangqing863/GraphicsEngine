#ifndef QZRT_CORE_SHAPELIST_H
#define QZRT_CORE_SHAPELIST_H
#include "../core/shape.h"
namespace raytracer {

	class ShapeList :public Shape {
	public:
		__device__ ShapeList() {}
		__device__ ShapeList(Shape** shapes, int n) { this->shapes = shapes; numShapes = n; }
		
		Shape** shapes;
		//int numShapes;

        // 通过 Shape 继承
        __device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;


        // 通过 Shape 继承
        __device__ virtual bool BoundingBox(Bounds3f& box) const override;
    };

    __device__ inline bool ShapeList::Hit(const Ray& ray, HitRecord& rec) const {
        HitRecord tempRec;
        bool hitAnything = false;
        Float closestSoFar = ray.tMax;
        for (int i = 0; i < numShapes; i++) {
            if (shapes[i]->Hit(ray, tempRec) && tempRec.t < closestSoFar) {
                hitAnything = true;
                closestSoFar = tempRec.t;
                rec = tempRec;
            }
        }
        return hitAnything;
    }

    __device__ inline bool ShapeList::BoundingBox(Bounds3f& box) const {
        return true;
    }
	// Shape *CreateShapeList(Shape** shapes);
}

#endif // QZRT_CORE_SHAPELIST_H