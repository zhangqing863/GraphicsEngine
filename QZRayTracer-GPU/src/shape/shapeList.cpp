#include "shapeList.h"

namespace raytracer {
    
    /*__device__ bool raytracer::ShapeList::Hit(const Ray& ray, HitRecord& rec) const {
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
    }*/
    /*Shape *CreateShapeList(Shape** shapes) {
        return new ShapeList(shapes);
    }*/
}

