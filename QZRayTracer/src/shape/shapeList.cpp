#include "shapeList.h"

namespace raytracer {
    bool raytracer::ShapeList::hit(const Ray& ray, HitRecord& rec) const {
        HitRecord tempRec;
        bool hitAnything = false;
        Float closestSoFar = ray.tMax;
        for (int i = 0; i < shapes.size(); i++) {
            if (shapes[i]->hit(ray, tempRec) && tempRec.t < closestSoFar) {
                hitAnything = true;
                closestSoFar = tempRec.t;
                rec = tempRec;
            }
        }
        return hitAnything;
    }
    std::shared_ptr<Shape> CreateShapeList(std::vector<std::shared_ptr<Shape>> shapes) {
        return std::make_shared<ShapeList>(shapes);
    }
}

