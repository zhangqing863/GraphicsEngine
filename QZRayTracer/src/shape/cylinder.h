#ifndef QZRT_SHAPE_CYLINDER_H
#define QZRT_SHAPE_CYLINDER_H
#include "../core/shape.h"

namespace raytracer {
	class Cylinder :public Shape {
	public:
		Point3f center;
		Float radius;
		Float zMin, zMax;
		Float invRadius;
		std::shared_ptr<Material> material;

		Cylinder() :center(Point3f(0, 0, 0)), radius(1.0), invRadius(1.0), zMin(0), zMax(0), material(nullptr) {}
		Cylinder(Point3f center, Float radius, Float zMin, Float zMax, std::shared_ptr<Material> mat)
			:center(center), zMin(std::min(zMin, zMax)), zMax(std::max(zMin, zMax)), radius(radius), material(mat) {
			Float halfHeight = (zMax - zMin) * .5f;
			this->zMin = center.y - halfHeight;
			this->zMax = center.y + halfHeight;
			invRadius = 1.0 / radius;
		};
		// Í¨¹ý Shape ¼Ì³Ð
		virtual bool Hit(const Ray& ray, HitRecord& rec) const override;
	};

	std::shared_ptr<Shape> CreateCylinderShape(Point3f center, Float radius, Float zMin, Float zMax, std::shared_ptr<Material> material);
}
#endif // QZRT_SHAPE_CYLINDER_H
