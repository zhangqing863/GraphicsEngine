#ifndef QZRT_SHAPE_SPHERE_H
#define QZRT_SHAPE_SPHERE_H
#include "../core/shape.h"

namespace raytracer {
	class Sphere :public Shape {
	public:
		Point3f center;
		Float radius;
		Float invRadius;
		std::shared_ptr<Material> material;

		Sphere() :center(Point3f(0, 0, 0)), radius(1.0), invRadius(1.0),material(nullptr) { }
		Sphere(Point3f center, Float radius, std::shared_ptr<Material> mat) :center(center), radius(radius), material(mat) {
			invRadius = 1.0 / radius;
		};
		// Í¨¹ý Shape ¼Ì³Ð
		virtual bool Hit(const Ray& ray, HitRecord& rec) const override;
	};

	std::shared_ptr<Shape> CreateSphereShape(Point3f center, Float radius, std::shared_ptr<Material> material);
}
#endif // QZRT_SHAPE_SPHERE_H