#ifndef QZRT_CORE_SPHERE_H
#define QZRT_CORE_SPHERE_H
#include "../core/QZRayTracer.h"
#include "../core/shape.h"

namespace raytracer {
	class Sphere :public Shape {
	public:
		Point3f center;
		Float radius;
		Float invRadius;
		Sphere() :center(Point3f(0, 0, 0)), radius(1.0), invRadius(1.0) {}
		Sphere(Point3f center, Float radius) :center(center), radius(radius) { invRadius = 1.0 / radius; };
		// Í¨¹ý Shape ¼Ì³Ð
		virtual bool hit(const Ray& ray, HitRecord& rec) const override;
	};

	std::shared_ptr<Shape> CreateSphereShape(Point3f center, Float radius);
}
#endif // QZRT_CORE_SPHERE_H