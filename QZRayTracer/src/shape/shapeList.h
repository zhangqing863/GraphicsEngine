#ifndef QZRT_CORE_SHAPELIST_H
#define QZRT_CORE_SHAPELIST_H
#include "../core/QZRayTracer.h"
#include "../core/shape.h"
namespace raytracer {

	class ShapeList :public Shape {
	public:
		ShapeList() {}
		ShapeList(std::vector<std::shared_ptr<Shape>> shapes) :shapes(shapes) {}
		// Í¨¹ý Shape ¼Ì³Ð
		virtual bool hit(const Ray& ray, HitRecord& rec) const override;
		std::vector<std::shared_ptr<Shape>> shapes;
	};

	std::shared_ptr<Shape> CreateShapeList(std::vector<std::shared_ptr<Shape>> shapes);
}

#endif // QZRT_CORE_SHAPELIST_H