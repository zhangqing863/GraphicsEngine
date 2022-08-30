#include "sphere.h"
namespace raytracer {
	bool Sphere::Hit(const Ray& ray, HitRecord& rec) const
	{
		Vector3f oc = ray.o - center;
		Float a = Dot(ray.d, ray.d);
		Float b = 2.0 * Dot(oc, ray.d);
		Float c = Dot(oc, oc) - radius * radius;
		Float discriminant = b * b - 4 * a * c;
		// 判断有根与否并求根，取小的根作为击中点所需要的时间(可以把t抽象成时间)
		Float t0, t1;
		if (!Quadratic(a, b, c, t0, t1)) return false;
		
		if (t0 > ray.tMax || t1 <= ShadowEpsilon) return false;
		Float tShapeHit = t0 < ShadowEpsilon ? t1 : t0;
		rec.t = tShapeHit;
		rec.p = ray(tShapeHit);
		rec.normal = Normal3f((rec.p - center) * invRadius);
		rec.mat = material;

		return true;
	}
	std::shared_ptr<Shape> CreateSphereShape(Point3f center, Float radius, std::shared_ptr<Material> material)
	{
		return std::make_shared<Sphere>(center, radius, material);
	}
}