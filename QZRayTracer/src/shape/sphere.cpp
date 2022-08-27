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
		if (discriminant > 0) {
			Float invA = 1.0 / (2.0 * a);
			Float temp = (-b - sqrt(discriminant)) * invA;
			if (temp < ShadowEpsilon) {
				temp = (-b + sqrt(discriminant)) * invA;
			}
			if (temp < ray.tMax && temp > ShadowEpsilon) {
				rec.t = temp;
				rec.p = ray(temp);
				rec.normal = Normal3f((rec.p - center) * invRadius);
				rec.mat = material;
				return true;
			}
		}
		return false;
	}
	std::shared_ptr<Shape> CreateSphereShape(Point3f center, Float radius, std::shared_ptr<Material> material)
	{
		return std::make_shared<Sphere>(center, radius, material);
	}
}