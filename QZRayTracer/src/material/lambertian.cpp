#include "lambertian.h"

namespace raytracer {
	bool Lambertian::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo) const {
		Point3f target = rec.p + Point3f(rec.normal) + RandomInUnitSphere();
		wo = Ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}
}