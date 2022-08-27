#include "metal.h"
namespace raytracer {
    bool raytracer::Metal::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo) const {
        Vector3f reflected = Reflect(Normalize(wi.d), Vector3f(rec.normal));
        wo = Ray(rec.p, reflected + Vector3f(fuzz * RandomInUnitSphere()));
        attenuation = albedo;
        return Dot(reflected, rec.normal) > 0; // 表明出射方向与法线必须在同一个半球内
    }
}