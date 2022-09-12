#ifndef QZRT_CORE_LAMBERTIAN_H
#define QZRT_CORE_LAMBERTIAN_H

#include "../core/material.h"

namespace raytracer {
	class Lambertian :public Material {
	public:
		// Point3f albedo;

		//__device__ Lambertian(const Point3f& color) :albedo(color) {}

		__device__ Lambertian(Texture* color) { albedo = color; }

		// Í¨¹ý Material ¼Ì³Ð
		__device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;

	};

	__device__ inline bool Lambertian::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
		Point3f target = rec.p + Point3f(rec.normal) + RandomInUnitSphere(local_rand_state);
		wo = Ray(rec.p, target - rec.p);
		attenuation = albedo->value(0, 0, rec.p);
		return true;
	}
}

#endif // QZRT_CORE_LAMBERTIAN_H