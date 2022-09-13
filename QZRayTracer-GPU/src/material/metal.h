#ifndef QZRT_CORE_METAL_H
#define QZRT_CORE_METAL_H

#include "../core/material.h"
namespace raytracer {
	class Metal : public Material {
	public:
		Float fuzz; // 模糊系数，用来偏移反射光

		__device__ Metal(Texture* color, Float f = 0.0f) :fuzz(f) { albedo = color; }
		// 通过 Material 继承
		__device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;
		
	};

	__device__ inline bool raytracer::Metal::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
		Vector3f reflected = Reflect(Normalize(wi.d), Vector3f(rec.normal));
		wo = Ray(rec.p, reflected + Vector3f(fuzz * RandomInUnitSphere(local_rand_state)));
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return Dot(wo.d, rec.normal) > 0.0f; // 表明出射方向与法线必须在同一个半球内
	}
}

#endif // QZRT_CORE_METAL_H
