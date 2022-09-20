#ifndef QZRT_MATERIAL_LAMBERTIAN_H
#define QZRT_MATERIAL_LAMBERTIAN_H

#include "../core/material.h"

namespace raytracer {
	class Lambertian :public Material {
	public:
		// Point3f albedo;

		//__device__ Lambertian(const Point3f& color) :albedo(color) {}

		__device__ Lambertian(Texture* color) { albedo = color; }

		// 通过 Material 继承
		__device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;

	};

	__device__ inline bool Lambertian::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
		// 生成随机点
		Point3f randomPosInHemisphere = RandomInUnitSphere(local_rand_state);

		//// 如果法线和随机生成的点没在同一面，就应该去负的随机值
		//if (Dot(rec.normal, Vector3f(randomPosInHemisphere)) < 0) {
		//	randomPosInHemisphere = -randomPosInHemisphere;
		//}
		Point3f target = rec.p + Point3f(rec.normal) + randomPosInHemisphere;
		wo = Ray(rec.p, Normalize(target - rec.p), wi.time, wi.tMax, wi.tMin);
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}
}

#endif // QZRT_MATERIAL_LAMBERTIAN_H