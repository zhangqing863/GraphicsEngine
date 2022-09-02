#ifndef QZRT_CORE_METAL_H
#define QZRT_CORE_METAL_H

#include "../core/material.h"
namespace raytracer {
	class Metal : public Material {
	public:
		Point3f albedo;
		Float fuzz; // ģ��ϵ��������ƫ�Ʒ����

		__device__ Metal(const Point3f& color, Float f = 0.0f) :albedo(color), fuzz(f) {}
		// ͨ�� Material �̳�
		__device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;
		
	};

	__device__ inline bool raytracer::Metal::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
		Vector3f reflected = Reflect(Normalize(wi.d), Vector3f(rec.normal));
		wo = Ray(rec.p, reflected + Vector3f(fuzz * RandomInUnitSphere(local_rand_state)));
		attenuation = albedo;
		return Dot(wo.d, rec.normal) > 0.0f; // �������䷽���뷨�߱�����ͬһ��������
	}
}

#endif // QZRT_CORE_METAL_H