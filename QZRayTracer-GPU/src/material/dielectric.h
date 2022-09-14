#ifndef QZRT_MATERIAL_Dielectric_H
#define QZRT_MATERIAL_Dielectric_H

#include "../core/material.h"
namespace raytracer {
	class Dielectric :public Material {
	public:
		/// <summary>
		/// ������
		/// </summary>
		Float refractionIndex;
		Float invRefractionIndex;
		__device__ Dielectric() { refractionIndex = invRefractionIndex = 1.0; };
		__device__ Dielectric(Float refIdx) :refractionIndex(refIdx) { invRefractionIndex = 1.0f / refractionIndex; };

		// ͨ�� Material �̳�
		__device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;
	};

    __device__ inline bool raytracer::Dielectric::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
        Vector3f outwardNormal;
        Vector3f originNormal = Vector3f(rec.normal);
        Vector3f reflected = Reflect(wi.d, originNormal);
        Float niOverNo;
        attenuation = Point3f(1.0, 1.0, 1.0);
        Vector3f refracted;

        Float reflectProb;
        Float cosine;


        // ������Ҫ�����жϹ��ߴ�������������л��Ǵӽ������䵽����ȥ����֤ʹ�õ��ǳ���ķ��ߣ�ͬʱ����������
        if (Dot(wi.d, rec.normal) > 0.0f) {
            outwardNormal = -originNormal;
            niOverNo = refractionIndex;
            cosine = refractionIndex * Dot(wi.d, originNormal) / wi.d.Length();
        }
        else {
            outwardNormal = originNormal;
            niOverNo = invRefractionIndex;
            cosine = -Dot(wi.d, originNormal) / wi.d.Length();
        }

        // �������Ƕ�̫С�ᵼ�������ɾ��淴��
        if (Refract(wi.d, outwardNormal, niOverNo, refracted)) {
            wo = Ray(rec.p, refracted);
            reflectProb = Schlick(cosine, refractionIndex);
        }
        else {
            wo = Ray(rec.p, reflected);
            reflectProb = 1.0f;
        }
        if (curand_uniform(local_rand_state) < reflectProb) {
            wo = Ray(rec.p, reflected);
        }
        else {
            wo = Ray(rec.p, refracted);
        }
        return true;
    }
}


#endif // QZRT_MATERIAL_Dielectric_H