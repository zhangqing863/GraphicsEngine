#ifndef QZRT_MATERIAL_ISOTROPIC_H
#define QZRT_MATERIAL_ISOTROPIC_H

#include "../core/material.h"
namespace raytracer {
    class Isotropic :public Material {
    public:
        __device__ Isotropic(Texture* tex) { albedo = tex; }

        // Í¨¹ý Material ¼Ì³Ð
        __device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;
    };

    __device__ inline bool raytracer::Isotropic::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
        wo = Ray(rec.p, Normalize(Vector3f(RandomInUnitSphere(local_rand_state))), wi.time, wi.tMax, wi.tMin);
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }
}


#endif // QZRT_MATERIAL_ISOTROPIC_H