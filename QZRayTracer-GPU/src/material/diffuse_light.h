#ifndef QZRT_MATERIAL_DIFFUSE_LIGHT_H
#define QZRT_MATERIAL_DIFFUSE_LIGHT_H

#include "../core/material.h"
namespace raytracer {
    class DiffuseLight :public Material {
    public:
        Texture* emit;
        __device__ DiffuseLight(Texture* a) :emit(a) {};

        // Í¨¹ý Material ¼Ì³Ð
        __device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const override;
        
        __device__ inline Point3f raytracer::DiffuseLight::Emitted(Float u, Float v, const Point3f& p) const override;
    };

    __device__ inline bool raytracer::DiffuseLight::Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state) const {
        return false;
    }

    __device__ inline Point3f raytracer::DiffuseLight::Emitted(Float u, Float v, const Point3f& p) const {
        return emit->value(u, v, p);
    }
}


#endif // QZRT_MATERIAL_DIFFUSE_LIGHT_H