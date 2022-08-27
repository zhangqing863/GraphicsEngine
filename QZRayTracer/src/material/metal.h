#ifndef QZRT_CORE_METAL_H
#define QZRT_CORE_METAL_H

#include "../core/material.h"
namespace raytracer {
	class Metal : public Material {
	public:
		Point3f albedo;
		Float fuzz; // 模糊系数，用来偏移反射光

		Metal(const Point3f& color, Float f = 0.0) :albedo(color), fuzz(f) {}
		// 通过 Material 继承
		virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo) const override;
		
	};
}

#endif // QZRT_CORE_METAL_H
