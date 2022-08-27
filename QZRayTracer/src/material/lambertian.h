#ifndef QZRT_CORE_LAMBERTIAN_H
#define QZRT_CORE_LAMBERTIAN_H

#include "../core/material.h"

namespace raytracer {
	class Lambertian :public Material {
	public:

		Point3f albedo;

		Lambertian(const Point3f& color) :albedo(color) {}


		// Í¨¹ý Material ¼Ì³Ð
		virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo) const override;

	};
}

#endif // QZRT_CORE_LAMBERTIAN_H