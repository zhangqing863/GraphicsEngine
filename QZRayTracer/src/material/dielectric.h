#ifndef QZRT_CORE_Dielectric_H
#define QZRT_CORE_Dielectric_H

#include "../core/material.h"
namespace raytracer {
	class Dielectric :public Material {
	public:
		/// <summary>
		/// ’€…‰¬ 
		/// </summary>
		Float refractionIndex;
		Float invRefractionIndex;
		Dielectric() { refractionIndex = invRefractionIndex = 1.0; };
		Dielectric(Float refIdx) :refractionIndex(refIdx) { invRefractionIndex = 1.0 / refractionIndex; };

		// Õ®π˝ Material ºÃ≥–
		virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo) const override;
	};
}


#endif // QZRT_CORE_Dielectric_H