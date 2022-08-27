#ifndef QZRT_CORE_METAL_H
#define QZRT_CORE_METAL_H

#include "../core/material.h"
namespace raytracer {
	class Metal : public Material {
	public:
		Point3f albedo;
		Float fuzz; // ģ��ϵ��������ƫ�Ʒ����

		Metal(const Point3f& color, Float f = 0.0) :albedo(color), fuzz(f) {}
		// ͨ�� Material �̳�
		virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo) const override;
		
	};
}

#endif // QZRT_CORE_METAL_H
