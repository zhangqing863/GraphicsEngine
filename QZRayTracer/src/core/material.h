#ifndef QZRT_CORE_MATERIAL_H
#define QZRT_CORE_MATERIAL_H
#include "QZRayTracer.h"
#include "geometry.h"
#include "shape.h"
namespace raytracer {
	class Material {
	public:
		Material(){}
		/// <summary>
		/// 材质如何反射入射而来的光线
		/// </summary>
		/// <param name="wi">入射光</param>
		/// <param name="rec">击中点的记录</param>
		/// <param name="attenuation">衰减程度</param>
		/// <param name="wo">出射光</param>
		/// <returns></returns>
		virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo)const = 0;

		
	};

	/// <summary>
	/// 镜面反射
	/// </summary>
	/// <param name="v">入射向量</param>
	/// <param name="n">法线向量</param>
	/// <returns>对称的出射向量</returns>
	inline Vector3f Reflect(const Vector3f& v, const Vector3f& n) { return v - 2 * Dot(v, n) * n; }
}

#endif  // QZRT_CORE_MATERIAL_H