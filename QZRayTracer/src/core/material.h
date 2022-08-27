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

	/// <summary>
	/// 折射
	/// </summary>
	/// <param name="v">入射向量</param>
	/// <param name="n">表面法线</param>
	/// <param name="niOverNo">折射率的比值</param>
	/// <param name="refracted">折射方向</param>
	/// <returns></returns>
	inline bool Refract(const Vector3f& v, const Vector3f& n, Float niOverNo, Vector3f& refracted){
		Vector3f uv = Normalize(v);
		// cos(theta) < 0，因为没有点乘 -n，但是并不影响，只有下式中 (uv - n * dt) 本来推导式应该是 (uv + n * dt) 
		Float dt = Dot(uv, n); 
		// 这里主要是判断能不能折射出来
		Float discriminant = 1.0 - niOverNo * niOverNo * (1 - dt * dt);
		if (discriminant > 0) {
			// 这里应该是（uv - n * dt）
			// 错误：（这里的 v 没有归一化）refracted = niOverNo * (v - n * dt) - n * sqrt(discriminant);
			refracted = niOverNo * (uv - n * dt) - n * sqrt(discriminant);
			return true;
		}
		return false;
	}

	/// <summary>
	/// 多项式逼近的玻璃折射率
	/// </summary>
	/// <param name="cosine">入射角度的余弦值</param>
	/// <param name="refIdx">折射率</param>
	/// <returns></returns>
	inline Float Schlick(Float cosine, Float refIdx) {
		Float r0 = (1 - refIdx) / (1 + refIdx);
		r0 *= r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
}

#endif  // QZRT_CORE_MATERIAL_H