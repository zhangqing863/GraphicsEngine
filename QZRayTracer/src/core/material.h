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
		/// ������η�����������Ĺ���
		/// </summary>
		/// <param name="wi">�����</param>
		/// <param name="rec">���е�ļ�¼</param>
		/// <param name="attenuation">˥���̶�</param>
		/// <param name="wo">�����</param>
		/// <returns></returns>
		virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo)const = 0;

		
	};

	/// <summary>
	/// ���淴��
	/// </summary>
	/// <param name="v">��������</param>
	/// <param name="n">��������</param>
	/// <returns>�ԳƵĳ�������</returns>
	inline Vector3f Reflect(const Vector3f& v, const Vector3f& n) { return v - 2 * Dot(v, n) * n; }
}

#endif  // QZRT_CORE_MATERIAL_H