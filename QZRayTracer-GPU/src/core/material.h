#ifndef QZRT_CORE_MATERIAL_H
#define QZRT_CORE_MATERIAL_H
#include "QZRayTracer.h"
#include "geometry.h"
#include "shape.h"
#include "texture.h"
namespace raytracer {
	class Material {
	public:

		Texture* albedo;

		/// <summary>
		/// ������η�����������Ĺ���
		/// </summary>
		/// <param name="wi">�����</param>
		/// <param name="rec">���е�ļ�¼</param>
		/// <param name="attenuation">˥���̶�</param>
		/// <param name="wo">�����</param>
		/// <returns></returns>
		__device__ virtual bool Scatter(const Ray& wi, const HitRecord& rec, Point3f& attenuation, Ray& wo, curandState* local_rand_state)const = 0;

		
	};

	/// <summary>
	/// ���淴��
	/// </summary>
	/// <param name="v">��������</param>
	/// <param name="n">��������</param>
	/// <returns>�ԳƵĳ�������</returns>
	__device__ inline Vector3f Reflect(const Vector3f& v, const Vector3f& n) { return v - 2.0f * Dot(v, n) * n; }

	/// <summary>
	/// ����
	/// </summary>
	/// <param name="v">��������</param>
	/// <param name="n">���淨��</param>
	/// <param name="niOverNo">�����ʵı�ֵ</param>
	/// <param name="refracted">���䷽��</param>
	/// <returns></returns>
	__device__ inline bool Refract(const Vector3f& v, const Vector3f& n, Float niOverNo, Vector3f& refracted){
		Vector3f uv = Normalize(v);
		// cos(theta) < 0����Ϊû�е�� -n�����ǲ���Ӱ�죬ֻ����ʽ�� (uv - n * dt) �����Ƶ�ʽӦ���� (uv + n * dt) 
		Float dt = Dot(uv, n); 
		// ������Ҫ���ж��ܲ����������
		Float discriminant = 1.0f - niOverNo * niOverNo * (1.0f - dt * dt);
		if (discriminant > 0.0f) {
			// ����Ӧ���ǣ�uv - n * dt��
			// ���󣺣������ v û�й�һ����refracted = niOverNo * (v - n * dt) - n * sqrt(discriminant);
			refracted = niOverNo * (uv - n * dt) - n * sqrt(discriminant);
			return true;
		}
		return false;
	}

	/// <summary>
	/// ����ʽ�ƽ��Ĳ���������
	/// </summary>
	/// <param name="cosine">����Ƕȵ�����ֵ</param>
	/// <param name="refIdx">������</param>
	/// <returns></returns>
	__device__ inline Float Schlick(Float cosine, Float refIdx) {
		Float r0 = (1.0f - refIdx) / (1.0f + refIdx);
		r0 *= r0;
		return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
	}
}

#endif  // QZRT_CORE_MATERIAL_H