#ifndef QZRT_TEXTURE_CHECKER_H
#define QZRT_TEXTURE_CHECKER_H

#include "../core/texture.h"

namespace raytracer {
	/// <summary>
	///  ��ͨ��ɫ
	/// </summary>
	class CheckerTexture :public Texture {
	public:
		Texture* odd; // �����õ�������
		Texture* even; // ż���õ�������
		Float invScale; // �����С
		__device__ CheckerTexture() {}
		__device__ CheckerTexture(Texture* a, Texture* b, Float scale = 1.0f) :odd(a), even(b) { this->invScale = 1.0f / scale; }

		// ͨ�� Texture �̳�
		__device__ virtual Point3f value(float u, float v, const Point3f& p) const override;

	};

	__device__ inline Point3f raytracer::CheckerTexture::value(float u, float v, const Point3f& p) const {
		Float x = abs(p.x) < ShadowEpsilon ? 1 : sin(10 * p.x * invScale);
		Float y = abs(p.y) < ShadowEpsilon ? 1 : sin(10 * p.y * invScale);
		Float z = abs(p.z) < ShadowEpsilon ? 1 : sin(10 * p.z * invScale);
		Float sines = x * y * z;
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}

}

#endif // QZRT_TEXTURE_CHECKER_H