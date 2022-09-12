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
		__device__ CheckerTexture() {}
		__device__ CheckerTexture(Texture* a, Texture* b) :odd(a), even(b) {}

		// ͨ�� Texture �̳�
		__device__ virtual Point3f value(float u, float v, const Point3f& p) const override;

	};

	__device__ inline Point3f raytracer::CheckerTexture::value(float u, float v, const Point3f& p) const {
		Float sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}

}

#endif // QZRT_TEXTURE_CHECKER_H