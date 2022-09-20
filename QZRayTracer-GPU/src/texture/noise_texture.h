#ifndef QZRT_TEXTURE_NOISE_H
#define QZRT_TEXTURE_NOISE_H

#include "../core/texture.h"

namespace raytracer {
	/// <summary>
	///  普通颜色
	/// </summary>
	class NoiseTexture :public Texture {
	public:
		Perlin* noise;
		Float scale;
		__device__ NoiseTexture() {}
		__device__ NoiseTexture(Perlin* noise_rand, Float scale) :noise(noise_rand), scale(scale) {}

		// 通过 Texture 继承
		__device__ virtual Point3f value(float u, float v, const Point3f& p) const override;

	};

	__device__ inline Point3f raytracer::NoiseTexture::value(float u, float v, const Point3f& p) const {
		//return Point3f(1, 1, 1) * noise->Noise(p * scale);
		//return Point3f(1, 1, 1) * noise->Turb(p * scale);
		//return Point3f(1, 1, 1) * 0.5f * (1 + noise->Turb(p * scale));
		return Point3f(1, 1, 1) * 0.5f * (1 + sin(noise->Turb(scale * p) * 10 + scale * p.z));
	}

}

#endif // QZRT_TEXTURE_NOISE_H