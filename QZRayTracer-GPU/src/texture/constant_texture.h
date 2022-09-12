#ifndef QZRT_TEXTURE_CONSTANT_H
#define QZRT_TEXTURE_CONSTANT_H

#include "../core/texture.h"

namespace raytracer {
	/// <summary>
	///  ��ͨ��ɫ
	/// </summary>
	class ConstantTexture:public Texture {
	public:
		Point3f color;
		__device__ ConstantTexture() {}
		__device__ ConstantTexture(Point3f c) :color(c) {}

		// ͨ�� Texture �̳�
		__device__ virtual Point3f value(float u, float v, const Point3f& p) const override;

	};

	__device__ inline Point3f raytracer::ConstantTexture::value(float u, float v, const Point3f& p) const {
		return color;
	}

}

#endif // QZRT_TEXTURE_CONSTANT_H