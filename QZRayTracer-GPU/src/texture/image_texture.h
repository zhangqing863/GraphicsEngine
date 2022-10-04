#ifndef QZRT_TEXTURE_IMAGE_H
#define QZRT_TEXTURE_IMAGE_H

#include "../core/texture.h"
#include "../core/QZRayTracer.h"

namespace raytracer {
	/// <summary>
	///  普通颜色
	/// </summary>
	class ImageTexture :public Texture {
	public:
		unsigned char* data;
		int width, height;
		//cudaTextureObject_t texs;
		__device__ ImageTexture() {}
		__device__ ImageTexture(/*cudaTextureObject_t texs*/unsigned char* data, int width, int height) :/*texs(texs)*/data(data), width(width), height(height) {}

		// 通过 Texture 继承
		__device__ virtual Point3f value(float u, float v, const Point3f& p) const override;

	};

	__device__ inline Point3f raytracer::ImageTexture::value(float u, float v, const Point3f& p) const {
		int i = int(u * width);
		int j = int((1 - v) * height - 0.001f);
		//printf("i:%d, j:%d, u:%f, v:%f\n", i, j, u, v);
		if (i > width - 1) {
			i = width - 1;
		}
		else if (i < 0) {
			i = 0;
		}
		if (j > height - 1) {
			j = height - 1;
		}
		else if (j < 0) {
			j = 0;
		}
		Float r =     int(data[3 * i + 3 * width * j]) / 255.0f;
		Float g = int(data[3 * i + 3 * width * j + 1]) / 255.0f;
		Float b = int(data[3 * i + 3 * width * j + 2]) / 255.0f;
#ifdef SRGB2LINEAR
		r = powf(r, invGamma);
		g = powf(g, invGamma);
		b = powf(b, invGamma);
#endif // SRGB2LINEAR

		//Float r = tex2DLayered<Float>(texs, u, 1 - v, 0), //R
		//	Float g = tex2DLayered<float>(texs, rec.u, 1 - rec.v, 1),//G
		//	Float b = tex2DLayered<float>(texs, rec.u, 1 - rec.v, 2));//B
		//printf("i:%d, j:%d, color:%f,%f,%f\n",i, j, r, g, b);
		return Point3f(r, g, b);
	}

}

#endif // QZRT_TEXTURE_IMAGE_H