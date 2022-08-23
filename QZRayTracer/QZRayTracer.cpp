
#include <iostream>
#include "src/core/QZRayTracer.h"
using namespace raytracer;
using namespace std;

int main()
{
    int width = 1920, height = 1080, channel = 3;

    auto* data = (unsigned char*)malloc(width * height * channel);

	for (int sy = height - 1; sy >= 0; sy--)
	{
		for (int sx = 0; sx < width; sx++)
		{
			Float r = Float(sx) / Float(width);
			Float g = Float(sy) / Float(height);
			Float b = 0.2;
			int ir = int(255.99 * r);
			int ig = int(255.99 * g);
			int ib = int(255.99 * b);
			int shadingPoint = ((height - sy - 1) * width + sx) * 3;
			data[shadingPoint] = ir;
			data[shadingPoint + 1] = ig;
			data[shadingPoint + 2] = ib;
		}
	}
	// 写入图像
	stbi_write_png("output-chapter01.png", width, height, channel, data, 0);

	stbi_image_free(data);
}

