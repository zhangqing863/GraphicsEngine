
#include "src/core/QZRayTracer.h"
#include "src/core/api.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "src/core/stb_image_write.h"
#include "src/core/stb_image.h"
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
			Float b = 1.0;
			
			Vector3f color(r, g, b);
			
			int ir = int(255.99 * color[0]);
			int ig = int(255.99 * color[1]);
			int ib = int(255.99 * color[2]);
			
			int shadingPoint = ((height - sy - 1) * width + sx) * 3;
			data[shadingPoint] = ir;
			data[shadingPoint + 1] = ig;
			data[shadingPoint + 2] = ib;
		}
	}
	// 写入图像
	stbi_write_png("output-chapter02.png", width, height, channel, data, 0);
	cout << "生成成功！" << endl;
	stbi_image_free(data);
}

