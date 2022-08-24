
#include "src/core/QZRayTracer.h"
#include "src/core/api.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "src/core/stb_image_write.h"
#include "src/core/stb_image.h"
using namespace raytracer;
using namespace std;

// Chapter03 : simple color function
Point3f Color(const Ray& ray) {
	Vector3f dir = Normalize(ray.d);
	Float t = 0.5 * (dir.x + 1.0);
	return Lerp(t, Point3f(1.0, 1.0, 1.0), Point3f(0.5, 0.7, 1.0));
}

int main()
{
    int width = 1920, height = 1080, channel = 3;

    auto* data = (unsigned char*)malloc(width * height * channel);

	// 构建一个简单的相机
	Vector3f lowerLeftCorner(-2.0, -1.0, -1.0);
	Vector3f horizontal(4.0, 0.0, 0.0);
	Vector3f vertical(0.0, 2.0, 0.0);
	Point3f origin;

	for (int sy = height - 1; sy >= 0; sy--)
	{
		for (int sx = 0; sx < width; sx++)
		{
			Float u = Float(sx) / Float(width);
			Float v = Float(sy) / Float(height);
			
			Ray ray(origin, lowerLeftCorner + u * horizontal + v * vertical);
			
			Point3f color = Color(ray);
			
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
	stbi_write_png("output-chapter03-1.png", width, height, channel, data, 0);
	cout << "生成成功！" << endl;
	stbi_image_free(data);
}

