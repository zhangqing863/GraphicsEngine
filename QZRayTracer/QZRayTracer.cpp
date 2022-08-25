
#include "src/core/QZRayTracer.h"
#include "src/core/api.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "src/core/stb_image_write.h"
#include "src/core/stb_image.h"
using namespace raytracer;
using namespace std;


// Chapter03-04 : simple color function
Point3f Color(const Ray& ray, shared_ptr<Shape> world) {
	HitRecord rec;
	
	if (world->hit(ray, rec)) {
		return 0.5 * Point3f(rec.normal.x + 1.0, rec.normal.y + 1.0, rec.normal.z + 1.0);
	}
	// 没击中就画个背景
	Vector3f dir = Normalize(ray.d);
	Float t = 0.5 * (dir.y + 1.0);
	return Lerp(t, Point3f(1.0, 1.0, 1.0), Point3f(0.5, 0.7, 1.0));
}


int main()
{
    int width = 2000, height = 1000, channel = 3;

    auto* data = (unsigned char*)malloc(width * height * channel);

	// 构建一个简单的相机
	Vector3f lowerLeftCorner(-2.0, -1.0, -1.0);
	Vector3f horizontal(4.0, 0.0, 0.0);
	Vector3f vertical(0.0, 2.0, 0.0);
	Point3f origin;

	vector<std::shared_ptr<Shape>> shapes;
	shapes.push_back(CreateSphereShape(Point3f(0, 0, -1), 0.5));
	shapes.push_back(CreateSphereShape(Point3f(0, -100.5, -1), 100));
	std::shared_ptr<Shape> world = CreateShapeList(shapes);
	for (int sy = height - 1; sy >= 0; sy--)
	{
		for (int sx = 0; sx < width; sx++)
		{
			Float u = Float(sx) / Float(width);
			Float v = Float(height - sy - 1) / Float(height);
			
			Ray ray(origin, lowerLeftCorner + u * horizontal + v * vertical);
			
			Point3f color = Color(ray, world);
			
			int ir = int(255.99 * color[0]);
			int ig = int(255.99 * color[1]);
			int ib = int(255.99 * color[2]);
			
			int shadingPoint = (sy * width + sx) * 3;
			data[shadingPoint] = ir;
			data[shadingPoint + 1] = ig;
			data[shadingPoint + 2] = ib;
		}
	}
	// 写入图像
	stbi_write_png("output-chapter05-2.png", width, height, channel, data, 0);
	cout << "生成成功！" << endl;
	stbi_image_free(data);
}

