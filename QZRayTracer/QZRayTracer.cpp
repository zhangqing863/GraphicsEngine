
#include "src/core/QZRayTracer.h"
#include "src/core/api.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "src/core/stb_image_write.h"
#include "src/core/stb_image.h"
using namespace raytracer;
using namespace std;

// 设置随机种子
std::default_random_engine seeds;

// Chapter07
Point3f RandomInUnitSphere() {
	Vector3f p;
	// 构建随机数
	std::uniform_real_distribution<Float> randomNum(0, 1); // 左闭右闭区间
	do {
		p = Vector3f(randomNum(seeds), randomNum(seeds), randomNum(seeds));
	} while (Dot(p, p) >= 1.0);
	
	
	return Point3f(p);
}

// Chapter03-04 : simple color function
Point3f Color(const Ray& ray, shared_ptr<Shape> world) {
	HitRecord rec;

	if (world->hit(ray, rec)) {
		Point3f target = rec.p + Point3f(rec.normal) + RandomInUnitSphere();
		return 0.5 * Color(Ray(rec.p, target - rec.p), world);
	}
	else {
		// 没击中就画个背景
		Vector3f dir = Normalize(ray.d);
		Float t = 0.5 * (dir.y + 1.0);
		return Lerp(t, Point3f(1.0, 1.0, 1.0), Point3f(0.5, 0.7, 1.0));
	}
}


void Renderer(const char* savePath) {
	// 参数设置
	int width = 1000, height = 500, channel = 3;
	Float gamma = 1.0 / 2.2;


	// 采样值，一个像素内采多少次样
	int spp = 4;
	Float invSpp = 1.0 / Float(spp);

	auto* data = (unsigned char*)malloc(width * height * channel);

	// 构建一个简单的相机
	Camera camera;

	// 搭建一个简单的场景
	vector<std::shared_ptr<Shape>> shapes;
	shapes.push_back(CreateSphereShape(Point3f(0, 0, -1), 0.5));
	shapes.push_back(CreateSphereShape(Point3f(0, -100.5, -1), 100));

	// 构建随机数
	// std::default_random_engine seeds;
	// seeds.seed(time(0));
	std::uniform_real_distribution<Float> randomNum(0, 1); // 左闭右闭区间


	std::shared_ptr<Shape> world = CreateShapeList(shapes);
	for (auto sy = height - 1; sy >= 0; sy--) {
		for (auto sx = 0; sx < width; sx++) {
			Point3f color;
			// 采样计算
			for (auto s = 0; s < spp; s++) {
				Float u = Float(sx + randomNum(seeds)) / Float(width);
				Float v = Float(height - sy - 1 + randomNum(seeds)) / Float(height);
				Ray ray = camera.GenerateRay(u, v);
				color += Color(ray, world);
			}
			color *= invSpp; // 求平均值
			color = Point3f(pow(color.x, gamma), pow(color.y, gamma), pow(color.z, gamma)); // gamma矫正
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
	stbi_write_png(savePath, width, height, channel, data, 0);
	cout << "渲染完成！" << endl;
	stbi_image_free(data);
}


int main() {
	// 记录用时
	clock_t start, end; 
	start = clock();
	seeds.seed(time(0));

	const char* savePath = "output-chapter07-spp4-gamma.png";

	Renderer(savePath);

	end = clock();   //结束时间
	cout << "Renderer time is " << double(end - start) << "ms" << endl;  //输出时间（单位：ms）
	
}

