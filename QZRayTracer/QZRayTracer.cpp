
#include "src/core/QZRayTracer.h"
#include "src/core/api.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "src/core/stb_image_write.h"
#include "src/core/stb_image.h"
using namespace raytracer;
using namespace std;

#define MAXBOUNDTIME 10
#define ELEGANT // 用来在控制台展示进度
/// <summary>
/// 着色器
/// </summary>
/// <param name="ray">光线</param>
/// <param name="world">渲染的对象</param>
/// <param name="depth">光线弹射次数</param>
/// <returns></returns>
Point3f Color(const Ray& ray, shared_ptr<Shape> world, int depth) {
	HitRecord rec;

	if (world->Hit(ray, rec)) {
		Ray wo;
		Point3f attenuation;
		if (depth < MAXBOUNDTIME && rec.mat->Scatter(ray, rec, attenuation, wo)) {
			return attenuation * Color(wo, world, depth + 1);
		}
		else {
			return Point3f();
		}
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
	int depth = 0;
	int width = 1000, height = 500, channel = 3;
	Float gamma = 1.0 / 2.2;


	// 采样值，一个像素内采多少次样
	int spp = 100;
	Float invSpp = 1.0 / Float(spp);

	auto* data = (unsigned char*)malloc(width * height * channel);

	// 构建一个简单的相机
	Camera camera;

	// 搭建一个简单的场景
	vector<std::shared_ptr<Shape>> shapes;
	std::shared_ptr<Material> lambRedMat = std::make_shared<Lambertian>(Point3f(0.8, 0.3, 0.3));
	std::shared_ptr<Material> lambBlueMat = std::make_shared<Lambertian>(Point3f(0.2, 0.596, 0.8588));
	std::shared_ptr<Material> lambPurpleMat = std::make_shared<Lambertian>(Point3f(0.557, 0.27, 0.678));
	std::shared_ptr<Material> lambGlassGreengreenMat = std::make_shared<Lambertian>(Point3f(0.8, 0.8, 0.0));
	std::shared_ptr<Material> metalGreenMat = std::make_shared<Metal>(Point3f(0.1, 0.74, 0.61), 0);
	std::shared_ptr<Material> metalBlueMat = std::make_shared<Metal>(Point3f(0.2, 0.596, 0.8588), 0.3);
	std::shared_ptr<Material> metalGlassGreenMat = std::make_shared<Metal>(Point3f(0.8, 0.6, 0.2), 0.6);
	std::shared_ptr<Material> metalWhiteMat = std::make_shared<Metal>(Point3f(0.8, 0.8, 0.8), 1.0);
	std::shared_ptr<Material> dlcMat = std::make_shared<Dielectric>(1.5);
	shapes.push_back(CreateSphereShape(Point3f(0, 0, -1), 0.5, lambBlueMat));
	shapes.push_back(CreateSphereShape(Point3f(0, -100.5, -1), 100, lambGlassGreengreenMat));
	shapes.push_back(CreateSphereShape(Point3f(1, 0, -1), 0.5, metalGlassGreenMat));
	shapes.push_back(CreateSphereShape(Point3f(-1, 0, -1), 0.5, dlcMat));
	shapes.push_back(CreateSphereShape(Point3f(-1, 0, -1), -0.45, dlcMat));
	shapes.push_back(CreateSphereShape(Point3f(-1, 0, -1), 0.3, metalGreenMat));

	// 包含所有Shape的场景
	std::shared_ptr<Shape> world = CreateShapeList(shapes);

#ifdef ELEGANT
	ProgressBar bar(height);
	bar.set_todo_char(" ");
	bar.set_done_char("█");
	bar.set_opening_bracket_char("Rendering:[");
	bar.set_closing_bracket_char("]");
#endif // ELEGANT
	for (auto sy = height - 1; sy >= 0; sy--) {
		for (auto sx = 0; sx < width; sx++) {
			Point3f color;
			// 采样计算
			for (auto s = 0; s < spp; s++) {
				Float u = Float(sx + randomNum(seeds)) / Float(width);
				Float v = Float(height - sy - 1 + randomNum(seeds)) / Float(height);
				Ray ray = camera.GenerateRay(u, v);
				color += Color(ray, world, depth);
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

#ifdef ELEGANT
		bar.update();
#endif // ELEGANT		
	}

	// 写入图像
	stbi_write_png(savePath, width, height, channel, data, 0);
	stbi_image_free(data);
}


int main() {
	// 记录用时
	clock_t start, end;
	start = clock();
	seeds.seed(time(0));

	std::cout << "        wWw  wWw(o)__(o)\\\\  //     .-.     ))           _oo  \\\\  //       \\\\\\  ///   \\/       .-.    wW  Ww\\\\\\  ///   \\/    " << std::endl;
	std::cout << "   /)   (O)  (O)(__  __)(o)(o)   c(O_O)c  (Oo)-.     >-(_  \\ (o)(o)   /)  ((O)(O))  (OO)    c(O_O)c  (O)(O)((O)(O))  (OO)   " << std::endl;
	std::cout << " (o)(O) / )  ( \\  (  )  ||  ||  ,'.---.`,  | (_))       / _/ ||  || (o)(O) | \\ || ,'.--.)  ,'.---.`,  (..)  | \\ || ,'.--.)" << std::endl;
	std::cout << "  //\\\\ / /    \\ \\  )(   |(__)| / /|_|_|\\ \\ |  .'       / /   |(__)|  //\\\\  ||\\\\||/ /|_|_\\ / /|_|_|\\ \\  ||   ||\\\\||/ /|_|_\\" << std::endl;
	std::cout << " |(__)|| \\____/ | (  )  /.--.\\ | \\_____/ | )|\\\\       / (    /.--.\\ |(__)| || \\ || \\_.--. | \\___.--.| _||_  || \\ || \\_.--." << std::endl;
	std::cout << " /,-. |'. `--' .`  )/  -'    `-'. `---' .`(/  \\)     (   `-.-'    `-/,-. | ||  ||'.   \\) \\'. `---\\) \\(_/\\_) ||  ||'.   \\) \\" << std::endl;
	std::cout << "-'   ''  `-..-'   (              `-...-'   )          `--.._)      -'   ''(_/  \\_) `-.(_.'  `-...(_.'      (_/  \\_) `-.(_.' " << std::endl << std::endl;

	const char* savePath = "./output/output-chapter09-spp100-hollowglass3-1000x500.png";

	Renderer(savePath);

	end = clock();   //结束时间
	cout << "\n\nRenderer time is " << double(end - start) << "ms" << endl;  //输出时间（单位：ms）

}

