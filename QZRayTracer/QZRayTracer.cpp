
#include "src/core/QZRayTracer.h"
#include "src/core/api.h"
#include "src/scene/example.h"

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
		return Lerp(t, Point3f(1.0, 1.0, 1.0), Point3f(0.8, 0.6, 0.6));
	}
}


void Renderer(RendererSet& set) {
	// 参数设置
	Camera camera = set.camera;
	int spp = set.spp;
	int depth = 0;
	int width = set.width, height = set.height, channel = 3;
	const char* savePath = set.savePath;
	Float invSpp = 1.0 / Float(spp);

	auto* data = (unsigned char*)malloc(width * height * channel);

	// 包含所有Shape的场景
	std::shared_ptr<Shape> world = set.shapes;

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
			color = Point3f(pow(color.x, Gamma), pow(color.y, Gamma), pow(color.z, Gamma)); // gamma矫正
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
	cout << endl;
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

	RendererSet renderSet = ShapeTestCylinderScene();
	
	
	Renderer(renderSet);
	// 渲染多帧来生成动画
	/*stringstream tempPath;
	for (int i = 0; i < frame; i++) {
		Camera camera(lookFrom, lookAt, worldUp, 20, 2.0, 1.0, i * unitFocusDis);
		tempPath << "./temp/focusDis" << i << ".png";
		Renderer(tempPath.str().c_str(), camera);
		tempPath.str("");
	}*/

	end = clock();   //结束时间
	cout << "\n\nRenderer time is " << Float(end - start) / CLOCKS_PER_SEC << "s" << endl;  //输出时间（单位：ms）

}

