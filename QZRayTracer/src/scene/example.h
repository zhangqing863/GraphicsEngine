#ifndef QZRT_SCENE_EXAMPLE_H
#define QZRT_SCENE_EXAMPLE_H

#include "../core/QZRayTracer.h"
#include "../core/geometry.h"
#include "../core/api.h"

namespace raytracer {
	RendererSet ShapeTestCylinderScene();
	RendererSet RandomScene();

	inline RendererSet RandomScene() {
		Point3f lookFrom = Point3f(13, 2, 3);
		Point3f lookAt = Point3f(0, 0, 0);
		Vector3f lookUp = WorldUp;
		Float aperture = 0.0;
		Float fov = 20.0;
		Float focusDis = 10.0;
		Float screenWidth = 1000;
		Float screenHeight = 500;
		Float aspect = screenWidth / screenHeight;
		Camera cam = Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis);
		int spp = 1000;
		const char* savePath = "./output/output-chapter12-test-1000x500.png";

		std::vector<std::shared_ptr<Shape>> shapes;
		shapes.push_back(CreateSphereShape(Point3f(0, -1000, 0), 1000, std::make_shared<Lambertian>(Point3f(0.5, 0.5, 0.5))));

		
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				Float chooseMat = randomNum(seeds);
				Point3f center = Point3f(a + 0.9 * randomNum(seeds), 0.2, b + 0.9 * randomNum(seeds));
				if ((center - Point3f(4, 0.2, 0)).Length() > 0.9) {
					if (chooseMat < 0.7) { // 选择漫反射材质
						shapes.push_back(
							CreateSphereShape(
								center,
								0.2,
								std::make_shared<Lambertian>(Point3f(randomNum(seeds) * randomNum(seeds), randomNum(seeds) * randomNum(seeds), randomNum(seeds) * randomNum(seeds)))));
					}
					else if(chooseMat < 0.85) { // 选择金属材质
						shapes.push_back(
							CreateSphereShape(
								center,
								0.2,
								std::make_shared<Metal>(Point3f(0.5 * (1 + randomNum(seeds)), 0.5 * (1 + randomNum(seeds)), 0.5 * (1 + randomNum(seeds))), randomNum(seeds))));
					}
					else if (chooseMat < 0.95) { // 选择玻璃材质
						shapes.push_back(
							CreateSphereShape(
								center,
								0.2,
								std::make_shared<Dielectric>(1 + randomNum(seeds))));
					}
					else { // 选择中空玻璃球材质
						shapes.push_back(
							CreateSphereShape(
								center,
								0.2,
								std::make_shared<Dielectric>(1.5)));
						shapes.push_back(
							CreateSphereShape(
								center,
								-0.15,
								std::make_shared<Dielectric>(1.5)));
					}
				}
			}

		}

		shapes.push_back(CreateSphereShape(Point3f(0, 1, 0), 1.0, std::make_shared<Dielectric>(1.5)));
		shapes.push_back(CreateSphereShape(Point3f(-4, 1, 0), 1.0, std::make_shared<Lambertian>(Point3f(0.4, 0.2, 0.1))));
		shapes.push_back(CreateSphereShape(Point3f(4, 1, 0), 1.0, std::make_shared<Metal>(Point3f(0.7, 0.6, 0.5), 0.0)));
		std::shared_ptr<Shape> shapeList = std::make_shared<ShapeList>(shapes);
		
		return RendererSet(cam, screenWidth, screenHeight, spp, savePath, shapeList);
	}

	inline RendererSet ShapeTestCylinderScene() {
		// 基础设置
		Point3f lookFrom = Point3f(-1, 3, 6);
		Point3f lookAt = Point3f(0, 0, -1);
		Vector3f lookUp = WorldUp;
		Float aperture = 0.0;
		Float fov = 80.0;
		Float focusDis = (lookFrom - lookAt).Length();
		Float screenWidth = 800;
		Float screenHeight = 500;
		Float aspect = screenWidth / screenHeight;
		Camera cam = Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis);
		int spp = 1000;
		const char* savePath = "./output/CustomAdd/cylinder-final.png";

		// 场景中的物体设置
		std::vector<std::shared_ptr<Shape>> shapes;

		shapes.push_back(CreateSphereShape(Point3f(0, -1000, 0), 1000, std::make_shared<Lambertian>(Point3f(0.980392, 0.694118, 0.627451))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 0.1, 0), 4.0, 0.0, 0.2, std::make_shared<Lambertian>(Point3f(0.882353, 0.439216, 0.333333))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 0.25, 0), 3.0, 0.0, 0.1, std::make_shared<Metal>(Point3f(0.423529, 0.360784, 0.905882), 0)));
		shapes.push_back(CreateCylinderShape(Point3f(2.5, 1.7, 2.5), 0.2, 0.0, 3.0, std::make_shared<Metal>(Point3f(0.0352941, 0.517647, 0.890196), 0.6)));
		shapes.push_back(CreateCylinderShape(Point3f(2.5, 1.7, -2.5), 0.2, 0.0, 3.0, std::make_shared<Lambertian>(Point3f(0, 0.807843, 0.788235))));
		shapes.push_back(CreateCylinderShape(Point3f(-2.5, 1.7, 2.5), 0.2, 0.0, 3.0, std::make_shared<Dielectric>(1.5)));
		shapes.push_back(CreateCylinderShape(Point3f(-2.5, 1.7, -2.5), 0.2, 0.0, 3.0, std::make_shared<Metal>(Point3f(0.992157, 0.47451, 0.658824), 0.3)));
		shapes.push_back(CreateCylinderShape(Point3f(0, 3.3, 0), 4.0, 0.0, 0.2, std::make_shared<Metal>(Point3f(0.839216, 0.188235, 0.192157), 0.5)));
		
		// 圆盘上的球
		for (int a = -3; a < 3; a++) {
			for (int b = -3; b < 3; b++) {
				Float chooseMat = randomNum(seeds);
				Point3f center = Point3f(a + 0.9 * randomNum(seeds), 0.3 + (0.1 + 0.2 * randomNum(seeds)), b + 0.9 * randomNum(seeds));
				if ((center - Point3f(0, center.y, 0)).Length() > 0.6 + (center.y - 0.35) && (center - Point3f(0, center.y, 0)).Length() <= 3.0 - (center.y - 0.35)) {
					if (chooseMat < 0.7) { // 选择漫反射材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.3,
								std::make_shared<Lambertian>(Point3f(randomNum(seeds) * randomNum(seeds), randomNum(seeds) * randomNum(seeds), randomNum(seeds) * randomNum(seeds)))));
					}
					else if (chooseMat < 0.85) { // 选择金属材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.3,
								std::make_shared<Metal>(Point3f(0.5 * (1 + randomNum(seeds)), 0.5 * (1 + randomNum(seeds)), 0.5 * (1 + randomNum(seeds))), randomNum(seeds))));
					}
					else if (chooseMat < 0.95) { // 选择玻璃材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.3,
								std::make_shared<Dielectric>(1 + randomNum(seeds))));
					}
					else { // 选择中空玻璃球材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.3,
								std::make_shared<Dielectric>(1.5)));
						shapes.push_back(
							CreateSphereShape(
								center,
								0.4 - center.y,
								std::make_shared<Dielectric>(1.5)));
					}
				}
			}

		}

		// 圆盘上的圆柱

		shapes.push_back(CreateCylinderShape(Point3f(0, 0.375, 0), 0.6, 0.0, 0.15, std::make_shared<Lambertian>(Point3f(1.0, 1.0, 1.0))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 0.525, 0), 0.5, 0.0, 0.15, std::make_shared<Lambertian>(Point3f(0.1, 0.1, 0.1))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 0.675, 0), 0.4, 0.0, 0.15, std::make_shared<Lambertian>(Point3f(0.9, 0.9, 0.9))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 0.825, 0), 0.3, 0.0, 0.15, std::make_shared<Metal>(Point3f(0.827451, 0.329412, 0), 0.3)));
		shapes.push_back(CreateCylinderShape(Point3f(0, 1.2625, 0), 0.2, 0.0, 0.575, std::make_shared<Dielectric>(1.5)));
		shapes.push_back(CreateSphereShape(Point3f(0, 1.75, 0), 0.20, std::make_shared<Metal>(Point3f(0.752941, 0.223529, 0.168627), 0)));
		shapes.push_back(CreateCylinderShape(Point3f(0, 2.2375, 0), 0.2, 0.0, 0.575, std::make_shared<Dielectric>(1.5)));
		shapes.push_back(CreateCylinderShape(Point3f(0, 2.6, 0), 0.3, 0.0, 0.15, std::make_shared<Metal>(Point3f(0.827451, 0.329412, 0), 0.3)));
		shapes.push_back(CreateCylinderShape(Point3f(0, 2.75, 0), 0.4, 0.0, 0.15, std::make_shared<Lambertian>(Point3f(0.9, 0.9, 0.9))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 2.9, 0), 0.5, 0.0, 0.15, std::make_shared<Lambertian>(Point3f(0.1, 0.1, 0.1))));
		shapes.push_back(CreateCylinderShape(Point3f(0, 3.05, 0), 0.6, 0.0, 0.15, std::make_shared<Lambertian>(Point3f(1.0, 1.0, 1.0))));

		// 圆盘下的球
		for (Float a = -4; a < 4; a+=0.5) {
			for (Float b = -4; b < 4; b++) {
				Float chooseMat = randomNum(seeds);
				Point3f center = Point3f(a + 0.9 * randomNum(seeds), 0.2 + (0.1 + 0.25 * randomNum(seeds)), b + 0.9 * randomNum(seeds));
				if ((center - Point3f(0, center.y, 0)).Length() <= 4.0 - (center.y - 0.2) &&
					(center - Point3f(0, center.y, 0)).Length() >= 3.0 + (center.y - 0.2) &&
					(center - Point3f(2.5, center.y, 2.5)).Length() >= (center.y - 0.2) &&
					(center - Point3f(-2.5, center.y, 2.5)).Length() >= (center.y - 0.2) &&
					(center - Point3f(2.5, center.y, -2.5)).Length() >= (center.y - 0.2) &&
					(center - Point3f(-2.5, center.y, -2.5)).Length() >= (center.y - 0.2)) {
					if (chooseMat < 0.7) { // 选择漫反射材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.2,
								std::make_shared<Lambertian>(Point3f(randomNum(seeds) * randomNum(seeds), randomNum(seeds) * randomNum(seeds), randomNum(seeds) * randomNum(seeds)))));
					}
					else if (chooseMat < 0.85) { // 选择金属材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.2,
								std::make_shared<Metal>(Point3f(0.5 * (1 + randomNum(seeds)), 0.5 * (1 + randomNum(seeds)), 0.5 * (1 + randomNum(seeds))), randomNum(seeds))));
					}
					else if (chooseMat < 0.95) { // 选择玻璃材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.2,
								std::make_shared<Dielectric>(1 + randomNum(seeds))));
					}
					else { // 选择中空玻璃球材质
						shapes.push_back(
							CreateSphereShape(
								center,
								center.y - 0.2,
								std::make_shared<Dielectric>(1.5)));
						shapes.push_back(
							CreateSphereShape(
								center,
								0.25 - center.y,
								std::make_shared<Dielectric>(1.5)));
					}
				}
			}

		}

		std::shared_ptr<Shape> shapeList = std::make_shared<ShapeList>(shapes);

		return RendererSet(cam, screenWidth, screenHeight, spp, savePath, shapeList);
	}
	
}


#endif // QZRT_SCENE_EXAMPLE_H