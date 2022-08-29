#ifndef QZRT_SCENE_EXAMPLE_H
#define QZRT_SCENE_EXAMPLE_H

#include "../core/QZRayTracer.h"
#include "../core/geometry.h"
#include "../core/api.h"

namespace raytracer {
	std::shared_ptr<ShapeList> RandomScene() {
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

		return std::make_shared<ShapeList>(shapes);
	}
	
}


#endif // QZRT_SCENE_EXAMPLE_H