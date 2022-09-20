#ifndef QZRT_SCENE_EXAMPLE_H
#define QZRT_SCENE_EXAMPLE_H

#include "../core/QZRayTracer.h"
#include "../core/geometry.h"
#include "../core/api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace raytracer {
#define RND (curand_uniform(&local_rand_state))

	__global__ void rand_init(curandState* rand_state, int num = 1) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			for (int i = 0; i < num; i++) {
				//curand_init(1908, 0, 0, rand_state);
				curand_init(2022, i, 0, &rand_state[i]);
			}
		}
	}

	__global__ void free_world(Shape** d_list, Shape** d_world, Camera** d_camera) {
		printf("free_world! n:%d\n", (*d_world)->numShapes);
		for (int i = 0; i < (*d_world)->numShapes; i++) {
			delete d_list[i]->material->albedo;
			delete d_list[i]->material;
			delete d_list[i];
		}
		delete* d_world;
		delete* d_camera;
	}

	
	__global__ void free_world_bvh(Shape** d_list, Shape** d_world, Camera** d_camera) {
		int numShapes = (*d_world)->numShapes;
		for (int i = 0; i < numShapes; i++) {
			// delete d_list[i]->material->albedo;
			delete d_list[i]->material;
			delete d_list[i];
		}
		//DeleteBVHNode(d_nodes, (*d_world)->numNodes);
		delete* d_camera;
	}

	

	__global__ void create_world(Shape** d_list, Shape** d_world, Camera** d_camera, int nx, int ny) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			int curNum = 0;
			d_list[0] = new Sphere(Point3f(0, 0, -1), 0.5,
				new Lambertian(new ConstantTexture(Point3f(0.1, 0.2, 0.5))));
			curNum++;
			d_list[1] = new Sphere(Point3f(0, -100.5, -1), 100,
				new Lambertian(new ConstantTexture(Point3f(0.8, 0.8, 0.0))));
			d_list[2] = new Sphere(Point3f(1, 0, -1), 0.5,
				new Metal(new ConstantTexture(Point3f(0.8, 0.6, 0.2)), 0.0));
			d_list[3] = new Sphere(Point3f(-1, 0, -1), 0.5,
				new Dielectric(1.5));
			d_list[4] = new Sphere(Point3f(-1, 0, -1), -0.45,
				new Dielectric(1.5));
			*d_world = new ShapeList(d_list, 5);
			Point3f lookfrom(-2, 2, 1);
			Point3f lookat(0, 0, -1);
			float dist_to_focus = (lookfrom - lookat).Length();
			float aperture = 0;
			*d_camera = new Camera(lookfrom,
				lookat,
				Vector3f(0, 1, 0),
				20.0,
				float(nx) / float(ny),
				aperture,
				dist_to_focus);
		}
	}

	

	__global__ void SampleScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis);

			int curNum = 0; // 记录创建的Shape数量
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(new ConstantTexture(Point3f(0.5, 0.5, 0.5))));


			for (int a = -11; a < 11; a++) {
				for (int b = -11; b < 11; b++) {
					Float chooseMat = RND;
					Point3f center = Point3f(a + 0.9 * RND, 0.2, b + 0.9 * RND);
					if ((center - Point3f(4, 0.2, 0)).Length() > 0.9) {
						if (chooseMat < 0.7) { // 选择漫反射材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Lambertian(new ConstantTexture(Point3f(RND * RND, RND * RND, RND * RND))));
						}
						else if (chooseMat < 0.85) { // 选择金属材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Metal(new ConstantTexture(Point3f(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND))), RND));
						}
						else if (chooseMat < 0.95) { // 选择玻璃材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1 + RND));
						}
						else { // 选择中空玻璃球材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1.5));
							shapes[curNum++] = new Sphere(
								center,
								-0.15,
								new Dielectric(1.5));
						}
					}
				}

			}

			shapes[curNum++] = new Sphere(Point3f(0, 1, 0), 1.0, new Dielectric(1.5));
			shapes[curNum++] = new Sphere(Point3f(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Point3f(0.4, 0.2, 0.1))));
			shapes[curNum++] = new Sphere(Point3f(4, 1, 0), 1.0, new Metal(new ConstantTexture(Point3f(0.7, 0.6, 0.5)), 0.0));
			*rand_state = local_rand_state;

			/*for (int i = 0; i < curNum; i++) {
				Bounds3f box;
				if (shapes[i]->BoundingBox(box)) {
					printf("The %dth Shape Bounding Box is:[%f, %f, %f]\n", i, box.pMin.x, box.pMin.y, box.pMin.z);
				}
			}*/
			//new BVHNode(shapes, curNum, &local_rand_state)
			*world = new ShapeList(shapes, curNum);


			


			printf("Create Successful!\n");
			//*world = root;
			int n = (*world)->numShapes;
			printf("n:%d\n", n);
		}
	}

	__global__ void Chapter1MotionBlurScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(new ConstantTexture(Point3f(0.5, 0.5, 0.5))));


			for (int a = -11; a < 11; a++) {
				for (int b = -11; b < 11; b++) {
					Float chooseMat = RND;
					Point3f center = Point3f(a + 0.9 * RND, 0.2, b + 0.9 * RND);
					if ((center - Point3f(4, 0.2, 0)).Length() > 0.9) {
						if (chooseMat < 0.7) { // 选择漫反射材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Lambertian(new ConstantTexture(Point3f(RND * RND, RND * RND, RND * RND))));
						}
						else if (chooseMat < 0.85) { // 选择金属材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Metal(new ConstantTexture(Point3f(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND))), RND));
						}
						else if (chooseMat < 0.95) { // 选择玻璃材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1 + RND));
						}
						else { // 选择中空玻璃球材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1.5));
							shapes[curNum++] = new Sphere(
								center,
								-0.15,
								new Dielectric(1.5));
						}
					}
				}

			}

			shapes[curNum++] = new Sphere(Point3f(0, 1, 0), 1.0, new Dielectric(1.5));
			shapes[curNum++] = new Sphere(Point3f(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Point3f(0.4, 0.2, 0.1))));
			shapes[curNum++] = new Sphere(Point3f(4, 1, 0), 1.0, new Metal(new ConstantTexture(Point3f(0.7, 0.6, 0.5)), 0.0));
			*rand_state = local_rand_state;

			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f); // 使用BVH 100s spp 1000
			int n = (*world)->numShapes;
			printf("n:%d\n", n);
		}
	}

	__global__ void ShapeTestCylinderScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			// 基础设置
			Point3f lookFrom = Point3f(-1, 3, 6);
			Point3f lookAt = Point3f(0, 0, -1);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 80.0;
			Float focusDis = (lookFrom - lookAt).Length();
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis);

			// 场景中的物体设置

			int curNum = 0; // 记录创建的Shape数量
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(new ConstantTexture(Point3f(0.980392, 0.694118, 0.627451))));
			shapes[curNum++] = new Cylinder(Point3f(0, 0.1, 0), 4.0, 0.0, 0.2, new Lambertian(new ConstantTexture(Point3f(0.882353, 0.439216, 0.333333))));
			shapes[curNum++] = new Cylinder(Point3f(0, 0.25, 0), 3.0, 0.0, 0.1, new Metal(new ConstantTexture(Point3f(0.423529, 0.360784, 0.905882)), 0));
			shapes[curNum++] = new Cylinder(Point3f(2.5, 1.7, 2.5), 0.2, 0.0, 3.0, new Metal(new ConstantTexture(Point3f(0.0352941, 0.517647, 0.890196)), 0.6));
			shapes[curNum++] = new Cylinder(Point3f(2.5, 1.7, -2.5), 0.2, 0.0, 3.0, new Lambertian(new ConstantTexture(Point3f(0, 0.807843, 0.788235))));
			shapes[curNum++] = new Cylinder(Point3f(-2.5, 1.7, 2.5), 0.2, 0.0, 3.0, new Dielectric(1.5));
			shapes[curNum++] = new Cylinder(Point3f(-2.5, 1.7, -2.5), 0.2, 0.0, 3.0, new Metal(new ConstantTexture(Point3f(0.992157, 0.47451, 0.658824)), 0.3));
			shapes[curNum++] = new Cylinder(Point3f(0, 3.3, 0), 4.0, 0.0, 0.2, new Metal(new ConstantTexture(Point3f(0.839216, 0.188235, 0.192157)), 0.5));

			// 圆盘上的球
			for (int a = -3; a < 3; a++) {
				for (int b = -3; b < 3; b++) {
					Float chooseMat = RND;
					Point3f center = Point3f(a + 0.9 * RND, 0.3 + (0.1 + 0.2 * RND), b + 0.9 * RND);
					if ((center - Point3f(0, center.y, 0)).Length() > 0.6 + (center.y - 0.35) && (center - Point3f(0, center.y, 0)).Length() <= 3.0 - (center.y - 0.35)) {
						if (chooseMat < 0.7) { // 选择漫反射材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.3,
								new Lambertian(new ConstantTexture(Point3f(RND * RND, RND * RND, RND * RND))));
						}
						else if (chooseMat < 0.85) { // 选择金属材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.3,
								new Metal(new ConstantTexture(Point3f(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND))), RND));
						}
						else if (chooseMat < 0.95) { // 选择玻璃材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.3,
								new Dielectric(1 + RND));
						}
						else { // 选择中空玻璃球材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.3,
								new Dielectric(1.5));
							shapes[curNum++] = new Sphere(
								center,
								0.4 - center.y,
								new Dielectric(1.5));
						}
					}
				}
			}



			// 圆盘上的圆柱

			shapes[curNum++] = new Cylinder(Point3f(0, 0.375, 0), 0.6, 0.0, 0.15, new Lambertian(new ConstantTexture(Point3f(1.0, 1.0, 1.0))));
			shapes[curNum++] = new Cylinder(Point3f(0, 0.525, 0), 0.5, 0.0, 0.15, new Lambertian(new ConstantTexture(Point3f(0.1, 0.1, 0.1))));
			shapes[curNum++] = new Cylinder(Point3f(0, 0.675, 0), 0.4, 0.0, 0.15, new Lambertian(new ConstantTexture(Point3f(0.9, 0.9, 0.9))));
			shapes[curNum++] = new Cylinder(Point3f(0, 0.825, 0), 0.3, 0.0, 0.15, new Metal(new ConstantTexture(Point3f(0.827451, 0.329412, 0)), 0.3));
			shapes[curNum++] = new Cylinder(Point3f(0, 1.2625, 0), 0.2, 0.0, 0.575, new Dielectric(1.5));
			shapes[curNum++] = new Sphere(Point3f(0, 1.75, 0), 0.20, new Metal(new ConstantTexture(Point3f(0.752941, 0.223529, 0.168627)), 0));
			shapes[curNum++] = new Cylinder(Point3f(0, 2.2375, 0), 0.2, 0.0, 0.575, new Dielectric(1.5));
			shapes[curNum++] = new Cylinder(Point3f(0, 2.6, 0), 0.3, 0.0, 0.15, new Metal(new ConstantTexture(Point3f(0.827451, 0.329412, 0)), 0.3));
			shapes[curNum++] = new Cylinder(Point3f(0, 2.75, 0), 0.4, 0.0, 0.15, new Lambertian(new ConstantTexture(Point3f(0.9, 0.9, 0.9))));
			shapes[curNum++] = new Cylinder(Point3f(0, 2.9, 0), 0.5, 0.0, 0.15, new Lambertian(new ConstantTexture(Point3f(0.1, 0.1, 0.1))));
			shapes[curNum++] = new Cylinder(Point3f(0, 3.05, 0), 0.6, 0.0, 0.15, new Lambertian(new ConstantTexture(Point3f(1.0, 1.0, 1.0))));

			// 圆盘下的球
			for (Float a = -4; a < 4; a += 0.5) {
				for (Float b = -4; b < 4; b++) {
					Float chooseMat = RND;
					Point3f center = Point3f(a + 0.9 * RND, 0.2 + (0.1 + 0.25 * RND), b + 0.9 * RND);
					if ((center - Point3f(0, center.y, 0)).Length() <= 4.0 - (center.y - 0.2) &&
						(center - Point3f(0, center.y, 0)).Length() >= 3.0 + (center.y - 0.2) &&
						(center - Point3f(2.5, center.y, 2.5)).Length() >= center.y  &&
						(center - Point3f(-2.5, center.y, 2.5)).Length() >= center.y  &&
						(center - Point3f(2.5, center.y, -2.5)).Length() >= center.y  &&
						(center - Point3f(-2.5, center.y, -2.5)).Length() >= center.y ) {
						if (chooseMat < 0.7) { // 选择漫反射材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.2,
								new Lambertian(new ConstantTexture(Point3f(RND * RND, RND * RND, RND * RND))));
						}
						else if (chooseMat < 0.85) { // 选择金属材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.2,
								new Metal(new ConstantTexture(Point3f(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND))), RND));
						}
						else if (chooseMat < 0.95) { // 选择玻璃材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.2,
								new Dielectric(1 + RND));
						}
						else { // 选择中空玻璃球材质
							shapes[curNum++] = new Sphere(
								center,
								center.y - 0.2,
								new Dielectric(1.5));
							shapes[curNum++] = new Sphere(
								center,
								0.25 - center.y,
								new Dielectric(1.5));
						}
					}
				}

			}
			*rand_state = local_rand_state;
			*world = new ShapeList(shapes, curNum);
		}
	}
	


	__global__ void Chapter2BVHScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(new ConstantTexture(Point3f(0.5, 0.5, 0.5))));


			for (int a = -11; a < 11; a++) {
				for (int b = -11; b < 11; b++) {
					Float chooseMat = RND;
					Point3f center = Point3f(a + 0.9 * RND, 0.2, b + 0.9 * RND);
					if ((center - Point3f(4, 0.2, 0)).Length() > 0.9) {
						if (chooseMat < 0.7) { // 选择漫反射材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Lambertian(new ConstantTexture(Point3f(RND * RND, RND * RND, RND * RND))));
						}
						else if (chooseMat < 0.85) { // 选择金属材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Metal(new ConstantTexture(Point3f(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND))), RND));
						}
						else if (chooseMat < 0.95) { // 选择玻璃材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1 + RND));
						}
						else { // 选择中空玻璃球材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1.5));
							shapes[curNum++] = new Sphere(
								center,
								-0.15,
								new Dielectric(1.5));
						}
					}
				}

			}

			shapes[curNum++] = new Sphere(Point3f(0, 1, 0), 1.0, new Dielectric(1.5));
			shapes[curNum++] = new Cylinder(Point3f(2, .25, 3), 1.0, 0.0, 0.5, new Lambertian(new ConstantTexture(Point3f(0.2, 0.2, 0.8))));
			shapes[curNum++] = new DSphere(Point3f(2, 0.8, 3), Point3f(2, 1.2, 3), 0.0, 1.0, 0.3, new Lambertian(new ConstantTexture(Point3f(0.8, 0.2, 0.2))));
			shapes[curNum++] = new Sphere(Point3f(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Point3f(0.4, 0.2, 0.1))));
			shapes[curNum++] = new Sphere(Point3f(4, 1, 0), 1.0, new Metal(new ConstantTexture(Point3f(0.7, 0.6, 0.5)), 0.0));

			/*for (int i = 0; i < curNum + curNum / 2 + 1; i++) {
				nodes[i] = new BVHNode(shapes, curNum, nodes);
			}*/

			*rand_state = local_rand_state;

			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f); // 使用BVH 100s spp 1000
			//*world = new ShapeList(shapes, curNum);// 不使用BVH 675s spp 1000
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter3TextureScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			Texture* checker = new CheckerTexture(new ConstantTexture(Point3f(0.607843, 0.34902, 0.713725)), new ConstantTexture(Point3f(0.9, 0.9, 0.9)));
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(checker));


			for (int a = -11; a < 11; a++) {
				for (int b = -11; b < 11; b++) {
					Float chooseMat = RND;
					Point3f center = Point3f(a + 0.9 * RND, 0.2, b + 0.9 * RND);
					if ((center - Point3f(4, 0.2, 0)).Length() > 0.9) {
						if (chooseMat < 0.7) { // 选择漫反射材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Lambertian(new ConstantTexture(Point3f(RND * RND, RND * RND, RND * RND))));
						}
						else if (chooseMat < 0.85) { // 选择金属材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Metal(new ConstantTexture(Point3f(0.5 * (1 + RND), 0.5 * (1 + RND), 0.5 * (1 + RND))), RND));
						}
						else if (chooseMat < 0.95) { // 选择玻璃材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1 + RND));
						}
						else { // 选择中空玻璃球材质
							shapes[curNum++] = new Sphere(
								center,
								0.2,
								new Dielectric(1.5));
							shapes[curNum++] = new Sphere(
								center,
								-0.15,
								new Dielectric(1.5));
						}
					}
				}

			}

			shapes[curNum++] = new Sphere(Point3f(0, 1, 0), 1.0, new Dielectric(1.5));
			shapes[curNum++] = new Cylinder(Point3f(2, .25, 3), 1.0, 0.0, 0.5, new Lambertian(new ConstantTexture(Point3f(0.2, 0.2, 0.8))));
			shapes[curNum++] = new DSphere(Point3f(2, 0.8, 3), Point3f(2, 1.2, 3), 0.0, 1.0, 0.3, new Lambertian(new ConstantTexture(Point3f(0.8, 0.2, 0.2))));
			shapes[curNum++] = new Sphere(Point3f(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Point3f(0.4, 0.2, 0.1))));
			shapes[curNum++] = new Sphere(Point3f(4, 1, 0), 1.0, new Metal(new ConstantTexture(Point3f(0.7, 0.6, 0.5)), 0.0));

			/*for (int i = 0; i < curNum + curNum / 2 + 1; i++) {
				nodes[i] = new BVHNode(shapes, curNum, nodes);
			}*/

			*rand_state = local_rand_state;

			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f); // 使用BVH 100s spp 1000
			//*world = new ShapeList(shapes, curNum);// 不使用BVH 675s spp 1000
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter3TextureScene2(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {

			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			Texture* checker = new CheckerTexture(new ConstantTexture(Point3f(0.607843, 0.34902, 0.713725)), new ConstantTexture(Point3f(0.9, 0.9, 0.9)));
			Texture* checker2 = new CheckerTexture(new ConstantTexture(Point3f(0.203922, 0.596078, 0.858824)), new ConstantTexture(Point3f(0.9, 0.9, 0.9)));
			shapes[curNum++] = new Sphere(Point3f(0, -11, 0), 10, new Lambertian(checker));
			shapes[curNum++] = new Cylinder(Point3f(0, 0, 0), 2, 0, 2, new Lambertian(checker2));
			shapes[curNum++] = new Sphere(Point3f(0, 11, 0), 10, new Lambertian(checker));
			/*for (int i = 0; i < curNum + curNum / 2 + 1; i++) {
				nodes[i] = new BVHNode(shapes, curNum, nodes);
			}*/

			*rand_state = local_rand_state;

			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f); // 使用BVH 100s spp 1000
			//*world = new ShapeList(shapes, curNum);// 不使用BVH 675s spp 1000
			printf("Create World Successful!\n");
		}
	}

	__global__ void Chapter4NoiseScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.0f);
			/*Texture* checker = new CheckerTexture(new ConstantTexture(Point3f(0.607843, 0.34902, 0.713725)), new ConstantTexture(Point3f(0.9, 0.9, 0.9)));
			Texture* checker2 = new CheckerTexture(new ConstantTexture(Point3f(0.203922, 0.596078, 0.858824)), new ConstantTexture(Point3f(0.9, 0.9, 0.9)));*/
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(pertext));
			//shapes[curNum++] = new Cylinder(Point3f(0, 0, 0), 2, 0, 2, new Lambertian(pertext));
			//shapes[curNum++] = new Sphere(Point3f(0, 2, 0), 2, new Lambertian(pertext));
			shapes[curNum++] = new Cylinder(Point3f(0, 2, 0), 2, 0, 4, new Lambertian(pertext));
			/*for (int i = 0; i < curNum + curNum / 2 + 1; i++) {
				nodes[i] = new BVHNode(shapes, curNum, nodes);
			}*/

			*rand_state = local_rand_state;

			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f); // 使用BVH 100s spp 1000
			//*world = new ShapeList(shapes, curNum);// 不使用BVH 675s spp 1000
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter5ImageScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 8, 3);
			Point3f lookAt = Point3f(0, 0, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 20.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* checker = new CheckerTexture(new ConstantTexture(Point3f(0.207843, 0.94902, 0.213725)), new ConstantTexture(Point3f(0.9, 0.9, 0.9)));
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.0f);
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(pertext));
			shapes[curNum++] = new Cylinder(Point3f(1, 1.5, 3), 1.0, 0, 3, new Lambertian(imgtext));
			shapes[curNum++] = new Sphere(Point3f(1, 1, 0), 1.1, new Dielectric(1.0));
			shapes[curNum++] = new Sphere(Point3f(1, 1, 0), 1, new Metal(checker, 1.0));
			shapes[curNum++] = new Sphere(Point3f(1, 0.7, -2.7), 0.7, new Lambertian(imgtext));



			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f); 
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter6LightScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(13, 2, 3);
			Point3f lookAt = Point3f(0, 1.5f, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0;
			Float fov = 40.0;
			Float focusDis = 10.0;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 0.1f);
			//Texture* light_intensity = new ConstantTexture(Point3f(4.f, 4.f, 4.f));
			//Material* light = new DiffuseLight(light_intensity);
			shapes[curNum++] = new Sphere(Point3f(0, -1000, 0), 1000, new Lambertian(pertext));
			shapes[curNum++] = new Cylinder(Point3f(1, 2, 4), 1.0, 0, 3, new DiffuseLight(new ConstantTexture(Point3f(.2f, 0.2f, 0.6f))));
			shapes[curNum++] = new Sphere(Point3f(0, 2, 0), 2, new Lambertian(imgtext));
			shapes[curNum++] = new Sphere(Point3f(0, 5, 0), .5, new DiffuseLight(new ConstantTexture(Point3f(0.6f, 0.2f, 0.2f))));
			shapes[curNum++] = new XYRect(3, 5, 1, 3, -2, new DiffuseLight(new ConstantTexture(Point3f(0.2f, 0.6f, 0.2f))));



			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter6LightScene2(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(278, 278, -800);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量
			
			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
			shapes[curNum++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white2);
			shapes[curNum++] = new XYRect(0.f, 555.f, 0.f, 555.f, 555.f, white3);



			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter7InstancesScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(278, 278, -800);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
			shapes[curNum++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white2);
			shapes[curNum++] = new XYRect(0.f, 555.f, 0.f, 555.f, 555.f, white3);
			shapes[curNum++] = new Box(Point3f(140.f, 10.f, 75.f), Point3f(285.f, 155.f, 220.f), new Dielectric(1.5f));
			shapes[curNum++] = new Box(Point3f(130.f, 0.f, 65.f), Point3f(295.f, 165.f, 230.f), new Dielectric(1.5f));
			shapes[curNum++] = new Cylinder(Point3f(430.f, 100.f, 150.f), 40.f, 0.f, 200.f, new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))));
			shapes[curNum++] = new Box(Point3f(265.f, 0.f, 295.f), Point3f(430.f, 330.f, 460.f),new Metal(new ConstantTexture(Point3f(0.73f, 0.73f, 0.93f)), 0.0f));



			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}



	__global__ void Chapter7InstancesScene2(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(278, 278, -800);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* imgtext2 = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.0f);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
			shapes[curNum++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white2);
			shapes[curNum++] = new XYRect(0.f, 555.f, 0.f, 555.f, 555.f, white3);


			Transform t00 = Translate(Vector3f(230, 380, 165)) * RotateY(-45) * Scale(40, 40, 40);
			Transform t01 = Translate(Vector3f(110, 40, 365)) * RotateY(-60) * Scale(80, 80, 40);
			shapes[curNum++] = new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.23f, 0.23f, 0.73f))), t00);
			shapes[curNum++] = new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Metal(imgtext2/*new ConstantTexture(Point3f(0.556863f, 0.266667f, 0.678431f))*/, 0.5f), t01);
			

			Transform t0 = Translate(Vector3f(230, 40, 165)) * RotateY(-90) * Scale(20, 20, 20);
			Transform t1 = Translate(Vector3f(430, 40, 165))  * Scale(20, 20, 20);
			Transform t2 = Translate(Vector3f(395, 40, 395)) * RotateY(15) * Scale(20, 40, 20);
			Transform t3 = Translate(Vector3f(230, 100, 165)) * RotateY(-18) * Scale(40, 40, 40);
			Transform t4 = Translate(Vector3f(355, 110, 395)) * RotateY(15) * Scale(40, 20, 20);
			shapes[curNum++] = new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Lambertian(imgtext), t0);
			shapes[curNum++] = new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Lambertian(new ConstantTexture(Point3f(0.556863f, 0.266667f, 0.678431f))), t1);
			shapes[curNum++] = new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Lambertian(pertext), t2);
			shapes[curNum++] = new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Dielectric(1.5), t3);
			shapes[curNum++] = new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Metal(new ConstantTexture(Point3f(0.0f, 0.9f, 0.0f)), 0.2f), t4);
			

			Transform t5 = Translate(Vector3f(230, 180, 165)) *  Scale(20, 20, 20);
			Transform t6 = Translate(Vector3f(375, 170, 395)) * Scale(20, 40, 20);
			Transform t7 = Translate(Vector3f(230, 240, 165)) *  Scale(40, 40, 40);
			Transform t8 = Translate(Vector3f(375, 240, 395)) * Scale(40, 20, 20);
			shapes[curNum++] = new DSphere(Point3f(0.f, 0.f, 0.f), Point3f(0.f, 1.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.9f, 0.3f, 0.9f))), t5);
			shapes[curNum++] = new DSphere(Point3f(0.f, 0.f, 0.f), Point3f(0.f, 0.f, 1.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.9f, 0.0f, 0.0f))), t6);
			shapes[curNum++] = new DSphere(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.0f, 0.9f))), t7);
			shapes[curNum++] = new DSphere(Point3f(0.f, 0.f, 0.f), Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.9f, 0.0f))),  t8);


			Transform t9 = Translate(Vector3f(130, 320, 165)) * Rotate(90, Vector3f(1, 1, 1)) * Scale(20, 20, 20);
			Transform t10 = Translate(Vector3f(335, 320, 395)) * RotateZ(90) * Scale(20, 40, 20);
			Transform t11 = Translate(Vector3f(330, 40, 165)) * RotateY(-18) * Scale(40, 40, 40);
			Transform t12 = Translate(Vector3f(465, 380, 395)) * RotateY(15) * Scale(40, 20, 20);
			shapes[curNum++] = new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.9f, 0.1f, 0.9f))), t9 );
			shapes[curNum++] = new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Metal(new ConstantTexture(Point3f(0.9f, 0.0f, 0.0f)), 0.2f), t10);
			shapes[curNum++] = new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.0f, 0.9f))), t11);
			shapes[curNum++] = new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.9f, 0.0f))), t12);

			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}


	__global__ void Chapter7InstancesScene3(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(278, 278, -800);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* imgtext2 = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.0f);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
			shapes[curNum++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white2);
			shapes[curNum++] = new XYRect(0.f, 555.f, 0.f, 555.f, 555.f, white3);


			Transform t00 = Translate(Vector3f(130, 0, 165)) * RotateY(-18) * Scale(165, 165, 165);
			Transform t01 = Translate(Vector3f(265, 40, 295)) * RotateY(15) * Scale(165, 330, 165);
			shapes[curNum++] = new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.23f, 0.23f, 0.73f))), t00);
			shapes[curNum++] = new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Metal(imgtext, 0.2f), t01);

			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}



	__global__ void Chapter8VolumeScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(278, 278, -800);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* imgtext2 = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.0f);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
			shapes[curNum++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white2);
			shapes[curNum++] = new XYRect(0.f, 555.f, 0.f, 555.f, 555.f, white3);


			Transform t00 = Translate(Vector3f(130, 0.0, 165)) * RotateY(-18) * Scale(165, 165, 165);
			Transform t01 = Translate(Vector3f(265, 0.0, 295)) * RotateY(15) * Scale(165, 330, 165);
			shapes[curNum++] = new ConstantMedium(new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))), t00), 0.01, new Isotropic(new ConstantTexture(Point3f(1.f, 1.f, 1.f))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))), t01), 0.01, new Isotropic(new ConstantTexture(Point3f(0.f, 0.f, 0.f))), &local_rand_state);
			//shapes[curNum++] = new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.23f, 0.23f, 0.73f))), t00);
			//shapes[curNum++] = new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Metal(imgtext, 0.2f), t01);

			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}



	__global__ void Chapter8VolumeScene2(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(278, 278, -800);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 5.0f);

			int curNum = 0; // 记录创建的Shape数量

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* imgtext2 = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.0f);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
			shapes[curNum++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
			shapes[curNum++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
			shapes[curNum++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white2);
			shapes[curNum++] = new XYRect(0.f, 555.f, 0.f, 555.f, 555.f, white3);


			Transform t00 = Translate(Vector3f(130, 0.0, 165)) * RotateY(-18) * Scale(165, 130, 165);
			Transform t01 = Translate(Vector3f(265, 0.0, 295)) * RotateY(15) * Scale(165, 170, 165);
			//shapes[curNum++] = new ConstantMedium(new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))), t00), 0.01, new Isotropic(new ConstantTexture(Point3f(1.f, 0.f, 0.f))), &local_rand_state);
			//shapes[curNum++] = new ConstantMedium(new Box(Point3f(0.f, 0.f, 0.f), Point3f(1.f, 1.f, 1.f), new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))), t01), 0.01, new Isotropic(new ConstantTexture(Point3f(0.f, 1.f, 0.f))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Cylinder(Point3f(0.5f, 0.5f, 0.5f), 0.5f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.0f, 0.9f))), t00), 0.01, new Isotropic(new ConstantTexture(Point3f(0.466667, 0.54902, 0.639216))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Cylinder(Point3f(0.5f, 0.5f, 0.5f), 0.5f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.9f, 0.0f))), t01), 0.01, new Isotropic(new ConstantTexture(Point3f(0.294118, 0.396078, 0.517647))), &local_rand_state);

			Transform t0 = Translate(Vector3f(230, 220, 265)) * Scale(80, 20, 40);
			Transform t1 = Translate(Vector3f(430, 250, 165)) * Scale(60, 20, 20);
			Transform t2 = Translate(Vector3f(170, 360, 295)) * Scale(100, 20, 40);
			Transform t3 = Translate(Vector3f(260, 270, 181)) * Scale(30, 20, 40);
			Transform t4 = Translate(Vector3f(400, 310, 350)) * Scale(70, 20, 40);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Lambertian(imgtext), t0), 0.04, new Isotropic(new ConstantTexture(Point3f(0.988235, 0.360784, 0.396078))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Lambertian(new ConstantTexture(Point3f(0.556863f, 0.266667f, 0.678431f))), t1), 0.02, new Isotropic(new ConstantTexture(Point3f(0.0352941, 0.517647, 0.890196))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Lambertian(pertext), t2), 0.05, new Isotropic(new ConstantTexture(Point3f(0.423529, 0.360784, 0.905882))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Dielectric(1.5), t3), 0.03, new Isotropic(new ConstantTexture(Point3f(0.333333, 0.937255, 0.768627))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0.f, 0.f, 0.f), 1.f, new Metal(new ConstantTexture(Point3f(0.0f, 0.9f, 0.0f)), 0.2f), t4), 0.01, new Isotropic(new ConstantTexture(Point3f(0.992157, 0.588235, 0.266667))), &local_rand_state);



			/*Transform t9 = Translate(Vector3f(130, 320, 165)) * Rotate(90, Vector3f(1, 1, 1)) * Scale(20, 20, 20);
			Transform t10 = Translate(Vector3f(335, 320, 395)) * RotateZ(90) * Scale(20, 40, 20);
			Transform t11 = Translate(Vector3f(330, 40, 165)) * RotateY(-18) * Scale(40, 40, 40);
			Transform t12 = Translate(Vector3f(465, 380, 395)) * RotateY(15) * Scale(40, 20, 20);
			shapes[curNum++] = new ConstantMedium(new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.9f, 0.1f, 0.9f))), t9), 0.01, new Isotropic(new ConstantTexture(Point3f(1.f, 1.f, 1.f))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Metal(new ConstantTexture(Point3f(0.9f, 0.0f, 0.0f)), 0.2f), t10), 0.01, new Isotropic(new ConstantTexture(Point3f(1.f, 1.f, 1.f))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.0f, 0.9f))), t11), 0.01, new Isotropic(new ConstantTexture(Point3f(1.f, 1.f, 1.f))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Cylinder(Point3f(0.f, 0.f, 0.f), 1.f, 0.f, 1.f, new Lambertian(new ConstantTexture(Point3f(0.0f, 0.9f, 0.0f))), t12), 0.01, new Isotropic(new ConstantTexture(Point3f(1.f, 1.f, 1.f))), &local_rand_state);*/


			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}

	__global__ void TestScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(478.0, 278.0, -600.0);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			int nb = 20;
			int ns = 10000;
			Material* ground = new Lambertian(new ConstantTexture(Point3f(0.48, 0.83, 0.53)));
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 10.f);


			shapes[curNum++] = new XZRect(123, 423, 147, 412, 554, new DiffuseLight(new ConstantTexture(Point3f(7.f, 7.f, 7.f))));
			shapes[curNum++] = new XZRect(0, 800, 0, 800, 0, ground);
			shapes[curNum++] = new DSphere(Point3f(300, 400, 200), Point3f(330, 400, 200), 0.f, 1.f, 50.f, new Lambertian(new ConstantTexture(Point3f(0.5, 0.5, 0.5))));
			/*shapes[curNum++] = new DSphere(Point3f(300, 200, 200), Point3f(300, 230, 200), 0.f, 1.f, 50.f, new Lambertian(new ConstantTexture(Point3f(0.7, 0.3, 0.1))));
			shapes[curNum++] = new DSphere(Point3f(200, 200, 200), Point3f(200, 200, 220), 0.f, 1.f, 50.f, new Lambertian(new ConstantTexture(Point3f(0.7, 0.3, 0.1))));*/


			shapes[curNum++] = new Sphere(Point3f(220, 280, 300), 80, new Lambertian(pertext));
			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}

	__global__ void RTNWScene(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(478.0, 278.0, -600.0);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			int nb = 20;
			int ns = 1000;

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* imgtext2 = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 0.1f);
			Material* ground = new Lambertian(new ConstantTexture(Point3f(0.48, 0.83, 0.53)));
			
			for (int i = 0; i < nb; i++) {
				for (int j = 0; j < nb; j++) {
					Float w = 100;
					Float x0 = -1000 + i * w;
					Float z0 = -1000 + j * w;
					Float y0 = 0;
					Float x1 = x0 + w;
					Float z1 = z0 + w;
					Float y1 = 100 * (curand_uniform(&local_rand_state) + 0.01);
					shapes[curNum++] = new Box(Point3f(x0, y0, z0), Point3f(x1, y1, z1), ground);
				}
			}

			shapes[curNum++] = new XZRect(123, 423, 147, 412, 554, new DiffuseLight(new ConstantTexture(Point3f(7.f, 7.f, 7.f))));
			shapes[curNum++] = new DSphere(Point3f(400, 400, 200), Point3f(430, 400, 200), 0.f, 1.f, 50.f, new Lambertian(new ConstantTexture(Point3f(0.7, 0.3, 0.1))));
			shapes[curNum++] = new Sphere(Point3f(260, 150, 45), 50, new Dielectric(1.5));
			shapes[curNum++] = new Sphere(Point3f(0, 150, 145), 50, new Metal(new ConstantTexture(Point3f(0.8, 0.8, 0.9)), 0.9));

			shapes[curNum++] = new Sphere(Point3f(360, 150, 145), 70, new Dielectric(1.5));
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(360, 150, 145), 70, new Dielectric(1.5)), 0.2, new Isotropic(new ConstantTexture(Point3f(0.2, 0.4, 0.9))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0, 0, 0), 5000, new Dielectric(1.5)), 0.0001, new Isotropic(new ConstantTexture(Point3f(1.0, 1.0, 1.0))), &local_rand_state);
			Transform trans = Translate(Vector3f(400, 200, 400)) * RotateY(90);/* * Rotate(15, Vector3f(82.5, srandPos.y, 82.5))*/;
			shapes[curNum++] = new Sphere(Point3f(0, 0, 0), 100, new Lambertian(imgtext), trans);
			shapes[curNum++] = new Sphere(Point3f(220, 280, 300), 80, new Lambertian(pertext));

			for (int i = 0; i < ns; i++) {
				Point3f srandPos = Point3f(165 * curand_uniform(&local_rand_state), 165 * curand_uniform(&local_rand_state), 165 * curand_uniform(&local_rand_state));
				Transform st = Translate(Vector3f(-100, 270, 395))/* * Rotate(15, Vector3f(82.5, srandPos.y, 82.5))*/;
				shapes[curNum++] = new  Sphere(srandPos, 10, new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))), st);
			}

			* rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}


	__global__ void RTNWScene2(Shape** shapes, Shape** nodes, Shape** world, Camera** camera, int width, int height, curandState* rand_state,
		cudaPitchedPtr image/*, cudaPitchedPtr image2*/) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			curandState local_rand_state = *rand_state;
			Point3f lookFrom = Point3f(478.0, 278.0, -600.0);
			Point3f lookAt = Point3f(278, 278, 0);
			Vector3f lookUp = Vector3f(0, 1, 0);
			Float aperture = 0.0f;
			Float fov = 40.0f;
			Float focusDis = 10.0f;
			Float screenWidth = width;
			Float screenHeight = height;
			Float aspect = screenWidth / screenHeight;
			*camera = new Camera(lookFrom, lookAt, lookUp, fov, aspect, aperture, focusDis, 0.0f, 1.0f);

			int curNum = 0; // 记录创建的Shape数量
			int nb = 20;
			int ns = 1000;

			Material* red = new Lambertian(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* redlight = new DiffuseLight(new ConstantTexture(Point3f(0.65f, 0.05f, 0.05f)));
			Material* white = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white2 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* white3 = new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f)));
			Material* green = new Lambertian(new ConstantTexture(Point3f(0.12f, 0.45f, 0.15f)));
			Material* light = new DiffuseLight(new ConstantTexture(Point3f(15.0f, 15.0f, 15.0f)));
			Texture* imgtext = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Texture* imgtext2 = new ImageTexture((unsigned char*)image.ptr, image.xsize, image.ysize);
			Perlin* noise = new Perlin(&local_rand_state);
			Texture* pertext = new NoiseTexture(noise, 0.1f);
			Material* ground = new Lambertian(new ConstantTexture(Point3f(0.48, 0.83, 0.53)));

			for (int i = 0; i < nb; i++) {
				for (int j = 0; j < nb; j++) {
					Float w = 100;
					Float x0 = -1000 + i * w;
					Float z0 = -1000 + j * w;
					Float y0 = 0;
					Float x1 = x0 + w;
					Float z1 = z0 + w;
					Float y1 = 100 * (curand_uniform(&local_rand_state) + 0.01);
					Float mm = curand_uniform(&local_rand_state);
					Float shapeType = curand_uniform(&local_rand_state);
					if (shapeType < 0.5) {
						
							shapes[curNum++] = new Box(Point3f(x0, y0, z0), Point3f(x1, y1, z1), new Lambertian(new ConstantTexture(Point3f(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state) / 2.f + 0.3, curand_uniform(&local_rand_state)))));
						
					}
					else {
							shapes[curNum++] = new Cylinder((Point3f(x0, y0, z0) + Point3f(x1, y1, z1)) / 2.f, w / 2.f, 0, (y1 - y0), new Lambertian(new ConstantTexture(Point3f(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state) / 2.f + 0.3, curand_uniform(&local_rand_state)))));
						
					}
				}
			}

			shapes[curNum++] = new XZRect(123, 423, 147, 412, 554, new DiffuseLight(new ConstantTexture(Point3f(10.f, 10.f, 10.f))));
			shapes[curNum++] = new DSphere(Point3f(400, 400, 200), Point3f(430, 400, 200), 0.f, 1.f, 50.f, new Lambertian(new ConstantTexture(Point3f(0.909804, 0.262745, 0.576471))));
			shapes[curNum++] = new Sphere(Point3f(260, 150, 45), 50, new Dielectric(1.5));
			shapes[curNum++] = new Sphere(Point3f(0, 150, 145), 50, new Metal(new ConstantTexture(Point3f(0.8, 0.8, 0.9)), 0.9));

			shapes[curNum++] = new Sphere(Point3f(360, 150, 145), 70, new Dielectric(1.5));
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(360, 150, 145), 70, new Dielectric(1.5)), 0.2, new Isotropic(new ConstantTexture(Point3f(0.423529, 0.360784, 0.905882))), &local_rand_state);
			shapes[curNum++] = new ConstantMedium(new Sphere(Point3f(0, 0, 0), 5000, new Dielectric(1.5)), 0.0001, new Isotropic(new ConstantTexture(Point3f(1.0, 1.0, 1.0))), &local_rand_state);
			Transform trans = Translate(Vector3f(400, 200, 400)) * RotateY(90);/* * Rotate(15, Vector3f(82.5, srandPos.y, 82.5))*/;
			shapes[curNum++] = new Sphere(Point3f(0, 0, 0), 100, new Lambertian(imgtext), trans);
			shapes[curNum++] = new Sphere(Point3f(220, 280, 300), 80, new Lambertian(pertext));

			for (int i = 0; i < ns; i++) {
				Point3f srandPos = Point3f(165 * curand_uniform(&local_rand_state), 165 * curand_uniform(&local_rand_state), 165 * curand_uniform(&local_rand_state));
				Transform st = Translate(Vector3f(-100, 270, 395))/* * Rotate(15, Vector3f(82.5, srandPos.y, 82.5))*/;
				shapes[curNum++] = new  Sphere(srandPos, 10, new Lambertian(new ConstantTexture(Point3f(0.73f, 0.73f, 0.73f))), st);
			}

			*rand_state = local_rand_state;
			*world = CreateBVHNode(shapes, curNum, nodes, &local_rand_state, 0.f, 0.f);
			printf("Create World Successful!\n");
		}
	}
}


#endif // QZRT_SCENE_EXAMPLE_H