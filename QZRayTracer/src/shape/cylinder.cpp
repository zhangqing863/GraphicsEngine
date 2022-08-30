#include "cylinder.h"

namespace raytracer {
	bool Cylinder::Hit(const Ray& ray, HitRecord& rec) const {
		Float dx = ray.d.x;
		Float dz = ray.d.z;
		Float cox = ray.o.x - center.x;
		Float coz = ray.o.z - center.z;

		Float a = dx * dx + dz * dz;
		Float b = 2 * (dx * cox + coz * dz);
		Float c = cox * cox + coz * coz - radius * radius;
		Float discriminant = b * b - 4 * a * c;
		Float t0, t1;
		if (!Quadratic(a, b, c, t0, t1)) return false;

		if (t0 > ray.tMax || t1 <= ShadowEpsilon) return false;
		Float tShapeHit = t0;
		// Point3f pHit = ray(tShapeHit);
		Point3f pHit;
		Normal3f normal;
		// case1 : 首先打到侧边朝前，表明光线从圆柱体外面射入的
		if (t0 >= ShadowEpsilon) {
			pHit = ray(tShapeHit);
			// 如果没打到，考虑上下两端能否打到
			if (pHit.y < zMin || pHit.y > zMax) {
				if (ray.o.y > zMax) {
					tShapeHit = (zMax - ray.o.y) / ray.d.y; // 击中上面时，光线需要朝下
				}
				else if (ray.o.y < zMin) {
					tShapeHit = (zMin - ray.o.y) / ray.d.y; // 击中下面时，光线需要朝上
				}
				else {
					return false;
				}
				if (tShapeHit < 0) return false; // 表示光线的方向无法到达这个平面
				pHit = ray(tShapeHit);
				// 在上下方击中的点如果没在圆柱的横截面范围内，则也表示没击中
				if ((pHit.x - center.x) * (pHit.x - center.x) + (pHit.z - center.z) * (pHit.z - center.z) > radius * radius) {
					return false;
				}
				pHit.y = ray.o.y > zMax ? zMax : zMin;
				normal = Normalize(Normal3f(pHit - Point3f(pHit.x, center.y, pHit.z)));
			}
			else {
				// 精确数值
				Float hitRad = std::sqrt((pHit.x - center.x) * (pHit.x - center.x) + (pHit.z - center.z) * (pHit.z - center.z));
				pHit.x *= radius / hitRad;
				pHit.z *= radius / hitRad;
				normal = Normalize(Normal3f(pHit - Point3f(center.x, pHit.y, center.z)));
			}
		}
		// case2 : 在圆柱内部
		else{
			if (ray.o.y < zMax && ray.o.y > zMin) {
				// 尝试能否击中上端
				tShapeHit = (zMax - ray.o.y) / ray.d.y; // 击中上面时，光线需要朝上，tShapeHit > 0
				if (tShapeHit < 0) { // 如果上面没击中，那么说明光线肯定是朝下的，可以直接计算朝下能否击中
					tShapeHit = (zMin - ray.o.y) / ray.d.y; // 击中下面时，光线需要朝下
				}
				if (tShapeHit < 0) return false; // 表示光线的方向无法到达这个平面
				pHit = ray(tShapeHit);
				// 在上下方击中的点如果没在圆柱的横截面范围内，则也表示没击中
				if ((pHit.x - center.x) * (pHit.x - center.x) + (pHit.z - center.z) * (pHit.z - center.z) > radius * radius) {
					return false;
				}
				pHit.y = ray.d.y > 0 ? zMax : zMin;
				normal = Normalize(Normal3f(Point3f(pHit.x, center.y, pHit.z) - pHit));
			}
			// 判断t1能否击中，即从内部击中其侧面
			else {
				tShapeHit = t1;
				if (t1 > ray.tMax) return false;
				pHit = ray(tShapeHit);

				if (pHit.y >= zMin && pHit.y <= zMax) {
					Float hitRad = std::sqrt(pHit.x * pHit.x + pHit.z * pHit.z);
					pHit.x *= radius / hitRad;
					pHit.z *= radius / hitRad;
					normal = Normalize(Normal3f(Point3f(center.x, pHit.y, center.z) - pHit));
				}
				else {
					return false;
				}

			}
		}

		rec.t = tShapeHit;
		rec.p = pHit;
		rec.normal = normal;
		rec.mat = material;

		return true;
	}
	std::shared_ptr<Shape> CreateCylinderShape(Point3f center, Float radius, Float zMin, Float zMax, std::shared_ptr<Material> material){
		return std::make_shared<Cylinder>(center, radius, zMin, zMax, material);
	}
}
