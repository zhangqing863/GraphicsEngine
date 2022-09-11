#ifndef QZRT_SHAPE_CYLINDER_H
#define QZRT_SHAPE_CYLINDER_H
#include "../core/shape.h"

namespace raytracer {
	class Cylinder :public Shape {
	public:
		Point3f center;
		Float radius;
		Float zMin, zMax;
		Float invRadius;
		//Material* material;

		__device__ Cylinder() :center(Point3f(0, 0, 0)), radius(1.0), invRadius(1.0), zMin(0), zMax(0) {}
		__device__ Cylinder(Point3f center, Float radius, Float bottom, Float top, Material* mat)
			:center(center), zMin(Min(bottom, top)), zMax(Max(bottom, top)), radius(radius){
			material = mat;
			Float halfHeight = (zMax - zMin) * .5f;
			this->zMin = center.y - halfHeight;
			this->zMax = center.y + halfHeight;
			invRadius = 1.0 / radius;
		};
		// ͨ�� Shape �̳�
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// ͨ�� Shape �̳�
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};

	__device__ inline bool Cylinder::Hit(const Ray& ray, HitRecord& rec) const {
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
		// case1 : ���ȴ򵽲�߳�ǰ���������ߴ�Բ�������������
		if (t0 >= ShadowEpsilon) {
			pHit = ray(tShapeHit);
			// ���û�򵽣��������������ܷ��
			if (pHit.y < zMin || pHit.y > zMax) {
				if (ray.o.y > zMax) {
					tShapeHit = (zMax - ray.o.y) / ray.d.y; // ��������ʱ��������Ҫ����
				}
				else if (ray.o.y < zMin) {
					tShapeHit = (zMin - ray.o.y) / ray.d.y; // ��������ʱ��������Ҫ����
				}
				else {
					return false;
				}
				if (tShapeHit < 0) return false; // ��ʾ���ߵķ����޷��������ƽ��
				pHit = ray(tShapeHit);
				// �����·����еĵ����û��Բ���ĺ���淶Χ�ڣ���Ҳ��ʾû����
				if ((pHit.x - center.x) * (pHit.x - center.x) + (pHit.z - center.z) * (pHit.z - center.z) > radius * radius) {
					return false;
				}
				pHit.y = ray.o.y > zMax ? zMax : zMin;
				normal = Normalize(Normal3f(pHit - Point3f(pHit.x, center.y, pHit.z)));
			}
			else {
				// ��ȷ��ֵ
				Float hitRad = std::sqrt((pHit.x - center.x) * (pHit.x - center.x) + (pHit.z - center.z) * (pHit.z - center.z));
				pHit.x *= radius / hitRad;
				pHit.z *= radius / hitRad;
				normal = Normalize(Normal3f(pHit - Point3f(center.x, pHit.y, center.z)));
			}
		}
		// case2 : ��Բ���ڲ�
		else {
			if (ray.o.y < zMax && ray.o.y > zMin) {
				// �����ܷ�����϶�
				tShapeHit = (zMax - ray.o.y) / ray.d.y; // ��������ʱ��������Ҫ���ϣ�tShapeHit > 0
				if (tShapeHit < 0) { // �������û���У���ô˵�����߿϶��ǳ��µģ�����ֱ�Ӽ��㳯���ܷ����
					tShapeHit = (zMin - ray.o.y) / ray.d.y; // ��������ʱ��������Ҫ����
				}
				if (tShapeHit < 0) return false; // ��ʾ���ߵķ����޷��������ƽ��
				pHit = ray(tShapeHit);
				// �����·����еĵ����û��Բ���ĺ���淶Χ�ڣ���Ҳ��ʾû����
				if ((pHit.x - center.x) * (pHit.x - center.x) + (pHit.z - center.z) * (pHit.z - center.z) > radius * radius) {
					return false;
				}
				pHit.y = ray.d.y > 0 ? zMax : zMin;
				normal = Normalize(Normal3f(Point3f(pHit.x, center.y, pHit.z) - pHit));
			}
			// �ж�t1�ܷ���У������ڲ����������
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

	__device__ inline bool Cylinder::BoundingBox(Bounds3f& box) const {
		box = Bounds3f(Point3f(center.x - radius, zMin, center.z - radius), Point3f(center.x + radius, zMax, center.z + radius));
		return true;
	}
	// Shape* CreateCylinderShape(Point3f center, Float radius, Float zMin, Float zMax, Material* material);
}
#endif // QZRT_SHAPE_CYLINDER_H
