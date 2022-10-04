#ifndef QZRT_SHAPE_TRANGLE_H
#define QZRT_SHAPE_TRANGLE_H
#include "../core/shape.h"

namespace raytracer {


	class Triangle :public Shape {
	public:
		// Triangle Private Data
		const TriangleMesh* mesh;
		const int faceIndex;

		Point3f p0, p1, p2;
		Point3f uvw0, uvw1, uvw2;
		Normal3f n0, n1, n2;
		__device__ Triangle(const TriangleMesh* mesh, const int triNumber, Material* mat, const  Transform& _trans = Transform())
			: mesh(mesh), faceIndex(triNumber) {
			material = mat;
			transform = _trans;

			
			p0 = Point3f(mesh->v[mesh->faceIndices[faceIndex * mesh->faceOffset] - 1]);
			p1 = Point3f(mesh->v[mesh->faceIndices[faceIndex * mesh->faceOffset + 1] - 1]);
			p2 = Point3f(mesh->v[mesh->faceIndices[faceIndex * mesh->faceOffset + 2] - 1]);

			if ((mesh->faceIndices[faceIndex * mesh->faceOffset + 3] >= 1 && mesh->faceIndices[faceIndex * mesh->faceOffset + 3] < mesh->nNormals) &&
				(mesh->faceIndices[faceIndex * mesh->faceOffset + 4] >= 1 && mesh->faceIndices[faceIndex * mesh->faceOffset + 4] < mesh->nNormals) &&
				(mesh->faceIndices[faceIndex * mesh->faceOffset + 5] >= 1 && mesh->faceIndices[faceIndex * mesh->faceOffset + 5] < mesh->nNormals)) {
				// 如果模型存在法线信息
				n0 = Normal3f(mesh->n[mesh->faceIndices[faceIndex * mesh->faceOffset + 3] - 1]);
				n1 = Normal3f(mesh->n[mesh->faceIndices[faceIndex * mesh->faceOffset + 4] - 1]);
				n2 = Normal3f(mesh->n[mesh->faceIndices[faceIndex * mesh->faceOffset + 5] - 1]);
			}
			else { // 不存在就通过点来叉乘计算，这里按照右手坐标系来计算的

				Vector3f dp02 = p0 - p2, dp12 = p1 - p2;
				Normal3f normal = Normal3f(Normalize(Cross(dp02, dp12)));

				n0 = Normal3f(normal);
				n1 = Normal3f(normal);
				n2 = Normal3f(normal);

			}

			Vector3f dp02 = p0 - p2, dp12 = p1 - p2;
			Normal3f normal = Normal3f(Normalize(Cross(dp02, dp12)));

			n0 = Normal3f(normal);
			n1 = Normal3f(normal);
			n2 = Normal3f(normal);

			if (mesh->faceIndices[faceIndex * mesh->faceOffset + 6] >= 1 && mesh->faceIndices[faceIndex * mesh->faceOffset + 6] < mesh->nUVWs &&
				mesh->faceIndices[faceIndex * mesh->faceOffset + 7] >= 1 && mesh->faceIndices[faceIndex * mesh->faceOffset + 7] < mesh->nUVWs &&
				mesh->faceIndices[faceIndex * mesh->faceOffset + 8] >= 1 && mesh->faceIndices[faceIndex * mesh->faceOffset + 8] < mesh->nUVWs) { // 如果有uv信息
				uvw0 = Point3f(mesh->uvw[mesh->faceIndices[faceIndex * mesh->faceOffset + 6] - 1]);
				uvw1 = Point3f(mesh->uvw[mesh->faceIndices[faceIndex * mesh->faceOffset + 7] - 1]);
				uvw2 = Point3f(mesh->uvw[mesh->faceIndices[faceIndex * mesh->faceOffset + 8] - 1]);
			}
			else {
				uvw0 = Point3f(0, 0, 0);
				uvw1 = Point3f(1, 0, 0);
				uvw2 = Point3f(1, 1, 0);
			}
			
				

			

		}
		// 通过 Shape 继承
		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec) const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;
	};
	__device__ inline bool Triangle::Hit(const Ray& ray, HitRecord& rec) const {
		Transform invTrans = Inverse(transform);
		Ray tansRay = Ray(invTrans(ray.o), invTrans(Normalize(ray.d)), ray.time, ray.tMax, ray.tMin);


		// 转换坐标系
		Point3f p0t = p0 - Vector3f(tansRay.o);
		Point3f p1t = p1 - Vector3f(tansRay.o);
		Point3f p2t = p2 - Vector3f(tansRay.o);

		int kz = MaxDimension(Abs(tansRay.d));
		int kx = kz + 1; if (kx == 3) kx = 0;
		int ky = kx + 1; if (ky == 3) ky = 0;
		Vector3f d = Permute(tansRay.d, kx, ky, kz);
		p0t = Permute(p0t, kx, ky, kz);
		p1t = Permute(p1t, kx, ky, kz);
		p2t = Permute(p2t, kx, ky, kz);

		Float Sx = -d.x / d.z;
		Float Sy = -d.y / d.z;
		Float Sz = 1.f / d.z;
		p0t.x += Sx * p0t.z;
		p0t.y += Sy * p0t.z;
		p1t.x += Sx * p1t.z;
		p1t.y += Sy * p1t.z;
		p2t.x += Sx * p2t.z;
		p2t.y += Sy * p2t.z;

		// 判断是否相交
		Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
		Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
		Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

		if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0)) {
			//printf("Hit1!\n");
			return false;
		}
		Float det = e0 + e1 + e2;
		if (abs(det) < ShadowEpsilon) {
			//printf("Hit2!\n");
			return false;
		}
		
		// 计算 t (通过重心坐标的原理插值)
		p0t.z *= Sz;
		p1t.z *= Sz;
		p2t.z *= Sz;
		Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
		if (det < 0 && (tScaled >= tansRay.tMin || tScaled < tansRay.tMax * det)) {
			//printf("Hit3!\n");
			return false;
		}
		else if (det > 0 && (tScaled <= tansRay.tMin || tScaled > tansRay.tMax * det)) {
			//printf("Hit4!\n");
			return false;
		}

		Float invDet = 1 / det;
		Float t = tScaled * invDet;
		//Float erro = transform(Vector3f(ShadowEpsilon, ShadowEpsilon, ShadowEpsilon)).LengthSquared();

		if (t < ShadowEpsilon * 100) return false;
		Float b0 = e0 * invDet;
		Float b1 = e1 * invDet;
		Float b2 = e2 * invDet;

		// Interpolate $(u,v)$ parametric coordinates and hit point
		Point3f pHit = b0 * p0 + b1 * p1 + b2 * p2;
		Point3f tempHit = ray(t);
		if ((pHit - tempHit).LengthSquared() < ShadowEpsilon)return false;
		Float u = b0 * uvw0.x + b1 * uvw1.x + b2 * uvw2.x;
		Float v = b0 * uvw0.y + b1 * uvw1.y + b2 * uvw2.y;

		//Vector3f dp02 = p0 - p2, dp12 = p1 - p2;
		//Normal3f normal = Normal3f(Cross(dp02, dp12));

		Normal3f pNormal = b0 * n0 + b1 * n1 + b2 * n2;
		if (Dot(tansRay.d, pNormal) > 0) {
			//printf("reverse normal!\n");
			pNormal = -pNormal;
		}
		
		rec.mat = material;
		rec.p = transform(pHit);
		rec.t = t;
		rec.t0 = t;
		rec.t1 = t;
		rec.u = u; 
		rec.v = v;
		rec.normal = Normalize(transform(pNormal));

		return true;
	}
	__device__ inline bool Triangle::BoundingBox(Bounds3f& box) const {
		box = transform(Union(Bounds3f(p0, p1), Bounds3f(p1, p2)));
		return true;
	}


	__device__ inline bool CreateModel(Shape** shapes, TriangleMesh* mesh, int& curNum, Material* mat, const  Transform& transform = Transform()) {

		printf("Test\n"); 
		int percent = int(mesh->nTriangles / 100);
		int curPer = 1;
		for (int i = 0; i < mesh->nTriangles; i++) {
			shapes[curNum++] = new Triangle(mesh, i, mat, transform);
			if (i == curPer * percent) {
				curPer++;
				printf("\rLoad the model to GPU : %d\%", curPer);
			}
		}
		printf("\rLoad the model to GPU : 100\%\n");
		return true;
	}
}
#endif // QZRT_SHAPE_TRANGLE_H