#ifndef  _QZRT_EXT_LOADOBJ_H_
#define  _QZRT_EXT_LOADOBJ_H_

#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

namespace raytracer {
	// Obj loader
	struct TriangleFace {
		int v[9]; // vertex indices
	};

	class float3 {
	public:
		float x;
		float y;
		float z;
	public:
		float3() { x = 0; y = 0; z = 0; }
		float3(float mx, float my, float mz) { x = mx; y = my; z = mz; }
		~float3() {}
	};

	struct TriangleMeshStruct {
		vector<float3> verts;
		vector<float3> normals;
		vector<float3> uvw;
		vector<TriangleFace> faces;
		int faceOffset = 9;
		int numFaces = 0;
	};

	//struct TriangleMeshStruct {
	//	float* verts;
	//	float* normals;
	//	float* uvw;
	//	int* faces;
	//	int faceOffset = 9;
	//	int numV, numN, numUVW, numF;

	//};


	int total_number_of_triangles = 0;

	int getCount(string mainStr, string subStr) {
		//确定主串，子串长度
		int i = mainStr.length(), j = subStr.length();
		if (i > j)//保证子串长度小于主串长度
		{
			int count = 0;//次数
			int pos = 0;//检索的起始位置
			int k = 0;//中间变量
			//子串在主串中从pos处开始最先出现的位置
			while ((k = mainStr.find(subStr, pos)) != -1) {
				count++;
				pos = k + j;//更新起始位置，检索位置+子串串长
			}
			return count;
		}
		return 0;
	}

	void loadObj(const std::string filename, TriangleMeshStruct& mesh);

	void loadObj(const std::string filename, TriangleMeshStruct& mesh) {
		//TriangleMeshTempStruct mesh;
		std::ifstream in(filename.c_str());

		if (!in.good()) {
			cout << "ERROR: loading obj:(" << filename << ") file is not good" << "\n";
			exit(0);
		}

		char buffer[256], str[255];
		float f1, f2, f3;

		while (!in.getline(buffer, 255).eof()) {
			buffer[255] = '\0';

			sscanf_s(buffer, "%s", str, 255);

			// reading a vertex
			if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
				if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3) {
					mesh.verts.push_back(float3(f1, f2, f3));
				}
				else {
					cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
					exit(-1);
				}
			}
			else if (buffer[0] == 'v' && buffer[1] == 'n' && (buffer[2] == ' ' || buffer[2] == 32)) {
				if (sscanf(buffer, "vn %f %f %f", &f1, &f2, &f3) == 3) {
					mesh.normals.push_back(float3(f1, f2, f3));
				}
				else {
					cout << "ERROR: normal not in wanted format in OBJLoader" << "\n";
					exit(-1);
				}
			}
			else if (buffer[0] == 'v' && buffer[1] == 't' && (buffer[2] == ' ' || buffer[2] == 32)) {
				if (sscanf(buffer, "vt %f %f", &f1, &f2) == 2) {
					mesh.uvw.push_back(float3(f1, f2, -1));
				}
				else {
					cout << "ERROR: uv not in wanted format in OBJLoader" << "\n";
					exit(-1);
				}
			}
			// reading FaceMtls 
			else if (buffer[0] == 'f') {
				TriangleFace f;
				//f v1 v2 v3 …, 仅由三个以上顶点索引组成；
				//f v1 / vt1 v2 / vt2 v3 / vt3 …, 由顶点和纹理索引组成；
				//f v1//vn1 v2//vn2 v3//vn3 …,由顶点和法线索引组成；
				//f v1 / vt1 / vn1 v2 / vt2 / vn2 v3 / vt3 / vn3 …, 由顶点, 纹理和法线索引组成；

				int numDSlash = getCount(buffer, "//");
				int numSlash = getCount(buffer, "/");

				if (numDSlash == 0 && numSlash == 0) {
					int nt = sscanf(buffer, "f %d %d %d", &f.v[0], &f.v[1], &f.v[2]);
					f.v[3] = -1;
					f.v[4] = -1;
					f.v[5] = -1;
					f.v[6] = -1;
					f.v[7] = -1;
					f.v[8] = -1;
					if (nt != 3) {
						cout << "ERROR: I don't know the format of that FaceMtl -- V" << "\n";
						exit(-1);
					}
				}
				else if (numDSlash == 0 && numSlash == 6) {
					int nt = sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d",
						&f.v[0], &f.v[3], &f.v[6],
						&f.v[1], &f.v[4], &f.v[7],
						&f.v[2], &f.v[5], &f.v[8]);
					if (nt != 9) {
						cout << "ERROR: I don't know the format of that FaceMtl -- V/N/T" << "\n";
						exit(-1);
					}
				}
				else if (numDSlash == 0 && numSlash == 3) {
					int nt = sscanf(buffer, "f %d/%d %d/%d %d/%d",
						&f.v[0], &f.v[6],
						&f.v[1], &f.v[7],
						&f.v[2], &f.v[8]);
					f.v[3] = -1;
					f.v[4] = -1;
					f.v[5] = -1;
					if (nt != 6) {
						cout << "ERROR: I don't know the format of that FaceMtl -- V/T" << "\n";
						exit(-1);
					}
				}
				else if (numDSlash == 3) {
					int nt = sscanf(buffer, "f %d//%d %d//%d %d//%d",
						&f.v[0], &f.v[3],
						&f.v[1], &f.v[4],
						&f.v[2], &f.v[5]);
					f.v[6] = -1;
					f.v[7] = -1;
					f.v[8] = -1;
					if (nt != 6) {
						cout << "ERROR: I don't know the format of that FaceMtl -- V//N" << "\n";
						exit(-1);
					}
				}
				else {
					cout << "ERROR: I don't know the format of that FaceMtl  !!!!!!" << "\n";
					continue;
				}
				mesh.faces.push_back(f);
				//for (int i = 0; i < mesh.faceOffset; i++) {
				//	mesh.faces.push_back(f.v[i]);
				//}

			}
		}

		int maxP = 0, maxN = 0, maxUVW = 0;
		for (int i = 0; i < mesh.faces.size(); i++) {
			maxP = max(max(mesh.faces[i].v[0], mesh.faces[i].v[1]), mesh.faces[i].v[2]);
			maxN = max(max(mesh.faces[i].v[3], mesh.faces[i].v[4]), mesh.faces[i].v[5]);
			maxUVW = max(max(mesh.faces[i].v[6], mesh.faces[i].v[7]), mesh.faces[i].v[8]);
		}
		printf("maxP = %d, maxN = %d, maxUVW = %d\n", maxP, maxN, maxUVW);
		mesh.numFaces = mesh.faces.size();

		cout << "----------obj file loaded-------------" << endl;
		cout << "number of faces:" << mesh.numFaces << " number of vertices:" << mesh.verts.size()
			<< " number of normals:" << mesh.normals.size() << " number of uvw:" << mesh.uvw.size() << endl;
	}
}



#endif 