#ifndef  _QZRT_SCENE_LOADOBJ_H_
#define  _QZRT_SCENE_LOADOBJ_H_

#include <iostream>
#include "../core/QZRayTracer.h"
#include "../ext/load_obj.h"
#include "../core/api.h"

using namespace raytracer;
using namespace std;


void LoadObj(char* fileName, TriangleMesh** triangleMesh, const int& modelId) {
	//TriangleMeshStruct mesh;
	//printf("Loading the %dth model meshes...\n", modelId + 1);
	//loadObj("./resource/model/bunny.obj", mesh);
	//int numVertices = mesh.verts.size();
	//int numNormals = mesh.normals.size();
	//int numUVWs = mesh.uvw.size();
	//int numFaces = mesh.faces.size();
	//int faceOffset = mesh.faceOffset;
	//Point3f* curVertices = new Point3f[numVertices];
	//Normal3f* curNormals = new Normal3f[numNormals];
	//Point3f* curUVWs = new Point3f[numUVWs];
	//int* faces = new int[numFaces * faceOffset];

	//for (int i = 0; i < numVertices; i++) {
	//	curVertices[i] = Point3f(mesh.verts[i].x, mesh.verts[i].y, mesh.verts[i].z);
	//	if(numNormals > 0)
	//		curNormals[i] = Normal3f(mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z);
	//	if (numUVWs > 0)
	//		curUVWs[i] = Point3f(mesh.uvw[i].x, mesh.uvw[i].y, mesh.uvw[i].z);
	//}

	//for (int i = 0; i < numFaces; i++) {
	//	for (int j = 0; j < faceOffset; j++) {
	//		faces[i * faceOffset + j] = mesh.faces[i].v[j];
	//	}
	//}

	//triangleMesh[modelId] = new TriangleMesh(numFaces, faceOffset, numVertices, curVertices, curNormals, curUVWs, faces);
	//printf("Load the %dth model meshes sucessfully!\n", modelId + 1);
}



#endif
