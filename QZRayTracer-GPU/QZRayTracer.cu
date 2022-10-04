
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./src/core/QZRayTracer.h"
#include "./src/core/api.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "./src/core/stb_image.h"
#include "./src/core/stb_image_write.h"
#include "./src/scene/scene.h"
#include "src/scene/example.h"

using namespace raytracer;
using namespace std;
#define MAXBOUNDTIME 10
#define MAXNUMSHAPE 2000000
#define MAXNUMTEXTURE 20
#define MAXNUMMODELS 20
#define MAXNUMVETEX 1000000
#define STATICNUMSEEDS 1 // 用于创建场景使用的随机种子数量

// GPU Mode
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


__device__ Point3f Color(const Ray& r, Shape** world, curandState* local_rand_state) {
    Ray cur_ray = r;
    Point3f cur_attenuation = Point3f(1.0f, 1.0f, 1.0f);
    Point3f cur_emitted = Point3f(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < MAXBOUNDTIME; i++) {
        HitRecord rec;

        if ((*world)->Hit(cur_ray, rec)) {
            //return Point3f(rec.normal);
            //if (i > 1) {
            //    printf("rec.t:%f, i:%d\n", rec.t, i);
            //}
            Ray scattered;
            Point3f attenuation;
            Point3f emitted = rec.mat->Emitted(rec.u, rec.v, rec.p);
            Point3f target = rec.p + Point3f(rec.normal) + RandomInUnitSphere(local_rand_state);
            if (rec.mat->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = cur_attenuation * attenuation + emitted;
                cur_emitted = Point3f(emitted);
                cur_ray = scattered;
                // return cur_attenuation;
            }
            else {
                return cur_attenuation * emitted;
            }
        }
        else {
            /*Vector3f unit_direction = Normalize(cur_ray.d);
            float t = 0.5f * (unit_direction.y + 1.0f);
            Point3f c = Lerp(t, Point3f(1.0, 1.0, 1.0), Point3f(0.5, 0.7, 1.0));*/
            return cur_attenuation * cur_emitted;
        }
    }

    return Point3f(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(2022, pixel_index, 0, &rand_state[pixel_index]);
}


__global__ void render(Point3f* fb, int max_x, int max_y, int ns, Camera** cam, Shape** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Point3f color;
    for (int s = 0; s < ns; s++) {
        Float u = Float(i + curand_uniform(&local_rand_state)) / Float(max_x);
        Float v = Float(/*max_y -*/ j /*- 1*/ + curand_uniform(&local_rand_state)) / Float(max_y);
        Ray ray = (*cam)->GenerateRay(u, v, &local_rand_state);
        //printf("GetColor。。。\n");
        color += Color(ray, world, &local_rand_state);

        //printf("GetColor done\n");
    }
    rand_state[pixel_index] = local_rand_state;
    color /= Float(ns);
    //color = Clamp(color, 0.f, 1.f);
    //color = color / (color + Point3f(1.0, 1.0, 1.0));
    fb[pixel_index] = color;
}

int main() {
    int nx = 3840;
    int ny = 2160;
    int ns = 100;
    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // allocate FB
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Point3f);

    size_t size;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    //printf("size: %d\n", size);

    Point3f* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, STATICNUMSEEDS * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables
    Shape** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, MAXNUMSHAPE * sizeof(Shape*)));
    Shape** d_nodes;
    checkCudaErrors(cudaMalloc((void**)&d_nodes, MAXNUMSHAPE * sizeof(Shape*)));
    Shape** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Shape*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));


    // add image
    int image_width, image_height, image_channel;
    unsigned char* image = stbi_load("./resource/texture/brown_photostudio_02_4k.hdr", &image_width, &image_height, &image_channel, 0);
    dim3  image_dimensions = dim3(image_width, image_height, image_channel);
    cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(unsigned char) * image_dimensions.x, image_dimensions.y, image_dimensions.z);
    cudaPitchedPtr devicePitchedPointer;
    cudaMalloc3D(&devicePitchedPointer, volumeSizeBytes);
    cudaMemcpy(devicePitchedPointer.ptr, image, image_width * image_height * image_channel * sizeof(unsigned char), cudaMemcpyHostToDevice);
    stbi_image_free(image);

    // add model
    //TriangleMesh** triangleMeshs = new TriangleMesh*[MAXNUMMODELS];

    TriangleMesh** d_triangleMeshs;
    checkCudaErrors(cudaMalloc((void**)&d_triangleMeshs, MAXNUMMODELS * sizeof(TriangleMesh*)));
    int modelId = 0;

#pragma region LoadOBJ
    TriangleMeshStruct mesh;
    loadObj("./resource/model/suzanne.obj", mesh);
    int numVertices = mesh.verts.size();
    int numNormals = mesh.normals.size();
    int numUVWs = mesh.uvw.size();
    int numFaces = mesh.numFaces;
    int faceOffset = mesh.faceOffset;

    Float* vertex;
    checkCudaErrors(cudaMalloc((void**)&vertex, numVertices * 3 * sizeof(Float)));
    cudaMemcpy(vertex, mesh.verts.data(), numVertices * 3 * sizeof(Float), cudaMemcpyHostToDevice);
    Float* normals;
    checkCudaErrors(cudaMalloc((void**)&normals, numNormals * 3 * sizeof(Float)));
    cudaMemcpy(normals, mesh.normals.data(), numNormals * 3 * sizeof(Float), cudaMemcpyHostToDevice);
    Float* uvws;
    checkCudaErrors(cudaMalloc((void**)&uvws, numUVWs * 3 * sizeof(Float)));
    cudaMemcpy(uvws, mesh.uvw.data(), numUVWs * 3 * sizeof(Float), cudaMemcpyHostToDevice);
    int* faces;
    checkCudaErrors(cudaMalloc((void**)&faces, faceOffset * numFaces * sizeof(int)));
    checkCudaErrors(cudaMemcpy(faces, mesh.faces.data(), faceOffset * numFaces * sizeof(int), cudaMemcpyHostToDevice));

    LoadOBJ << <1, 1 >> > (d_triangleMeshs, modelId++, vertex, normals, uvws, faces, numVertices, numNormals, numUVWs, numFaces, faceOffset);
#pragma endregion

    
    //cudaMemcpy(d_triangleMeshs, triangleMeshs, modelId * sizeof(TriangleMesh*), cudaMemcpyHostToDevice);


    /*--------------------------更换自己的场景--------------------------*/
    ModelScene << <1, 1 >> > (d_list, d_nodes, d_world, d_camera, nx, ny, d_rand_state2, devicePitchedPointer, d_triangleMeshs, modelId);
    //RTNWScene2 << <1, 1 >> > (d_list, d_nodes, d_world, d_camera, nx, ny, d_rand_state2, devicePitchedPointer);
    //SampleScene<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    // create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny);
    /*------------------------------end--------------------------------*/

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";
    auto* data = (unsigned char*)malloc(nx * ny * 3);
    Float hdr_max = 1.f;
#ifdef HDR
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            hdr_max = Max(Max(Max(fb[pixel_index].x, fb[pixel_index].y), fb[pixel_index].z), hdr_max);
        }
    }
    printf("hdr_max:%f\n", hdr_max);
    hdr_max = 1.f / hdr_max;
#endif // HDR

    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            fb[pixel_index] *= hdr_max;
            fb[pixel_index] = Point3f(pow(fb[pixel_index].x, Gamma), pow(fb[pixel_index].y, Gamma), pow(fb[pixel_index].z, Gamma)); // gamma矫正
            Float max_axis = Max(fb[pixel_index].z, Max(fb[pixel_index].x, fb[pixel_index].y));
            fb[pixel_index] = fb[pixel_index] / (max_axis > 1 ? max_axis : 1); // gamma矫正
            int ir = int(255.99 * fb[pixel_index].x) % 257;
            int ig = int(255.99 * fb[pixel_index].y) % 257;
            int ib = int(255.99 * fb[pixel_index].z) % 257;
            size_t shadingPoint = ((ny - j - 1) * nx + i) * 3;
            data[shadingPoint + 0] = ir;
            data[shadingPoint + 1] = ig;
            data[shadingPoint + 2] = ib;
        }
    }
    // 写入图像
    raytracer::stbi_write_png("./output/CustomAdd/test.png", nx, ny, 3, data, 0);
    raytracer::stbi_image_free(data);

    // clean up
    checkCudaErrors(cudaGetLastError());;
    checkCudaErrors(cudaDeviceSynchronize());
    free_world_bvh << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaFree(d_nodes));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    //checkCudaErrors(cudaFree(d_triangleMeshs));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_list));
    //checkCudaErrors(cudaFree(d_textures));
    //checkCudaErrors(cudaFree(devicePitchedPointer));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}

#pragma endregion




