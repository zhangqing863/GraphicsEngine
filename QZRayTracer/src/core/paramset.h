#ifndef QZRT_CORE_PARAMSET_H
#define QZRT_CORE_PARAMSET_H
#include "QZRayTracer.h"
#include "geometry.h"
#include "shape.h"
#include "camera.h"
namespace raytracer {
	class ParamSet {
    public:
        // ParamSet Public Methods
        ParamSet() {}
        

        Point3f lookAt, lookFrom;

    };

    struct RendererSet {
        RendererSet(Camera cam, Float resWidth, Float resHeight, int spp, const char* savePath, std::shared_ptr<Shape> shapes) {
            camera = cam;
            width = resWidth;
            height = resHeight;
            this->spp = spp;
            this->savePath = savePath;
            this->shapes = shapes;
        }
        Camera camera;
        Float width, height;
        int spp;
        const char* savePath;
        std::shared_ptr<Shape> shapes;
    };

}
#endif // QZRT_CORE_PARAMSET_H
