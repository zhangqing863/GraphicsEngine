#ifndef QZRT_CORE_QZRAYTRACER_H
#define QZRT_CORE_QZRAYTRACER_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


namespace raytracer {
#ifdef PBRT_FLOAT_AS_DOUBLE
	typedef double Float;
#else
	typedef float Float;
#endif  // PBRT_FLOAT_AS_DOUBLE


}

#endif  // QZRT_CORE_QZRAYTRACER_H