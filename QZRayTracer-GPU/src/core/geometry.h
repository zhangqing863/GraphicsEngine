#ifndef QZRT_CORE_GEOMETRY_H
#define QZRT_CORE_GEOMETRY_H

#include "QZRayTracer.h"

namespace raytracer {

    /*template <typename T>
    __host__ __device__ inline bool isNaN(const T x) {
        return std::isnan(x);
    }
    template <>
    __host__ __device__ inline bool isNaN(const int x) {
        return false;
    }*/

    // Vector Declarations
    template <typename T>
    class Vector2 {
    public:
        // Vector2 Public Methods
        __host__ __device__ Vector2() { x = y = 0; }
        __host__ __device__ Vector2(T xx, T yy) : x(xx), y(yy) { /*DCHECK(!HasNaNs());*/ }
        //__host__ __device__ bool HasNaNs() const { return isNaN(x) || isNaN(y); }
        __host__ __device__ explicit Vector2(const Point2<T>& p);
        __host__ __device__ explicit Vector2(const Point3<T>& p);
#ifndef NDEBUG
        // The default versions of these are fine for release builds; for debug
        // we define them so that we can add the Assert checks.
        __host__ __device__ Vector2(const Vector2<T>& v) {
            //DCHECK(!v.HasNaNs());
            x = v.x;
            y = v.y;
        }
        __host__ __device__ Vector2<T>& operator=(const Vector2<T>& v) {
            //DCHECK(!v.HasNaNs());
            x = v.x;
            y = v.y;
            return *this;
        }
#endif  // !NDEBUG

        __host__ __device__ Vector2<T> operator+(const Vector2<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Vector2(x + v.x, y + v.y);
        }

        __host__ __device__ Vector2<T>& operator+=(const Vector2<T>& v) {
            //DCHECK(!v.HasNaNs());
            x += v.x;
            y += v.y;
            return *this;
        }
        __host__ __device__ Vector2<T> operator-(const Vector2<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Vector2(x - v.x, y - v.y);
        }

        __host__ __device__ Vector2<T>& operator-=(const Vector2<T>& v) {
            //DCHECK(!v.HasNaNs());
            x -= v.x;
            y -= v.y;
            return *this;
        }
        __host__ __device__ bool operator==(const Vector2<T>& v) const { return x == v.x && y == v.y; }
        __host__ __device__ bool operator!=(const Vector2<T>& v) const { return x != v.x || y != v.y; }
        template <typename U>
        __host__ __device__ Vector2<T> operator*(U f) const {
            return Vector2<T>(f * x, f * y);
        }

        template <typename U>
        __host__ __device__ Vector2<T>& operator*=(U f) {
            //DCHECK(!isNaN(f));
            x *= f;
            y *= f;
            return *this;
        }
        template <typename U>
        __host__ __device__ Vector2<T> operator/(U f) const {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            return Vector2<T>(x * inv, y * inv);
        }

        template <typename U>
        __host__ __device__ Vector2<T>& operator/=(U f) {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            x *= inv;
            y *= inv;
            return *this;
        }
        __host__ __device__ Vector2<T> operator-() const { return Vector2<T>(-x, -y); }
        T operator[](int i) const {
            //DCHECK(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }

        T& operator[](int i) {
            //DCHECK(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }
        __host__ __device__ Float LengthSquared() const { return x * x + y * y; }
        __host__ __device__ Float Length() const { return std::sqrt(LengthSquared()); }

        // Vector2 Public Data
        T x, y;
    };

    template <typename T>
    __host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Vector2<T>& v) {
        os << "[ " << v.x << ", " << v.y << " ]";
        return os;
    }

    template <typename T>
    class Vector3 {
    public:
        // Vector3 Public Methods
        __host__ __device__ T operator[](int i) const {
            //DCHECK(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }
        __host__ __device__ T& operator[](int i) {
            //DCHECK(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }
        __host__ __device__ Vector3() { x = y = z = 0; }
        __host__ __device__ Vector3(T x, T y, T z) : x(x), y(y), z(z) { /*DCHECK(!HasNaNs());*/ }
        //__host__ __device__ bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
        __host__ __device__ explicit Vector3(const Point3<T>& p);

#ifndef NDEBUG
        // The default versions of these are fine for release builds; for debug
        // we define them so that we can add the Assert checks.
        __host__ __device__ Vector3(const Vector3<T>& v) {
            //DCHECK(!v.HasNaNs());
            x = v.x;
            y = v.y;
            z = v.z;
        }

        __host__ __device__ Vector3<T>& operator=(const Vector3<T>& v) {
            //DCHECK(!v.HasNaNs());
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }
#endif  // !NDEBUG
        __host__ __device__ Vector3<T> operator+(const Vector3<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Vector3(x + v.x, y + v.y, z + v.z);
        }
        __host__ __device__ Vector3<T>& operator+=(const Vector3<T>& v) {
            //DCHECK(!v.HasNaNs());
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }
        __host__ __device__ Vector3<T> operator-(const Vector3<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Vector3(x - v.x, y - v.y, z - v.z);
        }
        __host__ __device__ Vector3<T>& operator-=(const Vector3<T>& v) {
            //DCHECK(!v.HasNaNs());
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }
        __host__ __device__ bool operator==(const Vector3<T>& v) const {
            return x == v.x && y == v.y && z == v.z;
        }
        __host__ __device__ bool operator!=(const Vector3<T>& v) const {
            return x != v.x || y != v.y || z != v.z;
        }
        template <typename U>
        __host__ __device__ Vector3<T> operator*(U s) const {
            return Vector3<T>(s * x, s * y, s * z);
        }
        template <typename U>
        __host__ __device__ Vector3<T>& operator*=(U s) {
            //DCHECK(!isNaN(s));
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }
        template <typename U>
        __host__ __device__ Vector3<T> operator/(U f) const {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            return Vector3<T>(x * inv, y * inv, z * inv);
        }

        template <typename U>
        __host__ __device__ Vector3<T>& operator/=(U f) {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }
        __host__ __device__ Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }
        __host__ __device__ Float LengthSquared() const { return x * x + y * y + z * z; }
        __host__ __device__ Float Length() const { return std::sqrt(LengthSquared()); }
        __host__ __device__ explicit Vector3(const Normal3<T>& n);

        // Vector3 Public Data
        T x, y, z;
    };

    template <typename T>
    __host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Vector3<T>& v) {
        os << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
        return os;
    }


    typedef Vector2<Float> Vector2f;
    typedef Vector2<int> Vector2i;
    typedef Vector3<Float> Vector3f;
    typedef Vector3<int> Vector3i;

    // Point Declarations
    template <typename T>
    class Point2 {
    public:
        // Point2 Public Methods
        __host__ __device__ explicit Point2(const Point3<T>& p) : x(p.x), y(p.y) { /*DCHECK(!HasNaNs()); */ }
        __host__ __device__ Point2() { x = y = 0; }
        __host__ __device__ Point2(T xx, T yy) : x(xx), y(yy) { /*DCHECK(!HasNaNs());*/ }

        template <typename U>
        __host__ __device__ explicit Point2(const Point2<U>& p) {
            x = (T)p.x;
            y = (T)p.y;
            //DCHECK(!HasNaNs());
        }

        template <typename U>
        __host__ __device__ explicit Point2(const Vector2<U>& p) {
            x = (T)p.x;
            y = (T)p.y;
            //DCHECK(!HasNaNs());
        }

        template <typename U>
        __host__ __device__ explicit operator Vector2<U>() const {
            return Vector2<U>(x, y);
        }

#ifndef NDEBUG
        __host__ __device__ Point2(const Point2<T>& p) {
            //DCHECK(!p.HasNaNs());
            x = p.x;
            y = p.y;
        }

        __host__ __device__ Point2<T>& operator=(const Point2<T>& p) {
            //DCHECK(!p.HasNaNs());
            x = p.x;
            y = p.y;
            return *this;
        }
#endif  // !NDEBUG
        __host__ __device__ Point2<T> operator+(const Vector2<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Point2<T>(x + v.x, y + v.y);
        }

        __host__ __device__ Point2<T>& operator+=(const Vector2<T>& v) {
            //DCHECK(!v.HasNaNs());
            x += v.x;
            y += v.y;
            return *this;
        }
        __host__ __device__ Vector2<T> operator-(const Point2<T>& p) const {
            //DCHECK(!p.HasNaNs());
            return Vector2<T>(x - p.x, y - p.y);
        }

        __host__ __device__ Point2<T> operator-(const Vector2<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Point2<T>(x - v.x, y - v.y);
        }
        __host__ __device__ Point2<T> operator-() const { return Point2<T>(-x, -y); }
        __host__ __device__ Point2<T>& operator-=(const Vector2<T>& v) {
            //DCHECK(!v.HasNaNs());
            x -= v.x;
            y -= v.y;
            return *this;
        }
        __host__ __device__ Point2<T>& operator+=(const Point2<T>& p) {
            //DCHECK(!p.HasNaNs());
            x += p.x;
            y += p.y;
            return *this;
        }
        __host__ __device__ Point2<T> operator+(const Point2<T>& p) const {
            //DCHECK(!p.HasNaNs());
            return Point2<T>(x + p.x, y + p.y);
        }
        template <typename U>
        __host__ __device__ Point2<T> operator*(U f) const {
            return Point2<T>(f * x, f * y);
        }
        template <typename U>
        __host__ __device__ Point2<T>& operator*=(U f) {
            x *= f;
            y *= f;
            return *this;
        }
        template <typename U>
        __host__ __device__ Point2<T> operator/(U f) const {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            return Point2<T>(inv * x, inv * y);
        }
        template <typename U>
        __host__ __device__ Point2<T>& operator/=(U f) {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            x *= inv;
            y *= inv;
            return *this;
        }
        __host__ __device__ T operator[](int i) const {
            //DCHECK(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }

        __host__ __device__ T& operator[](int i) {
            //DCHECK(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }
        __host__ __device__ bool operator==(const Point2<T>& p) const { return x == p.x && y == p.y; }
        __host__ __device__ bool operator!=(const Point2<T>& p) const { return x != p.x || y != p.y; }
        //__host__ __device__ bool HasNaNs() const { /*return isNaN(x) || isNaN(y);*/ }

        // Point2 Public Data
        T x, y;
    };

    template <typename T>
    __host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Point2<T>& v) {
        os << "[ " << v.x << ", " << v.y << " ]";
        return os;
    }


    template <typename T>
    class Point3 {
    public:
        // Point3 Public Methods
        __host__ __device__ Point3() { x = y = z = 0; }
        __host__ __device__ Point3(T x, T y, T z) : x(x), y(y), z(z) { /*DCHECK(!HasNaNs());*/ }
        template <typename U>
        __host__ __device__ explicit Point3(const Point3<U>& p)
            : x((T)p.x), y((T)p.y), z((T)p.z) {
            //DCHECK(!HasNaNs());
        }
        __host__ __device__ explicit Point3(const Vector3<T>& v);
        __host__ __device__ explicit Point3(const Normal3<T>& v);
        template <typename U>
        __host__ __device__ explicit operator Vector3<U>() const {
            return Vector3<U>(x, y, z);
        }

#ifndef NDEBUG
        __host__ __device__ Point3(const Point3<T>& p) {
            //DCHECK(!p.HasNaNs());
            x = p.x;
            y = p.y;
            z = p.z;
        }

        __host__ __device__ Point3<T>& operator=(const Point3<T>& p) {
            //DCHECK(!p.HasNaNs());
            x = p.x;
            y = p.y;
            z = p.z;
            return *this;
        }
#endif  // !NDEBUG
        __host__ __device__ Point3<T> operator+(const Vector3<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Point3<T>(x + v.x, y + v.y, z + v.z);
        }
        __host__ __device__ Point3<T>& operator+=(const Vector3<T>& v) {
            //DCHECK(!v.HasNaNs());
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }
        __host__ __device__ Vector3<T> operator-(const Point3<T>& p) const {
            //DCHECK(!p.HasNaNs());
            return Vector3<T>(x - p.x, y - p.y, z - p.z);
        }
        __host__ __device__ Point3<T> operator-(const Vector3<T>& v) const {
            //DCHECK(!v.HasNaNs());
            return Point3<T>(x - v.x, y - v.y, z - v.z);
        }
        __host__ __device__ Point3<T>& operator-=(const Vector3<T>& v) {
            //DCHECK(!v.HasNaNs());
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }
        __host__ __device__ Point3<T>& operator+=(const Point3<T>& p) {
            //DCHECK(!p.HasNaNs());
            x += p.x;
            y += p.y;
            z += p.z;
            return *this;
        }
        __host__ __device__ Point3<T> operator+(const Point3<T>& p) const {
            //DCHECK(!p.HasNaNs());
            return Point3<T>(x + p.x, y + p.y, z + p.z);
        }

        __host__ __device__ Point3<T> operator*(const Point3<T>& p) const {
            //DCHECK(!p.HasNaNs());
            return Point3<T>(x * p.x, y * p.y, z * p.z);
        }

        template <typename U>
        __host__ __device__ Point3<T> operator*(U f) const {
            return Point3<T>(f * x, f * y, f * z);
        }
        // template <typename U>
        __host__ __device__ Point3<T>& operator*=(Float f) {
            x *= f;
            y *= f;
            z *= f;
            return *this;
        }
        template <typename U>
        __host__ __device__ Point3<T> operator/(U f) const {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            return Point3<T>(inv * x, inv * y, inv * z);
        }
        template <typename U>
        __host__ __device__ Point3<T>& operator/=(U f) {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }
        __host__ __device__ T operator[](int i) const {
            //DCHECK(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        __host__ __device__ T& operator[](int i) {
            //DCHECK(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }
        __host__ __device__ bool operator==(const Point3<T>& p) const {
            return x == p.x && y == p.y && z == p.z;
        }
        __host__ __device__ bool operator!=(const Point3<T>& p) const {
            return x != p.x || y != p.y || z != p.z;
        }
        //__host__ __device__ bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
        __host__ __device__ Point3<T> operator-() const { return Point3<T>(-x, -y, -z); }

        // Point3 Public Data
        T x, y, z;
    };

    template <typename T>
    __host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Point3<T>& v) {
        os << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
        return os;
    }



    typedef Point2<Float> Point2f;
    typedef Point2<int> Point2i;
    typedef Point3<Float> Point3f;
    typedef Point3<int> Point3i;

    // Normal Declarations
    template <typename T>
    class Normal3 {
    public:
        // Normal3 Public Methods
        __host__ __device__ Normal3() { x = y = z = 0; }
        __host__ __device__ Normal3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) { /*DCHECK(!HasNaNs());*/ }
        __host__ __device__ Normal3<T> operator-() const { return Normal3(-x, -y, -z); }
        __host__ __device__ Normal3<T> operator+(const Normal3<T>& n) const {
            //DCHECK(!n.HasNaNs());
            return Normal3<T>(x + n.x, y + n.y, z + n.z);
        }

        __host__ __device__ Normal3<T>& operator+=(const Normal3<T>& n) {
            //DCHECK(!n.HasNaNs());
            x += n.x;
            y += n.y;
            z += n.z;
            return *this;
        }
        __host__ __device__ Normal3<T> operator-(const Normal3<T>& n) const {
            //DCHECK(!n.HasNaNs());
            return Normal3<T>(x - n.x, y - n.y, z - n.z);
        }

        __host__ __device__ Normal3<T>& operator-=(const Normal3<T>& n) {
            //DCHECK(!n.HasNaNs());
            x -= n.x;
            y -= n.y;
            z -= n.z;
            return *this;
        }
        //__host__ __device__ bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
        template <typename U>
        __host__ __device__ Normal3<T> operator*(U f) const {
            return Normal3<T>(f * x, f * y, f * z);
        }

        template <typename U>
        __host__ __device__ Normal3<T>& operator*=(U f) {
            x *= f;
            y *= f;
            z *= f;
            return *this;
        }
        template <typename U>
        __host__ __device__ Normal3<T> operator/(U f) const {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            return Normal3<T>(x * inv, y * inv, z * inv);
        }

        template <typename U>
        __host__ __device__ Normal3<T>& operator/=(U f) {
            //CHECK_NE(f, 0);
            Float inv = (Float)1 / f;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }
        __host__ __device__ Float LengthSquared() const { return x * x + y * y + z * z; }
        __host__ __device__ Float Length() const { return std::sqrt(LengthSquared()); }

#ifndef NDEBUG
        __host__ __device__ Normal3<T>(const Normal3<T>& n) {
            //DCHECK(!n.HasNaNs());
            x = n.x;
            y = n.y;
            z = n.z;
        }

        __host__ __device__ Normal3<T>& operator=(const Normal3<T>& n) {
            //DCHECK(!n.HasNaNs());
            x = n.x;
            y = n.y;
            z = n.z;
            return *this;
        }
#endif  // !NDEBUG
        __host__ __device__ explicit Normal3<T>(const Vector3<T>& v) : x(v.x), y(v.y), z(v.z) {
            //DCHECK(!v.HasNaNs());
        }
        __host__ __device__ bool operator==(const Normal3<T>& n) const {
            return x == n.x && y == n.y && z == n.z;
        }
        __host__ __device__ bool operator!=(const Normal3<T>& n) const {
            return x != n.x || y != n.y || z != n.z;
        }

        __host__ __device__ T operator[](int i) const {
            //DCHECK(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        __host__ __device__ T& operator[](int i) {
            //DCHECK(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        // Normal3 Public Data
        T x, y, z;
    };

    template <typename T>
    __host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Normal3<T>& v) {
        os << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
        return os;
    }

    typedef Normal3<Float> Normal3f;


    // Geometry Inline Functions
    template <typename T>
    __host__ __device__ inline Vector3<T>::Vector3(const Point3<T>& p)
        : x(p.x), y(p.y), z(p.z) {
        /*DCHECK(!HasNaNs());*/
    }

    template <typename T>
    __host__ __device__ inline Point3<T>::Point3(const Vector3<T>& v)
        : x(v.x), y(v.y), z(v.z) {
        //DCHECK(!HasNaNs());
    }

    template <typename T, typename U>
    __host__ __device__ inline Vector3<T> operator*(U s, const Vector3<T>& v) {
        return v * s;
    }
    template <typename T>
    __host__ __device__ Vector3<T> Abs(const Vector3<T>& v) {
        return Vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
    }

    template <typename T>
    __host__ __device__ inline T Dot(const Vector3<T>& v1, const Vector3<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    template <typename T>
    __host__ __device__ inline T AbsDot(const Vector3<T>& v1, const Vector3<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        return std::abs(Dot(v1, v2));
    }

    /// <summary>
    /// 右手坐标系
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="v1"></param>
    /// <param name="v2"></param>
    /// <returns></returns>
    template <typename T>
    __host__ __device__ inline Vector3<T> Cross(const Vector3<T>& v1, const Vector3<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        double v1x = v1.x, v1y = v1.y, v1z = v1.z;
        double v2x = v2.x, v2y = v2.y, v2z = v2.z;
        return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
            (v1x * v2y) - (v1y * v2x));
    }

    template <typename T>
    __host__ __device__ inline Vector3<T> Cross(const Vector3<T>& v1, const Normal3<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        double v1x = v1.x, v1y = v1.y, v1z = v1.z;
        double v2x = v2.x, v2y = v2.y, v2z = v2.z;
        return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
            (v1x * v2y) - (v1y * v2x));
    }

    template <typename T>
    __host__ __device__ inline Vector3<T> Cross(const Normal3<T>& v1, const Vector3<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        double v1x = v1.x, v1y = v1.y, v1z = v1.z;
        double v2x = v2.x, v2y = v2.y, v2z = v2.z;
        return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
            (v1x * v2y) - (v1y * v2x));
    }

    template <typename T>
    __host__ __device__ inline Vector3<T> Normalize(const Vector3<T>& v) {
        return v / v.Length();
    }
    template <typename T>
    __host__ __device__ T MinComponent(const Vector3<T>& v) {
        return Min(v.x, Min(v.y, v.z));
    }

    template <typename T>
    __host__ __device__ T MaxComponent(const Vector3<T>& v) {
        return Max(v.x, Max(v.y, v.z));
    }

    template <typename T>
    __host__ __device__ int MaxDimension(const Vector3<T>& v) {
        return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
    }

    template <typename T>
    __host__ __device__ Vector3<T> Min(const Vector3<T>& p1, const Vector3<T>& p2) {
        return Vector3<T>(Min(p1.x, p2.x), Min(p1.y, p2.y),
            Min(p1.z, p2.z));
    }

    template <typename T>
    __host__ __device__ Vector3<T> Max(const Vector3<T>& p1, const Vector3<T>& p2) {
        return Vector3<T>(Max(p1.x, p2.x), Max(p1.y, p2.y),
            Max(p1.z, p2.z));
    }

    template <typename T>
    __host__ __device__ Vector3<T> Permute(const Vector3<T>& v, int x, int y, int z) {
        return Vector3<T>(v[x], v[y], v[z]);
    }

    template <typename T>
    __host__ __device__ inline void CoordinateSystem(const Vector3<T>& v1, Vector3<T>* v2,
        Vector3<T>* v3) {
        if (std::abs(v1.x) > std::abs(v1.y))
            *v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            *v2 = Vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
        *v3 = Cross(v1, *v2);
    }

    template <typename T>
    __host__ __device__ Vector2<T>::Vector2(const Point2<T>& p)
        : x(p.x), y(p.y) {
        //DCHECK(!HasNaNs());
    }

    template <typename T>
    __host__ __device__ Vector2<T>::Vector2(const Point3<T>& p)
        : x(p.x), y(p.y) {
        //DCHECK(!HasNaNs());
    }

    template <typename T, typename U>
    __host__ __device__ inline Vector2<T> operator*(U f, const Vector2<T>& v) {
        return v * f;
    }
    template <typename T>
    __host__ __device__ inline Float Dot(const Vector2<T>& v1, const Vector2<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        return v1.x * v2.x + v1.y * v2.y;
    }

    template <typename T>
    __host__ __device__ inline Float AbsDot(const Vector2<T>& v1, const Vector2<T>& v2) {
        //DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
        return std::abs(Dot(v1, v2));
    }

    template <typename T>
    __host__ __device__ inline Vector2<T> Normalize(const Vector2<T>& v) {
        return v / v.Length();
    }
    template <typename T>
    __host__ __device__ Vector2<T> Abs(const Vector2<T>& v) {
        return Vector2<T>(std::abs(v.x), std::abs(v.y));
    }

    template <typename T>
    __host__ __device__ inline Float Distance(const Point3<T>& p1, const Point3<T>& p2) {
        return (p1 - p2).Length();
    }

    template <typename T>
    __host__ __device__ inline Float DistanceSquared(const Point3<T>& p1, const Point3<T>& p2) {
        return (p1 - p2).LengthSquared();
    }

    template <typename T, typename U>
    __host__ __device__ inline Point3<T> operator*(U f, const Point3<T>& p) {
        //DCHECK(!p.HasNaNs());
        return p * f;
    }

    template <typename T>
    __host__ __device__ Point3<T> Lerp(Float t, const Point3<T>& p0, const Point3<T>& p1) {
        return (1 - t) * p0 + t * p1;
    }

    template <typename T>
    __host__ __device__ Point3<T> Min(const Point3<T>& p1, const Point3<T>& p2) {
        return Point3<T>(Min(p1.x, p2.x), Min(p1.y, p2.y),
            Min(p1.z, p2.z));
    }

    template <typename T>
    __host__ __device__ Point3<T> Max(const Point3<T>& p1, const Point3<T>& p2) {
        return Point3<T>(Max(p1.x, p2.x), Max(p1.y, p2.y),
            Max(p1.z, p2.z));
    }

    template <typename T>
    __host__ __device__ Point3<T> Floor(const Point3<T>& p) {
        return Point3<T>(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    }

    template <typename T>
    __host__ __device__ Point3<T> Ceil(const Point3<T>& p) {
        return Point3<T>(std::ceil(p.x), std::ceil(p.y), std::ceil(p.z));
    }

    template <typename T>
    __host__ __device__ Point3<T> Abs(const Point3<T>& p) {
        return Point3<T>(std::abs(p.x), std::abs(p.y), std::abs(p.z));
    }

    template <typename T>
    __host__ __device__ inline Float Distance(const Point2<T>& p1, const Point2<T>& p2) {
        return (p1 - p2).Length();
    }

    template <typename T>
    __host__ __device__ inline Float DistanceSquared(const Point2<T>& p1, const Point2<T>& p2) {
        return (p1 - p2).LengthSquared();
    }

    template <typename T, typename U>
    __host__ __device__ inline Point2<T> operator*(U f, const Point2<T>& p) {
        //DCHECK(!p.HasNaNs());
        return p * f;
    }

    template <typename T>
    __host__ __device__ Point2<T> Floor(const Point2<T>& p) {
        return Point2<T>(std::floor(p.x), std::floor(p.y));
    }

    template <typename T>
    __host__ __device__ Point2<T> Ceil(const Point2<T>& p) {
        return Point2<T>(std::ceil(p.x), std::ceil(p.y));
    }

    template <typename T>
    __host__ __device__ Point2<T> Lerp(Float t, const Point2<T>& v0, const Point2<T>& v1) {
        return (1 - t) * v0 + t * v1;
    }

    template <typename T>
    __host__ __device__ Point2<T> Min(const Point2<T>& pa, const Point2<T>& pb) {
        return Point2<T>(Min(pa.x, pb.x), Min(pa.y, pb.y));
    }

    template <typename T>
    __host__ __device__ Point2<T> Max(const Point2<T>& pa, const Point2<T>& pb) {
        return Point2<T>(Max(pa.x, pb.x), Max(pa.y, pb.y));
    }

    template <typename T>
    __host__ __device__ Point3<T> Permute(const Point3<T>& p, int x, int y, int z) {
        return Point3<T>(p[x], p[y], p[z]);
    }

    template <typename T, typename U>
    __host__ __device__ inline Normal3<T> operator*(U f, const Normal3<T>& n) {
        return Normal3<T>(f * n.x, f * n.y, f * n.z);
    }

    template <typename T>
    __host__ __device__ inline Normal3<T> Normalize(const Normal3<T>& n) {
        return n / n.Length();
    }

    template <typename T>
    __host__ __device__ inline Vector3<T>::Vector3(const Normal3<T>& n)
        : x(n.x), y(n.y), z(n.z) {
        //DCHECK(!n.HasNaNs());
    }

    template <typename T>
    __host__ __device__ inline Point3<T>::Point3(const Normal3<T>& n)
        : x(n.x), y(n.y), z(n.z) {
        //DCHECK(!n.HasNaNs());
    }

    template <typename T>
    __host__ __device__ inline T Dot(const Normal3<T>& n1, const Vector3<T>& v2) {
        //DCHECK(!n1.HasNaNs() && !v2.HasNaNs());
        return n1.x * v2.x + n1.y * v2.y + n1.z * v2.z;
    }

    template <typename T>
    __host__ __device__ inline T Dot(const Vector3<T>& v1, const Normal3<T>& n2) {
        //DCHECK(!v1.HasNaNs() && !n2.HasNaNs());
        return v1.x * n2.x + v1.y * n2.y + v1.z * n2.z;
    }

    template <typename T>
    __host__ __device__ inline T Dot(const Normal3<T>& n1, const Normal3<T>& n2) {
        //DCHECK(!n1.HasNaNs() && !n2.HasNaNs());
        return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
    }

    template <typename T>
    __host__ __device__ inline T AbsDot(const Normal3<T>& n1, const Vector3<T>& v2) {
        //DCHECK(!n1.HasNaNs() && !v2.HasNaNs());
        return std::abs(n1.x * v2.x + n1.y * v2.y + n1.z * v2.z);
    }

    template <typename T>
    __host__ __device__ inline T AbsDot(const Vector3<T>& v1, const Normal3<T>& n2) {
        //DCHECK(!v1.HasNaNs() && !n2.HasNaNs());
        return std::abs(v1.x * n2.x + v1.y * n2.y + v1.z * n2.z);
    }

    template <typename T>
    __host__ __device__ inline T AbsDot(const Normal3<T>& n1, const Normal3<T>& n2) {
        //DCHECK(!n1.HasNaNs() && !n2.HasNaNs());
        return std::abs(n1.x * n2.x + n1.y * n2.y + n1.z * n2.z);
    }

    template <typename T>
    __host__ __device__ inline Normal3<T> Faceforward(const Normal3<T>& n, const Vector3<T>& v) {
        return (Dot(n, v) < 0.f) ? -n : n;
    }

    template <typename T>
    __host__ __device__ inline Normal3<T> Faceforward(const Normal3<T>& n, const Normal3<T>& n2) {
        return (Dot(n, n2) < 0.f) ? -n : n;
    }

    template <typename T>
    __host__ __device__ inline Vector3<T> Faceforward(const Vector3<T>& v, const Vector3<T>& v2) {
        return (Dot(v, v2) < 0.f) ? -v : v;
    }

    template <typename T>
    __host__ __device__ inline Vector3<T> Faceforward(const Vector3<T>& v, const Normal3<T>& n2) {
        return (Dot(v, n2) < 0.f) ? -v : v;
    }

    template <typename T>
    __host__ __device__ Normal3<T> Abs(const Normal3<T>& v) {
        return Normal3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
    }


    template <typename T>
    __host__ __device__ Point3<T> Clamp(const Point3<T>& v, T a, T b) {
        T left = min(a, b);
        T right = max(a, b);
        T x = min(max(v.x, left), right);
        T y = min(max(v.y, left), right);
        T z = min(max(v.z, left), right);
        return Point3<T>(x, y, z);
    }

    template <typename T>
    __host__ __device__ Normal3<T> Clamp(const Normal3<T>& v, T a, T b) {
        T left = min(a, b);
        T right = max(a, b);
        T x = min(max(v.x, left), right);
        T y = min(max(v.y, left), right);
        T z = min(max(v.z, left), right);
        return Normal3<T>(x, y, z);
    }

    template <typename T>
    __host__ __device__ Vector3<T> Clamp(const Vector3<T>& v, T a, T b) {
        T left = min(a, b);
        T right = max(a, b);
        T x = min(max(v.x, left), right);
        T y = min(max(v.y, left), right);
        T z = min(max(v.z, left), right);
        return Vector3<T>(x, y, z);
    }


    // Bounds Declarations
    template <typename T>
    class Bounds2 {
    public:
        // Bounds2 Public Methods
        __device__ Bounds2() {
            T minNum = std::numeric_limits<T>::lowest();
            T maxNum = std::numeric_limits<T>::max();
            pMin = Point2<T>(maxNum, maxNum);
            pMax = Point2<T>(minNum, minNum);
        }
        __device__ explicit Bounds2(const Point2<T>& p) : pMin(p), pMax(p) {}
        __device__ Bounds2(const Point2<T>& p1, const Point2<T>& p2) {
            pMin = Point2<T>(Min(p1.x, p2.x), Min(p1.y, p2.y));
            pMax = Point2<T>(Max(p1.x, p2.x), Max(p1.y, p2.y));
        }
        template <typename U>
        __device__ explicit operator Bounds2<U>() const {
            return Bounds2<U>((Point2<U>)pMin, (Point2<U>)pMax);
        }

        __device__ Vector2<T> Diagonal() const { return pMax - pMin; }
        __device__ T Area() const {
            Vector2<T> d = pMax - pMin;
            return (d.x * d.y);
        }
        __device__ int MaximumExtent() const {
            Vector2<T> diag = Diagonal();
            if (diag.x > diag.y)
                return 0;
            else
                return 1;
        }
        __device__ inline const Point2<T>& operator[](int i) const {
            DCHECK(i == 0 || i == 1);
            return (i == 0) ? pMin : pMax;
        }
        __device__ inline Point2<T>& operator[](int i) {
            DCHECK(i == 0 || i == 1);
            return (i == 0) ? pMin : pMax;
        }
        __device__ bool operator==(const Bounds2<T>& b) const {
            return b.pMin == pMin && b.pMax == pMax;
        }
        __device__ bool operator!=(const Bounds2<T>& b) const {
            return b.pMin != pMin || b.pMax != pMax;
        }
        __device__ Point2<T> Lerp(const Point2f& t) const {
            return Point2<T>(Lerp(t.x, pMin.x, pMax.x),
                Lerp(t.y, pMin.y, pMax.y));
        }
        __device__ Vector2<T> Offset(const Point2<T>& p) const {
            Vector2<T> o = p - pMin;
            if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
            if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
            return o;
        }
        __device__ void BoundingSphere(Point2<T>* c, Float* rad) const {
            *c = (pMin + pMax) / 2;
            *rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
        }

        // Bounds2 Public Data
        Point2<T> pMin, pMax;
    };

    template <typename T>
    class Bounds3 {
    public:
        // Bounds3 Public Methods
        __device__ Bounds3() {
            T minNum = MinFloat;
            T maxNum = MaxFloat;
            pMin = Point3<T>(maxNum, maxNum, maxNum);
            pMax = Point3<T>(minNum, minNum, minNum);
        }
        __device__ explicit Bounds3(const Point3<T>& p) : pMin(p), pMax(p) {}
        __device__ Bounds3(const Point3<T>& p1, const Point3<T>& p2)
            : pMin(Min(p1.x, p2.x), Min(p1.y, p2.y),
                Min(p1.z, p2.z)),
            pMax(Max(p1.x, p2.x), Max(p1.y, p2.y),
                Max(p1.z, p2.z)) {
        }
        __device__ const Point3<T>& operator[](int i) const;
        __device__ Point3<T>& operator[](int i);
        __device__ bool operator==(const Bounds3<T>& b) const {
            return b.pMin == pMin && b.pMax == pMax;
        }
        __device__ bool operator!=(const Bounds3<T>& b) const {
            return b.pMin != pMin || b.pMax != pMax;
        }
        __device__ Point3<T> Corner(int corner) const {
            DCHECK(corner >= 0 && corner < 8);
            return Point3<T>((*this)[(corner & 1)].x,
                (*this)[(corner & 2) ? 1 : 0].y,
                (*this)[(corner & 4) ? 1 : 0].z);
        }
        __device__ Vector3<T> Diagonal() const { return pMax - pMin; }
        __device__ T SurfaceArea() const {
            Vector3<T> d = Diagonal();
            return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
        }
        __device__ T Volume() const {
            Vector3<T> d = Diagonal();
            return d.x * d.y * d.z;
        }
        __device__ int MaximumExtent() const {
            Vector3<T> d = Diagonal();
            if (d.x > d.y && d.x > d.z)
                return 0;
            else if (d.y > d.z)
                return 1;
            else
                return 2;
        }
        __device__ Point3<T> Lerp(const Point3f& t) const {
            return Point3<T>(Lerp(t.x, pMin.x, pMax.x),
                Lerp(t.y, pMin.y, pMax.y),
                Lerp(t.z, pMin.z, pMax.z));
        }
        __device__ Vector3<T> Offset(const Point3<T>& p) const {
            Vector3<T> o = p - pMin;
            if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
            if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
            if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
            return o;
        }
        __device__ void BoundingSphere(Point3<T>* center, Float* radius) const {
            *center = (pMin + pMax) / 2;
            *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
        }
        template <typename U>
        __device__ explicit operator Bounds3<U>() const {
            return Bounds3<U>((Point3<U>)pMin, (Point3<U>)pMax);
        }

        __device__ bool IntersectP(const Ray& ray, Float* hitt0 = nullptr,
            Float* hitt1 = nullptr) const;

        // Bounds3 Public Data
        Point3<T> pMin, pMax;
    };

    typedef Bounds2<Float> Bounds2f;
    typedef Bounds2<int> Bounds2i;
    typedef Bounds3<Float> Bounds3f;
    typedef Bounds3<int> Bounds3i;


    template <typename T>
    __device__ inline const Point3<T>& Bounds3<T>::operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }

    template <typename T>
    __device__ inline Point3<T>& Bounds3<T>::operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }

    template <typename T>
    __device__ Bounds3<T> Union(const Bounds3<T>& b, const Point3<T>& p) {
        Bounds3<T> ret;
        ret.pMin = Min(b.pMin, p);
        ret.pMax = Max(b.pMax, p);
        return ret;
    }

    template <typename T>
    __device__ Bounds3<T> Union(const Bounds3<T>& b1, const Bounds3<T>& b2) {
        Bounds3<T> ret;
        ret.pMin = Min(b1.pMin, b2.pMin);
        ret.pMax = Max(b1.pMax, b2.pMax);
        return ret;
    }

    template <typename T>
    __device__ Bounds3<T> Intersect(const Bounds3<T>& b1, const Bounds3<T>& b2) {
        // Important: assign to pMin/pMax directly and don't run the Bounds2()
        // constructor, since it takes min/max of the points passed to it.  In
        // turn, that breaks returning an invalid bound for the case where we
        // intersect non-overlapping bounds (as we'd like to happen).
        Bounds3<T> ret;
        ret.pMin = Max(b1.pMin, b2.pMin);
        ret.pMax = Min(b1.pMax, b2.pMax);
        return ret;
    }

    template <typename T>
    __device__ bool Overlaps(const Bounds3<T>& b1, const Bounds3<T>& b2) {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return (x && y && z);
    }

    template <typename T>
    __device__ bool Inside(const Point3<T>& p, const Bounds3<T>& b) {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }

    template <typename T>
    __device__ bool InsideExclusive(const Point3<T>& p, const Bounds3<T>& b) {
        return (p.x >= b.pMin.x && p.x < b.pMax.x&& p.y >= b.pMin.y &&
            p.y < b.pMax.y&& p.z >= b.pMin.z && p.z < b.pMax.z);
    }

    template <typename T, typename U>
    __device__ inline Bounds3<T> Expand(const Bounds3<T>& b, U delta) {
        return Bounds3<T>(b.pMin - Vector3<T>(delta, delta, delta),
            b.pMax + Vector3<T>(delta, delta, delta));
    }

    // Minimum squared distance from point to box; returns zero if point is
    // inside.
    template <typename T, typename U>
    __device__ inline Float DistanceSquared(const Point3<T>& p, const Bounds3<U>& b) {
        Float dx = Max({ Float(0), b.pMin.x - p.x, p.x - b.pMax.x });
        Float dy = Max({ Float(0), b.pMin.y - p.y, p.y - b.pMax.y });
        Float dz = Max({ Float(0), b.pMin.z - p.z, p.z - b.pMax.z });
        return dx * dx + dy * dy + dz * dz;
    }

    template <typename T, typename U>
    __device__ inline Float Distance(const Point3<T>& p, const Bounds3<U>& b) {
        return std::sqrt(DistanceSquared(p, b));
    }

    template <typename T>
    __device__ Bounds2<T> Union(const Bounds2<T>& b, const Point2<T>& p) {
        Bounds2<T> ret;
        ret.pMin = Min(b.pMin, p);
        ret.pMax = Max(b.pMax, p);
        return ret;
    }

    template <typename T>
    __device__ Bounds2<T> Union(const Bounds2<T>& b, const Bounds2<T>& b2) {
        Bounds2<T> ret;
        ret.pMin = Min(b.pMin, b2.pMin);
        ret.pMax = Max(b.pMax, b2.pMax);
        return ret;
    }

    template <typename T>
    __device__ Bounds2<T> Intersect(const Bounds2<T>& b1, const Bounds2<T>& b2) {
        // Important: assign to pMin/pMax directly and don't run the Bounds2()
        // constructor, since it takes min/max of the points passed to it.  In
        // turn, that breaks returning an invalid bound for the case where we
        // intersect non-overlapping bounds (as we'd like to happen).
        Bounds2<T> ret;
        ret.pMin = Max(b1.pMin, b2.pMin);
        ret.pMax = Min(b1.pMax, b2.pMax);
        return ret;
    }

    template <typename T>
    __device__ bool Overlaps(const Bounds2<T>& ba, const Bounds2<T>& bb) {
        bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
        bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
        return (x && y);
    }

    template <typename T>
    __device__ bool Inside(const Point2<T>& pt, const Bounds2<T>& b) {
        return (pt.x >= b.pMin.x && pt.x <= b.pMax.x && pt.y >= b.pMin.y &&
            pt.y <= b.pMax.y);
    }

    template <typename T>
    __device__ bool InsideExclusive(const Point2<T>& pt, const Bounds2<T>& b) {
        return (pt.x >= b.pMin.x && pt.x < b.pMax.x&& pt.y >= b.pMin.y &&
            pt.y < b.pMax.y);
    }

    template <typename T, typename U>
    __device__ Bounds2<T> Expand(const Bounds2<T>& b, U delta) {
        return Bounds2<T>(b.pMin - Vector2<T>(delta, delta),
            b.pMax + Vector2<T>(delta, delta));
    }

    template <typename T>
    __device__ inline bool Bounds3<T>::IntersectP(const Ray& ray, Float* hitt0,
        Float* hitt1) const {
        Float t0 = MinFloat, t1 = ray.tMax;
        for (int i = 0; i < 3; ++i) {
            // Update interval for _i_th bounding box slab
            Float invRayDir = 1.f / ray.d[i];
            Float tNear = (pMin[i] - ray.o[i]) * invRayDir;
            Float tFar = (pMax[i] - ray.o[i]) * invRayDir;

            // Update parametric interval from slab intersection $t$ values
            // 做这步的原因就是因为光线方向分量为负，导致近的在远平面找到
            // 近远平面的定义主要是按照boundbox的轴，分量值越小，在该分量轴上就定义为近平面
            if (tNear > tFar) {
                Float temp = tNear;
                tNear = tFar;
                tFar = temp;
            }

            // Update _tFar_ to ensure robust ray--bounds intersection
            // tFar *= 1 + 2 * gamma(3);
            // 判断区间是否重叠，没重叠就返回false
            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if (t0 > t1) return false;
        }
        if (hitt0) *hitt0 = t0;
        if (hitt1) *hitt1 = t1;
        return true;
    }

    


    class Ray {
    public:
        // Ray Public Methods
        __device__ Ray() : tMax(Infinity), time(0.f), tMin(ShadowEpsilon) {}
        __device__ Ray(const Point3f& o, const Vector3f& d,
            Float time = 0.f, Float tMax = Infinity, Float tMin = ShadowEpsilon)
            : o(o), d(d), tMax(tMax), time(time), tMin(tMin) {}
        __device__ Point3f operator()(Float t) const { return o + d * t; }
        //__device__ bool HasNaNs() const { return (o.HasNaNs() || d.HasNaNs() || isNaN(tMax)); }
        /*friend std::ostream& operator<<(std::ostream& os, const Ray& r) {
            os << "[o=" << r.o << ", d=" << r.d << ", tMax=" << r.tMax
                << ", time=" << r.time << "]";
            return os;
        }*/

        // Ray Public Data
        Point3f o;
        Vector3f d;
        mutable Float tMax, tMin; // 突破const的限制，即使是 const Ray &r，也能更改tMax
        Float time;
    };


    __device__ inline Float TrilinearLerp(Float c[2][2][2], Float u, Float v, Float w) {
        Float accum = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    accum += (i * u + (1 - i) * (1 - u)) * (j * v + (1 - j) * (1 - v)) * (k * w + (1 - k) * (1 - w)) * c[i][j][k];
                }
            }
        }
        return accum;
    }


#ifdef GPUMODE
    __device__ inline Point3<Float> RandomInUnitSphere(curandState* local_rand_state) {
        Vector3f p;
        do {
            p = 2.0f * Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - Vector3f(1, 1, 1);
        } while (Dot(p, p) >= 1.0f);
        return Point3<Float>(p);
    }
    __device__ inline Point3<Float> RandomInUnitDisk(curandState* local_rand_state) {
        Vector3f p;
        do {
            p = 2.0f * Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vector3f(1, 1, 0);
        } while (Dot(p, p) >= 1.0f);
        return Point3<Float>(p);
    }

#else
    /// <summary>
    /// 在一个单位球内产生一个随机点
    /// </summary>
    /// <returns></returns>
    inline Point3<Float> RandomInUnitSphere() {
        Vector3f p;
        // 构建随机数
        // std::uniform_real_distribution<Float> randomNum(0, 1); // 左闭右闭区间
        do {
            p = 2.0 * Vector3f(RND, RND, RND) - Vector3f(1, 1, 1);
        } while (Dot(p, p) >= 1.0);


        return Point3<Float>(p);
    }

    /// <summary>
    /// 在一个单位圆内产生一个随机点
    /// </summary>
    /// <returns></returns>
    inline Point3<Float> RandomInUnitDisk() {
        Vector3f p;
        do {
            p = 2.0 * Vector3f(RND, RND, 0) - Vector3f(1, 1, 0);
        } while (Dot(p, p) >= 1.0);


        return Point3<Float>(p);
    }
#endif // GPUMODE

#pragma region SortFunction
    /// <summary>
    /// 交换
    /// </summary>
    /// <param name="buf1"></param>
    /// <param name="buf2"></param>
    /// <param name="width"></param>
    /// <returns></returns>
    __device__ inline void QSwap(char* buf1, char* buf2, int size) {
        char tmp = 0;
        for (int i = 0; i < size; i++) {
            tmp = *buf1;
            *buf1 = *buf2;
            *buf2 = tmp;
            *buf1++;
            *buf2++;
        }
    }


    __device__ inline int Partition(void* base, int num, int size, int(*comparator)(const void*, const void*)) {

        int first = 0, end = num - 1, pivot = first;

        printf("Partition start\n");
        while (first < end)

        {
            //  while (first < end && comparator(base + pivot, base + end) <= 0)
            //  {
            //   --end;
            //  }
            //  swap(base + first,base + end);

            pivot = end;

            while (first < end && comparator((char*)base + first * size, (char*)base + pivot * size) <= 0) {

                ++first;

            }

            QSwap((char*)base + first * size, (char*)base + end * size, size);

            pivot = first;

        }
        printf("Partition End\n");
        return first;
    }

    __device__ inline void Qsort(void* base, int num, int size, int(*comparator)(const void*, const void*)) {

        /*printf("numShapes:%d \n", num);

        if (num > 0) {

            printf("Sort\n");
            int pivot = Partition(base, num, size, comparator);
            printf("pivot:%d \n", pivot);

            printf("AfterPartition\n");
            Qsort((char*)base, pivot, size, comparator);

            if (pivot + 1 > num) {
                printf("NULL!!!\n");
            }

            Qsort((char*)base + (pivot + 1) * size, num - pivot - 1, size, comparator);

        }*/

        int i = 0;
        for (i = 0; i < num - 1; i++) {
            int j = 0;
            for (j = 0; j < num - 1 - i; j++) {
                //两个元素比较
                //base强制类型转换成char*，
                if (comparator((char*)base + j * size, (char*)base + (j + 1) * size) > 0) {
                    //交换函数
                    QSwap((char*)base + j * size, (char*)base + (j + 1) * size, size);
                }
            }
        }
    }
#pragma endregion
    

    
    // Global Constants
    static Vector3f WorldUp(0.0, 1.0, 0.0);
    static Vector3f WorldRight(1.0, 0.0, 0.0);


#pragma region 柏林噪声
    class Perlin {

    public:
        __device__ Perlin() {}

        __device__ Perlin(curandState* local_rand_state) {
            ranVec = PerlinGenerate(local_rand_state);
            perm_x = PerlinGeneratePermute(local_rand_state);
            perm_y = PerlinGeneratePermute(local_rand_state);
            perm_z = PerlinGeneratePermute(local_rand_state);
        }

        __device__ Float Turb(const Point3f& p, int depth=7) {
            Float accum = 0.f;
            Point3f temp = p;
            Float weight = 1.0f;
            for (int i = 0; i < depth; i++) {
                accum += weight * Noise(temp);
                weight *= 0.5f;
                temp *= 2;
            }
            return abs(accum);
        }

        __device__ Float Noise(const Point3f& p) {
            Point3f uvw = Point3f(p - Floor(p));
            Point3f ijk = Floor(p);
            Vector3f c[2][2][2];
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        c[i][j][k] = ranVec[perm_x[((int)ijk.x + i) & 255] ^ perm_y[((int)ijk.y + j) & 255] ^ perm_z[((int)ijk.z + k) & 255]];
                    }
                }
            }
            return PerlinLerp(c, uvw.x, uvw.y, uvw.z);
        }

        Vector3f* ranVec = nullptr;
        int* perm_x = nullptr;
        int* perm_y = nullptr;
        int* perm_z = nullptr;

    private:
        __device__ inline Vector3f* PerlinGenerate(curandState* local_rand_state) {
            Vector3f* p = new Vector3f[256];
            for (int i = 0; i < 256; i++) {
                p[i] = Normalize(Vector3f(-1 + 2 * curand_uniform(local_rand_state), -1 + 2 * curand_uniform(local_rand_state), -1 + 2 * curand_uniform(local_rand_state)));
            }
            return p;
        }

        __device__ inline Float PerlinLerp(Vector3f c[2][2][2], Float u, Float v, Float w) {
            Float uu = u * u * (3.f - 2.f * u);
            Float vv = v * v * (3.f - 2.f * v);
            Float ww = w * w * (3.f - 2.f * w);
            Float accum = 0;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        Vector3f weight(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) * (k * ww + (1 - k) * (1 - ww)) * Dot(c[i][j][k], weight);
                    }
                }
            }
            return accum;
        }

        __device__ inline void PerlinPermute(int* p, int n, curandState* local_rand_state) {
            for (int i = n - 1; i > 0; i--) {
                int target = int(curand_uniform(local_rand_state) * (i + 1));
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }

        __device__ inline int* PerlinGeneratePermute(curandState* local_rand_state) {
            int* p = new int[256];
            for (int i = 0; i < 256; i++) {
                p[i] = i;
            }
            PerlinPermute(p, 256, local_rand_state);
            return p;
        }

    };
    

#pragma endregion

    
    
}

#endif  // QZRT_CORE_GEOMETRY_H
