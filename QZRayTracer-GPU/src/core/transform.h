#ifndef QZRT_CORE_TRANSFORM_H
#define QZRT_CORE_TRANSFORM_H

#include "QZRayTracer.h"
#include "geometry.h"


namespace raytracer {
    // Matrix4x4 Declarations
    struct Matrix4x4 {
        // Matrix4x4 Public Methods
        __host__ __device__ Matrix4x4() {
            m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
            m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
                m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
        }
        __host__ __device__ Matrix4x4(Float mat[4][4]);
        __host__ __device__ Matrix4x4(Float t00, Float t01, Float t02, Float t03, Float t10, Float t11,
            Float t12, Float t13, Float t20, Float t21, Float t22, Float t23,
            Float t30, Float t31, Float t32, Float t33);
        __host__ __device__ bool operator==(const Matrix4x4& m2) const {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (m[i][j] != m2.m[i][j]) return false;
            return true;
        }
        __host__ __device__ bool operator!=(const Matrix4x4& m2) const {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (m[i][j] != m2.m[i][j]) return true;
            return false;
        }
        __host__ __device__ friend Matrix4x4 Transpose(const Matrix4x4&);

        __host__ __device__ static Matrix4x4 Mul(const Matrix4x4& m1, const Matrix4x4& m2) {
            Matrix4x4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
                    m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
            return r;
        }
        __host__ __device__ friend Matrix4x4 Inverse(const Matrix4x4&);

        Float m[4][4];
    };

    // Transform Declarations
    class Transform {
    public:
        // Transform Public Methods
        __host__ __device__ Transform() {}
        __host__ __device__ Transform(const Float mat[4][4]) {
            m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                mat[3][3]);
            mInv = Inverse(m);
        }
        __host__ __device__ Transform(const Matrix4x4& m) : m(m), mInv(Inverse(m)) {}
        __host__ __device__ Transform(const Matrix4x4& m, const Matrix4x4& mInv) : m(m), mInv(mInv) {}
        __host__ __device__ friend Transform Inverse(const Transform& t) {
            return Transform(t.mInv, t.m);
        }
        __host__ __device__ friend Transform Transpose(const Transform& t) {
            return Transform(Transpose(t.m), Transpose(t.mInv));
        }
        __host__ __device__ bool operator==(const Transform& t) const {
            return t.m == m && t.mInv == mInv;
        }
        __host__  __device__ bool operator!=(const Transform& t) const {
            return t.m != m || t.mInv != mInv;
        }
        __host__ __device__ bool operator<(const Transform& t2) const {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    if (m.m[i][j] < t2.m.m[i][j]) return true;
                    if (m.m[i][j] > t2.m.m[i][j]) return false;
                }
            return false;
        }
        __host__ __device__ bool IsIdentity() const {
            return (m.m[0][0] == 1.f && m.m[0][1] == 0.f && m.m[0][2] == 0.f &&
                m.m[0][3] == 0.f && m.m[1][0] == 0.f && m.m[1][1] == 1.f &&
                m.m[1][2] == 0.f && m.m[1][3] == 0.f && m.m[2][0] == 0.f &&
                m.m[2][1] == 0.f && m.m[2][2] == 1.f && m.m[2][3] == 0.f &&
                m.m[3][0] == 0.f && m.m[3][1] == 0.f && m.m[3][2] == 0.f &&
                m.m[3][3] == 1.f);
        }
        __host__ __device__ const Matrix4x4& GetMatrix() const { return m; }
        __host__ __device__ const Matrix4x4& GetInverseMatrix() const { return mInv; }
        __host__ __device__ bool HasScale() const {
            Float la2 = (*this)(Vector3f(1, 0, 0)).LengthSquared();
            Float lb2 = (*this)(Vector3f(0, 1, 0)).LengthSquared();
            Float lc2 = (*this)(Vector3f(0, 0, 1)).LengthSquared();
#define NOT_ONE(x) ((x) < .999f || (x) > 1.001f)
            return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
        }
        template <typename T>
        __host__ __device__ inline Point3<T> operator()(const Point3<T>& p) const;
        template <typename T>
        __host__ __device__ inline Vector3<T> operator()(const Vector3<T>& v) const;
        template <typename T>
        __host__ __device__ inline Normal3<T> operator()(const Normal3<T>&) const;

        __device__ Bounds3f operator()(const Bounds3f& b) const;
        __host__ __device__ Transform operator*(const Transform& t2) const;
        __host__ __device__ bool SwapsHandedness() const;

    private:
        // Transform Private Data
        Matrix4x4 m, mInv;
        //friend class AnimatedTransform;
        friend struct Quaternion;
    };


     //__host__ __device__ Transform Translate(const Vector3f& delta);
     //__host__ __device__ Transform Scale(Float x, Float y, Float z);
     //__host__ __device__ Transform RotateX(Float theta);
     //__host__ __device__ Transform RotateY(Float theta);
     //__host__ __device__ Transform RotateZ(Float theta);
     //__host__ __device__ Transform Rotate(Float theta, const Vector3f& axis);
     //__device__ Transform LookAt(const Point3f& pos, const Point3f& look, const Vector3f& up);
     //__device__ Transform Orthographic(Float znear, Float zfar);
     //__device__ Transform Perspective(Float fov, Float znear, Float zfar);
     //__device__ bool SolveLinearSystem2x2(const Float A[2][2], const Float B[2], Float* x0,
     //   Float* x1);


     // Transform Inline Functions
     template <typename T>
     __host__ __device__ inline Point3<T> Transform::operator()(const Point3<T>& p) const {
         T x = p.x, y = p.y, z = p.z;
         T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
         T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
         T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
         T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
         if (wp == 0)return Point3f(p);
         if (wp == 1)
             return Point3<T>(xp, yp, zp);
         else
             return Point3<T>(xp, yp, zp) / wp;
     }

     template <typename T>
     __host__ __device__ inline Vector3<T> Transform::operator()(const Vector3<T>& v) const {
         T x = v.x, y = v.y, z = v.z;
         return Vector3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
             m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
             m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
     }

     template <typename T>
     __host__ __device__ inline Normal3<T> Transform::operator()(const Normal3<T>& n) const {
         T x = n.x, y = n.y, z = n.z;
         return Normal3<T>(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
             mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
             mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
     }

     __host__ __device__ inline Matrix4x4::Matrix4x4(Float mat[4][4]) { memcpy(m, mat, 16 * sizeof(Float)); }

     __host__ __device__ inline Matrix4x4::Matrix4x4(Float t00, Float t01, Float t02, Float t03, Float t10,
         Float t11, Float t12, Float t13, Float t20, Float t21,
         Float t22, Float t23, Float t30, Float t31, Float t32,
         Float t33) {
         m[0][0] = t00;
         m[0][1] = t01;
         m[0][2] = t02;
         m[0][3] = t03;
         m[1][0] = t10;
         m[1][1] = t11;
         m[1][2] = t12;
         m[1][3] = t13;
         m[2][0] = t20;
         m[2][1] = t21;
         m[2][2] = t22;
         m[2][3] = t23;
         m[3][0] = t30;
         m[3][1] = t31;
         m[3][2] = t32;
         m[3][3] = t33;
     }

     __host__ __device__ inline Matrix4x4 Transpose(const Matrix4x4& m) {
         return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
             m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
             m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
             m.m[3][3]);
     }

     __host__ __device__ inline Matrix4x4 Inverse(const Matrix4x4& m) {
         int indxc[4], indxr[4];
         int ipiv[4] = { 0, 0, 0, 0 };
         Float minv[4][4];
         memcpy(minv, m.m, 4 * 4 * sizeof(Float));
         for (int i = 0; i < 4; i++) {
             int irow = 0, icol = 0;
             Float big = 0.f;
             // Choose pivot
             for (int j = 0; j < 4; j++) {
                 if (ipiv[j] != 1) {
                     for (int k = 0; k < 4; k++) {
                         if (ipiv[k] == 0) {
                             if (std::abs(minv[j][k]) >= big) {
                                 big = Float(std::abs(minv[j][k]));
                                 irow = j;
                                 icol = k;
                             }
                         }
                         else if (ipiv[k] > 1)
                             printf("Singular matrix in MatrixInvert\n");
                     }
                 }
             }
             ++ipiv[icol];
             // Swap rows _irow_ and _icol_ for pivot
             if (irow != icol) {
                 for (int k = 0; k < 4; ++k) std::swap(minv[irow][k], minv[icol][k]);
             }
             indxr[i] = irow;
             indxc[i] = icol;
             if (minv[icol][icol] == 0.f) printf("Singular matrix in MatrixInvert\n");

             // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
             Float pivinv = 1. / minv[icol][icol];
             minv[icol][icol] = 1.;
             for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;

             // Subtract this row from others to zero out their columns
             for (int j = 0; j < 4; j++) {
                 if (j != icol) {
                     Float save = minv[j][icol];
                     minv[j][icol] = 0;
                     for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k] * save;
                 }
             }
         }
         // Swap columns to reflect permutation
         for (int j = 3; j >= 0; j--) {
             if (indxr[j] != indxc[j]) {
                 for (int k = 0; k < 4; k++)
                     std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
             }
         }
         return Matrix4x4(minv);
     }


     __device__ inline Bounds3f Transform::operator()(const Bounds3f& b) const {
         const Transform& M = *this;
         Bounds3f ret(M(Point3f(b.pMin.x, b.pMin.y, b.pMin.z)));
         ret = Union(ret, M(Point3f(b.pMax.x, b.pMin.y, b.pMin.z)));
         ret = Union(ret, M(Point3f(b.pMin.x, b.pMax.y, b.pMin.z)));
         ret = Union(ret, M(Point3f(b.pMin.x, b.pMin.y, b.pMax.z)));
         ret = Union(ret, M(Point3f(b.pMin.x, b.pMax.y, b.pMax.z)));
         ret = Union(ret, M(Point3f(b.pMax.x, b.pMax.y, b.pMin.z)));
         ret = Union(ret, M(Point3f(b.pMax.x, b.pMin.y, b.pMax.z)));
         ret = Union(ret, M(Point3f(b.pMax.x, b.pMax.y, b.pMax.z)));
         return ret;
     }

     __host__ __device__ inline Transform Transform::operator*(const Transform& t2) const {
         return Transform(Matrix4x4::Mul(m, t2.m), Matrix4x4::Mul(t2.mInv, mInv));
     }

     __host__ __device__ inline bool Transform::SwapsHandedness() const {
         Float det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
             m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
             m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
         return det < 0;
     }


     __host__ __device__ inline Transform Translate(const Vector3f& delta) {
         Matrix4x4 m(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0,
             1);
         Matrix4x4 minv(1, 0, 0, -delta.x, 0, 1, 0, -delta.y, 0, 0, 1, -delta.z, 0,
             0, 0, 1);
         return Transform(m, minv);
     }

     __host__ __device__ inline Transform Scale(Float x, Float y, Float z) {
         Matrix4x4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
         Matrix4x4 minv(1 / x, 0, 0, 0, 0, 1 / y, 0, 0, 0, 0, 1 / z, 0, 0, 0, 0, 1);
         return Transform(m, minv);
     }

     __host__ __device__ inline Transform RotateX(Float theta) {
         Float sinTheta = std::sin(Radians(theta));
         Float cosTheta = std::cos(Radians(theta));
         Matrix4x4 m(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0,
             0, 0, 0, 1);
         return Transform(m, Transpose(m));
     }

     __host__ __device__ inline Transform RotateY(Float theta) {
         Float sinTheta = std::sin(Radians(theta));
         Float cosTheta = std::cos(Radians(theta));
         Matrix4x4 m(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0,
             0, 0, 0, 1);
         return Transform(m, Transpose(m));
     }

     __host__ __device__ inline Transform RotateZ(Float theta) {
         Float sinTheta = std::sin(Radians(theta));
         Float cosTheta = std::cos(Radians(theta));
         Matrix4x4 m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1);
         return Transform(m, Transpose(m));
     }

     __host__ __device__ inline Transform Rotate(Float theta, const Vector3f& axis) {
         Vector3f a = Normalize(axis);
         Float sinTheta = std::sin(Radians(theta));
         Float cosTheta = std::cos(Radians(theta));
         Matrix4x4 m;
         // Compute rotation of first basis vector
         m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
         m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
         m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
         m.m[0][3] = 0;

         // Compute rotations of second and third basis vectors
         m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
         m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
         m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
         m.m[1][3] = 0;

         m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
         m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
         m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
         m.m[2][3] = 0;
         return Transform(m, Transpose(m));
     }
}

#endif // QZRT_CORE_TRANSFORM_H