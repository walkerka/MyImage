#ifndef GLRENDERER_H
#define GLRENDERER_H
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <list>
#include <math.h>

class GLRenderer;
class ShapeEffect;
class MixGreyscaleEffect;
class ColorOverrideEffect;
class MaskExpandEffect;
class MaskBlurEffect;
class SoftenMaskEffect;
class BlendEffect;
class FXAAEffect;
class SmaaEffect;
class BlitEffect;
class PaletteEffect;
class PaletteMixEffect;
class PaletteAddEffect;
class DetectHoleEffect;
class FillHoleEffect;
class Canvas2d;
class Canvas3d;
class GLMesh;
class Font;

float DegreeToRadian(float value);
float RadianToDegree(float value);

char* LoadFile(const char* path, int& size);
bool SaveFile(const char* path, const void* buf, int size);
unsigned char* LoadImg(const char* path, int& w, int& h, int& pixelSize);
unsigned char* LoadImgFromMemory(const void* buf, int len, int& w, int& h, int& pixelSize);
bool IsGif(const char* path);
unsigned char* LoadAnim(const char* path, int& w, int& h, int& pixelSize, int& frames);
void ReleaseImg(unsigned char* img);
bool SaveImg(const char* path, int w, int h, int pixelSize, unsigned char* data);
unsigned char* SaveImgToMemory(int w, int h, int pixelSize, unsigned char* data, int& resultSize);


class Vector2
{
public:
    float x;
    float y;
public:
    Vector2();
    Vector2(float x, float y);

    bool operator == (const Vector2& v) const;
    bool operator != (const Vector2& v) const;
    bool operator < (const Vector2& v) const;
    bool operator > (const Vector2& v) const;
    bool operator <= (const Vector2& v) const;
    bool operator >= (const Vector2& v) const;
    Vector2 operator * (float s) const;
    Vector2 operator / (float s) const;
    Vector2 operator - () const;
    Vector2 operator + (const Vector2& v) const;
    Vector2 operator - (const Vector2& v) const;
    Vector2 operator * (const Vector2& v) const;
    Vector2 operator / (const Vector2& v) const;
    Vector2& operator *= (float s);
    Vector2& operator /= (float s);
    Vector2& operator += (const Vector2& v);
    Vector2& operator -= (const Vector2& v);
    float Length() const;
    float LengthSq() const;
    float DistanceTo(const Vector2& v) const;
    float DistanceToSq(const Vector2& v) const;
    float Normalise();
    Vector2 GetNormalized() const;
    float Dot(const Vector2& v) const;
    Vector2 GetPerpendicular() const;
    bool IsPerpendicular(const Vector2& v, float epsilon = 0.0001f) const;
    bool IsParellel(const Vector2& v, float epsilon = 0.0001f) const;
    bool IsEqual(const Vector2& v, float epsilon = 0.0001f) const;
    bool IsZero(float epsilon = 0.0001f) const;
    Vector2 Lerp(const Vector2& v, float t) const;
    Vector2 Lerp(const Vector2& v, Vector2 t) const;
    void Set(float x, float y);
    void Truncate(float x0, float y0, float x1, float y1);
    void Truncate(const Vector2& low, const Vector2& high);
    float GetAngle(const Vector2& pivot);
};

class AABB2
{
public:
	float xMin;
	float yMin;
	float xMax;
	float yMax;

public:
	AABB2();
    AABB2(float xMin, float yMin, float xMax, float yMax);
	float Width() const;
	float Height() const;
	bool IsNull() const;
	bool HitTest(const AABB2& v) const;
	bool HitTest(const Vector2& p) const;
	void Union(const AABB2& v);
	void Union(const Vector2& v);
    void Extend(float size);
    AABB2 Intersect(const AABB2& v);
	void BuildFrom(Vector2* points, int n);
	void SetNull();
};


class Matrix3
{
public:
    float m00;
    float m10;
    float m20;

    float m01;
    float m11;
    float m21;

    float m02;
    float m12;
    float m22;
public:
    Matrix3()
    {
        Identity();
    }

    ~Matrix3()
    {
    }

    Matrix3(const float* m)
        :m00(m[0]),m10(m[1]),m20(m[2])
        ,m01(m[3]),m11(m[4]),m21(m[5])
        ,m02(m[6]),m12(m[7]),m22(m[8])
    {
    }

    Matrix3(float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22)
        :m00(m00),m10(m10),m20(m20)
        ,m01(m01),m11(m11),m21(m21)
        ,m02(m02),m12(m12),m22(m22)
    {
    }

    Matrix3 operator * (const Matrix3& m) const
    {
        return Matrix3(
            m00 * m.m00 + m01 * m.m10 + m02 * m.m20,   m00 * m.m01 + m01 * m.m11 + m02 * m.m21,  m00 * m.m02 + m01 * m.m12 + m02 * m.m22,
            m10 * m.m00 + m11 * m.m10 + m12 * m.m20,   m10 * m.m01 + m11 * m.m11 + m12 * m.m21,  m10 * m.m02 + m11 * m.m12 + m12 * m.m22,
            m20 * m.m00 + m21 * m.m10 + m22 * m.m20,   m20 * m.m01 + m21 * m.m11 + m22 * m.m21,  m20 * m.m02 + m12 * m.m12 + m22 * m.m22
            );
    }

    Vector2 operator * (const Vector2& vec) const
    {
        float x = vec.x * m00 + vec.y * m01 + m02;
        float y = vec.x * m10 + vec.y * m11 + m12;
        return Vector2(x, y);
    }

    void Transform(Vector2& vec) const
    {
        float x = vec.x * m00 + vec.y * m01 + m02;
        float y = vec.x * m10 + vec.y * m11 + m12;
        vec.Set(x, y);
    }

    void Transform(Vector2* vectors, int count) const
    {
        for (int i = 0; i < count; ++i)
        {
            float x = vectors[i].x;
            float y = vectors[i].y;
            vectors[i].x = x * m00 + y * m01 + m02;
            vectors[i].y = x * m10 + y * m11 + m12;
        }
    }

	AABB2 Transform(const AABB2& aabb) const
	{
		Vector2 vs[4] = {
			Vector2(aabb.xMin, aabb.yMin),
			Vector2(aabb.xMax, aabb.yMin),
			Vector2(aabb.xMax, aabb.yMax),
			Vector2(aabb.xMin, aabb.yMax)
		};
		Transform(vs, 4);
		AABB2 result;
		result.BuildFrom(vs, 4);
		return result;
	}

    Matrix3 Inverse() const
    {
        float det = 
            m00 * m11 * m22 
            + m10 * m21 * m02 
            + m20 * m01 * m12 
            - m20 * m11 * m02 
            - m10 * m01 * m22 
            - m00 * m21 * m12 
            ;

        float invDet = 1.0f / det;

        return Matrix3(
            (m11 * m22 - m21 * m12) * invDet,
            (m21 * m02 - m01 * m22) * invDet,
            (m01 * m12 - m11 * m02) * invDet,

            (m20 * m12 - m10 * m22) * invDet,
            (m00 * m22 - m20 * m02) * invDet,
            (m10 * m02 - m00 * m12) * invDet,

            (m10 * m21 - m20 * m11) * invDet,
            (m20 * m01 - m00 * m21) * invDet,
            (m00 * m11 - m10 * m01) * invDet
            );
    }

    void Identity()
    {
        m00 = 1.0f; m10 = 0.0f; m20 = 0.0f;
        m01 = 0.0f; m11 = 1.0f; m21 = 0.0f;
        m02 = 0.0f; m12 = 0.0f; m22 = 1.0f;
    }

    Vector2 GetXAxis() const
    {
        return Vector2(m00, m10);
    }

    Vector2 GetYAxis() const
    {
        return Vector2(m01, m11);
    }

    Vector2 GetTranslate() const
    {
        return Vector2(m02, m12);
    }

    void Translate(float x, float y)
    {
        *this = BuildTranslate(x, y) * *this;
    }

    void Scale(float x, float y)
    {
        *this = BuildScale(x, y) * *this;
    }

    void Rotate(float radians)
    {
        *this = BuildRotate(radians) * *this;
    }

    void LocalTranslateX(float delta)
    {
        const Vector2& d = GetXAxis().GetNormalized() * delta;
        m02 += d.x;
        m12 += d.y;
    }

    void LocalTranslateY(float delta)
    {
        const Vector2& d = GetYAxis().GetNormalized() * delta;
        m02 += d.x;
        m12 += d.y;
    }

    void LocalRotate(float deltaRadians)
    {
        float cosa = cosf(deltaRadians);
        float sina = sinf(deltaRadians);
        float r00 = cosa * m00 - sina * m10;
        float r01 = cosa * m01 - sina * m11;
        float r10 = sina * m00 + cosa * m10;
        float r11 = sina * m01 + cosa * m11;
        m00 = r00;
        m10 = r10;
        m01 = r01;
        m11 = r11;
    }

    static Matrix3 BuildTranslate(float x, float y)
    {
        return Matrix3(
                1.0f, 0.0f, x,
                0.0f, 1.0f, y,
                0.0f, 0.0f, 1.0f);
    }

    static Matrix3 BuildScale(float sx, float sy)
    {
        return Matrix3(
                  sx, 0.0f, 0.0f,
                0.0f,   sy, 0.0f,
                0.0f, 0.0f, 1.0f);
    }

    static Matrix3 BuildRotate(float radians)
    {
        float cosa = cosf(radians);
        float sina = sinf(radians);

        return Matrix3(
            cosa, -sina, 0.0f,
            sina,  cosa, 0.0f,
            0.0f,  0.0f, 1.0f);

        
    }
};


class Vector3
{
public:
    float x;
    float y;
    float z;
public:
    Vector3();
    Vector3(float x, float y, float z);
    bool operator == (const Vector3& v) const;
    bool operator != (const Vector3& v) const;
    bool operator < (const Vector3& v) const;
    bool operator > (const Vector3& v) const;
    bool operator <= (const Vector3& v) const;
    bool operator >= (const Vector3& v) const;
    Vector3 operator * (float s) const;
    Vector3 operator / (float s) const;
    Vector3 operator - () const;
    Vector3 operator + (const Vector3& v) const;
    Vector3 operator - (const Vector3& v) const;
    Vector3 operator * (const Vector3& v) const;
    Vector3 operator / (const Vector3& v) const;
    Vector3& operator *= (float s);
    Vector3& operator /= (float s);
    Vector3& operator += (const Vector3& v);
    Vector3& operator -= (const Vector3& v);
    float Length() const;
    float LengthSq() const;
    float DistanceTo(const Vector3& v) const;
    float DistanceToSq(const Vector3& v) const;
    float Normalise();
    Vector3 GetNormalized() const;
    float Dot(const Vector3& v) const;
    Vector3 Cross(const Vector3& v) const;
    bool IsPerpendicular(const Vector3& v, float epsilon = 0.0001f) const;
    bool IsParellel(const Vector3& v, float epsilon = 0.0001f) const;
    bool IsEqual(const Vector3& v, float epsilon = 0.0001f) const;
    bool IsZero(float epsilon = 0.0001f) const;
    Vector3 Lerp(const Vector3& v, float t) const;
    void Set(float x, float y, float z);
};

class Vector2i
{
public:
    int x;
    int y;

public:
    Vector2i() :x(0), y(0) {}
    Vector2i(int x, int y) :x(x), y(y) {}
};

class Vector3i
{
public:
    int x;
    int y;
    int z;

public:
    Vector3i():x(0),y(0),z(0) {}
    Vector3i(int x, int y, int z) :x(x), y(y), z(z) {}
};

class Matrix4
{
public:
    // column 0
    float m00;
    float m10;
    float m20;
    float m30;
    // column 1
    float m01;
    float m11;
    float m21;
    float m31;
    // column 2
    float m02;
    float m12;
    float m22;
    float m32;
    // column 3
    float m03;
    float m13;
    float m23;
    float m33;

public:
    Matrix4(void);
    ~Matrix4(void);
    Matrix4(const float* m, bool isColumnMajor);
    Matrix4(const Matrix4& m);
    Matrix4(
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33
        );
    void CopyTo(float* buf, bool isColumnMajor = true) const;
    Matrix4 operator * (const Matrix4& m) const;
    Vector3 operator * (const Vector3& v) const;
    Matrix4 Inverse() const;
    void Identity();
    void Transpose();
    Vector3 GetXAxis() const;
    Vector3 GetYAxis() const;
    Vector3 GetZAxis() const;
    Vector3 GetTranslate() const;
    void Translate(float x, float y, float z);
    void Scale(float x, float y, float z);
    void Rotate(float radians, float x, float y, float z);
    void RotateAt(float radians, float x, float y, float z, float posx, float posy, float posz);
    void LocalTranslateX(float delta);
    void LocalTranslateY(float delta);
    void LocalTranslateZ(float delta);
    void LocalRotate(float deltaRadians, float x, float y, float z);
    static Matrix4 BuildTranslate(float x, float y, float z);
    static Matrix4 BuildScale(float sx, float sy, float sz);
    static Matrix4 BuildRotate(float radians, float x, float y, float z);
    static Matrix4 BuildRotateAt(float radians, float x, float y, float z, float posx, float posy, float posz);
    static Matrix4 BuildFrustum(float left, float right, float bottom, float top, float nearZ, float farZ);
    static Matrix4 BuildPerspective(float fovy, float aspect, float nearZ, float farZ);
    static Matrix4 BuildOrtho(float left, float right, float bottom, float top, float nearZ, float farZ);
    static Matrix4 BuildBiasMatrix();
};

class AABB3
{
public:
    float xMin;
    float yMin;
    float zMin;
    float xMax;
    float yMax;
    float zMax;

public:
    AABB3();
    AABB3(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax);
    float Width() const;
    float Height() const;
    float Depth() const;
    bool IsNull() const;
    bool HitTest(const AABB3& v) const;
    bool HitTest(const Vector3& p) const;
    void Union(const AABB3& v);
    void Union(const Vector3& v);
    AABB3 Intersect(const AABB3& v);
    void BuildFrom(Vector3* points, int n);
    void SetNull();
};

class Plane3
{
public:
    Plane3(const Vector3& n, float p);
public:
    Vector3 mNormal;
    float mP;
};

class Ray3
{
public:
    Ray3(const Vector3& origin, const Vector3 direction);
    bool GetIntersection(const Plane3& plane, Vector3& intersection);
public:
    Vector3 mOrigin;
    Vector3 mDirection;
};

class Frustum
{
public:
    Frustum() {}
    Frustum(const Matrix4& mvp) { Extract(mvp); }
    ~Frustum() {}

    void Extract(const Matrix4& mvp);
    bool HitTest(const Vector3& point);
    bool HitTest(const AABB3& box);
private:
    float frustum[6][4];
};

class Transform3
{
public:
    Transform3();
    ~Transform3();
    const Vector3& GetXAxis() const { return mXAxis; }
    const Vector3& GetYAxis() const { return mYAxis; }
    const Vector3& GetZAxis() const { return mZAxis; }
    const Vector3& GetPosition() const { return mPosition; }
    void SetXAxis(const Vector3& x) { mXAxis = x; }
    void SetYAxis(const Vector3& y) { mYAxis = y; }
    void SetZAxis(const Vector3& z) { mZAxis = z; }
    void SetPosition(const Vector3& p) { mPosition = p; }
    Matrix4 ToMatrix();
    void BuildFrom(const Matrix4& mat);

    void LookAt(const Vector3& eye, const Vector3& look, const Vector3& up);
private:
    Vector3 mXAxis;
    Vector3 mYAxis;
    Vector3 mZAxis;
    Vector3 mPosition;
};

class Camera3
{
public:
    enum ProjectionType
    {
        ProjectionTypeNone,
        ProjectionTypePerspective,
        ProjectionTypeOrtho,
    };

    struct ProjectionParam
    {
        ProjectionType type;
        float nearZ;
        float farZ;
        float fovy;
        float aspect;
        float left;
        float right;
        float bottom;
        float top;
    };

    Camera3();
    ~Camera3();
    void SetPerspective(float fovy, float aspect, float nearZ, float farZ);
    void SetOrtho(float left, float right, float bottom, float top, float nearZ, float farZ);
    const Matrix4& GetProjectionMatrix() const { return mProjectMatrix; }
    const Matrix4& GetViewMatrix() const { return mViewMatrix; }
    Ray3 GetPickingRay(const Vector2& screenPos);
protected:
    Matrix4 mProjectMatrix;
    Matrix4 mViewMatrix;
    ProjectionParam mProjection;
};


class Camera3Free : public Camera3
{
public:
    Camera3Free();
    ~Camera3Free();
    void Reset();
    void LookAt(const Vector3& eye, const Vector3& look, const Vector3& up);
    void MoveForward(float delta);
    void MoveRight(float delta);
    void MoveUp(float delta);
    void Rotate(const Vector3& delta);
    void UpdateViewMatrix();
protected:
    Vector3 mRight;
    Vector3 mUp;
    Vector3 mBack;
    Vector3 mPosition;
};

class Camera3Flight : public Camera3
{
public:
    Camera3Flight(const Vector3& right, const Vector3& up);
    ~Camera3Flight();
    void MoveForward(float delta);
    void MoveRight(float delta);
    void MoveUp(float delta);
    void TurnUp(float delta);
    void TurnRight(float delta);
    void SetPosition(const Vector3& position);
    void LookAt(const Vector3& eye, const Vector3& look);
    void UpdateViewMatrix();
    void SetTransform(const Transform3& t);
    const Transform3& GetTransform();
private:
    Vector3 mRight;
    Vector3 mUp;
    Vector3 mBack;
    Vector3 mPosition;
    Transform3 mTransform;
    Vector3 mWorldUp;
};

class Camera3FirstPerson : public Camera3
{
public:
    Camera3FirstPerson(const Vector3& right, const Vector3& up);
    ~Camera3FirstPerson();
    void MoveForward(float delta);
    void MoveRight(float delta);
    void MoveUp(float delta);
    void TurnUp(float delta);
    void TurnRight(float delta);
    void SetPosition(const Vector3& position);
    void LookAt(const Vector3& eye, const Vector3& look);
    void UpdateViewMatrix();
private:
    Vector3 mRight;
    Vector3 mUp;
    Vector3 mBack;
    Vector3 mPosition;
    Vector3 mWorldUp;
};

class Camera3ThirdPerson : public Camera3
{
public:
    Camera3ThirdPerson(const Vector3& right, const Vector3& up);
    ~Camera3ThirdPerson();
    void SetPosition(const Vector3& position);
    void SetDistance(float distance);
    void SetTilt(float degree);
    void SetRotate(float degree);
    void Pan(const Vector3& delta);
    void TurnUp(float delta);
    void TurnRight(float delta);
    void Zoom(float delta);
    void UpdateViewMatrix();
    void SetTransform(const Transform3& t);
    const Transform3& GetTransform() { return mTransform; }
    Vector2 ScreenToGroundPoint(const Vector2& screenPos);
private:
    Vector3 mWorldRight;
    Vector3 mWorldUp;
    Vector3 mLookAt;
    float mDistance;
    float mRotate;
    float mTilt;
    Transform3 mTransform;
    
};

class Color
{
public:
    float r;
    float g;
    float b;
    float a;
public:
    Color(float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f):r(r), g(g), b(b), a(a) {}
    bool operator == (const Color& c) { return r == c.r && g == c.g && b == c.b && a == c.a; }
};

class GLTexture
{
    friend class GLRenderer;
    friend class GLRenderTarget;
    friend class GLShaderProgram;
    friend class GBuffer;
public:
    ~GLTexture();
    int GetWidth() const { return mWidth; }
    int GetHeight() const { return mHeight; }
    int GetChannels() const { return mChannels; }
    int GetId() const { return mTextureId; }
    void WriteTexture(const void* data, int w, int h, bool useMipmap);
    void Save(const char* path);

private:
    GLTexture(int width, int height, void* data, int channels, bool minSmooth, bool magSmooth, bool repeat, bool useMipmap, bool useAnisotropicFiltering, bool isDepth, int componentType);
    
    int mWidth;
    int mHeight;
    int mChannels;
    int mTextureId;
};

class GLRenderTarget
{
    friend class GLRenderer;
public:
    ~GLRenderTarget();
    int GetWidth() const { return mWidth; }
    int GetHeight() const { return mHeight; }
    int GetId() const { return mFrameBufferId; }
    void Clear(const Color& color);
    void Bind();
    void Unbind();
    void Blit(const GLRenderTarget* src, int x, int y, bool smooth);
    void Blit(const GLRenderTarget* src, int dstX, int dstY, int dstW, int dstH, int srcX, int srcY, int srcW, int srcH, bool smooth);
    void FloodFill(GLTexture* target, int* pts, int numPts, const Color& color, int expand);
    void FloodFill(GLTexture* target, GLRenderTarget* input);
    Color GetPixel(int x, int y);
    GLTexture* GetDepthTexture() { return mDepthTexture; }
    GLTexture* GetTexture(int index = 0) { return mTextures[index]; }
    void GetImage(void* data);
    void Save(const char* path);
    void Replace(int fboId, int w, int h);
    bool IsEqual(GLRenderTarget* src);
    bool IsSmoothMin() const { return mSmoothMin; }
    bool IsSmoothMag() const { return mSmoothMag; }

private:
    GLRenderTarget(int defaultFrameBufferId);
    GLRenderTarget(int width, int height, bool hasDepth, bool enableAA, void* data, bool smoothMin, bool smoothMag, int channels, int numTextures);

private:
    int mWidth;
    int mHeight;
    int mFrameBufferId;
    std::vector<GLTexture*> mTextures;
    GLTexture* mDepthTexture;
    bool mEnableAA;
    int mAABufferId;
    bool mDisableRelease;
    bool mSmoothMin;
    bool mSmoothMag;
};

class GLShaderProgram
{
    friend class GLRenderer;
public:
    GLShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource, const std::map<std::string, int>& bindLocations, int debugLine, const char* name = NULL);
    ~GLShaderProgram();

    void Bind();
    void Unbind();
    int GetUniformLocation(const char* name);

    void SetTexture(const char* name, int index, GLTexture* texture);
    void SetMatrix(const char* name, const float* mat);
    void SetFloat4(const char* name, float* v);
    void SetFloat3(const char* name, float* v);
    void SetFloat2(const char* name, float* v);
    void SetFloat(const char* name, float v);
    void SetFloat2(const char* name, float x, float y);
    void SetFloat3(const char* name, float x, float y, float z);
    void SetFloat4(const char* name, float x, float y, float z, float w);
    void SetFloat2Array(const char* name, const float* v, int count);

    void SetTexture(int loc, int index, GLTexture* texture);
    void SetMatrix(int loc, const float* mat);
    void SetFloat4(int loc, float* v);
    void SetFloat3(int loc, float* v);
    void SetFloat2(int loc, float* v);
    void SetFloat(int loc, float v);
    void SetFloat2(int loc, float x, float y);
    void SetFloat3(int loc, float x, float y, float z);
    void SetFloat4(int loc, float x, float y, float z, float w);
    void SetFloatArray(int loc, const float* v, int count);
    void SetFloat2Array(int loc, const float* v, int count);
    void SetFloat3Array(int loc, const float* v, int count);
    void SetFloat4Array(int loc, const float* v, int count);
    void SetInt(int loc, int v);

private:
    int mProgramId;
};

class GLUniformValue
{
    struct TextureValue
    {
        GLTexture* texture;
        int index;
    };

    struct FloatArray
    {
        const float* values;
        int count;
    };

    union Value
    {
        float value;
        float vec2[2];
        float vec3[3];
        float vec4[4];
        float mat4[16];
        FloatArray fv;
        TextureValue texture;
    };

    enum ValueType
    {
        ValueTypeNone,
        ValueTypeTexture,
        ValueTypeFloat,
        ValueTypeFloat2,
        ValueTypeFloat3,
        ValueTypeFloat4,
        ValueTypeFloat2Array,
        ValueTypeMatrix4
    };

public:
    GLUniformValue();
    GLUniformValue(int index, GLTexture* texture);
    GLUniformValue(float x, float y, float z, float w);
    GLUniformValue(float x, float y, float z);
    GLUniformValue(float x, float y);
    GLUniformValue(float v);
    GLUniformValue(const float* mat);
    GLUniformValue(const Matrix4& mat);
    GLUniformValue(const float* vecs, int components, int count);
    ~GLUniformValue();

    void Apply(const char* name, GLShaderProgram* program);

private:
    ValueType mType;
    Value mValue;
};

class GLShaderState
{
    typedef std::map<std::string, GLUniformValue> ValueMap;
public:
    GLShaderState();
    ~GLShaderState();
    void SetValue(const char* name, const GLUniformValue& value);
    void Apply(GLShaderProgram* program);
    void Reset();

private:
    ValueMap mValues;
};



class GLAttribute
{
    friend class GLMesh;
    friend class GLRenderer;
public:
    GLAttribute(int index, int components, int count, int offset);
    ~GLAttribute();

    int GetIndex() { return mIndex; }
    int GetComponent() { return mComponets; }
    int GetCount() { return mCount; }
    int GetVertexSize() { return mVertexSize; }

private:
    int mIndex;
    int mComponets;
    int mVertexSize;
    int mCount;
    int mOffset;
};

#define MAX_GL_ATTRIBUTES 8
class GLMesh
{
    friend class GLRenderer;
	friend class Canvas2d;
    typedef std::vector<GLAttribute*> AttributeMap;
public:

    enum GLAttributeType
    {
        GLAttributeTypePosition = 0,
        GLAttributeTypeTexcoord = 1,
        GLAttributeTypeColor = 2,
        GLAttributeTypeNormal = 3,
        GLAttributeTypeUserDefined = 4
    };

    struct AttributeDesc
    {
        AttributeDesc(int index = 0, int components = 3):index(index), components(components) {}
        int index;
        int components;
    };

    enum GLPrimitiveType
    {
        GLPrimitiveTypeTriangleList,
        GLPrimitiveTypeLineList,
    };
    
    GLMesh(int vertexCapacity, int indexCapacity, AttributeDesc* attributes, int attrNum, GLPrimitiveType primitiveType = GLPrimitiveTypeTriangleList);
    ~GLMesh();

    void* GetVertex(int index) { return mVertices + mVertexSize * index; }
    unsigned short* GetIndex(int index) { return mIndices + index; }
    
    void DrawSubset(int offset, int indexCount);
    void Draw();

    void SetFloat2(int attr, int at, float x, float y);
    void SetVector2(int attr, int at, const Vector2& value);
    void SetFloat3(int attr, int at, float x, float y, float z);
    void SetVector3(int attr, int at, const Vector3& value);
    void SetFloat4(int attr, int at, float x, float y, float z, float w);
    void SetIndex(int value) { mIndices[mIndexCount++] = (unsigned short)value; }

    void Clear();
    void AddRect(const AABB2& rect, const AABB2& texcoord, const Color& color);
    void AddBlendRect(const AABB2& rect, const AABB2& texcoordSrc, const AABB2& texcoordDst);
    void AddCircle(const Vector2& center, float radius, const Color& c0, const Color& c1);
    void AddLineV(const Vector3& v0, const Vector3& v1, const Color& color);
    void AddFreeLine(const Vector3* vertices, int num, const Color& color);
    float AddTextureLine(const Vector3* vertices, int num, const Vector2& size, float step, float offset
        , GLTexture* texture, const AABB2& texcoord, const Color& color);
    void AddPolygon(const Vector2* vertices, int num, const Color& color);
    void AddCube(const Vector3& center, const Vector3& size, const Color& color);
    void AddSphere(const Vector3& center, float r, const Color& color, int xDivide, int yDivide);
    void AddCone(const Vector3& center, float radius, float height, const Color& color, int xDivide, int yDivide);
    void AddCylinder(const Vector3& center, float radius, float height, const Color& color, int xDivide, int yDivide);
    void AddCircleCap(const Vector3& center, float radius, const Color& color, int samples, bool isUp);
    void AddArrow(const Vector3& from, const Vector3& to, float arrowRadius, float arrowLength, float axisRadius, const Color& color, int samples);
    void AddFrameIndicator(const Vector3& center, const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis, float length, int samples);
    void AddGrid(const Vector3& center, const Vector3& xDir, const Vector3& yDir, const Vector2& size, const Color& color, int xSegments, int ySegments);
    void ApplyTransform(const Matrix4& mat, int vertexFrom, int vertexTo);
    void BuildVBO();

    int GetVertexCount() const { return mVertexCount; }
    int GetIndexCount() const { return mIndexCount; }
    int GetVertexCapcity() const { return mVertexCapacity; }
    int GetIndexCapacity() const { return mIndexCapacity; }
    void SetVertexCount(int n);

    static GLMesh* LoadObj(const char* path);
	bool SaveObj(const char* path);
private:
    int mVertexCapacity;
    int mIndexCapacity;
    int mVertexSize;
    unsigned char* mVertices;
    unsigned short* mIndices;
    GLAttribute* mAttributes[MAX_GL_ATTRIBUTES];
    int mVertexCount;
    unsigned short mIndexCount;
    int mVertexBufferId;
    int mIndexBufferId;
    int mVao;
    GLPrimitiveType mPrimitiveType;
};

class GLRenderer
{
public:
    GLRenderer();
    ~GLRenderer();

    GLTexture* CreateTexture(int width, int height, void* data, int channels = 4, bool minSmooth = true, bool magSmooth = true, bool repeat = true, bool useMipmap = true);
    GLTexture* CreateTexture(const char* path);
    GLRenderTarget* CreateTarget(int width, int height, bool hasDepth, bool enableAA, void* data, bool minSmooth, bool magSmooth, int channels = 4);
    GLRenderTarget* CreateTarget(int width, int height, bool hasDepth, bool enableAA);
    GLRenderTarget* CreateTarget(int width, int height, void* data);
    GLRenderTarget* CreateTarget(const char* path);
    GLRenderTarget* CreateTarget(int fboId);
    GLRenderTarget* CreateGBuffer(int width, int height, int numTextures);
    GLShaderProgram* CreateShaderFromFile(const char* vsPath, const char* fsPath, std::map<std::string, int>& attrLocs);

    GLMesh* CreateCube(const Vector3& center, const Vector3& size, const Color& color);
    GLMesh* CreateSphere(const Vector3& center, float radius, const Color& color, int xDivide, int yDivide);
    GLMesh* CreateFrameIndicator(const Vector3& center, const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis, float length, int samples);
    GLMesh* CreateScreenAlignedQuad();
    GLMesh* CreateGrid(const Vector3& center, const Vector3& xDir, const Vector3& yDir, const Vector2& size, const Color& color, int xSegments, int ySegments);
    GLMesh* CreateText(const char* text, float xPos, float yPos, float fontHeight, const Color& color);
    AABB2 GetTextSize(const char* text, float fontHeight);

    GLTexture* GetDefaultTexture() { return mDefaultTexture; }
    GLRenderTarget* GetDefaultTarget() { return mDefaultTarget; }
    GLRenderTarget* SetDefaultTarget(GLRenderTarget* target);
    GLRenderTarget* GetTempTarget() { return mTempTarget; }
    GLTexture* GetFontTexture() { return mFontTexture; }
    static GLRenderer* GetInstance() { return sInstance; }

    ShapeEffect* GetShapeEffect() { return mShapeEffect; }
    MixGreyscaleEffect* GetMixGreyscaleEffect() { return mMixGreyscaleEffect; }
    ColorOverrideEffect* GetColorOverrideEffect() { return mColorOverrideEffect; }
    FXAAEffect* GetFXAAEffect() { return mFxaaEffect; }
    SmaaEffect* GetSMAAEffect() { return mSmaaEffect; }

    BlendEffect* GetMixEffect() { return mMixEffect; }
    BlendEffect* GetBehindEffect() { return mBehindEffect; }
    BlendEffect* GetEraseEffect() { return mEraseEffect; }
    BlendEffect* GetInvertEffect() { return mInvertEffect; }
    BlendEffect* GetMaskEffect() { return mMaskEffect; }
    BlendEffect* GetCutEffect() { return mCutEffect; }

    MaskBlurEffect* GetMaskBlurEffect() { return mMaskBlurEffect; }

    BlendEffect* GetMaskMixEffect() { return mMaskMixEffect; }
    BlendEffect* GetMaskBehindEffect() { return mMaskBehindEffect; }
    BlendEffect* GetMaskEraseEffect() { return mMaskEraseEffect; }
    BlendEffect* GetMaskAddEffect() { return mMaskAddEffect; }
    BlendEffect* GetMaskMultiplyEffect() { return mMaskMultiplyEffect; }
    BlendEffect* GetMaskInverseMixEffect() { return mMaskInverseMixEffect; }
    BlendEffect* GetMaskMaskEffect() { return mMaskMaskEffect; }
    BlendEffect* GetMaskCutEffect() { return mMaskCutEffect; }

    BlendEffect* GetReplaceAlhpaEffect() { return mReplaceAlhpaEffect; }

    MaskExpandEffect* GetMaskExpandMixEffect() { return mMaskExpandMixEffect; }
    MaskExpandEffect* GetMaskExpandBehindEffect() { return mMaskExpandBehindEffect; }
    MaskExpandEffect* GetMaskExpandEraseEffect() { return mMaskExpandEraseEffect; }
    MaskExpandEffect* GetMaskExpandAddEffect() { return mMaskExpandAddEffect; }
    MaskExpandEffect* GetMaskExpandMultiplyEffect() { return mMaskExpandMultiplyEffect; }
    
    SoftenMaskEffect* GetSoftenMaskMixEffect() { return mSoftenMaskMixEffect; }
    SoftenMaskEffect* GetSoftenMaskBehindEffect() { return mSoftenMaskBehindEffect; }
    SoftenMaskEffect* GetSoftenMaskEraseEffect() { return mSoftenMaskEraseEffect; }
    SoftenMaskEffect* GetSoftenMaskAddEffect() { return mSoftenMaskAddEffect; }
    SoftenMaskEffect* GetSoftenMaskMultiplyEffect() { return mSoftenMaskMultiplyEffect; }

    PaletteEffect* GetPaletteEffect() { return mPaletteEffect; }
    PaletteMixEffect* GetPaletteMixEffect() { return mPaletteMixEffect; }
    PaletteAddEffect* GetPaletteAddEffect() { return mPaletteAddEffect; }
    DetectHoleEffect* GetDetectHoleEffect() { return mDetectHoleEffect; }
    FillHoleEffect* GetFillHoleEffect() { return mFillHoleEffect; }
    
    BlitEffect* GetBlitEffect() { return mBlitEffect; }
    GLMesh* GetBlitMesh() { return mBlitMesh; }

    Canvas2d* GetCanvas2d() { return mCanvas2d; }
    Canvas3d* GetCanvas3d() { return mCanvas3d; }

    void Reset();


private:
    GLTexture* mDefaultTexture;
    GLRenderTarget* mDefaultTarget;
    GLRenderTarget* mTempTarget;
    GLShaderProgram* mDefaultProgram;
    typedef std::map<std::string, GLShaderProgram*> ProgramCollection;
    ProgramCollection mPrograms;

    BlitEffect* mBlitEffect;
    GLMesh* mBlitMesh;

    ShapeEffect* mShapeEffect;
    MixGreyscaleEffect* mMixGreyscaleEffect;
    ColorOverrideEffect* mColorOverrideEffect;
    FXAAEffect* mFxaaEffect;
    SmaaEffect* mSmaaEffect;

    BlendEffect* mMixEffect;
    BlendEffect* mBehindEffect;
    BlendEffect* mEraseEffect;
    BlendEffect* mInvertEffect;
    BlendEffect* mMaskEffect;
    BlendEffect* mCutEffect;

    MaskBlurEffect* mMaskBlurEffect;

    BlendEffect* mMaskMixEffect;
    BlendEffect* mMaskBehindEffect;
    BlendEffect* mMaskEraseEffect;
    BlendEffect* mMaskAddEffect;
    BlendEffect* mMaskMultiplyEffect;
    BlendEffect* mMaskInverseMixEffect;
    BlendEffect* mMaskMaskEffect;
    BlendEffect* mMaskCutEffect;

    BlendEffect* mReplaceAlhpaEffect;

    MaskExpandEffect* mMaskExpandMixEffect;
    MaskExpandEffect* mMaskExpandBehindEffect;
    MaskExpandEffect* mMaskExpandEraseEffect;
    MaskExpandEffect* mMaskExpandAddEffect;
    MaskExpandEffect* mMaskExpandMultiplyEffect;

    SoftenMaskEffect* mSoftenMaskMixEffect;
    SoftenMaskEffect* mSoftenMaskBehindEffect;
    SoftenMaskEffect* mSoftenMaskEraseEffect;
    SoftenMaskEffect* mSoftenMaskAddEffect;
    SoftenMaskEffect* mSoftenMaskMultiplyEffect;

    PaletteEffect* mPaletteEffect;
    PaletteMixEffect* mPaletteMixEffect;
    PaletteAddEffect* mPaletteAddEffect;

    DetectHoleEffect* mDetectHoleEffect;
    FillHoleEffect* mFillHoleEffect;

    Canvas2d* mCanvas2d;
    Canvas3d* mCanvas3d;
    Font* mFont;
    GLTexture* mFontTexture;
    static GLRenderer* sInstance;
};

class BlitEffect
{
public:
	BlitEffect(GLRenderer* renderer);
    ~BlitEffect();

    void Blit(GLRenderTarget* dst, int dstX, int dstY, int dstW, int dstH, GLRenderTarget* src, int srcX, int srcY, int srcW, int srcH, bool smooth);

private:
    void CreateDirectBlitShader();
    void CreateSmoothBlitShader();

private:
    GLShaderProgram* mProgramDirect;
    int mMvpLocDirect;
    int mTex0LocDirect;
    GLShaderProgram* mProgramSmooth;
    int mMvpLocSmooth;
    int mTex0LocSmooth;
    int mInvSizeLocSmooth;
};

class SuperSamplingEffect
{
public:
    SuperSamplingEffect(GLRenderer* renderer);
    ~SuperSamplingEffect();

    void SetMvp(const Matrix4& mvp);
    void SetTexture(GLTexture* texture);
    void SetSamples(int samples);
    void SetPixelSize(float w, float h);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mTex0Loc;
    int mPixelSizeLoc;
    int mSamplesLoc;
    int mNumSamplesLoc;
};

class ShapeEffect
{
public:
    ShapeEffect(GLRenderer* renderer);
    ~ShapeEffect();

    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
    void SetTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mShapeColorLoc;
    int mTex0Loc;
};


class MixGreyscaleEffect
{
public:
    MixGreyscaleEffect(GLRenderer* renderer);
    ~MixGreyscaleEffect();

    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
    void SetTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mShapeColorLoc;
    int mTex0Loc;
};

class PaletteEffect
{
public:
	PaletteEffect(GLRenderer* renderer);
	~PaletteEffect();

    void SetColor(const Color& color);
    void SetMv(const Matrix4& mv);
	void SetMvp(const Matrix4& mvp);
    void SetDstTexture(GLTexture* texture);
    void SetSrcTexture(GLTexture* texture);
    void SetPaletteTexture(GLTexture* texture);
	void Bind();
	void Unbind();

private:
	GLShaderProgram* mProgram;
    int mShapeColorLoc;
    int mMvLoc;
	int mMvpLoc;
	int mTexSrcLoc;
    int mTexPaletteLoc;
    int mTexDstLoc;
    int mDstSizeInvLoc;
};


class PaletteMixEffect
{
public:
    PaletteMixEffect(GLRenderer* renderer);
    ~PaletteMixEffect();

    void SetMv(const Matrix4& mv);
    void SetMvp(const Matrix4& mvp);
    void SetDstTexture(GLTexture* texture);
    void SetSrcTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvLoc;
    int mMvpLoc;
    int mTexSrcLoc;
    int mTexDstLoc;
    int mDstSizeInvLoc;
};

class PaletteAddEffect
{
public:
    PaletteAddEffect(GLRenderer* renderer);
    ~PaletteAddEffect();

    void SetMv(const Matrix4& mv);
    void SetMvp(const Matrix4& mvp);
    void SetDstTexture(GLTexture* texture);
    void SetSrcTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvLoc;
    int mMvpLoc;
    int mTexSrcLoc;
    int mTexDstLoc;
    int mDstSizeInvLoc;
};

class DetectHoleEffect
{
public:
    DetectHoleEffect(GLRenderer* renderer);
    ~DetectHoleEffect();
    
    void Render(GLRenderTarget* dst, GLTexture* src, GLRenderTarget* temp);

private:
    typedef std::pair<int,int> Position;
    void CreatePatternDetectionProgram();
    void CreatePatternExpandProgram();
    void AddPatternImage(std::vector<std::vector<Position> >& patterns, int patternSize, int* patternImg);
    void LoadPatterns(std::vector<std::vector<Position> >& patterns, int& patternSize);
private:
    GLShaderProgram* mProgram;
    int mTexSrcLoc;
    GLShaderProgram* mExpandProgram;
    int mExpandTexHoleLoc;
    int mExpandTexSrcLoc;
    GLMesh* mQuad;
};

class FillHoleEffect
{
public:
    FillHoleEffect(GLRenderer* renderer);
    ~FillHoleEffect();
    void Render(GLRenderTarget* dst, GLTexture* holeTexture, GLTexture* colorTexture);

private:
    GLShaderProgram* mProgram;
    int mTexHoleLoc;
    int mTexColorLoc;
    GLMesh* mQuad;
};

class FreelineEffect
{
public:
    FreelineEffect(GLRenderer* renderer);
    ~FreelineEffect();

    void SetPoints(Vector3* points, int numPoints);
    void SetMvp(const Matrix4& mvp);
    void SetColor(const Color& color);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mPointsLoc;
    int mNumPointsLoc;
    int mShapeColorLoc;
};


class PointCloudEffect
{
public:
    PointCloudEffect(GLRenderer* renderer);
    ~PointCloudEffect();

    void SetPoints(Vector3* points, int numPoints);
    void SetMvp(const Matrix4& mvp);
    void SetColor(const Color& color);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mPointsLoc;
    int mNumPointsLoc;
    int mShapeColorLoc;
};

class ColorOverrideEffect
{
public:
    ColorOverrideEffect(GLRenderer* renderer);
    ~ColorOverrideEffect();
    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
    void SetTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mShapeColorLoc;
    int mTex0Loc;
};

class BlendEffect
{
public:
    BlendEffect(GLRenderer* renderer, const char* blendStr);
    virtual ~BlendEffect();

    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
	void SetMv(const Matrix4& mv);
    void SetSrcTexture(GLTexture* texture);
    void SetDstTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
	int mMvLoc;
    int mShapeColorLoc;
    int mTexDstLoc;
    int mTexSrcLoc;
	int mDstSizeInvLoc;
};

class MaskExpandEffect
{
public:
    MaskExpandEffect(GLRenderer* renderer, const char* blendStr);
    virtual ~MaskExpandEffect();

    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
    void SetMv(const Matrix4& mv);
    void SetSrcTexture(GLTexture* texture);
    void SetDstTexture(GLTexture* texture);
    void SetExpand(int expand);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mShapeColorLoc;
    int mTexDstLoc;
    int mTexSrcLoc;
    int mDstSizeInvLoc;
    int mExpandLoc;
    int mExpandTexcoordStep;
};

class MaskBlurEffect
{
public:
    MaskBlurEffect(GLRenderer* renderer, const char* blendStr);
    virtual ~MaskBlurEffect();

    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
    void SetMv(const Matrix4& mv);
    void SetSrcTexture(GLTexture* texture);
    void SetDstTexture(GLTexture* texture);
    void SetRadius(int r);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mShapeColorLoc;
    int mTexDstLoc;
    int mTexSrcLoc;
    int mDstSizeInvLoc;
    int mExpandLoc;
    int mExpandTexcoordStep;
};

class SoftenMaskEffect
{
public:
    SoftenMaskEffect(GLRenderer* renderer, const char* blendStr);
    virtual ~SoftenMaskEffect();

    void SetColor(const Color& color);
    void SetMvp(const Matrix4& mvp);
    void SetMv(const Matrix4& mv);
    void SetSrcTexture(GLTexture* texture);
    void SetDstTexture(GLTexture* texture);
    void SetRadius(int r);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mShapeColorLoc;
    int mTexDstLoc;
    int mTexSrcLoc;
    int mDstSizeInvLoc;
    int mExpandLoc;
    int mExpandTexcoordStep;
};

class FXAAEffect
{
public:
    FXAAEffect(GLRenderer* renderer);
    ~FXAAEffect();

    void SetMvp(const Matrix4& mvp);
    void SetTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mTexLoc;
    int mRcpFrameLoc;
};

class Fxaa3Effect
{
public:
    Fxaa3Effect(const char* dir);
    ~Fxaa3Effect(void);
    void Render(GLRenderTarget* dst, GLTexture* src);

private:
    GLShaderProgram* mProgram0;
    GLMesh* mQuad;
    int mScreenSizeLoc0;
    int mColorTexLoc0;
};

class SmaaEffect
{
public:
    SmaaEffect(const char* dir);
    ~SmaaEffect(void);

    void RenderRGB(GLRenderTarget* dst, GLTexture* src, GLRenderTarget* temp0, GLRenderTarget* temp1);
    void RenderRGBA(GLRenderTarget* dst, GLTexture* src, GLRenderTarget* temp0, GLRenderTarget* temp1, GLRenderTarget* temp2, GLRenderTarget* temp3, GLRenderTarget* temp4);

private:
    GLTexture* mAreaTexture;
    GLTexture* mSearchTexture;

    GLShaderProgram* mProgram0;
    GLShaderProgram* mProgram1;
    GLShaderProgram* mProgram2;
    GLMesh* mQuad;

    int mScreenSizeLoc0;
    int mColorTexLoc0;

    int mScreenSizeLoc1;
    int mEdgeTexLoc1;
    int mAreaTexLoc1;
    int mSearchTexLoc1;

    int mScreenSizeLoc2;
    int mColorTexLoc2;
    int mBlendTexLoc2;

    GLShaderProgram* mProgramAlphaToMask;
    int mColorTexLoc3;

    GLShaderProgram* mProgramCombine;
    int mColorTexLoc4;
    int mAlphaTexLoc4;
    int mScreenSizeLoc4;

    GLShaderProgram* mProgramExpand;
    int mColorTexLoc5;
    int mScreenSizeLoc5;
};

class PresentEffect
{
public:
    PresentEffect(GLRenderer* renderer);
    ~PresentEffect();
    void SetTexture(GLTexture* texture);
    void Bind();
    void Unbind();

private:
    GLShaderProgram* mProgram;
    int mTex0Loc;
};

class FloatTestEffect
{
public:
	FloatTestEffect();
    ~FloatTestEffect();
    void Render();

private:
    GLRenderer* mRenderer;
    GLMesh* mQuad;
    GLShaderProgram* mProgram;
};

class Transform2
{
public:
    Transform2();
    Transform2(const Matrix3& mat);
    ~Transform2();
    
    void ResetTransform();
    void SetTranslate(float x, float y);
    void SetScale(float x, float y);
    void SetRotate(float radians);
    void ModTranslate(float x, float y);
    void ModScale(float x, float y);
    void ModRotate(float radians);
    void Merge(const Transform2& t);

    const Vector2& GetTranslate() const { return mTranslate; }
    const Vector2& GetScale() const { return mScale; }
    float GetRotate() const { return mRotate; }
    void GetMatrix(Matrix3& matrix) const;

    bool operator == (const Transform2& t);

public:
    static Transform2 IDENTITY;
private:
    Vector2 mTranslate;
    Vector2 mScale;
    float mRotate;
};

class Camera2
{
public:
    Camera2();
    ~Camera2();

    const Transform2& GetTransform();
    void Reset();
    void Set(const Vector2 position, float scale, float rotate);
    void SetTransform(const Transform2& tran);

    Vector2 WorldToView(const Vector2& position);
    Vector2 ViewToWorld(const Vector2& position);
    
    void OnPanBegin(float x, float y);
    void OnPanMove(float x, float y);
    void OnPanEnd(float x, float y);
    void OnZoomBegin(float x, float y);
    void OnZoomMove(float x, float y);
    void OnZoomEnd(float x, float y);
    void OnRotateBegin(float x, float y);
    void OnRotateMove(float x, float y);
    void OnRotateEnd(float x, float y);
    
    void OnPinchZoomBegin(float x, float y);
    void OnPinchZoomMove(float s);
    void OnPinchZoomEnd(float s);

    // combined pan/zoom/rotate
    void OnPinchBegin(const Vector2& p0, const Vector2& p1);
    void OnPinchMove(const Vector2& p0, const Vector2& p1);
    void OnPinchEnd(const Vector2& p0, const Vector2& p1);

private:
    enum PinchState
    {
        PinchStateEvaluate,
        PinchStateMove,
        PinchStateScale,
        PinchStateRotate
    };

    Transform2 mTransform;
    Vector2 mPanAnchor;
    Vector2 mZoomWorldAnchor;
    Vector2 mZoomScreenAnchor;
    float mZoomScale;
    float mZoomInitLength;
    Vector2 mRoateScreenAnchor;
    Vector2 mRoateWorldAnchor;
    bool mRotateBegin;
    float mRotateRefAngle;
    float mRotateStartAngle;
    Vector2 mPinchPoints[2];
    PinchState mPinchState;
    float mInitalPinchDistance;
    float mTotalPinchDistance;
    int mPinchUpdates;
};

enum BlendMode
{
    BlendModeMix,
    BlendModeGreyscaleMix,
    BlendModeAdd,
    BlendModeSource,
    BlendModeColorOverride,
    BlendModeInvert,
    BlendModeFXAA,
    BlendModeMaskMix,
    BlendModeMaskInverseMix,
    BlendModeTransparentMix,
    BlendModeTransparentBehind,
    BlendModeTransparentErase,
    BlendModeTransparentAdd,
    BlendModeTransparentMultiply,
    BlendModeTransparentInvert,
    BlendModeTransparentMask,
    BlendModeTransparentCut,
    BlendModeTransparentBlur,
    BlendModeTransparentReplaceAlpha,
    BlendModeTransparentMaskBlur,
    BlendModeTransparentMaskMix,
    BlendModeTransparentMaskBehind,
    BlendModeTransparentMaskErase,
    BlendModeTransparentMaskAdd,
    BlendModeTransparentMaskMultiply,
    BlendModeTransparentMaskInverseMix,
    BlendModeTransparentMaskMask,
    BlendModeTransparentMaskCut,
    BlendModeTransparentMaskExpandMix,
    BlendModeTransparentMaskExpandBehind,
    BlendModeTransparentMaskExpandErase,
    BlendModeTransparentMaskExpandAdd,
    BlendModeTransparentMaskExpandMultiply,
    BlendModeTransparentSoftenMaskMix,
    BlendModeTransparentSoftenMaskBehind,
    BlendModeTransparentSoftenMaskErase,
    BlendModeTransparentSoftenMaskAdd,
    BlendModeTransparentSoftenMaskMultiply,
    BlendModePalettePresent,
    BlendModePaletteMix,
    BlendModePaletteAdd,
};

enum AntiAliasingMode
{
    AntiAliasingModeNone,
    AntiAliasingModeMSAA,
    AntiAliasingModeFSAA
};

struct Canvas2dState
{
    Canvas2dState()
        :target(GLRenderer::GetInstance()->GetDefaultTarget())
        ,texture(GLRenderer::GetInstance()->GetDefaultTexture())
        ,texture2(GLRenderer::GetInstance()->GetDefaultTexture())
        ,color(1, 1, 1, 1)
        ,blendMode(BlendModeMix)
        ,aaMode(AntiAliasingModeNone)
        ,samples(2)
        ,radius(10)
    {
    }

    Canvas2dState(GLRenderTarget* target,
                GLTexture* texture,
                GLTexture* texture2,
                Color color,
                Transform2 transform,
                BlendMode blendMode,
                AntiAliasingMode aaMode,
                int samples,
                int radius)
        :target(target)
        ,texture(texture)
        ,texture2(texture2)
        ,color(color)
        ,transform(transform)
        ,blendMode(blendMode)
        ,aaMode(aaMode)
        ,samples(samples)
        ,radius(radius)
    {
    }

    GLRenderTarget* target;
    GLTexture* texture;
    GLTexture* texture2;
    Color color;
    Transform2 transform;
    BlendMode blendMode;
    AntiAliasingMode aaMode;
    int samples;
    int radius;
};

class Canvas2d
{
public:
    enum OutlineDirection
    {
        OutlineDirectionCenter,
        OutlineDirectionOuter,
        OutlineDirectionInner
    };
    Canvas2d(GLRenderer* renderer);
    ~Canvas2d();

    void SetState(const Canvas2dState& state);

    void BeginBatchDraw();
    void DrawCircle(float x, float y, float r, const Color& color);
    void DrawGradientCircle(float x, float y, float r, const Color* colors, float* stops, int stopNum);
    void DrawRect(float x, float y, float w, float h, const Color& color);
    void DrawRectOutline(const AABB2& rect, const Color& color, float outlineWidth, OutlineDirection direction = OutlineDirectionCenter);
    void DrawLine(float x0, float y0, float w0, float x1, float y1, float w1, const Color& color);
    void DrawLine(float x0, float y0, float x1, float y1, float width, const Color& color);
    void DrawLineJoint(float x0, float y0, float x1, float y1, float x2, float y2, float w, const Color& color);
    void DrawImage(float x, float y, AABB2 srcRect, const Color& color);
    void DrawImage(AABB2 dstRect, AABB2 srcRect, const Color& color);
    void DrawImage(AABB2 dstRect, AABB2 srcRect, const Color& color, const Transform2& trans);
    void DrawPolygon(const Vector2* vertices, int num, const Color& color);
    void EndBatchDraw();

    void DrawFreeline(Vector3* vertices, int num);
    void DrawString(const char* text, float xPos, float yPos, float fontHeight, const Color& color);

private:
    void Draw(float* positions, float* texcoords, float* colors, int vertexNum, unsigned short* indices, int indexNum);
    void Flush();
    void RenderTile(int w, int h, int x, int y, int tw, int th, int samples);

private:
    GLRenderer* mRenderer;
    GLMesh* mMesh;
    BlendMode mBlendMode;
    AntiAliasingMode mAAMode;
    GLRenderTarget* mTempTarget;
    GLRenderTarget* mAATarget;
    GLRenderTarget* mAATargetHalf;
    GLRenderTarget* mFSAATarget;
    GLRenderTarget* mFSAABlitTarget;
    GLRenderTarget* mTarget;
    GLTexture* mSrcTexture;
    GLTexture* mSrcTexture2;
    Color mColor;
    Transform2 mTransform;
    Matrix4 mMvp;
    Matrix4 mMv;
    int mSamples;
    int mRadius;
    bool mHasMatrixChange;
    bool mDeferredDraw;
    AABB2 mBounds;
    FreelineEffect* mFreelineEffect;
};

class Effect3d
{
public:
    virtual ~Effect3d() {}
    virtual void Bind() = 0;
    virtual void SetProjectionMatrix(const Matrix4& /*mat*/) {}
    virtual void SetViewMatrix(const Matrix4& /*mat*/) {}
    virtual void SetWorldMatrix(const Matrix4& /*mat*/) {}
    virtual void SetTextures(GLTexture** /*textures*/) {}
};

class Effect3dUnlighted: public Effect3d
{
public:
    Effect3dUnlighted();
    virtual ~Effect3dUnlighted();
    virtual void Bind();
    virtual void SetProjectionMatrix(const Matrix4& mat);
    virtual void SetViewMatrix(const Matrix4& mat);
    virtual void SetWorldMatrix(const Matrix4& mat);
    virtual void SetTextures(GLTexture** textures);
    virtual void SetColor(const Color& color);
private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mTexDiffuseLoc;
    int mColorLoc;
    Matrix4 mViewMat;
    Matrix4 mWorldMat;
    Matrix4 mProjMat;
    GLTexture* mTexDiffuse;
    Color mColor;
};

class Effect3dForwardShading: public Effect3d
{
public:
    Effect3dForwardShading();
    virtual ~Effect3dForwardShading();
    virtual void Bind();
    virtual void SetProjectionMatrix(const Matrix4& mat);
    virtual void SetViewMatrix(const Matrix4& mat);
    virtual void SetWorldMatrix(const Matrix4& mat);
    virtual void SetTextures(GLTexture** textures);
private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
	int mNormalMatLoc;
    int mTexDiffuseLoc;
    int mLightDir;
    int mLightColor;
    int mAmbientFactor;
    int mShiness;
    Matrix4 mViewMat;
    Matrix4 mWorldMat;
    Matrix4 mProjMat;
    GLTexture* mTexDiffuse;
};

class Effect3dGPass : public Effect3d
{
public:
    Effect3dGPass();
    virtual ~Effect3dGPass();
    virtual void Bind();
    virtual void SetProjectionMatrix(const Matrix4& mat);
    virtual void SetViewMatrix(const Matrix4& mat);
    virtual void SetWorldMatrix(const Matrix4& mat);
    virtual void SetTextures(GLTexture** textures);
private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mNormalMatLoc;
    int mTexDiffuseLoc;
    Matrix4 mViewMat;
    Matrix4 mWorldMat;
    Matrix4 mProjMat;
    GLTexture* mTexDiffuse;
};


class Effect3dDSDiHemisphereLighting : public Effect3d
{
public:
    Effect3dDSDiHemisphereLighting();
    virtual ~Effect3dDSDiHemisphereLighting();
    virtual void Bind();
    virtual void SetProjectionMatrix(const Matrix4& mat);
    virtual void SetViewMatrix(const Matrix4& mat);
    virtual void SetWorldMatrix(const Matrix4& mat);
    virtual void SetTextures(GLTexture** textures);

    void SetSkyDirection(const Vector3& dir);
    void SetSkyColor(const Color& color);
    void SetGroundColor(const Color& color);
private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mNormalMatLoc;
    int mShadowMatLoc;
    int mTexDiffuseLoc;
    int mTexNormalLoc;
    int mTexPositionLoc;
    int mTexAOLoc;
    int mLightDirLoc;
    int mSkyColorLoc;
    int mGroundColorLoc;
    Matrix4 mViewMat;
    Matrix4 mWorldMat;
    Matrix4 mProjMat;
    GLTexture* mTexDiffuse;
    GLTexture* mTexNormal;
    GLTexture* mTexPosition;
    GLTexture* mTexAO;
    Vector3 mLightDirection;
    Color mSkyLightColor;
    Color mGroundLightColor;
};

class Effect3dDSDirectionalLighting : public Effect3d
{
public:
	Effect3dDSDirectionalLighting();
	virtual ~Effect3dDSDirectionalLighting();
	virtual void Bind();
	virtual void SetProjectionMatrix(const Matrix4& mat);
	virtual void SetViewMatrix(const Matrix4& mat);
	virtual void SetWorldMatrix(const Matrix4& mat);
	virtual void SetTextures(GLTexture** textures);

	void SetLightDirection(const Vector3& dir);
	void SetLightColor(const Color& color);
    void SetShadowMatrix(const Matrix4& mat);
private:
	GLShaderProgram* mProgram;
	int mMvpLoc;
	int mMvLoc;
	int mNormalMatLoc;
    int mShadowMatLoc;
	int mTexDiffuseLoc;
	int mTexNormalLoc;
	int mTexPositionLoc;
    int mTexShadowLoc;
	int mLightDirLoc;
	int mLightColorLoc;
	Matrix4 mViewMat;
	Matrix4 mWorldMat;
	Matrix4 mProjMat;
    Matrix4 mShadowMat;
	GLTexture* mTexDiffuse;
	GLTexture* mTexNormal;
	GLTexture* mTexPosition;
    GLTexture* mTexShadow;
	Vector3 mLightDirection;
	Color mLightColor;
};


class Effect3dDSPointLighting : public Effect3d
{
public:
    Effect3dDSPointLighting();
    virtual ~Effect3dDSPointLighting();
    virtual void Bind();
    virtual void SetProjectionMatrix(const Matrix4& mat);
    virtual void SetViewMatrix(const Matrix4& mat);
    virtual void SetWorldMatrix(const Matrix4& mat);
    virtual void SetTextures(GLTexture** textures);

    void SetLightPosition(const Vector3& position);
    void SetLightColor(const Color& color);
    void SetShadowMatrix(const Matrix4& mat);
private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mNormalMatLoc;
    int mShadowMatLoc;
    int mTexDiffuseLoc;
    int mTexNormalLoc;
    int mTexPositionLoc;
    int mTexShadowLoc;
    int mLightPosLoc;
    int mLightColorLoc;
    Matrix4 mViewMat;
    Matrix4 mWorldMat;
    Matrix4 mProjMat;
    Matrix4 mShadowMat;
    GLTexture* mTexDiffuse;
    GLTexture* mTexNormal;
    GLTexture* mTexPosition;
    GLTexture* mTexShadow;
    Vector3 mLightPosition;
    Color mLightColor;
};


class Effect3dSSAO: public Effect3d
{
public:
	Effect3dSSAO(GLRenderer* renderer, bool depthOnly, int maxSamples, int randTextureSize, float radius, float falloff, float bias, float density);
	~Effect3dSSAO();

	void SetSamples(int maxSample);
	void SetRadius(float radius);
	void SetFalloff(float falloff);
	void SetBias(float bias);
	void SetDensity(float density);

	virtual void Bind();
	virtual void SetProjectionMatrix(const Matrix4& mat);
	virtual void SetTextures(GLTexture** textures);

private:
	void Build();

private:
	GLRenderer* mRenderer;
	bool mDepthOnly;
	int mMaxSamples;
	int mRandTextureSize;
	float mRadius;
	float mFalloff;
	float mBias;
	float mDensity;
	GLShaderProgram* mAOProgram;
	GLTexture* mRandTexture;
	std::vector<float> mRandDirs;
	Matrix4 mProjectMatrix;
	GLTexture* mDepthTexture;
	GLTexture* mNormalTexture;
	GLTexture* mPositionTexture;
};


class Effect3dAOBlur : public Effect3d
{
public:
    Effect3dAOBlur(GLRenderer* renderer, int samples, float radius);
    ~Effect3dAOBlur();

    virtual void Bind();
    virtual void SetTextures(GLTexture** textures);
    void SetSamples(int maxSample);
    void SetRadius(float radius);

private:
    void Build();

private:
    GLRenderer* mRenderer;
    int mMaxSamples;
    float mRadius;
    GLShaderProgram* mProgram;
    GLTexture* mColorTexture;
    GLTexture* mNormalTexture;
    std::vector<float> mRandDirs;
};

class Effect3dShadowPass : public Effect3d
{
public:
    Effect3dShadowPass();
    virtual ~Effect3dShadowPass();
    virtual void Bind();
    virtual void SetProjectionMatrix(const Matrix4& mat);
    virtual void SetViewMatrix(const Matrix4& mat);
    virtual void SetWorldMatrix(const Matrix4& mat);
private:
    GLShaderProgram* mProgram;
    int mMvpLoc;
    int mMvLoc;
    int mNormalMatLoc;
    Matrix4 mViewMat;
    Matrix4 mWorldMat;
    Matrix4 mProjMat;
};

struct Canvas3dState
{
public:
    Canvas3dState();

public:
    GLRenderTarget* target;
    Matrix4 transform;
    GLTexture* textures[8];
    Effect3d* effect;
    Camera3* camera;
};

class Canvas3d
{
public:
    Canvas3d(GLRenderer* renderer);
    ~Canvas3d();

    void SetState(const Canvas3dState& state);
    void Draw(GLMesh* mesh);
	void DrawFullScreenQuad();
	void RenderSSAO(GLRenderTarget* target, GLTexture* depthTexture, Matrix4 projectionMatrix, float radius, float falloff, float bias, float density);
	void RenderSSAO(GLRenderTarget* target, GLTexture* positionTexture, GLTexture* normalTexture, const Matrix4& projectionMatrix, float radius, float falloff, float bias, float density);
    Effect3dUnlighted* GetUnlightedEffect() { return mUnlightedEffect; }
    Effect3dForwardShading* GetForwardShadingEffect() { return mForwardShadingEffect; }
    Camera3* GetDefaultCamera() { return &mDefaultCamera; }
    Effect3dGPass* GetGPassEffect() { return mGPassEffect; }
	Effect3dDSDirectionalLighting* GetDSDirectionalLightingEffect() { return mDSDirectionalLightingEffect; }
    Effect3dDSPointLighting* GetDSPointLightingEffect() { return mDSPointLightingEffect; }
    Effect3dDSDiHemisphereLighting* GetDSDiHemisphereLightingEffect() { return mDSHemisphereLightingEffect; }
	Effect3dSSAO* GetDeferredSSAOEffect() { return mDeferredSSAOEffect; }
    Effect3dAOBlur* GetAOBlurEffect() { return mAOBlurEffect; }
    Effect3dShadowPass* GetShadowPassEffect() { return mShadowPassEffect; }
private:
    GLRenderer* mRenderer;
    Effect3dUnlighted* mUnlightedEffect;
    Effect3dForwardShading* mForwardShadingEffect;
    Camera3 mDefaultCamera;
	Effect3dSSAO* mDeferredSSAOEffect;
    Effect3dAOBlur* mAOBlurEffect;
    Effect3dGPass* mGPassEffect;
	Effect3dDSDirectionalLighting* mDSDirectionalLightingEffect;
    Effect3dDSPointLighting* mDSPointLightingEffect;
    Effect3dDSDiHemisphereLighting* mDSHemisphereLightingEffect;
    Effect3dShadowPass* mShadowPassEffect;
	GLMesh* mQuad;
};

class UIView;
class UIApplication;

enum UIEventType
{
    UIEventTypeTouchDown,
    UIEventTypeTouchMove,
    UIEventTypeTouchUp,
    UIEventTypeResize,
    UIEventTypeClick,
    UIEventTypeClose,
    UIEventTypeDestroy,
    UIEventTypeRepaint,
    UIEventTypeValueChange,
};

class UIEvent
{
public:
    virtual ~UIEvent() {};
    virtual UIEventType GetType() = 0;
};

class UIEventListener
{
public:
    virtual ~UIEventListener() {}
    virtual bool OnEvent(UIEvent* event, UIView* sender) = 0;
};

class UITouchEvent: public UIEvent
{
public:
    UITouchEvent(UIEventType type, float x, float y, float pressure):mType(type), mX(x),mY(y),mPressure(pressure) {}
    ~UITouchEvent() {}
    virtual UIEventType GetType() { return mType; }
    float GetX() const { return mX; }
    float GetY() const { return mY; }
    float GetPressure() const { return mPressure; }

private:
    UIEventType mType;
    float mX;
    float mY;
    float mPressure;
};


class UIClickEvent: public UIEvent
{
public:
    UIClickEvent(float x, float y):mX(x),mY(y) {}
    ~UIClickEvent() {}
    virtual UIEventType GetType() { return UIEventTypeClick; }
    float GetX() const { return mX; }
    float GetY() const { return mY; }

private:
    float mX;
    float mY;
};

class UIResizeEvent: public UIEvent
{
public:
    UIResizeEvent(float w, float h):mWidth(w),mHeight(h) {}
    ~UIResizeEvent() {}
    virtual UIEventType GetType() { return UIEventTypeResize; }
    float GetWidth() const { return mWidth; }
    float GetHeight() const { return mHeight; }

private:
    float mWidth;
    float mHeight;
};

class UICloseEvent: public UIEvent
{
public:
    UICloseEvent() {}
    ~UICloseEvent() {}
    virtual UIEventType GetType() { return UIEventTypeClose; }
};

class UIDestroyEvent: public UIEvent
{
public:
    UIDestroyEvent() {}
    ~UIDestroyEvent() {}
    virtual UIEventType GetType() { return UIEventTypeDestroy; }
};


class UIRepaintEvent: public UIEvent
{
public:
    UIRepaintEvent() {}
    ~UIRepaintEvent() {}
    virtual UIEventType GetType() { return UIEventTypeRepaint; }
};

class UIValueChangeEvent: public UIEvent
{
public:
    UIValueChangeEvent(float value):mValue(value) {}
    ~UIValueChangeEvent() {}
    virtual UIEventType GetType() { return UIEventTypeValueChange; }
    float GetValue() const { return mValue; }
private:
    float mValue;
};

enum UIRenderableType
{
    UIRenderableTypeRect,
    UIRenderableTypePolygon,
    UIRenderableTypeLabel,
    UIRenderableTypeBitmap,
};

class UIRenderable
{
public:
    virtual ~UIRenderable() {}
    virtual UIRenderableType GetType() = 0;
};

class UIRect:
    public UIRenderable
{
public:
    UIRect(float width, float height, const Color& color);
    ~UIRect();
    virtual UIRenderableType GetType() { return UIRenderableTypeRect; }

    void SetPosition(const Vector2& pos);
    void SetSize(float width, float height);
    void SetColor(const Color& color);
    const Vector2& GetPosition() const { return mPosition; }
    const Vector2& GetSize() const { return mSize; }
    const Color& GetColor() const { return mColor; }

private:
    Vector2 mPosition;
    Vector2 mSize;
    Color mColor;
};

class UIPolygon:
    public UIRenderable
{
public:
    UIPolygon(const std::vector<Vector2>& vertices, const Color& color);
    ~UIPolygon();
    virtual UIRenderableType GetType() { return UIRenderableTypePolygon; }

    void SetPosition(const Vector2& pos);
    void SetColor(const Color& color);
    const Vector2& GetPosition() const { return mPosition; }
    const Color& GetColor() const { return mColor; }
    const std::vector<Vector2>& GetVertices() const { return mVertices; }

private:
    Vector2 mPosition;
    std::vector<Vector2> mVertices;
    Color mColor;
};

struct UITextureRect
{
    std::string name;
    int x;
    int y;
    int width;
    int height;
};

class UIBitmap:
    public UIRenderable
{
public:
    UIBitmap(float w, float h, const UITextureRect& texture);
    virtual UIRenderableType GetType() { return UIRenderableTypeBitmap; }
    void SetTexture(const UITextureRect& texture);
    void SetPosition(const Vector2& pos);
    void SetSize(float width, float height);
    void SetOpacity(float opacity);
    const Vector2& GetPosition() const { return mPosition; }
    const Vector2& GetSize() const { return mSize; }
    float GetOpacity() const { return mOpacity; }
    const UITextureRect& GetTexture() const { return mTexture; }

private:
    Vector2 mPosition;
    Vector2 mSize;
    float mOpacity;
    UITextureRect mTexture;
};

class UIText:
    public UIRenderable
{
public:
    UIText(const std::string& text, float fontHeight);
    virtual UIRenderableType GetType() { return UIRenderableTypeLabel; }
    void SetText(const std::string& text);
    void SetPosition(const Vector2& pos);
    void SetColor(const Color& color);
    void SetFontHeight(float fontHeight);
    const std::string& GetText() const { return mText; }
    const Vector2& GetPosition() const { return mPosition; }
    const Color& GetColor() const { return mColor; }
    float GetFontHeight() const { return mFontHeight; }
    float GetWidth() const { return mWidth; }
    float GetHeight() const { return mHeight; }
    
private:
    std::string mText;
    Vector2 mPosition;
    Color mColor;
    float mFontHeight;
    float mWidth;
    float mHeight;
};

class UILayout;

struct UIMessage
{
public:
    UIMessage(UIEvent* e, UIView* s):event(e), sender(s) {}

    UIEvent* event;
    UIView* sender;
};

class UIView
{
public:
    UIView(UIApplication* app, float x, float y, float w, float h);
    virtual ~UIView();

    void AddChildView(UIView* v);
    void RemoveChildView(UIView* v);
    int GetChildViewCount();
    UIView* GetChildView(int index);
    void SetPosition(float x, float y);
    void SetPosition(const Vector2& position);
    void ModPosition(const Vector2& deltaPosition);
    float GetWidth() const { return mWidth; }
    float GetHeight() const { return mHeight; }
    const Vector2& GetPosition() const { return mPosition; }
    void AddListener(UIEventListener* listener);
    void RemoveListener(UIEventListener* listener);
    void SetLayout(UILayout* layout);
    void UpdateLayout();
    UIView* GetTopViewAt(float x, float y);
    Vector2 WorldToLocal(const Vector2& worldPos);
    Vector2 LocalToWorld(const Vector2& localPos);
    void GetRenderables(std::vector<UIRenderable*>& objects);
    void PostEvent(UIEvent* ev, UIView* sender);
    void Destroy();
    void Update();

    virtual void Render(std::vector<UIRenderable*>& objects);
    virtual void OnEvent(UIEvent* e);
    virtual bool HitTest(float x, float y);
    virtual void Resize(float w, float h);
    virtual Vector2 GetMinSize();
    virtual Vector2 GetMaxSize();

protected:
    UIApplication* mApplication;
    Vector2 mPosition;
    float mWidth;
    float mHeight;
    UIView* mParentView;
    UILayout* mLayout;

    std::vector<UIView*> mChildViews;
    std::vector<UIEventListener*> mListeners;
};


class UILayout
{
public:
    virtual ~UILayout() {}
    virtual void Update() = 0;

    void SetView(UIView* view) { mView = view; }
protected:
    UIView* mView;
};

class UIRenderTarget
{
public:
    virtual ~UIRenderTarget() {}
    virtual int GetWidth() const = 0;
    virtual int GetHeight() const = 0;
    virtual const std::string& GetName() const = 0;
};

class UIRenderer
{
public:
    virtual ~UIRenderer() {}
    virtual void Render(std::vector<UIRenderable*>& objects) = 0;
    virtual void SetSize(int w, int h) = 0;
    virtual void AddTexture(const std::string& name, const std::string& path) = 0;
    virtual UIRenderTarget* AddRenderTarget(const std::string& name, int width, int height) = 0;
    virtual void AddFont(const std::string& name, const std::string& path) = 0;
};

class UIApplicationListener
{
public:
    virtual ~UIApplicationListener() {}
    virtual void OnRepaint() = 0;
    virtual void OnQuit() = 0;
};

class UIApplication
{
public:
    UIApplication(UIRenderer* renderer, int w, int h, UIApplicationListener* listener);
    ~UIApplication();
    void Render();
    void OnEvent(UIEvent* e);
    UIView* GetRoot() { return mRoot; }
    void PostEvent(UIEvent* ev, UIView* sender);
    void Quit();
    void Update();
private:
    UITouchEvent* GetLocalTouchEvent(UITouchEvent* e);
    void InitUIResources();

private:
    UIApplicationListener* mListener;
    UIView* mRoot;
    UIView* mFocusView;
    UIRenderer* mRenderer;
    bool mNeedRepaint;
    std::deque<UIMessage> mEventQueue;
};


class UIRootView: public UIView
{
public:
    UIRootView(UIApplication* app, float x, float y, float w, float h);
    ~UIRootView();
    void Render(std::vector<UIRenderable*>& objects);

private:
    UIRect* mUiRect;
};

class UIWindow: public UIView
{
public:
    UIWindow(UIApplication* app, float x, float y, float w, float h);
    ~UIWindow();
    void Render(std::vector<UIRenderable*>& objects);
    void AddContentView(UIView* view);

private:
    class WinTitleListener: public UIEventListener
    {
    public:
        WinTitleListener(UIWindow* win);
        virtual bool OnEvent(UIEvent* event, UIView* sender);
    private:
        UIWindow* mWindow;
    };
    friend class UIWindowTitleBar;
    UIRect* mBgRect;
    WinTitleListener mTitleListener;
    UIView* mContainerView;
};

class UIWindowTitleBar: public UIView
{
public:
    UIWindowTitleBar(UIApplication* app, UIWindow* window);
    ~UIWindowTitleBar();
    void OnEvent(UIEvent* e);
    void Render(std::vector<UIRenderable*>& objects);

private:
    UIRect* mBgRect;
    UIText* mTitleLabel;
    Vector2 mDragStartPosition;
    Vector2 mWindowStartPosition;
};


class UIButton: public UIView
{
public:
    UIButton(UIApplication* app, float w, float h, const UITextureRect& normalTexture, const UITextureRect& pressedTexture);
    UIButton(UIApplication* app, const char* text);
    ~UIButton();
    void OnEvent(UIEvent* e);
    void Render(std::vector<UIRenderable*>& objects);
    std::string GetText() { return mText ? mText->GetText() : ""; }
private:
    UITextureRect mNormalTexture;
    UITextureRect mPressedTexture;
    bool mPressed;
    UIBitmap* mBitmap;
    UIText* mText;
    UIRect* mRect;
    float mPadding;
};

class UISlider: public UIView
{
public:
    UISlider(UIApplication* app, float maxValue = 100.0f, float w=200, float h=32);
    ~UISlider();
    void OnEvent(UIEvent* e);
    void Render(std::vector<UIRenderable*>& objects);
    float GetValue();
    void SetValue(float v);
    void UpdateValue(float x, float y);

private:
    float mMaxValue;
    float mValue;
    UIText* mText;
    UIRect* mRect;
    UIRect* mCursor;
    float mPadding;
};

class UIImageView : public UIView
{
public:
    UIImageView(UIApplication* app, UIRenderTarget* target);
    UIImageView(UIApplication* app, float w, float h, const UITextureRect& texture);
    UIImageView(UIApplication* app, UIBitmap* target);
    ~UIImageView();
    virtual void Render(std::vector<UIRenderable*>& objects);
    UIBitmap* GetImage() { return mBitmap; }
private:
    UIBitmap* mBitmap;
};

class UIGLRenderTarget:
    public UIRenderTarget
{
public:
    UIGLRenderTarget(const std::string& name, int width, int height);
    virtual ~UIGLRenderTarget();
    
    virtual int GetWidth() const { return mTarget->GetWidth(); }
    virtual int GetHeight() const { return mTarget->GetHeight(); }
    virtual const std::string& GetName() const { return mName; }
    GLRenderTarget* GetTarget() { return mTarget; }
    void SetTarget(GLRenderTarget* target);

private:
    std::string mName;
    GLRenderTarget* mTarget;
};

class UIGLRenderer: public UIRenderer
{
public:
    UIGLRenderer();
    ~UIGLRenderer();
    virtual void Render(std::vector<UIRenderable*>& objects);
    virtual void SetSize(int w, int h);
    virtual void AddTexture(const std::string& name, const std::string& path);
    virtual UIRenderTarget* AddRenderTarget(const std::string& name, int width, int height);
    virtual void AddFont(const std::string& name, const std::string& path);

    GLTexture* GetTexture(const std::string& name);

private:
    int mWidth;
    int mHeight;
    std::map<std::string, GLTexture*> mTextures;
    std::map<std::string, UIGLRenderTarget*> mTargets;
};

class Node;

class Component
{
public:
    virtual ~Component();
    Node* GetNode();
    virtual bool Save();
    virtual bool Load();
};

class NodeListener
{
public:
    virtual ~NodeListener() {}
    virtual void OnChildNodeAdded(Node* node) = 0;
    virtual void OnChildNodeRemoved(Node* node) = 0;
};

class Node
{
public:
    Node();
    ~Node();
    
    bool Save();
    bool Load();

    Node* GetParentNode();
    Node* GetChildNodeAt(int index);
    void AddChildNode(Node* node);
    void RemoveChildNode(Node* node);

    void AddComponent(Component* comp);
    void RemoveComponent(Component* comp);
    Component* GetComponentAt(int index);
    int GetNumComponents();

    void AddNodeListener(NodeListener* l);
    void RemoveNodeListener(NodeListener* l);
};

class AnimationClip
{
public:
    virtual ~AnimationClip() {}
    virtual bool Load() = 0;
    virtual bool Save() = 0;
    virtual bool isLooped() = 0;
};

class NodeAnimationClip:
    public AnimationClip
{
public:
    void GetTransform(float frame, Matrix4& transform);
};

class SkeletonAnimationClip:
    public AnimationClip
{
public:
    void GetBoneTransformations(float frame, std::vector<Matrix4>& boneTransforms);
};

class SpriteAnimationClip:
    public AnimationClip
{
public:
    int GetSpriteId(float frame);
    int AddSpriteRect(const AABB2& rect);
    const AABB2& GetSpriteRect();
};

class AnimationController
{
public:
    bool Load();
    bool Save();
    void Play();
    void Pause();
    void Stop();
};

class StaticMesh
{
public:
    bool Load();
    bool Save();
};

class SkinnedMesh
{
public:
    bool Load();
    bool Save();
};

class Sprite
{
public:
    bool Load();
    bool Save();
};

class AnimatedSprite
{
public:
    bool Load();
    bool Save();
};

class Material
{
public:
    Material();
    virtual ~Material();
};

class SpriteMaterial:
    public Material
{
public:
    void SetTexture(GLTexture* texture);
    void AddSpriteRect(const AABB2& rect);
    void SetSpriteIds(std::vector<int>& ids);
};

class PhongMaterial:
    public Material
{
public:
    bool IsTransparent();
    bool IsNormalMapped();
    bool IsShadowMapped();
    void SetTransparent(bool isTransparent);
    void SetColorMap(GLTexture* texture);
    void SetNormalSpecularMap(GLTexture* texture);
    void SetMvp();
    void SetNormalMatrix();
};

class SkinnedMeshMaterial:
    public PhongMaterial
{
public:
    void SetBoneTransforms(std::vector<Matrix4> transforms);
};

class StaticMeshComponent:
    public Component
{
public:
    Material* GetMatrial();
    StaticMesh* GetMesh();
    void SetMaterial(Material* material);
    void SetMesh(GLMesh* mesh);
};

class SkinnedMeshComponent:
    public Component
{
public:
    Material* GetMatrial();
    SkinnedMesh* GetMesh();
    void SetMaterial(Material* material);
    void SetMesh(GLMesh* mesh);
    void SetAnimationController(AnimationController* animController);
};

class SpriteComponent:
    public Component
{
public:
};

class SpatialPartitionComponent:
        public Component
{
public:
    virtual void Add(Node* node) = 0;
    virtual void Remove(Node* node) = 0;
    virtual void Query(const Frustum& frustum, std::vector<Node*>& nodes) = 0;
    virtual void Query(const AABB3& bounds, std::vector<Node*>& nodes) = 0;
    virtual void Query(const Ray3& ray, std::vector<Node*>& nodes) = 0;
};

class LinearPartitionComponent:
    public Component
{
public:
    virtual void Add(Node* node);
    virtual void Remove(Node* node);
    virtual void Query(const Frustum& frustum, std::vector<Node*>& nodes);
    virtual void Query(const AABB3& bounds, std::vector<Node*>& nodes);
    virtual void Query(const Ray3& ray, std::vector<Node*>& nodes);
};

class OctreeComponent:
    public Component
{
public:
    virtual void Add(Node* node);
    virtual void Remove(Node* node);
    virtual void Query(const Frustum& frustum, std::vector<Node*>& nodes);
    virtual void Query(const AABB3& bounds, std::vector<Node*>& nodes);
    virtual void Query(const Ray3& ray, std::vector<Node*>& nodes);
};

// Component base node hierarchy
class Scene
{
public:
    Scene();
    ~Scene();

    bool Save();
    bool Load();

    Node* GetRoot();
};

class Spatial3dComponent:
        public Component
{
public:
    const Matrix4& GetMatrix();
    const AABB3& GetBounds();
    const Matrix4& GetTransform();
};

class Camera3dComponent:
    public Component
{
public:
    Camera3dComponent();
    ~Camera3dComponent();

    const Matrix4& GetViewMatrix();
    const Matrix4& GetProjectMatrix();

    void SetPerspective(float fovY, float aspect, float near, float far);
    void SetOrtho(float left, float right, float bottom, float top, float near, float far);

    void LookAt(const Vector3& eye, const Vector3& center, const Vector3& up);
};

class Light3dComponent:
    public Component
{
public:
    Light3dComponent();
    ~Light3dComponent();

    bool IsShadowCaster();
    void SetShadowCaster(bool enabled);
};

class ParticleSystemComponent:
    public Component
{
public:
    ParticleSystemComponent();
    ~ParticleSystemComponent();
};

// deferred shading
// frustum culling
// state sort
// depth sort
// occlusion culling
// instanced rendering
// dynamic batching
class GLSceneRenderer
{
public:
    GLSceneRenderer();
    ~GLSceneRenderer();

    GLRenderTarget* GetRenderTarget();
    void ResizeRenderTarget(int width, int height);
    void SetCamera(Camera3dComponent* camera);
    void SetScene(Scene* scene);
    void Render();
private:
    GLRenderTarget* mTarget;
    GLRenderTarget* mGBufferTarget;
    Camera3dComponent* mCamera;
    Scene* mScene;
};

// class RayTracingSceneRenderer:
//     public SceneRenderer
// {
// public:
//     RayTracingSceneRenderer();
//     ~RayTracingSceneRenderer();
// };



#endif
