#include "glrenderer.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <set>
#include <fstream>
#include "stb_image.h"
#include "stb_image_write.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef __ANDROID__
#ifdef USE_OPENGL3_ES
#include <GLES3/gl3.h>
#else
#include <GLES2/gl2.h>
#endif
#include <jni.h>
#include <android/log.h>
#define NO_TESS2
#ifndef NO_TESS2
#include "libtess2.h"
#endif

#define  LOG_TAG    "libgl2jni"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOG_LONG LOGE
#else

#ifdef _WIN32
#include <windows.h>
#ifdef USE_OPENGL3
#include "GL/gl3w.h"
#else
#include "GL/glew.h"
#include <GL/gl.h>
#endif
#include <GL/glu.h>

#define  LOGE(...) { char buf[512] = {0}; sprintf(buf, __VA_ARGS__); OutputDebugStringA(buf);}
#define  LOG_LONG(buf) {OutputDebugStringA(buf);}

#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if (TARGET_OS_IPHONE)
#include <GLES2/gl2.h>
#define NO_TESS2
#else
#include <gl3.h>
#define NO_TESS2
#ifndef USE_OPENGL3
#define USE_OPENGL3
#endif
#endif

#include <qdebug.h>
#define LOGE qDebug
#define  LOG_LONG LOGE
#endif


#ifdef __linux__
#include "GL/gl3w.h"
#include "libtess2.h"

#define  LOGE printf
#define  LOG_LONG LOGE

#endif

#endif

#ifdef QT_WIDGETS_LIB
#include <QFile>
char* LoadFile(const char* path, int& size)
{
    QFile f(path);
    if (f.open(QIODevice::ReadOnly))
    {
        qint64 s = f.size();
        char* buf = new char[s + 1];
        f.read(buf, s);
        buf[s] = '\0';
        f.close();
        size = s;
        f.close();
        return buf;
    }
    else
    {
        size = 0;
        return NULL;
    }

}

bool SaveFile(const char* path, const void* buf, int size)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
    {
        return false;
    }
    f.write((const char*)buf, size);
    f.close();
    return true;
}

#include <QImage>
#include <QByteArray>
#include <QBuffer>
#include <QTime>
#include <QDebug>

unsigned char* LoadImg(const char* path, int& w, int& h, int& pixelSize)
{
    unsigned char* data = NULL;
    QImage img(path);
    if (img.isNull())
    {
        unsigned char* sd = stbi_load(path, &w, &h, &pixelSize, 4);
        if (sd)
        {
            data = new unsigned char[w * h * pixelSize];
            for (int i = 0; i < h; ++i)
            {
                memcpy(data + i * w * pixelSize, sd + (h - 1 - i) * w * pixelSize, w * pixelSize);
            }
            stbi_image_free(sd);
        }
    }
    else
    {
        img = img.mirrored();
        w = img.width();
        h = img.height();
        if (img.format() == QImage::Format_RGB888)
        {
            pixelSize = 3;
        }
        else if (img.format() == QImage::Format_Grayscale8)
        {
            pixelSize = 1;
        }
        else
        {
            pixelSize = 4;
            if (img.format() != QImage::Format_RGBA8888 && img.format() != QImage::Format_RGB888)
            {
                img = img.convertToFormat(QImage::Format_RGBA8888);
            }
        }

        data = new unsigned char[w * h * pixelSize];
        for (int i = 0; i < h; ++i)
        {
            memcpy(data + i * w * pixelSize, img.scanLine(i), w * pixelSize);
        }
    }
    return data;
}

static void FlipY(unsigned char* p, int w, int h, int pixelSize)
{
	int rowSize = w * pixelSize;
	unsigned char* temp = new unsigned char[rowSize];
	for (int i = 0; i < h / 2; ++i)
	{
		unsigned char* r0 = p + rowSize * i;
		unsigned char* r1 = p + rowSize * (h - 1 - i);
		memcpy(temp, r0, rowSize);
		memcpy(r0, r1, rowSize);
		memcpy(r1, temp, rowSize);
	}
	delete[] temp;
}

unsigned char* LoadImgFromMemory(const void* buf, int len, int& w, int& h, int& pixelSize)
{
    QTime time;
    time.start();
	unsigned char* mData = NULL;
#if 0
    QImage img;
    if (img.loadFromData((const uchar*)buf, len, "JPG  "))
    {
        qDebug() << "qt=" << time.elapsed();
        time.restart();
        w = img.width();
        h = img.height();
        pixelSize = img.pixelFormat().channelCount();
        mData = new unsigned char[w * h * pixelSize];
        int rowBytes = img.bytesPerLine();
        for (int i = 0; i < h; ++i)
        {
            const void* row = img.scanLine(i);
            memcpy(mData + (h - 1 - i) * w * pixelSize, row, rowBytes);
        }
        qDebug() << "copyFlip=" << time.elapsed();
    }
#else
    unsigned char* stbImg = stbi_load_from_memory((const stbi_uc*)buf, len, &w, &h, &pixelSize, 0);
    qDebug() << "stb=" << time.elapsed();
    time.restart();
    if (stbImg)
    {
        FlipY(stbImg, w, h, pixelSize);
        qDebug() << "flip=" << time.elapsed();
        time.restart();
        mData = new unsigned char[w * h * pixelSize];
        memcpy(mData, stbImg, w * h * pixelSize);
        stbi_image_free(stbImg);
        qDebug() << "copy=" << time.elapsed();
        time.restart();
    }
#endif
	return mData;
}

extern "C" {
STBIDEF int stbi_is_gif(char const *filename);
STBIDEF unsigned char *stbi_xload(char const *filename, int *x, int *y, int *frames);
}

bool IsGif(const char* path)
{
    return stbi_is_gif(path);
}

unsigned char* LoadAnim(const char* path, int& w, int& h, int& pixelSize, int& frames)
{
    pixelSize = 4;
	unsigned char* result = stbi_xload(path, &w, &h, &frames);
	if (frames > 0)
	{
		int frameSize = w * h * pixelSize + 2;
		for (int i = 0; i < frames; ++i)
		{
			unsigned char* p = result + frameSize * i;
			FlipY(p, w, h, pixelSize);
		}
	}
	return result;
}

void ReleaseImg(unsigned char* img)
{
    delete[] img;
}

bool SaveImg(const char* path, int w, int h, int pixelSize, unsigned char* data)
{
    QImage::Format format = QImage::Format_RGBA8888;
    if (pixelSize == 3)
    {
        format = QImage::Format_RGB888;
    }
    else if (pixelSize == 1)
    {
        format = QImage::Format_Grayscale8;
    }
    QImage img(data, w, h, pixelSize * w, format);
    return img.save(path);
}

unsigned char* SaveImgToMemory(int w, int h, int pixelSize, unsigned char* data, int& resultSize)
{
    unsigned char* result = NULL;
    resultSize = 0;
    QImage::Format format = QImage::Format_RGBA8888;
    if (pixelSize == 3)
    {
        format = QImage::Format_RGB888;
    }
    else if (pixelSize == 1)
    {
        format = QImage::Format_Grayscale8;
    }
    QByteArray ba;
    QBuffer buffer(&ba);
    buffer.open(QIODevice::ReadWrite);
    QImage img(data, w, h, pixelSize * w, format);
    img = img.mirrored();
    if (img.save(&buffer, "PNG"))
    {
        result = new unsigned char[ba.size()];
        resultSize = ba.size();
        memcpy(result, ba.data(), ba.size());
    }
    return result;
}

#define GL_RENDERER_SHADER_DIR ":/shader"
#else
char* LoadFile(const char* path, int& size)
{
    FILE* fp = fopen(path, "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        long s = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        char* buf = new char[s + 1];
        fread(buf, s, 1, fp);
        buf[s] = '\0';
        size = s;
        return buf;
    }
    size = 0;
    return NULL;
}
#define GL_RENDERER_SHADER_DIR "shader"

static unsigned char* LoadImg(const char* path, int& w, int& h, int& pixelSize)
{
    unsigned char* data = stbi_load(path, &w, &h, &pixelSize, 4);
    return data;
}

static void ReleaseImg(unsigned char* img)
{
    stbi_image_free(img);
}

static bool SaveImg(const char* path, int mWidth, int mHeight, int pixelSize, unsigned char* data)
{
    int rowBytes = mWidth * pixelSize;
    return stbi_write_png(path, mWidth, mHeight, 4, data, rowBytes) != 0;
}
#endif

#if defined(USE_OPENGL3) || defined(USE_OPENGL3_ES)
#define USE_VAO
#endif

#if defined(USE_OPENGL3) || defined(USE_OPENGL3_ES) || defined(_WIN32)
//#define HAVE_BLIT
#endif

const char* vsFSLight =
    "attribute vec3 position;\n"
	"attribute vec2 texcoord;\n"
	"attribute vec3 normal;\n"
	"uniform mat4 mvp;\n"
	"uniform mat4 mv;\n"
	"varying vec3 positionOut;\n"
	"varying vec3 normalOut;\n"
	"varying vec2 uvOut;\n"
    "void main() {\n"
	"   positionOut = (mv * vec4(position, 1.0)).xyz;\n"
	"   normalOut = (mv * vec4(normal, 1.0)).xyz;\n"
	"   normalOut = normal;\n"
	"	uvOut = texcoord;\n"
    "   gl_Position = mvp * vec4(position, 1.0);\n"
    "}\n";

const char* psFSLight =
    "#extension GL_OES_standard_derivatives : enable\n"
    "#ifdef GL_ES\n"
    "precision highp float;\n"
    "#endif\n"
	"uniform sampler2D texDiffuse;\n"
    "uniform vec3 lightDir;\n"
    "uniform vec3 lightColor;\n"
    "uniform float ambientFactor;\n"
    "uniform float shiness;\n"
	"varying vec3 positionOut;\n"
	"varying vec3 normalOut;\n"
	"varying vec2 uvOut;\n"
    "void main() {\n"
    "   vec3 P = positionOut;\n"
    //"   vec3 N = normalize(cross(dFdx(P),dFdy(P)));\n"
	"   vec3 N = normalOut;\n"
    "   vec3 L = normalize(-lightDir);\n"
    "   vec3 E = normalize(-P);\n"
    "   vec4 D = texture2D(texDiffuse, uvOut) * vec4(lightColor * max(0.0, dot(N,L) * (1.0 - ambientFactor) + ambientFactor), 1.0);\n"
    "   float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
    "   vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
    "   gl_FragColor = mix(D,S,specular);\n"
	//"	gl_FragColor = vec4(N, 1.0);\n"
    "}\n";


const char* vsMultiply =
    "attribute vec2 position;\n"
    "varying vec2 texcoordOut;\n"
    "void main() {\n"
    "   texcoordOut = position * 0.5 + 0.5;\n"
    "   gl_Position = vec4(position, 0.0, 1.0);\n"
    "}\n";

const char* psMultiply =
    "#ifdef GL_ES\n"
    "precision highp float;\n"
    "#endif\n"
    "uniform sampler2D tex0;\n"
    "uniform sampler2D tex1;\n"
    "varying vec2 texcoordOut;\n"
    "void main() {\n"
    "   vec4 dst = texture2D(tex0, texcoordOut);\n"
    "   vec4 src = texture2D(tex1, texcoordOut);\n"
    "   gl_FragColor = vec4(src.rgb * src.a + dst.rgb * (1-src.a), 1.0);\n"
    //"   gl_FragColor = texture2D(tex0, texcoordOut);\n"
    "}\n";

#define PI 3.1415926f

float DegreeToRadian(float value)
{
    return value * PI / 180.0f;
}

float RadianToDegree(float value)
{
    return value * 180.0f / PI;
}

Vector2::Vector2()
    :x(0)
    ,y(0)
{
}

Vector2::Vector2(float x, float y)
    :x(x)
    ,y(y)
{
}

bool Vector2::operator == (const Vector2& v) const
{
    return x == v.x && y == v.y;
}

bool Vector2::operator != (const Vector2& v) const
{
    return x != v.x || y != v.y;
}

bool Vector2::operator < (const Vector2& v) const
{
    return x < v.x && y < v.y;
}

bool Vector2::operator > (const Vector2& v) const
{
    return x > v.x && y > v.y;
}

bool Vector2::operator <= (const Vector2& v) const
{
    return x <= v.x && y <= v.y;
}

bool Vector2::operator >= (const Vector2& v) const
{
    return x >= v.x && y >= v.y;
}

Vector2 Vector2::operator * (float s) const
{
    return Vector2(x * s, y * s);
}

Vector2 Vector2::operator / (float s) const
{
    return Vector2(x / s, y / s);
}

Vector2 Vector2::operator - () const
{
    return Vector2(-x, -y);
}

Vector2 Vector2::operator + (const Vector2& v) const
{
    return Vector2(x + v.x, y + v.y);
}

Vector2 Vector2::operator - (const Vector2& v) const
{
    return Vector2(x - v.x, y - v.y);
}

Vector2 Vector2::operator * (const Vector2& v) const
{
    return Vector2(x * v.x, y * v.y);
}

Vector2 Vector2::operator / (const Vector2& v) const
{
    return Vector2(x / v.x, y / v.y);
}

Vector2& Vector2::operator *= (float s)
{
    x *= s;
    y *= s;
    return *this;
}

Vector2& Vector2::operator /= (float s)
{
    x /= s;
    y /= s;
    return *this;
}

Vector2& Vector2::operator += (const Vector2& v)
{
    x += v.x;
    y += v.y;
    return *this;
}

Vector2& Vector2::operator -= (const Vector2& v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

float Vector2::Length() const
{
    return sqrtf(x * x + y * y);
}

float Vector2::LengthSq() const
{
    return x * x + y * y;
}

float Vector2::DistanceTo(const Vector2& v) const
{
    float dx = x - v.x;
    float dy = y - v.y;
    return sqrtf(dx * dx + dy * dy);
}

float Vector2::DistanceToSq(const Vector2& v) const
{
    float dx = x - v.x;
    float dy = y - v.y;
    return dx * dx + dy * dy;
}

float Vector2::Normalise()
{
    float length = sqrtf(x * x + y * y);
    if (length > 0)
    {
        float invLength = 1.0f / length;
        x *= invLength;
        y *= invLength;
    }
    return length;
}

Vector2 Vector2::GetNormalized() const
{
    Vector2 n(x, y);
    n.Normalise();
    return n;
}

float Vector2::Dot(const Vector2& v) const
{
    return x * v.x + y * v.y;
}

Vector2 Vector2::GetPerpendicular() const
{
    return Vector2(-y, x);
}

bool Vector2::IsPerpendicular(const Vector2& v, float epsilon) const
{
    return fabs(GetNormalized().Dot(v.GetNormalized())) < epsilon;
}

bool Vector2::IsParellel(const Vector2& v, float epsilon) const
{
    float d = GetNormalized().Dot(v.GetNormalized());
    return 1.0f - fabs(d) < epsilon;
}

bool Vector2::IsEqual(const Vector2& v, float epsilon) const
{
    return fabs(v.x - x) < epsilon && fabs(v.y - y) < epsilon;
}

bool Vector2::IsZero(float epsilon) const
{
    return fabs(x) < epsilon && fabs(y) < epsilon;
}

Vector2 Vector2::Lerp(const Vector2& v, float t) const
{
    float invT = 1.0f - t;
    return Vector2(x * invT + v.x * t, y * invT + v.y * t);
}

Vector2 Vector2::Lerp(const Vector2& v, Vector2 t) const
{
    return Vector2(x * (1.0f - t.x) + v.x * t.x, y * (1.0f - t.y) + v.y * t.y);
}

void Vector2::Set(float x, float y)
{
    this->x = x;
    this->y = y;
}

void Vector2::Truncate(float x0, float y0, float x1, float y1)
{
    if (x < x0)
    {
        x = x0;
    }
    else if (x > x1)
    {
        x = x1;
    }

    if (y < y0)
    {
        y = y0;
    }
    else if (y > y1)
    {
        y = y1;
    }
}

void Vector2::Truncate(const Vector2& low, const Vector2& high)
{
    Truncate(low.x, low.y, high.x, high.y);
}

float Vector2::GetAngle(const Vector2& pivot)
{
    Vector2 dir(x - pivot.x, y - pivot.y);
    if (dir.LengthSq() == 0)
    {
        return 0;
    }
    dir.Normalise();
    float angle = acosf(dir.Dot(Vector2(1.0f, 0)));
    if (dir.y < 0)
    {
        angle = - angle;
    }
    return angle;
}

AABB2::AABB2()
    :xMin(1.0f)
    ,yMin(1.0f)
    ,xMax(-1.0f)
    ,yMax(-1.0f)
{
}

AABB2::AABB2(float xMin, float yMin, float xMax, float yMax)
    :xMin(xMin)
    ,yMin(yMin)
    ,xMax(xMax)
    ,yMax(yMax)
{
}

float AABB2::Width() const
{
    return xMax - xMin;
}

float AABB2::Height() const
{
    return yMax - yMin;
}

bool AABB2::IsNull() const
{
    return !(xMin <= xMax) || !(yMin <= yMax);
}

bool AABB2::HitTest(const AABB2& v) const
{
    return !(xMin > v.xMax || yMin > v.yMax || xMax < v.xMin || yMax < v.yMin);
}

bool AABB2::HitTest(const Vector2& p) const
{
    return p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax;
}

void AABB2::Union(const AABB2& v)
{
    if (v.IsNull())
    {
        return;
    }
    if (IsNull())
    {
        xMin = v.xMin;
        yMin = v.yMin;
        xMax = v.xMax;
        yMax = v.yMax;
    }
    else
    {
        if (v.xMin < xMin)
        {
            xMin = v.xMin;
        }
        if (v.yMin < yMin)
        {
            yMin = v.yMin;
        }
        if (v.xMax > xMax)
        {
            xMax = v.xMax;
        }
        if (v.yMax > yMax)
        {
            yMax = v.yMax;
        }
    }
}

void AABB2::Extend(float size)
{
    if (!IsNull())
    {
        xMin -= size;
        yMin -= size;
        xMax += size;
        yMax += size;
    }
}

void AABB2::Union(const Vector2& v)
{
    if (IsNull())
    {
        xMin = xMax = v.x;
        yMin = yMax = v.y;
    }
    else
    {
        if (v.x < xMin)
        {
            xMin = v.x;
        }
        else if (v.x > xMax)
        {
            xMax = v.x;
        }
        if (v.y < yMin)
        {
            yMin = v.y;
        }
        else if (v.y > yMax)
        {
            yMax = v.y;
        }
    }
}

AABB2 AABB2::Intersect(const AABB2& v)
{
    AABB2 result = v;
    if (xMin > result.xMin)
    {
        result.xMin = xMin;
    }
    if (xMax < result.xMax)
    {
        result.xMax = xMax;
    }
    if (yMin > result.yMin)
    {
        result.yMin = yMin;
    }
    if (yMax < result.yMax)
    {
        result.yMax = yMax;
    }
    return result;
}

void AABB2::BuildFrom(Vector2* points, int n)
{
    if (n <= 0)
    {
        return;
    }
    xMin = xMax = points[0].x;
    yMin = yMax = points[0].y;
    for (int i = 1; i < n; ++i)
    {
        const Vector2& v = points[i];
        if (v.x < xMin)
        {
            xMin = v.x;
        }
        else if (v.x > xMax)
        {
            xMax = v.x;
        }
        if (v.y < yMin)
        {
            yMin = v.y;
        }
        else if (v.y > yMax)
        {
            yMax = v.y;
        }
    }
}

void AABB2::SetNull()
{
	xMin = 1;
	yMin = 1;
	xMax = -1;
	yMax = -1;
}

Vector3::Vector3()
    :x(0)
    ,y(0)
    ,z(0)
{
}

Vector3::Vector3(float x, float y, float z)
    :x(x)
    ,y(y)
    ,z(z)
{
}

bool Vector3::operator == (const Vector3& v) const
{
    return x == v.x && y == v.y && z == v.z;
}

bool Vector3::operator != (const Vector3& v) const
{
    return x != v.x || y != v.y || z != v.z;
}

bool Vector3::operator < (const Vector3& v) const
{
    return x < v.x && y < v.y && z < v.z;
}

bool Vector3::operator > (const Vector3& v) const
{
    return x > v.x && y > v.y && z > v.z;
}

bool Vector3::operator <= (const Vector3& v) const
{
    return x <= v.x && y <= v.y && z <= v.z;
}

bool Vector3::operator >= (const Vector3& v) const
{
    return x >= v.x && y >= v.y && z >= v.z;
}

Vector3 Vector3::operator * (float s) const
{
    return Vector3(x * s, y * s, z * s);
}

Vector3 Vector3::operator / (float s) const
{
    return Vector3(x / s, y / s, z / s);
}

Vector3 Vector3::operator - () const
{
    return Vector3(-x, -y, -z);
}

Vector3 Vector3::operator + (const Vector3& v) const
{
    return Vector3(x + v.x, y + v.y, z + v.z);
}

Vector3 Vector3::operator - (const Vector3& v) const
{
    return Vector3(x - v.x, y - v.y, z - v.z);
}

Vector3 Vector3::operator * (const Vector3& v) const
{
    return Vector3(x * v.x, y * v.y, z * v.z);
}

Vector3 Vector3::operator / (const Vector3& v) const
{
    return Vector3(x / v.x, y / v.y, z / v.z);
}

Vector3& Vector3::operator *= (float s)
{
    x *= s;
    y *= s;
    z *= s;
    return *this;
}

Vector3& Vector3::operator /= (float s)
{
    x /= s;
    y /= s;
    z /= s;
    return *this;
}

Vector3& Vector3::operator += (const Vector3& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

Vector3& Vector3::operator -= (const Vector3& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

float Vector3::Length() const
{
    return sqrtf(x * x + y * y + z * z);
}

float Vector3::LengthSq() const
{
    return x * x + y * y + z * z;
}

float Vector3::DistanceTo(const Vector3& v) const
{
    float dx = x - v.x;
    float dy = y - v.y;
    float dz = z - v.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

float Vector3::DistanceToSq(const Vector3& v) const
{
    float dx = x - v.x;
    float dy = y - v.y;
    float dz = z - v.z;
    return dx * dx + dy * dy + dz * dz;
}

float Vector3::Normalise()
{
    float length = sqrtf(x * x + y * y + z * z);
    if (length > 0)
    {
        float invLength = 1.0f / length;
        x *= invLength;
        y *= invLength;
        z *= invLength;
    }
    return length;
}

Vector3 Vector3::GetNormalized() const
{
    Vector3 n(x, y, z);
    n.Normalise();
    return n;
}

float Vector3::Dot(const Vector3& v) const
{
    return x * v.x + y * v.y + z * v.z;
}

Vector3 Vector3::Cross(const Vector3& v) const
{
    return Vector3(
        y * v.z - z * v.y
        , z * v.x - x * v.z
        , x * v.y - y * v.x);
}


bool Vector3::IsPerpendicular(const Vector3& v, float epsilon) const
{
    return fabs(GetNormalized().Dot(v.GetNormalized())) < epsilon;
}

bool Vector3::IsParellel(const Vector3& v, float epsilon) const
{
    float d = GetNormalized().Dot(v.GetNormalized());
    return 1.0f - fabs(d) < epsilon;
}

bool Vector3::IsEqual(const Vector3& v, float epsilon) const
{
    return fabs(v.x - x) < epsilon && fabs(v.y - y) < epsilon && fabs(v.z - z) < epsilon;
}

bool Vector3::IsZero(float epsilon) const
{
    return fabs(x) < epsilon && fabs(y) < epsilon && fabs(z) < epsilon;
}

Vector3 Vector3::Lerp(const Vector3& v, float t) const
{
    float invT = 1.0f - t;
    return Vector3(x * invT + v.x * t, y * invT + v.y * t, z * invT + v.z * t);
}

void Vector3::Set(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

Matrix4::Matrix4()
{
    Identity();
}

Matrix4::~Matrix4()
{
}

Matrix4::Matrix4(const float* m, bool isColumnMajor)
{
    if (isColumnMajor)
    {
        m00 = m[0];m01 = m[4];m02 = m[8];m03 = m[12];
        m10 = m[1];m11 = m[5];m12 = m[9];m13 = m[13];
        m20 = m[2];m21 = m[6];m22 = m[10];m23 = m[14];
        m30 = m[3];m31 = m[7];m32 = m[11];m33 = m[15];
    }
    else
    {
        m00 = m[0];m01 = m[1];m02 = m[2];m03 = m[3];
        m10 = m[4];m11 = m[5];m12 = m[6];m13 = m[7];
        m20 = m[8];m21 = m[9];m22 = m[10];m23 = m[11];
        m30 = m[12];m31 = m[13];m32 = m[14];m33 = m[15];
    }
}

Matrix4::Matrix4(const Matrix4& m)
    :m00(m.m00), m10(m.m10), m20(m.m20), m30(m.m30)
    ,m01(m.m01), m11(m.m11), m21(m.m21), m31(m.m31)
    ,m02(m.m02), m12(m.m12), m22(m.m22), m32(m.m32)
    ,m03(m.m03), m13(m.m13), m23(m.m23), m33(m.m33)
{
}

Matrix4::Matrix4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33
    )
    :m00(m00), m10(m10), m20(m20), m30(m30)
    ,m01(m01), m11(m11), m21(m21), m31(m31)
    ,m02(m02), m12(m12), m22(m22), m32(m32)
    ,m03(m03), m13(m13), m23(m23), m33(m33)
{
}

void Matrix4::CopyTo(float* buf, bool isColumnMajor) const
{
    if (isColumnMajor)
    {
        buf[0] = m00; buf[1] = m10; buf[2] = m20; buf[3] = m30;
        buf[4] = m01; buf[5] = m11; buf[6] = m21; buf[7] = m31;
        buf[8] = m02; buf[9] = m12; buf[10] = m22; buf[11] = m32;
        buf[12] = m03; buf[13] = m13; buf[14] = m23; buf[15] = m33;
    }
    else
    {
        buf[0] = m00; buf[4] = m10; buf[8] = m20; buf[12] = m30;
        buf[1] = m01; buf[5] = m11; buf[9] = m21; buf[13] = m31;
        buf[2] = m02; buf[6] = m12; buf[10] = m22; buf[14] = m32;
        buf[3] = m03; buf[7] = m13; buf[11] = m23; buf[15] = m33;
    }
}

Matrix4 Matrix4::operator * (const Matrix4& m) const
{
    return Matrix4(
        // row 0
        m00 * m.m00 + m01 * m.m10 + m02 * m.m20 + m03 * m.m30,
        m00 * m.m01 + m01 * m.m11 + m02 * m.m21 + m03 * m.m31,
        m00 * m.m02 + m01 * m.m12 + m02 * m.m22 + m03 * m.m32,
        m00 * m.m03 + m01 * m.m13 + m02 * m.m23 + m03 * m.m33,
        // row 1
        m10 * m.m00 + m11 * m.m10 + m12 * m.m20 + m13 * m.m30,
        m10 * m.m01 + m11 * m.m11 + m12 * m.m21 + m13 * m.m31,
        m10 * m.m02 + m11 * m.m12 + m12 * m.m22 + m13 * m.m32,
        m10 * m.m03 + m11 * m.m13 + m12 * m.m23 + m13 * m.m33,
        // row 2
        m20 * m.m00 + m21 * m.m10 + m22 * m.m20 + m23 * m.m30,
        m20 * m.m01 + m21 * m.m11 + m22 * m.m21 + m23 * m.m31,
        m20 * m.m02 + m21 * m.m12 + m22 * m.m22 + m23 * m.m32,
        m20 * m.m03 + m21 * m.m13 + m22 * m.m23 + m23 * m.m33,
        // row 3
        m30 * m.m00 + m31 * m.m10 + m32 * m.m20 + m33 * m.m30,
        m30 * m.m01 + m31 * m.m11 + m32 * m.m21 + m33 * m.m31,
        m30 * m.m02 + m31 * m.m12 + m32 * m.m22 + m33 * m.m32,
        m30 * m.m03 + m31 * m.m13 + m32 * m.m23 + m33 * m.m33
        );
}

Vector3 Matrix4::operator * (const Vector3& v) const
{
    float wInv = 1.0f;// / (v.x * m03 + v.y * m13 + v.z * m23 + m33);
    return Vector3
        ((v.x * m00 + v.y * m01 + v.z * m02 + m03) * wInv
        ,(v.x * m10 + v.y * m11 + v.z * m12 + m13) * wInv
        ,(v.x * m20 + v.y * m21 + v.z * m22 + m23) * wInv);
}

Matrix4 Matrix4::Inverse() const
{
    float d =
        (m00 * m11 - m01 * m10) * (m22 * m33 - m23 * m32) -
        (m00 * m12 - m02 * m10) * (m21 * m33 - m23 * m31) +
        (m00 * m13 - m03 * m10) * (m21 * m32 - m22 * m31) +
        (m01 * m12 - m02 * m11) * (m20 * m33 - m23 * m30) -
        (m01 * m13 - m03 * m11) * (m20 * m32 - m22 * m30) +
        (m02 * m13 - m03 * m12) * (m20 * m31 - m21 * m30);

    d = 1.0f / d;

    return Matrix4(
        d * (m11 * (m22 * m33 - m23 * m32) + m12 * (m23 * m31 - m21 * m33) + m13 * (m21 * m32 - m22 * m31)),
        d * (m21 * (m02 * m33 - m03 * m32) + m22 * (m03 * m31 - m01 * m33) + m23 * (m01 * m32 - m02 * m31)),
        d * (m31 * (m02 * m13 - m03 * m12) + m32 * (m03 * m11 - m01 * m13) + m33 * (m01 * m12 - m02 * m11)),
        d * (m01 * (m13 * m22 - m12 * m23) + m02 * (m11 * m23 - m13 * m21) + m03 * (m12 * m21 - m11 * m22)),

        d * (m12 * (m20 * m33 - m23 * m30) + m13 * (m22 * m30 - m20 * m32) + m10 * (m23 * m32 - m22 * m33)),
        d * (m22 * (m00 * m33 - m03 * m30) + m23 * (m02 * m30 - m00 * m32) + m20 * (m03 * m32 - m02 * m33)),
        d * (m32 * (m00 * m13 - m03 * m10) + m33 * (m02 * m10 - m00 * m12) + m30 * (m03 * m12 - m02 * m13)),
        d * (m02 * (m13 * m20 - m10 * m23) + m03 * (m10 * m22 - m12 * m20) + m00 * (m12 * m23 - m13 * m22)),

        d * (m13 * (m20 * m31 - m21 * m30) + m10 * (m21 * m33 - m23 * m31) + m11 * (m23 * m30 - m20 * m33)),
        d * (m23 * (m00 * m31 - m01 * m30) + m20 * (m01 * m33 - m03 * m31) + m21 * (m03 * m30 - m00 * m33)),
        d * (m33 * (m00 * m11 - m01 * m10) + m30 * (m01 * m13 - m03 * m11) + m31 * (m03 * m10 - m00 * m13)),
        d * (m03 * (m11 * m20 - m10 * m21) + m00 * (m13 * m21 - m11 * m23) + m01 * (m10 * m23 - m13 * m20)),

        d * (m10 * (m22 * m31 - m21 * m32) + m11 * (m20 * m32 - m22 * m30) + m12 * (m21 * m30 - m20 * m31)),
        d * (m20 * (m02 * m31 - m01 * m32) + m21 * (m00 * m32 - m02 * m30) + m22 * (m01 * m30 - m00 * m31)),
        d * (m30 * (m02 * m11 - m01 * m12) + m31 * (m00 * m12 - m02 * m10) + m32 * (m01 * m10 - m00 * m11)),
        d * (m00 * (m11 * m22 - m12 * m21) + m01 * (m12 * m20 - m10 * m22) + m02 * (m10 * m21 - m11 * m20))
        );
}

void Matrix4::Identity()
{
    m00 = 1.0f; m01 = 0.0f; m02 = 0.0f; m03 = 0.0f;
    m10 = 0.0f; m11 = 1.0f; m12 = 0.0f; m13 = 0.0f;
    m20 = 0.0f; m21 = 0.0f; m22 = 1.0f; m23 = 0.0f;
    m30 = 0.0f; m31 = 0.0f; m32 = 0.0f; m33 = 1.0f;
}

void Matrix4::Transpose()
{
    ; m01 = m10; m02 = m20; m03 = m30;
    m10 = m01; ; m12 = m21; m13 = m31;
    m20 = m02; m21 = m12; ; m23 = m32;
    m30 = m03; m31 = m13; m32 = m32; ;
}

Vector3 Matrix4::GetXAxis() const
{
    return Vector3(m00, m10, m20);
}

Vector3 Matrix4::GetYAxis() const
{
    return Vector3(m01, m11, m21);
}

Vector3 Matrix4::GetZAxis() const
{
    return Vector3(m02, m12, m22);
}

Vector3 Matrix4::GetTranslate() const
{
    return Vector3(m03, m13, m23);
}

void Matrix4::Translate(float x, float y, float z)
{
    const Matrix4& m = BuildTranslate(x, y, z) * *this;
    *this = m;
}

void Matrix4::Scale(float x, float y, float z)
{
    const Matrix4& m = BuildScale(x, y, z) * *this;
    *this = m;
}

void Matrix4::Rotate(float radians, float x, float y, float z)
{
    const Matrix4& m = BuildRotate(radians, x, y, z) * *this;
    *this = m;
}

void Matrix4::RotateAt(float radians, float x, float y, float z, float posx, float posy, float posz)
{
    const Matrix4& m = BuildRotateAt(radians, x, y, z, posx, posy, posz) * *this;
    *this = m;
}

void Matrix4::LocalTranslateX(float delta)
{
    const Vector3& d = GetXAxis().GetNormalized() * delta;
    m03 += d.x;
    m13 += d.y;
    m23 += d.z;
}

void Matrix4::LocalTranslateY(float delta)
{
    const Vector3& d = GetYAxis().GetNormalized() * delta;
    m03 += d.x;
    m13 += d.y;
    m23 += d.z;
}

void Matrix4::LocalTranslateZ(float delta)
{
    const Vector3& d = GetZAxis().GetNormalized() * delta;
    m03 += d.x;
    m13 += d.y;
    m23 += d.z;
}

void Matrix4::LocalRotate(float deltaRadians, float x, float y, float z)
{
    float tx = m03;
    float ty = m13;
    float tz = m23;
    Translate(-tx, -ty, -tz);
    Rotate(deltaRadians, x, y, z);
    Translate(tx, ty, tz);
}

Matrix4 Matrix4::BuildTranslate(float x, float y, float z)
{
    return Matrix4(
        1.0f, 0.0f, 0.0f, x,
        0.0f, 1.0f, 0.0f, y,
        0.0f, 0.0f, 1.0f, z,
        0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::BuildScale(float sx, float sy, float sz)
{
    return Matrix4(
        sx, 0.0f, 0.0f, 0.0f,
        0.0f,   sy, 0.0f, 0.0f,
        0.0f, 0.0f,   sz, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::BuildRotate(float radians, float x, float y, float z)
{
    float cosa = cosf(radians);
    float sina = sinf(radians);
    float mc = 1.0f - cosa;
    float xyc = x * y * mc;
    float yzc = y * z * mc;
    float xzc = x * z * mc;
    float xxc = x * x * mc;
    float yyc = y * y * mc;
    float zzc = z * z * mc;
    float xs = x * sina;
    float ys = y * sina;
    float zs = z * sina;

    return Matrix4(
        xxc + cosa,  xyc - zs,    xzc + ys,   0.0f,
        xyc + zs,    yyc + cosa,  yzc - xs,   0.0f,
        xzc - ys,    yzc + xs,    zzc + cosa, 0.0f,
        0.0f,        0.0f,        0.0f,       1.0f);
}

Matrix4 Matrix4::BuildRotateAt(float radians, float x, float y, float z, float posx, float posy, float posz)
{
    return BuildTranslate(posx, posy, posz) * BuildRotate(radians, x, y, z) * BuildTranslate(-posx, -posy, -posz);
}

Matrix4 Matrix4::BuildFrustum(float left, float right, float bottom, float top, float nearZ, float farZ)
{
    float dx = 1.0f / (right - left);
    float dy = 1.0f / (top - bottom);
    float dz = 1.0f / (farZ - nearZ);
    float n2 = nearZ * 2;
    return Matrix4(
        n2 * dx, 0,        (right + left) * dx,  0,
        0,       n2 * dy,  (top + bottom) * dy,  0,
        0,       0,       -(farZ + nearZ) * dz, -n2 * farZ * dz,
        0,       0,       -1,                   0
        );
}

Matrix4 Matrix4::BuildPerspective(float fovy, float aspect, float nearZ, float farZ)
{
    float top = tanf(fovy * 0.5f * PI / 180.0f) * nearZ;
    float right = top * aspect;
    return BuildFrustum(-top, top, -right, right, nearZ, farZ);
}

Matrix4 Matrix4::BuildOrtho(float left, float right, float bottom, float top, float nearZ, float farZ)
{
    float dx = 1.0f / (right - left);
    float dy = 1.0f / (top - bottom);
    float dz = 1.0f / (farZ - nearZ);
    return Matrix4(
        2 * dx,      0,       0, -(right + left) * dx,
        0, 2 * dy,       0, -(top + bottom) * dy,
        0,      0, -2 * dz, -(farZ + nearZ) * dz,
        0,      0,       0,                    1
        );
}

Matrix4 Matrix4::BuildBiasMatrix()
{
    return Matrix4(
        0.5f, 0.0f, 0.0f, 0.5f,
        0.0f, 0.5f, 0.0f, 0.5f,
        0.0f, 0.0f, 0.5f, 0.5f,
        0.0f, 0.0f, 0.0f, 1.0f
        );
}

AABB3::AABB3()
    :xMin(1.0f)
    ,yMin(1.0f)
    ,zMin(1.0f)
    ,xMax(-1.0f)
    ,yMax(-1.0f)
    ,zMax(-1.0f)
{
}

AABB3::AABB3(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax)
    :xMin(xMin)
    ,yMin(yMin)
    ,zMin(zMin)
    ,xMax(xMax)
    ,yMax(yMax)
    ,zMax(zMax)
{
}

float AABB3::Width() const
{
    return xMax - xMin;
}

float AABB3::Height() const
{
    return yMax - yMin;
}

float AABB3::Depth() const
{
    return zMax - zMin;
}

bool AABB3::IsNull() const
{
    return !(xMin <= xMax) || !(yMin <= yMax) || !(zMin <= zMax);
}

bool AABB3::HitTest(const AABB3& v) const
{
    return !(xMin > v.xMax || yMin > v.yMax 
          || xMax < v.xMin || yMax < v.yMin
          || zMax < v.zMin || zMax < v.zMin);
}

bool AABB3::HitTest(const Vector3& p) const
{
    return p.x >= xMin && p.x <= xMax 
        && p.y >= yMin && p.y <= yMax
        && p.z >= zMin && p.z <= zMax;
}

void AABB3::Union(const AABB3& v)
{
    if (v.IsNull())
    {
        return;
    }
    if (IsNull())
    {
        xMin = v.xMin;
        yMin = v.yMin;
        zMin = v.zMin;
        xMax = v.xMax;
        yMax = v.yMax;
        zMax = v.zMax;
    }
    else
    {
        if (v.xMin < xMin)
        {
            xMin = v.xMin;
        }
        if (v.yMin < yMin)
        {
            yMin = v.yMin;
        }
        if (v.zMin < zMin)
        {
            zMin = v.zMin;
        }
        if (v.xMax > xMax)
        {
            xMax = v.xMax;
        }
        if (v.yMax > yMax)
        {
            yMax = v.yMax;
        }
        if (v.zMax > zMax)
        {
            zMax = v.zMax;
        }
    }
}

void AABB3::Union(const Vector3& v)
{
    if (IsNull())
    {
        xMin = xMax = v.x;
        yMin = yMax = v.y;
        zMin = zMax = v.z;
    }
    else
    {
        if (v.x < xMin)
        {
            xMin = v.x;
        }
        else if (v.x > xMax)
        {
            xMax = v.x;
        }
        if (v.y < yMin)
        {
            yMin = v.y;
        }
        else if (v.y > yMax)
        {
            yMax = v.y;
        }
        if (v.z < zMin)
        {
            zMin = v.z;
        }
        else if (v.z > zMax)
        {
            zMax = v.z;
        }
    }
}

AABB3 AABB3::Intersect(const AABB3& v)
{
    AABB3 result = v;
    if (xMin > result.xMin)
    {
        result.xMin = xMin;
    }
    if (xMax < result.xMax)
    {
        result.xMax = xMax;
    }
    if (yMin > result.yMin)
    {
        result.yMin = yMin;
    }
    if (yMax < result.yMax)
    {
        result.yMax = yMax;
    }
    if (zMin > result.zMin)
    {
        result.zMin = zMin;
    }
    if (zMax < result.zMax)
    {
        result.zMax = zMax;
    }
    return result;
}

void AABB3::BuildFrom(Vector3* points, int n)
{
    if (n <= 0)
    {
        return;
    }
    xMin = xMax = points[0].x;
    yMin = yMax = points[0].y;
    zMin = zMax = points[0].z;
    for (int i = 1; i < n; ++i)
    {
        const Vector3& v = points[i];
        if (v.x < xMin)
        {
            xMin = v.x;
        }
        else if (v.x > xMax)
        {
            xMax = v.x;
        }
        if (v.y < yMin)
        {
            yMin = v.y;
        }
        else if (v.y > yMax)
        {
            yMax = v.y;
        }
        if (v.z < zMin)
        {
            zMin = v.z;
        }
        else if (v.z > zMax)
        {
            zMax = v.z;
        }
    }
}

void AABB3::SetNull()
{
    xMin = 1;
    yMin = 1;
    zMin = 1;
    xMax = -1;
    yMax = -1;
    zMax = -1;
}

Plane3::Plane3(const Vector3& n, float p)
    :mNormal(n)
    , mP(p)
{
    mNormal.Normalise();
}

Ray3::Ray3(const Vector3& origin, const Vector3 direction)
    :mOrigin(origin)
    ,mDirection(direction)
{
}

bool Ray3::GetIntersection(const Plane3& plane, Vector3& intersection)
{
    float d = mDirection.Dot(plane.mNormal);
    float epsilon = 0.0001f;
    if (d < epsilon && d > -epsilon)
    {
        return false;
    }
    float t = (plane.mNormal.Dot(mOrigin) + plane.mP) / -d;
    if (t < 0.0f)
    {
        return false;
    }
    intersection = mOrigin + mDirection * t;
    return true;
}

void Frustum::Extract(const Matrix4& mvp)
{
    const float* clip = (const float*)&mvp;
    /* Extract the numbers for the RIGHT plane */
    frustum[0][0] = clip[ 3] - clip[ 0];
    frustum[0][1] = clip[ 7] - clip[ 4];
    frustum[0][2] = clip[11] - clip[ 8];
    frustum[0][3] = clip[15] - clip[12];

    float t;
    /* Normalize the result */
    t = sqrt( frustum[0][0] * frustum[0][0] + frustum[0][1] * frustum[0][1] + frustum[0][2] * frustum[0][2] );
    frustum[0][0] /= t;
    frustum[0][1] /= t;
    frustum[0][2] /= t;
    frustum[0][3] /= t;

    /* Extract the numbers for the LEFT plane */
    frustum[1][0] = clip[ 3] + clip[ 0];
    frustum[1][1] = clip[ 7] + clip[ 4];
    frustum[1][2] = clip[11] + clip[ 8];
    frustum[1][3] = clip[15] + clip[12];

    /* Normalize the result */
    t = sqrt( frustum[1][0] * frustum[1][0] + frustum[1][1] * frustum[1][1] + frustum[1][2] * frustum[1][2] );
    frustum[1][0] /= t;
    frustum[1][1] /= t;
    frustum[1][2] /= t;
    frustum[1][3] /= t;

    /* Extract the BOTTOM plane */
    frustum[2][0] = clip[ 3] + clip[ 1];
    frustum[2][1] = clip[ 7] + clip[ 5];
    frustum[2][2] = clip[11] + clip[ 9];
    frustum[2][3] = clip[15] + clip[13];

    /* Normalize the result */
    t = sqrt( frustum[2][0] * frustum[2][0] + frustum[2][1] * frustum[2][1] + frustum[2][2] * frustum[2][2] );
    frustum[2][0] /= t;
    frustum[2][1] /= t;
    frustum[2][2] /= t;
    frustum[2][3] /= t;

    /* Extract the TOP plane */
    frustum[3][0] = clip[ 3] - clip[ 1];
    frustum[3][1] = clip[ 7] - clip[ 5];
    frustum[3][2] = clip[11] - clip[ 9];
    frustum[3][3] = clip[15] - clip[13];

    /* Normalize the result */
    t = sqrt( frustum[3][0] * frustum[3][0] + frustum[3][1] * frustum[3][1] + frustum[3][2] * frustum[3][2] );
    frustum[3][0] /= t;
    frustum[3][1] /= t;
    frustum[3][2] /= t;
    frustum[3][3] /= t;

    /* Extract the FAR plane */
    frustum[4][0] = clip[ 3] - clip[ 2];
    frustum[4][1] = clip[ 7] - clip[ 6];
    frustum[4][2] = clip[11] - clip[10];
    frustum[4][3] = clip[15] - clip[14];

    /* Normalize the result */
    t = sqrt( frustum[4][0] * frustum[4][0] + frustum[4][1] * frustum[4][1] + frustum[4][2] * frustum[4][2] );
    frustum[4][0] /= t;
    frustum[4][1] /= t;
    frustum[4][2] /= t;
    frustum[4][3] /= t;

    /* Extract the NEAR plane */
    frustum[5][0] = clip[ 3] + clip[ 2];
    frustum[5][1] = clip[ 7] + clip[ 6];
    frustum[5][2] = clip[11] + clip[10];
    frustum[5][3] = clip[15] + clip[14];

    /* Normalize the result */
    t = sqrt( frustum[5][0] * frustum[5][0] + frustum[5][1] * frustum[5][1] + frustum[5][2] * frustum[5][2] );
    frustum[5][0] /= t;
    frustum[5][1] /= t;
    frustum[5][2] /= t;
}

bool Frustum::HitTest(const Vector3& point)
{
    for( int p = 0; p < 6; p++ )
    {
        if( frustum[p][0] * point.x + frustum[p][1] * point.y + frustum[p][2] * point.z + frustum[p][3] <= 0 )
        {
            return false;
        }
    }
    return true;
}

bool Frustum::HitTest(const AABB3& box)
{
    for(int p = 0; p < 6; p++)
    {
        if( frustum[p][0] * box.xMin + frustum[p][1] * box.yMin + frustum[p][2] * box.zMin + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMax + frustum[p][1] * box.yMin + frustum[p][2] * box.zMin + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMin + frustum[p][1] * box.yMax + frustum[p][2] * box.zMin + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMax + frustum[p][1] * box.yMax + frustum[p][2] * box.zMin + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMin + frustum[p][1] * box.yMin + frustum[p][2] * box.zMax + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMax + frustum[p][1] * box.yMin + frustum[p][2] * box.zMax + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMin + frustum[p][1] * box.yMax + frustum[p][2] * box.zMax + frustum[p][3] > 0 )
            continue;
        if( frustum[p][0] * box.xMax + frustum[p][1] * box.yMax + frustum[p][2] * box.zMax + frustum[p][3] > 0 )
            continue;
        return false;
    }
    return true;
}


class FontImpl;

class Glyph
{
    int width;
    int height;
    int xOffset;
    int yOffset;
    int xAdvance;
    int atlasX;
    int atlasY;

public:
    Glyph(int width, int height, int xOffset, int yOffset, int xAdvance, int atlasX, int atlasY)
        :width(width), height(height), xOffset(xOffset), yOffset(yOffset), xAdvance(xAdvance), atlasX(atlasX), atlasY(atlasY)
    {
    }
    ~Glyph() {}
    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetXOffset() const { return xOffset; }
    int GetYOffset() const { return yOffset; }
    int GetXAdvance() const { return xAdvance; }
    int GetAtlasX() const { return atlasX; }
    int GetAtlasY() const { return atlasY; }
};

class Font
{
public:
    struct AtlasRect
    {
        int x;
        int y;
        int width;
        int height;
    };

public:
    Font(int fontHeight, int atlasSize = 1024, const char* defaultFontPath = 0);
    ~Font();

    Glyph* GetGlyph(int glyphId);
    int GetXAdvance(int glyphId);
    int GetFontHeight();
    int GetAscent();
    int GetDescent();
    int GetAtlasSize();
    void* GetAtlasBitmap();
    void RenderText(const char* text, float xPos, float yPos, float fontHeight, std::vector<float>& vertices);
    void BakeText(const char* text, void** bitmap, int* width, int* height);
    void GetAtlasUpdateRects(std::vector<Font::AtlasRect>& updatedRects);
    void ExportAtlas(const char* path);

private:
    FontImpl* impl;
};



#ifdef __glu_h__
#ifndef CALLBACK
#define CALLBACK
#endif

#define GluFloat GLdouble

struct TriangulateVertex
{
    GluFloat x;
    GluFloat y;
    GluFloat s;
    GluFloat t;
};

struct TriangulateContext
{
public:
    TriangulateContext(std::vector<TriangulateVertex>* outputVertices)
        :outputVertices(outputVertices), outputContours(NULL)
    {
    }

    TriangulateContext(std::vector<std::vector<TriangulateVertex> >* outputContours)
        :outputVertices(NULL), outputContours(outputContours)
    {
    }

    ~TriangulateContext()
    {
        for (size_t i = 0; i < generatedVertices.size(); ++i)
        {
            delete generatedVertices[i];
        }
    }

    void BeginPrimitiveBatch(GLenum primitiveType)
    {
        this->primitiveType = primitiveType;
        vertices.clear();
    }

    void EndPrimitiveBatch()
    {
        switch (primitiveType)
        {
        case GL_TRIANGLE_FAN:
            for (size_t i = 2; i < vertices.size(); ++i)
            {
                outputVertices->push_back(vertices[0]);
                outputVertices->push_back(vertices[i - 1]);
                outputVertices->push_back(vertices[i]);
            }
            break;
        case GL_TRIANGLE_STRIP:
            for (size_t i = 2; i < vertices.size(); ++i)
            {
                if (i % 2 == 0)
                {
                    outputVertices->push_back(vertices[i - 2]);
                    outputVertices->push_back(vertices[i - 1]);
                    outputVertices->push_back(vertices[i]);
                }
                else
                {
                    outputVertices->push_back(vertices[i - 1]);
                    outputVertices->push_back(vertices[i - 2]);
                    outputVertices->push_back(vertices[i]);
                }
            }
            break;
        case GL_TRIANGLES:
            for (size_t i = 0; i < vertices.size(); ++i)
            {
                outputVertices->push_back(vertices[i]);
            }
            break;
        case GL_LINE_LOOP:
            {
                std::vector<TriangulateVertex> contour;
                for (size_t i = 0; i < vertices.size(); ++i)
                {
                    contour.push_back(vertices[i]);
                }
                outputContours->push_back(contour);
            }
            break;
        default:
            // Unknown type
            break;
        }
    }

    GluFloat* NewVertex(GluFloat x, GluFloat y, GluFloat z, GluFloat s, GluFloat t)
    {
        GluFloat* v = new GluFloat[5];
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v[3] = s;
        v[4] = t;
        generatedVertices.push_back(v);
        return v;
    }

    void AddBatchVertex(GluFloat* v)
    {
        TriangulateVertex vt;
        vt.x = (float)v[0];
        vt.y = (float)v[1];
        vt.s = (float)v[3];
        vt.t = (float)v[4];
        vertices.push_back(vt);
    }
public:
    std::vector<GluFloat*> generatedVertices;
    GLenum primitiveType;
    std::vector<TriangulateVertex>* outputVertices;
    std::vector<std::vector<TriangulateVertex> >* outputContours;
    std::vector<TriangulateVertex> vertices;
};

void CALLBACK tessBeginCB(GLenum which, void* polygon)
{
    TriangulateContext* context = (TriangulateContext*)polygon;
    context->BeginPrimitiveBatch(which);
}

void CALLBACK tessEndCB(void* polygon)
{
    TriangulateContext* context = (TriangulateContext*)polygon;
    context->EndPrimitiveBatch();
}

void CALLBACK tessVertexCB(void *data, void* polygon)
{
    TriangulateContext* context = (TriangulateContext*)polygon;
    context->AddBatchVertex((GluFloat*)data);
}

void CALLBACK tessCombineCB(const GluFloat newVertex[3], const GluFloat *neighborVertex[4],
                            const GLfloat neighborWeight[4], GluFloat **outData, void* polygon)
{
    TriangulateContext* context = (TriangulateContext*)polygon;
    GluFloat s = 0;
    GluFloat t = 0;

    for (size_t i = 0; i < 4; ++i)
    {
        if (neighborVertex[i])
        {
            s += neighborWeight[i] * neighborVertex[i][3];
            t += neighborWeight[i] * neighborVertex[i][4];
        }
    }

    *outData = context->NewVertex(newVertex[0], newVertex[1], newVertex[2], s, t);
}

void CALLBACK tessErrorCB(GLenum /*errorCode*/, void* /*polygon*/)
{
}

struct TriangulatePolygonList
{
    const Vector2* positions;
    const Vector2* texcoords;
    size_t count;
};

void GluRegulateContoursWinding(const TriangulatePolygonList* polygons, size_t num, std::vector<std::vector<TriangulateVertex> >& contours)
{
    GLUtesselator *tessContour = gluNewTess();
    gluTessProperty(tessContour, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_NONZERO);
    gluTessProperty(tessContour, GLU_TESS_BOUNDARY_ONLY, GL_TRUE);
    gluTessCallback(tessContour, GLU_TESS_BEGIN_DATA, (void (__stdcall *)(void))tessBeginCB);
    gluTessCallback(tessContour, GLU_TESS_END_DATA, (void (__stdcall *)(void))tessEndCB);
    gluTessCallback(tessContour, GLU_TESS_ERROR_DATA, (void (__stdcall *)(void))tessErrorCB);
    gluTessCallback(tessContour, GLU_TESS_VERTEX_DATA, (void (__stdcall *)(void))tessVertexCB);
    gluTessCallback(tessContour, GLU_TESS_COMBINE_DATA, (void (__stdcall *)(void))tessCombineCB);

    TriangulateContext context(&contours);
    gluTessNormal(tessContour, 0, 0, 1);

    gluTessBeginPolygon(tessContour, &context);
    for (size_t j = 0; j < num; ++j)
    {
        const TriangulatePolygonList& pg = polygons[j];
        gluTessBeginContour(tessContour);
        for (size_t i = 0; i < pg.count; ++i)
        {
            float s = pg.texcoords ? pg.texcoords[i].x : 0.0f;
            float t = pg.texcoords ? pg.texcoords[i].y : 0.0f;
            GluFloat* vert = context.NewVertex(pg.positions[i].x, pg.positions[i].y, 0, s, t);
            gluTessVertex(tessContour, vert, vert);
        }
        gluTessEndContour(tessContour);
    }
    gluTessEndPolygon(tessContour);

    gluDeleteTess(tessContour);
}

bool Triangulation(const TriangulatePolygonList* polygons, size_t num, std::vector<float>& positionsOut, std::vector<unsigned short>& indexOut, std::vector<float>* texcoordOut = NULL)
{
    GLUtesselator *tess = gluNewTess();
    if (!tess)
    {
        return false;
    }

    std::vector<TriangulateVertex> triVertices;

    try
    {
        std::vector<std::vector<TriangulateVertex> > outerContours;
        GluRegulateContoursWinding(polygons, num, outerContours);

        TriangulateContext context(&triVertices);
        gluTessCallback(tess, GLU_TESS_BEGIN_DATA, (void (__stdcall *)(void))tessBeginCB);
        gluTessCallback(tess, GLU_TESS_END_DATA, (void (__stdcall *)(void))tessEndCB);
        gluTessCallback(tess, GLU_TESS_ERROR_DATA, (void (__stdcall *)(void))tessErrorCB);
        gluTessCallback(tess, GLU_TESS_VERTEX_DATA, (void (__stdcall *)(void))tessVertexCB);
        gluTessCallback(tess, GLU_TESS_COMBINE_DATA, (void (__stdcall *)(void))tessCombineCB);
        gluTessProperty(tess, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_POSITIVE);
        
        gluTessNormal(tess, 0, 0, 1);
        gluTessBeginPolygon(tess, &context);

        for (size_t i = 0; i < outerContours.size(); ++i)
        {
            gluTessBeginContour(tess);
            std::vector<TriangulateVertex>& c = outerContours[i];
            for (size_t j = 0; j < c.size(); ++j)
            {
                GluFloat* v = context.NewVertex(c[j].x, c[j].y, 0, c[j].s, c[j].t);
                gluTessVertex(tess, v, v);
            }
            gluTessEndContour(tess);
        }
        gluTessEndPolygon(tess);

        gluDeleteTess(tess);
    }
    catch (...)
    {
    }

    if (triVertices.size() > 0)
    {
        unsigned short base = (unsigned short)(positionsOut.size() / 2);
        for (size_t i = 0; i < triVertices.size(); ++i)
        {
            positionsOut.push_back((float)triVertices[i].x);
            positionsOut.push_back((float)triVertices[i].y);
            if (texcoordOut)
            {
                texcoordOut->push_back((float)triVertices[i].s);
                texcoordOut->push_back((float)triVertices[i].t);
            }

            indexOut.push_back((unsigned short)(base + i));
        }

        return true;
    }
    else
    {
        return false;
    }
}
#endif

void CatMullRomSampling(const float* p0, const float* p1, const float* p2, const float* p3, int components, int slice, std::vector<float>& samples)
{
    for (int j = 0; j < components; ++j)
    {
        float x0 = p0[j];
        float x1 = p1[j];
        float x2 = p2[j];
        float x3 = p3[j];
        float X0 = x1;
        float X1 = (-x0 + x2) * 0.5f;
        float X2 = x0 - 2.5f * x1 + 2.0f * x2 - 0.5f * x3;
        float X3 = -0.5f * x0 + 1.5f * x1- 1.5f * x2 + 0.5f * x3;

        for (int k = 0; k <= slice; ++k)
        {
            float t = k / (float)slice;
            float t2 = t * t;
            float t3 = t2 * t;
            samples.push_back(X0 + X1 * t + X2 * t2 + X3 * t3);
        }
    }
}

void CatMullRomSampling(const float* vertices, int num, int components, int slice, std::vector<float>& samples)
{
    for (int i = 1; i < num; ++i)
    {
        int idx0 = i - 2 >= 0 ? i - 2 : 0;
        int idx1 = i - 1;
        int idx2 = i;
        int idx3 = i + 1 < num ? i + 1 : num -1;
        const float* p0 = vertices + idx0 * components;
        const float* p1 = vertices + idx1 * components;
        const float* p2 = vertices + idx2 * components;
        const float* p3 = vertices + idx3 * components;

        CatMullRomSampling(p0, p1, p2, p3, components, slice, samples);
    }
}

float GetPolylineSamples(const float* vertices, int num, int vertexComponents, float step, float offset, std::vector<float>& result)
{
    float total = 0.0f;
    int i = 1;
    while (i < num)
    {
        const float* p0 = vertices + (i - 1) * vertexComponents;
        const float* p1 = vertices + i * vertexComponents;

        float dx = p1[0] - p0[0];
        float dy = p1[1] - p0[1];
        float segLength = sqrtf(dx * dx + dy * dy);

        if (segLength == 0)
        {
            ++i;
            continue;
        }

        if (offset > total + segLength)
        {
            ++i;
            total += segLength;
        }
        else
        {
            float t = (offset - total) / segLength;
            float invT = 1.0f - t;

            for (int j = 0; j < vertexComponents; ++j)
            {
                result.push_back(p0[j] * invT + p1[j] * t);
            }

            offset += step;
        }
    }
    return total - offset;
}


GLRenderTarget::GLRenderTarget(int width, int height, bool hasDepth, bool enableAA, void* data, bool smoothMin, bool smoothMag, int channels, int numTextures)
    :mWidth(width)
    ,mHeight(height)
    ,mFrameBufferId(0)
    ,mDepthTexture(0)
    ,mEnableAA(enableAA)
    ,mAABufferId(0)
    ,mDisableRelease(false)
    ,mSmoothMin(smoothMin)
    ,mSmoothMag(smoothMag)
{
    GLuint frameBufferId = 0;
    glGenFramebuffers(1, &frameBufferId);

    if (frameBufferId == 0)
    {
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
#ifdef glRenderbufferStorageMultisample
    if (mEnableAA)
    {
        GLuint rbo = 0;
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        GLint maxSample;
        glGetIntegerv(GL_MAX_SAMPLES, &maxSample);
        if (maxSample > 4)
        {
            maxSample = 4;
        }
        GLenum format = GL_RGBA;
        if (channels == 1)
        {
#if defined(__ANDROID__) || defined(TARGET_OS_IPHONE)
            format = GL_LUMINANCE;
#else
            format = GL_RED;
#endif
        }
        else if (channels == 3)
        {
            format = GL_RGB;
        }
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, maxSample, format, mWidth, mHeight);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
        mAABufferId = rbo;
    }
    else
#endif
    {
#if defined(USE_OPENGL3) || defined(USE_OPENGL3_ES)
        if (numTextures > 1)
        {
            GLenum bufs[16] = { 0 };
            for (int i = 0; i < numTextures; ++i)
            {
                mTextures.push_back(new GLTexture(mWidth, mHeight, data, channels, smoothMin, smoothMag, false, false, false, false, 1));
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, mTextures[i]->mTextureId, 0);
                bufs[i] = GL_COLOR_ATTACHMENT0 + i;
            }
            glDrawBuffers(numTextures, bufs);
        }
        else
#endif
        {
            mTextures.push_back(new GLTexture(mWidth, mHeight, data, channels, smoothMin, smoothMag, false, false, false, false, 0));
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTextures[0]->mTextureId, 0);
        }
        
        if (hasDepth)
        {
            mDepthTexture = new GLTexture(mWidth, mHeight, NULL, 1, false, false, false, false, false, true, 1);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mDepthTexture->mTextureId, 0);
        }
    }

    GLuint status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        LOGE("Can't create FBO: GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT width=%d height=%d", width, height);
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        LOGE("Can't create FBO: GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT width=%d height=%d", width, height);
        break;
    case GL_FRAMEBUFFER_UNSUPPORTED:
        LOGE("Can't create FBO: GL_FRAMEBUFFER_UNSUPPORTED width=%d height=%d", width, height);
        break;
    default:
        LOGE("Can't create FBO: unknown error width=%d height=%d", width, height);
        break;
    }

    if (data == NULL)
    {
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    mFrameBufferId = frameBufferId;
}

GLRenderTarget::GLRenderTarget(int defaultFrameBufferId)
    :mWidth(0)
    ,mHeight(0)
    ,mFrameBufferId(defaultFrameBufferId)
    ,mDepthTexture(0)
    ,mEnableAA(false)
    ,mDisableRelease(true)
    ,mSmoothMin(false)
    ,mSmoothMag(false)
{
}

GLRenderTarget::~GLRenderTarget()
{
    if (mDisableRelease)
    {
        return;
    }
    GLuint fid = mFrameBufferId;
    glDeleteFramebuffers(1, &fid);

    for (size_t i = 0; i < mTextures.size(); ++i)
    {
        delete mTextures[i];
    }
    if (mDepthTexture)
    {
        delete mDepthTexture;
    }
    if (mAABufferId)
    {
        GLuint rbo = mAABufferId;
        glDeleteBuffers(1, &rbo);
    }
}

void GLRenderTarget::Clear(const Color& color)
{
    Bind();
    glClearColor(color.r, color.g, color.b, color.a);
    GLbitfield flags = GL_COLOR_BUFFER_BIT;
    if (mFrameBufferId == 0 || mDepthTexture)
    {
        flags |= GL_DEPTH_BUFFER_BIT;
    }
    glClear(flags);
}

void GLRenderTarget::Bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBufferId);
    glViewport(0, 0, mWidth, mHeight);
}

void GLRenderTarget::Unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLRenderTarget::Blit(const GLRenderTarget* src, int x, int y, bool smooth)
{
    Blit(src, x, y, src->mWidth, src->mHeight, 0, 0, src->mWidth, src->mHeight, smooth);
}

void GLRenderTarget::Blit(const GLRenderTarget* src, int dstX, int dstY, int dstW, int dstH, int srcX, int srcY, int srcW, int srcH, bool smooth)
{
#ifdef HAVE_BLIT
    // ensure same size
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mFrameBufferId);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src->mFrameBufferId);
    glBlitFramebuffer(srcX, srcY, srcX + srcW, srcY + srcH,
                      dstX, dstY, dstX + dstW, dstY + dstH, GL_COLOR_BUFFER_BIT, smooth ? GL_LINEAR : GL_NEAREST);
#else
	GLRenderer* renderer = GLRenderer::GetInstance();
	BlitEffect* fx = renderer->GetBlitEffect();
    fx->Blit(this, dstX, dstY, dstW, dstH, (GLRenderTarget*)src, srcX, srcY, srcW, srcH, smooth);
#endif
}

Color GLRenderTarget::GetPixel(int x, int y)
{
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBufferId);
    GLubyte p[4] = {0};
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    Color result(p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f, p[3] / 255.0f);
    return result;
}

void GLRenderTarget::GetImage(void* data)
{
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBufferId);
    GLenum format = GL_RGBA;
    glReadPixels(0, 0, mWidth, mHeight, format, GL_UNSIGNED_BYTE, data);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLRenderTarget::Save(const char* path)
{
    int rowBytes = mWidth * 4;
    GLubyte* data = new GLubyte[rowBytes * mHeight];
    GLubyte* row = new GLubyte[rowBytes];
    GetImage(data);
    for (int i = 0; i < mHeight / 2; ++i)
    {
        GLubyte* l0 = data + i * rowBytes;
        GLubyte* l1 = data + (mHeight - 1 - i) * rowBytes;
        memcpy(row, l0, rowBytes);
        memcpy(l0, l1, rowBytes);
        memcpy(l1, row, rowBytes);
    }
    SaveImg(path, mWidth, mHeight, 4, data);
 
    delete[] row;
    delete[] data;
}

void GLRenderTarget::Replace(int fboId, int width, int height)
{
    mFrameBufferId = fboId;
    mWidth = width;
    mHeight = height;
}

bool GLRenderTarget::IsEqual(GLRenderTarget* src)
{
    if (mWidth != src->mWidth || mHeight != src->mHeight 
        || mTextures.size() != 1
        || src->mTextures.size() != 1)
    {
        return false;
    }

    int s = mWidth * mHeight * 4;
    GLubyte* data = new GLubyte[s];
    GLubyte* data2 = new GLubyte[s];
    GetImage(data);
    src->GetImage(data2);
    bool result = memcmp(data, data2, s) == 0;
    delete[] data;
    delete[] data2;
    return result;
}

namespace floodfill
{

typedef struct {		/* window: a discrete 2-D rectangle */
    int x0, y0;			/* xmin and ymin */
    int x1, y1;			/* xmax and ymax (inclusive) */
} Window;

typedef unsigned int Pixel;		/* 1-channel frame buffer assumed */

static Pixel* pixelBuffer = NULL;
static Pixel* maskBuffer = NULL;
static int pixelBufferWidth = 0;
static int pixelBufferHeight = 0;
static Window maskBound = {0, 0, 0, 0};

static inline Pixel pixelread(int x, int y)
{
    return pixelBuffer[y * pixelBufferWidth + x];
}

static inline bool pixelEquals(int x, int y, Pixel p0, int /*threshold*/)
{
    Pixel p1 = pixelBuffer[y * pixelBufferWidth + x];

    if (p1 != p0)
    {
        return false;
    }

    if (maskBuffer[y * pixelBufferWidth + x])
    {
        return false;
    }

    return true;
}

static inline void pixelwrite(int x, int y)
{
    maskBuffer[y * pixelBufferWidth + x] = 0xFFFFFFFF;
    if (x < maskBound.x0)
    {
        maskBound.x0 = x;
    }
    else if (x > maskBound.x1)
    {
        maskBound.x1 = x;
    }
    if (y < maskBound.y0)
    {
        maskBound.y0 = y;
    }
    else if (y > maskBound.y1)
    {
        maskBound.y1 = y;
    }
}

typedef struct {short y, xl, xr, dy;} Segment;
/*
* Filled horizontal segment of scanline y for xl<=x<=xr.
* Parent segment was on line y-dy.  dy=1 or -1
*/

#define MAX 10000		/* max depth of stack */

#define PUSH(Y, XL, XR, DY)	/* push new segment on stack */ \
    if (sp<stack+MAX && Y+(DY)>=win->y0 && Y+(DY)<=win->y1) \
{sp->y = Y; sp->xl = XL; sp->xr = XR; sp->dy = DY; sp++;}

#define POP(Y, XL, XR, DY)	/* pop segment off stack */ \
{sp--; Y = sp->y+(DY = sp->dy); XL = sp->xl; XR = sp->xr;}

/*
* fill: set the pixel at (x,y) and all of its 4-connected neighbors
* with the same pixel value to the new pixel value nv.
* A 4-connected neighbor is a pixel above, below, left, or right of a pixel.
*/

void fill(int x, int y,	/* seed point */
            Window *win,	/* screen window */
            int threshold
)
{
    int l, x1, x2, dy;
    Pixel ov;	/* old pixel value */
    Segment stack[MAX], *sp = stack;	/* stack of filled segments */

    if (x<win->x0 || x>win->x1 || y<win->y0 || y>win->y1)
    {
        return;
    }

    ov = pixelread(x, y);		/* read pv at seed point */

    PUSH(y, x, x, 1);			/* needed in some cases */
    PUSH(y+1, x, x, -1);		/* seed segment (popped 1st) */

    while (sp>stack)
    {
        /* pop segment off stack and fill a neighboring scan line */
        POP(y, x1, x2, dy);
        /*
        * segment of scan line y-dy for x1<=x<=x2 was previously filled,
        * now explore adjacent pixels in scan line y
        */
        for (x=x1; x>=win->x0 && pixelEquals(x, y, ov, threshold); x--)
        {
            pixelwrite(x, y);
        }

        if (x>=x1)
        {
            goto skip;
        }
        l = x+1;
        if (l<x1)
        {
            PUSH(y, l, x1-1, -dy);		/* leak on left? */
        }

        x = x1+1;
        do
        {
            for (; x<=win->x1 && pixelEquals(x, y, ov, threshold); x++)
            {
                pixelwrite(x, y);
            }

            PUSH(y, l, x-1, dy);
            if (x>x2+1)
            {
                PUSH(y, x2+1, x-1, -dy);	/* leak on right? */
            }

skip:	    for (x++; x<=x2 && !pixelEquals(x, y, ov, threshold); x++)
            {
            }
            l = x;
        } while (x<=x2);
    }
}

void fill(unsigned int* pixels, int width, int height, int* pts, int numPts, int threshold, unsigned int* mask, int& xMin, int& xMax, int& yMin, int& yMax)
{
    if (numPts < 1)
    {
        xMin = 0;
        xMax = 0;
        yMin = 0;
        yMax = 0;
        return;
    }

    Window win;
    win.x0 = 0;
    win.y0 = 0;
    win.x1 = width - 1;
    win.y1 = height - 1;
    pixelBuffer = pixels;
    maskBuffer = mask;
    pixelBufferWidth = width;
    pixelBufferHeight = height;

    maskBound.x0 = maskBound.x1 = pts[0];
    maskBound.y0 = maskBound.y1 = pts[1];
    for (int idx = 1; idx < numPts; ++idx)
    {
        int* pp = pts + (idx - 1) * 2;
        int x1 = pp[0];
        int y1 = pp[1];
        int x2 = pp[2];
        int y2 = pp[3];
#define sgn(v) ((v > 0) - (v < 0))
        int d, x, y, ax, ay, sx, sy, dx, dy;

        dx = x2-x1;  ax = abs(dx)<<1;  sx = sgn(dx);
        dy = y2-y1;  ay = abs(dy)<<1;  sy = sgn(dy);

        x = x1;
        y = y1;
        if (ax>ay) {		/* x dominant */
            d = ay-(ax>>1);
            for (;;) {
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    fill(x, y, &win, threshold);
                }
                if (x==x2) break;
                if (d>=0) {
                    y += sy;
                    d -= ax;
                }
                x += sx;
                d += ay;
            }
        }
        else {			/* y dominant */
            d = ax-(ay>>1);
            for (;;) {
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    fill(x, y, &win, threshold);
                }
                if (y==y2) break;
                if (d>=0) {
                    x += sx;
                    d -= ay;
                }
                y += sy;
                d += ax;
            }
        }
#undef sgn
    }

    if (numPts == 1)
    {
        fill(pts[0], pts[1], &win, threshold);
    }

    xMin = maskBound.x0;
    xMax = maskBound.x1;
    yMin = maskBound.y0;
    yMax = maskBound.y1;
}

void fillPts(unsigned int* pixels, int width, int height, int* pts, int numPts, int threshold, unsigned int* mask, int& xMin, int& xMax, int& yMin, int& yMax)
{
    if (numPts < 1)
    {
        xMin = 0;
        xMax = 0;
        yMin = 0;
        yMax = 0;
        return;
    }

    Window win;
    win.x0 = 0;
    win.y0 = 0;
    win.x1 = width - 1;
    win.y1 = height - 1;
    pixelBuffer = pixels;
    maskBuffer = mask;
    pixelBufferWidth = width;
    pixelBufferHeight = height;

    maskBound.x0 = maskBound.x1 = pts[0];
    maskBound.y0 = maskBound.y1 = pts[1];
    for (int idx = 0; idx < numPts; ++idx)
    {
        int* pp = pts + idx * 2;
        int x = pp[0];
        int y = pp[1];
        if ((pixels[y * width + x] & 0xFFFFFF) != 0)
        {
            continue;
        }
        fill(x, y, &win, threshold);
    }

    xMin = maskBound.x0;
    xMax = maskBound.x1;
    yMin = maskBound.y0;
    yMax = maskBound.y1;
}

void expandFill(unsigned int* /*bits*/, unsigned int* mask, int w, int h,
                int xMin, int yMin, int xMax, int yMax,
                int expand)
{
    for (int y = yMin; y <= yMax; ++y)
    {
        for (int x = xMin; x <= xMax; ++x)
        {
            bool hit = false;
            for (int dy = -expand; dy <= expand; ++dy)
            {
                int py = y + dy;
                if (py < 0 || py >= h)
                {
                    continue;
                }
                for (int dx = -expand; dx <= expand; ++dx)
                {
                    int px = x + dx;
                    if (px < 0 || px >= w)
                    {
                        continue;
                    }

                    if (mask[py * w + px] == 0xFFFFFFFF)
                    {
                        hit = true;
                        goto done;
                    }
                }
            }
done:
            if (hit)
            {
                mask[y * w + x] = 0xFFFFFFFE;
            }
        }
    }
}

bool isHole(unsigned char* bits, unsigned int* mask, int w, int h, int x, int y, int expand)
{
	int px = x;
	int py = y;

	if (mask[y * w + x])
	{
		return false;
	}

	{
		px = x + 1;
		py = y;
		while (px <= x + expand)
		{
			if (px >= w)
			{
				break;
			}
			if (bits[(py * w + px) * 4 + 3] || mask[py * w + px] == 0xFFFFFFFF)
			{
				break;
			}
			++px;
		}
		if (px > x + expand)
		{
			return false;
		}
	}

	{
		px = x;
		py = y + 1;
		while (py <= y + expand)
		{
			if (py >= h)
			{
				break;
			}
			if (bits[(py * w + px) * 4 + 3] || mask[py * w + px] == 0xFFFFFFFF)
			{
				break;
			}
			++py;
		}
		if (py > y + expand)
		{
			return false;
		}
	}

	{
		px = x - 1;
		py = y;
		while (px >= x - expand)
		{
			if (px < 0)
			{
				break;
			}
			if (bits[(py * w + px) * 4 + 3] || mask[py * w + px] == 0xFFFFFFFF)
			{
				break;
			}
			--px;
		}
		if (px < x - expand)
		{
			return false;
		}
	}

	{
		px = x;
		py = y - 1;
		while (py >= y - expand)
		{
			if (py < 0)
			{
				break;
			}
			if (bits[(py * w + px) * 4 + 3] || mask[py * w + px] == 0xFFFFFFFF)
			{
				break;
			}
			--py;
		}
		if (py < y - expand)
		{
			return false;
		}
	}
	return true;
}

bool isUnfilled(unsigned char* bits, unsigned char* mask, int w, int h, int x, int y, int expand)
{
#define getA(x,y) bits[((y)*w+(x))*4+3]
#define maskA(x,y) mask[((y)*w+(x))*4+3]
	// abrupt
	int a = getA(x, y);
	int leak = 0;
	if (x - 1 >= 0)
	{
		int pa = maskA(x - 1, y);
		if (pa == 0)
		{
			pa = getA(x - 1, y);
		}
		if (pa <= a)
		{
			// find leak
			int i = 0;
			for (i = 1; i <= expand; ++i)
			{
				int dx = x + i * (-1);
				int la = getA(dx, y);
				if (la > a)
				{
					break;
				}
			}
			if (i > expand)
			{
				return false;
			}
 			++leak;
		}
	}

	if (x + 1 < w)
	{
		int pa = maskA(x + 1, y);
		if (pa == 0)
		{
			pa = getA(x + 1, y);
		}
		if (pa <= a)
		{
			if (leak)
			{
				return false;
			}
			// find leak
			int i = 0;
			for (i = 1; i <= expand; ++i)
			{
				int dx = x + i * (1);
				int la = getA(dx, y);
				if (la > a)
				{
					break;
				}
			}
			if (i > expand)
			{
				return false;
			}
 			++leak;
		}
	}

	if (y - 1 >= 0)
	{
		int pa = maskA(x, y - 1);
		if (pa == 0)
		{
			pa = getA(x, y - 1);
		}
		if (pa <= a)
		{
			if (leak)
			{
				return false;
			}
			// find leak
			int i = 0;
			for (i = 1; i <= expand; ++i)
			{
				int dy = y + i * (-1);
				int la = getA(x, dy);
				if (la > a)
				{
					break;
				}
			}
			if (i > expand)
			{
				return false;
			}
 			++leak;
		}
	}

	if (y + 1 < h)
	{
		int pa = maskA(x, y + 1);
		if (pa == 0)
		{
			pa = getA(x, y + 1);
		}
		if (pa <= a)
		{

			if (leak)
			{
				return false;
			}
			// find leak
			int i = 0;
			for (i = 1; i <= expand; ++i)
			{
				int dy = y + i * (1);
				int la = getA(x, dy);
				if (la > a)
				{
					break;
				}
			}
			if (i > expand)
			{
				return false;
			}
 			++leak;
		}
	}

	return leak <= 3;
#undef getA
#undef maskA
}

void fillHoles(unsigned int* bits, unsigned int* mask, int w, int h,
               int xMin, int yMin, int xMax, int yMax,
               int /*expand*/)
{
	unsigned char* pb = (unsigned char*)bits;
	int iteration = 0;
	bool changed = false;
	
	do 
	{
		long contourAlpha = 0;
		int count = 0;
		std::vector<int> pts;
		changed = false;
		for (int y = yMin - 1; y <= yMax + 1; ++y)
		{
			if (y < 0 || y >= h)
			{
				continue;
			}
			for (int x = xMin - 1; x <= xMax + 1; ++x)
			{
				if (x < 0 || x >= w)
				{
					continue;
				}

				if (mask[y * w + x])
				{
					continue;
				}

				int ma = pb[(y * w + x) * 4 + 3];
				if (ma)
				{
					contourAlpha += ma;
					count++;
				}
				bool isInRange = false;
				for (int dy = -1; dy <= 1; ++dy)
				{
					int py = y + dy;
					if (py < 0 || py >= h)
					{
						continue;
					}
					for (int dx = -1; dx <= 1; ++dx)
					{
						int px = x + dx;
						if (px < 0 || px >= w)
						{
							continue;
						}
						if (dx == 0 && dy == 0)
						{
							continue;
						}

						unsigned int pix = mask[py * w + px];
						if (pix)
						{
							isInRange = true;
							goto done;
						}
					}
				}
			done:
				if (!isInRange || !isUnfilled(pb, (unsigned char*)mask, w, h, x, y, 2))
				{
					continue;
				}

				pts.push_back(x);
				pts.push_back(y);
			}
		}

		if (pts.size() > 0)
		{
			contourAlpha /= count;

			for (size_t i = 0; i < pts.size()/2; ++i)
			{
				int x = pts[i * 2];
				int y = pts[i * 2 + 1];

				int pa = pb[(y * w + x) * 4 + 3];
				if (pa > contourAlpha)
				{
					continue;
				}

				mask[y * w + x] = 0xFFFFFFFF;

				if (x > xMax)
				{
					xMax = x;
				}
				if (x < xMin)
				{
					xMin = x;
				}
				if (y > yMax)
				{
					yMax = y;
				}
				if (y < yMin)
				{
					yMin = y;
				}
			}
			
			changed = true;
		}

		++iteration;
	} 
	while (changed && iteration < 30);
// 
// 	std::vector<int> holePts;
// 	for (int y = yMin - expand; y <= yMax + expand; ++y)
// 	{
// 		if (y < 0 || y >= h)
// 		{
// 			continue;
// 		}
// 
// 		for (int x = xMin - expand; x <= xMax + expand; ++x)
// 		{
// 			if (x < 0 || x >= w)
// 			{
// 				continue;
// 			}
// 
// 			if (mask[y * w + x] == 0xFFFFFFFF)
// 			{
// 				// filled
// 				continue;
// 			}
// 
// 			bool inRange = false;
// 			for (int dy = -expand; dy <= expand; ++dy)
// 			{
// 				int py = y + dy;
// 				if (py < 0 || py >= h)
// 				{
// 					continue;
// 				}
// 				for (int dx = -expand; dx <= expand; ++dx)
// 				{
// 					int px = x + dx;
// 					if (px < 0 || px >= w)
// 					{
// 						continue;
// 					}
// 
// 					if (dx == 0 && dy == 0)
// 					{
// 						continue;
// 					}
// 
// 					if (mask[py * w + px] == 0xFFFFFFFF)
// 					{
// 						inRange = true;
// 						goto done2;
// 					}
// 				}
// 			}
// 		done2:
// 			if (!inRange)
// 			{
// 				continue;
// 			}
// 
// 			if (isHole(pb, mask, w, h, x, y, 2) || pb[(y * w + x) * 4 + 3])
// 			{
// 				holePts.push_back(x);
// 				holePts.push_back(y);
// 			}
// 		}
// 	}
// 
// 	for (size_t i = 0; i < holePts.size() / 2; ++i)
// 	{
// 		int x = holePts[i * 2];
// 		int y = holePts[i * 2 + 1];
// 
// 		mask[y * w + x] = 0xFFFFFF88;
// 	}


}


}

void GLRenderTarget::FloodFill(GLTexture* target, int* pts, int numPts, const Color& /*color*/, int expand)
{
    int rowBytes = mWidth * 4;
    GLubyte* bits = new GLubyte[rowBytes * mHeight];
    GetImage(bits);

    unsigned int* mask = new unsigned int[mWidth * mHeight];
    memset(mask, 0, mWidth * mHeight * 4);
    int xMin, xMax, yMin, yMax;
    floodfill::fill((unsigned int*) bits, mWidth, mHeight, pts, numPts, 0, mask, xMin, xMax, yMin, yMax);
    if (expand)
    {
        floodfill::expandFill((unsigned int*) bits, mask, mWidth, mHeight, xMin, yMin, xMax, yMax, expand);
    }
	//floodfill::fillHoles((unsigned int*)bits, mask, mWidth, mHeight, xMin, yMin, xMax, yMax, 10);
    target->WriteTexture(mask, mWidth, mHeight, false);

    delete[] mask;
    delete[] bits;
}

void GLRenderTarget::FloodFill(GLTexture* target, GLRenderTarget* input)
{
    typedef std::map<unsigned int, std::vector<Vector2i>*> ColorPointMap;
    ColorPointMap fillPoints;
    {
        unsigned int* bits = new unsigned int[input->mWidth * input->mHeight];
        input->GetImage(bits);
        for (int y = 0; y < input->mHeight; ++y)
        {
            for (int x = 0; x < input->mWidth; ++x)
            {
                unsigned int color = bits[y * mWidth + x];
                if (color & 0xFFFFFF)
                {
                    ColorPointMap::iterator it = fillPoints.find(color);
                    if (it != fillPoints.end())
                    {
                        it->second->push_back(Vector2i(x,y));
                    }
                    else
                    {
                        std::vector<Vector2i>* pts = new std::vector<Vector2i>();
                        pts->push_back(Vector2i(x,y));
                        fillPoints[color] = pts;
                    }
                }
            }
        }
        delete[] bits;
    }

    if (fillPoints.empty())
    {
        return;
    }

    GLubyte* bits = new GLubyte[mWidth * mHeight * 4];
    GetImage(bits);

    unsigned int* mask = new unsigned int[mWidth * mHeight];

    unsigned int* result = new unsigned int[mWidth * mHeight];
    memset(result, 0, mWidth * mHeight * 4);

    for (ColorPointMap::iterator it = fillPoints.begin(); it != fillPoints.end(); ++it)
    {
        unsigned int color = it->first;
        const std::vector<Vector2i>& pts = *it->second;
        if (pts.size() > 0)
        {
            int xMin, xMax, yMin, yMax;
            memset(mask, 0, mWidth * mHeight * 4);
            floodfill::fillPts((unsigned int*) bits, mWidth, mHeight, (int*)&pts.front(), pts.size(), 0, mask, xMin, xMax, yMin, yMax);
            for (int y = yMin; y <= yMax; ++y)
            {
                for (int x = xMin; x <= xMax; ++x)
                {
                    if (mask[y * mWidth + x])
                    {
                        unsigned int* p = result + (y * mWidth + x);
                        *p = color;
                    }
                }
            }
        }
        delete it->second;
    }
    target->WriteTexture(result, mWidth, mHeight, false);

    delete[] mask;
    delete[] bits;
    delete[] result;
}

GLTexture::GLTexture(int width, int height, void* data, int channels, bool minSmooth, bool magSmooth, bool repeat, bool useMipmap, bool /*useAnisotropicFiltering*/, bool isDepth, int componentType)
    :mWidth(width)
    ,mHeight(height)
	,mChannels(channels)
    ,mTextureId(0)
{
    GLuint textureId = 0;
    glGenTextures(1, &textureId);
    if (textureId == 0)
    {
        return;
    }
    glBindTexture(GL_TEXTURE_2D, textureId);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    GLenum format = GL_RGBA;
    GLenum compType = GL_UNSIGNED_BYTE;
    GLenum pixelFormat = GL_RGBA;

    if (isDepth)
    {
        format = GL_DEPTH_COMPONENT;
        pixelFormat = GL_DEPTH_COMPONENT;
        compType = GL_UNSIGNED_INT;
    }
    else
    {
        if (channels == 1)
        {
#if defined(__ANDROID__) || (defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE)
            format = GL_LUMINANCE;
            pixelFormat = GL_LUMINANCE;
#else
            format = GL_RED;
            pixelFormat = GL_RED;
#endif
        }
        else if (channels == 3)
        {
            format = GL_RGB;
            pixelFormat = GL_RGB;
        }

        if (componentType == 1)
        {
            compType = GL_FLOAT;
#if !defined(__ANDROID__) && !defined(TARGET_OS_IPHONE)
            if (channels == 3)
            {
                format = GL_RGB32F;
            }
            else
            {
                format = GL_RGBA32F;
            }
#endif
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, pixelFormat, compType, data);

    if (useMipmap)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minSmooth ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minSmooth ? GL_LINEAR : GL_NEAREST);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magSmooth ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    mTextureId = textureId;
}

GLTexture::~GLTexture()
{
    if (mTextureId != 0)
    {
        GLuint t = mTextureId;
        glDeleteTextures(1, &t);
    }
}

void GLTexture::WriteTexture(const void* data, int w, int h, bool useMipmap)
{
    glBindTexture(GL_TEXTURE_2D, mTextureId);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    GLenum format = GL_RGBA;
    if (mChannels == 1)
    {
#if defined(__ANDROID__) || (defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE)
        format = GL_LUMINANCE;
#else
        format = GL_RED;
#endif
    }
    else if (mChannels == 3)
    {
        format = GL_RGB;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, format, GL_UNSIGNED_BYTE, data);

    if (useMipmap)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
}

void GLTexture::Save(const char* path)
{
#ifdef _WIN32
    int rowBytes = mWidth * 4;
    GLubyte* data = new GLubyte[rowBytes * mHeight];
    GLubyte* row = new GLubyte[rowBytes];

    glBindTexture(GL_TEXTURE_2D, mTextureId);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    for (int i = 0; i < mHeight / 2; ++i)
    {
        GLubyte* l0 = data + i * rowBytes;
        GLubyte* l1 = data + (mHeight - 1 - i) * rowBytes;
        memcpy(row, l0, rowBytes);
        memcpy(l0, l1, rowBytes);
        memcpy(l1, row, rowBytes);
    }
    SaveImg(path, mWidth, mHeight, 4, data);

    delete[] row;
    delete[] data;
#endif
}

static int CreateShader(GLuint programId, const char* source, GLenum shaderType, int debugLine, const char* name)
{
    GLint status = 0;
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const char**)&source, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status)
    {
        glAttachShader(programId, shader);
        return shader;
    }
    else
    {
        const int MAX_INFO_LOG_SIZE = 1000;
        GLchar log[MAX_INFO_LOG_SIZE];
        glGetShaderInfoLog(shader, MAX_INFO_LOG_SIZE, NULL, log);
        LOGE("Failed to compile shader. name=%s, line=%d:\n%s\n", name ? name : "untitled", debugLine, log);
        return 0;
    }
}

GLShaderProgram::GLShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource, const std::map<std::string, int>& bindLocations, int debugLine, const char* name)
    :mProgramId(0)
{
    GLuint programId = 0;
    GLuint vertexShaderId = 0;
    GLuint fragmentShaderId = 0;

    bool failed = false;
    do
    {
        programId = glCreateProgram();
        if (programId == 0)
        {
            failed = true;
            break;
        }

        vertexShaderId = CreateShader(programId, vertexShaderSource, GL_VERTEX_SHADER, debugLine, name);
        if (vertexShaderId == 0)
        {
            failed = true;
            break;
        }

        fragmentShaderId = CreateShader(programId, fragmentShaderSource, GL_FRAGMENT_SHADER, debugLine, name);
        if (fragmentShaderId == 0)
        {
            failed = true;
            break;
        }

        for (std::map<std::string, int>::const_iterator it = bindLocations.begin(); it != bindLocations.end(); ++it)
        {
            glBindAttribLocation(programId, it->second, it->first.c_str());
        }

        GLint status = 0;
        glLinkProgram(programId);
        glGetProgramiv(programId, GL_LINK_STATUS, &status);
        if (!status)
        {
            const int MAX_INFO_LOG_SIZE = 1000;
            GLchar log[MAX_INFO_LOG_SIZE];
            glGetProgramInfoLog(programId, MAX_INFO_LOG_SIZE, NULL, log);
            LOGE("link shdader failed. name=%s, line=%d\n%s\n", name ? name : "untitled", debugLine, log);
        }
    }
    while(0);

    if (vertexShaderId)
    {
        glDeleteShader(vertexShaderId);
    }
    if (fragmentShaderId)
    {
        glDeleteShader(fragmentShaderId);
    }

    if (failed && programId)
    {
        glDeleteProgram(programId);
        return;
    }

    mProgramId = programId;
}

GLShaderProgram::~GLShaderProgram()
{
    if (mProgramId <= 0)
    {
        return;
    }
    glDeleteProgram((GLuint)mProgramId);
}

int GLShaderProgram::GetUniformLocation(const char* name)
{
    return glGetUniformLocation(mProgramId, name);
}

void GLShaderProgram::Bind()
{
    glUseProgram(mProgramId);
}

void GLShaderProgram::Unbind()
{
    glUseProgram(0);
}

void GLShaderProgram::SetTexture(const char* name, int index, GLTexture* texture)
{
    glActiveTexture(GL_TEXTURE0 + index);
    glBindTexture(GL_TEXTURE_2D, texture->mTextureId);
    glUniform1i(GetUniformLocation(name), index);
}

void GLShaderProgram::SetTexture(int loc, int index, GLTexture* texture)
{
    glActiveTexture(GL_TEXTURE0 + index);
    glBindTexture(GL_TEXTURE_2D, texture->mTextureId);
    glUniform1i(loc, index);
}

void GLShaderProgram::SetMatrix(const char* name, const float* mat)
{
    glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, mat);
}

void GLShaderProgram::SetMatrix(int loc, const float* mat)
{
    glUniformMatrix4fv(loc, 1, GL_FALSE, mat);
}

void GLShaderProgram::SetFloat4(const char* name, float* v)
{
    glUniform4fv(GetUniformLocation(name), 1, v);
}

void GLShaderProgram::SetFloat4(int loc, float* v)
{
    glUniform4fv(loc, 1, v);
}

void GLShaderProgram::SetFloat3(const char* name, float* v)
{
    glUniform3fv(GetUniformLocation(name), 1, v);
}

void GLShaderProgram::SetFloat3(int loc, float* v)
{
    glUniform3fv(loc, 1, v);
}

void GLShaderProgram::SetFloat2(const char* name, float* v)
{
    glUniform2fv(GetUniformLocation(name), 1, v);
}

void GLShaderProgram::SetFloat2(int loc, float* v)
{
    glUniform2fv(loc, 1, v);
}

void GLShaderProgram::SetFloat(const char* name, float v)
{
    glUniform1f(GetUniformLocation(name), v);
}

void GLShaderProgram::SetFloat(int loc, float v)
{
    glUniform1f(loc, v);
}

void GLShaderProgram::SetFloat2(const char* name, float x, float y)
{
    glUniform2f(GetUniformLocation(name), x, y);
}

void GLShaderProgram::SetFloat2(int loc, float x, float y)
{
    glUniform2f(loc, x, y);
}

void GLShaderProgram::SetFloat3(const char* name, float x, float y, float z)
{
    glUniform3f(GetUniformLocation(name), x, y, z);
}

void GLShaderProgram::SetFloat3(int loc, float x, float y, float z)
{
    glUniform3f(loc, x, y, z);
}

void GLShaderProgram::SetFloat4(const char* name, float x, float y, float z, float w)
{
    glUniform4f(GetUniformLocation(name), x, y, z, w);
}

void GLShaderProgram::SetFloat4(int loc, float x, float y, float z, float w)
{
    glUniform4f(loc, x, y, z, w);
}

void GLShaderProgram::SetFloat2Array(const char* name, const float* v, int count)
{
    glUniform2fv(GetUniformLocation(name), count, v);
}

void GLShaderProgram::SetFloatArray(int loc, const float* v, int count)
{
    glUniform1fv(loc, count, v);
}

void GLShaderProgram::SetFloat2Array(int loc, const float* v, int count)
{
    glUniform2fv(loc, count, v);
}

void GLShaderProgram::SetFloat3Array(int loc, const float* v, int count)
{
    glUniform3fv(loc, count, v);
}

void GLShaderProgram::SetFloat4Array(int loc, const float* v, int count)
{
    glUniform4fv(loc, count, v);
}

void GLShaderProgram::SetInt(int loc, int v)
{
    glUniform1i(loc, v);
}

GLUniformValue::GLUniformValue()
    :mType(ValueTypeNone)
{
}

GLUniformValue::GLUniformValue(int index, GLTexture* texture)
    :mType(ValueTypeTexture)
{
    TextureValue tv;
    tv.index = index;
    tv.texture = texture;
    mValue.texture = tv;
}

GLUniformValue::GLUniformValue(float x, float y, float z, float w)
    :mType(ValueTypeFloat4)
{
    mValue.vec4[0] = x;
    mValue.vec4[1] = y;
    mValue.vec4[2] = z;
    mValue.vec4[3] = w;
}

GLUniformValue::GLUniformValue(float x, float y, float z)
    :mType(ValueTypeFloat3)
{
    mValue.vec3[0] = x;
    mValue.vec3[1] = y;
    mValue.vec3[2] = z;
}

GLUniformValue::GLUniformValue(float x, float y)
    :mType(ValueTypeFloat2)
{
    mValue.vec2[0] = x;
    mValue.vec2[1] = y;
}

GLUniformValue::GLUniformValue(float v)
    :mType(ValueTypeFloat)
{
    mValue.value = v;
}

GLUniformValue::GLUniformValue(const float* mat)
    :mType(ValueTypeMatrix4)
{
    memcpy(mValue.mat4, mat, sizeof(float) * 16);
}

GLUniformValue::GLUniformValue(const float* vecs, int components, int count)
    :mType(ValueTypeNone)
{
    switch (components)
    {
    case 2:
        mType = ValueTypeFloat2Array;
        break;
    default:
        break;
    }

    if (mType == ValueTypeNone)
    {
        mValue.fv.values = NULL;
        mValue.fv.count = 0;
        return;
    }

    mValue.fv.values = vecs;
    mValue.fv.count = count;
}

GLUniformValue::GLUniformValue(const Matrix4& mat)
    :mType(ValueTypeMatrix4)
{
    memcpy(mValue.mat4, &mat.m00, sizeof(float) * 16);
}

GLUniformValue::~GLUniformValue()
{
}

void GLUniformValue::Apply(const char* name, GLShaderProgram* program)
{
    switch (mType)
    {
    case GLUniformValue::ValueTypeTexture:
        program->SetTexture(name, mValue.texture.index, mValue.texture.texture);
        break;
    case GLUniformValue::ValueTypeFloat:
        program->SetFloat(name, mValue.value);
        break;
    case GLUniformValue::ValueTypeFloat2:
        program->SetFloat2(name, mValue.vec2);
        break;
    case GLUniformValue::ValueTypeFloat3:
        program->SetFloat3(name, mValue.vec3);
        break;
    case GLUniformValue::ValueTypeFloat4:
        program->SetFloat4(name, mValue.vec4);
        break;
    case GLUniformValue::ValueTypeMatrix4:
        program->SetMatrix(name, mValue.mat4);
        break;
    case GLUniformValue::ValueTypeFloat2Array:
        program->SetFloat2Array(name, mValue.fv.values, mValue.fv.count);
        break;
    default:
        break;
    }
}

GLShaderState::GLShaderState()
{
}

GLShaderState::~GLShaderState()
{
}

void GLShaderState::SetValue(const char* name, const GLUniformValue& value)
{
    mValues[name] = value;
}

void GLShaderState::Apply(GLShaderProgram* program)
{
    for (ValueMap::iterator it = mValues.begin(); it != mValues.end(); ++it)
    {
        it->second.Apply(it->first.c_str(), program);
    }
}

void GLShaderState::Reset()
{
    mValues.clear();
}



GLAttribute::GLAttribute(int index, int components, int count, int offset)
    :mIndex(index)
    ,mComponets(components)
    ,mVertexSize(sizeof(float) * components)
    ,mCount(count)
    ,mOffset(offset)
{
    assert(mComponets > 0 && mComponets <= 4);
}

GLAttribute::~GLAttribute()
{
}

GLMesh::GLMesh(int vertexCapacity, int indexCapacity, AttributeDesc* attributes, int attrNum, GLPrimitiveType primitiveType)
    :mVertexCapacity(vertexCapacity)
    ,mIndexCapacity(indexCapacity)
    ,mVertexSize(0)
    ,mVertices(0)
    ,mIndices(0)
    ,mVertexCount(0)
    ,mIndexCount(0)
    ,mVertexBufferId(0)
    ,mIndexBufferId(0)
    ,mVao(0)
    ,mPrimitiveType(primitiveType)
{
    int offset = 0;
    memset(mAttributes, 0, sizeof(GLAttribute*) * MAX_GL_ATTRIBUTES);
    for (int i = 0; i < attrNum; ++i)
    {
        GLAttribute* attr = new GLAttribute(attributes[i].index, attributes[i].components, mVertexCapacity, offset);
        mAttributes[attributes[i].index] = attr;
        offset += attr->GetVertexSize();
    }
    mVertexSize = offset;

    mVertices = new unsigned char[mVertexSize * vertexCapacity];
    mIndices = new unsigned short[indexCapacity];
    memset(mVertices, 0, mVertexSize * mVertexCapacity);
    memset(mIndices, 0, mIndexCapacity * sizeof(unsigned short));
}

GLMesh::~GLMesh()
{
    for (size_t i = 0; i < MAX_GL_ATTRIBUTES; ++i)
    {
        delete mAttributes[i];
    }
    delete[] mVertices;
    delete[] mIndices;

    if (mVertexBufferId)
    {
        GLuint id = (GLuint)mVertexBufferId;
        glDeleteBuffers(1, &id);
    }
    if (mIndexBufferId)
    {
        GLuint id = (GLuint)mIndexBufferId;
        glDeleteBuffers(1, &id);
    }
#ifdef USE_VAO
    if (mVao)
    {
        GLuint id = (GLuint)mVao;
        glDeleteVertexArrays(1, &id);
    }
#endif
}

void GLMesh::SetVertexCount(int n)
{
    assert(n <= mVertexCapacity);
    mVertexCount = n;
}

void GLMesh::DrawSubset(int offset, int indexCount)
{
    if (mIndexCount == 0)
    {
        return;
    }
    GLenum primitveType = GL_TRIANGLES;
    if (mPrimitiveType == GLPrimitiveTypeLineList)
    {
        primitveType = GL_LINES;
    }

#ifdef USE_VAO
    if (!mVao)
    {
        BuildVBO();
    }
    if (mVao)
    {
        glBindVertexArray(mVao);
        glDrawElements(primitveType, indexCount, GL_UNSIGNED_SHORT, (const void*)offset);
    }
#else
    {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        for (size_t i = 0; i < MAX_GL_ATTRIBUTES; ++i)
        {
            GLAttribute* attr = mAttributes[i];
            if (attr)
            {
                glVertexAttribPointer(i, attr->mComponets, GL_FLOAT, GL_FALSE, mVertexSize, mVertices + attr->mOffset);
                glEnableVertexAttribArray(i);
            }
            else
            {
                glDisableVertexAttribArray(i);
            }
        }
        glDrawElements(primitveType, indexCount, GL_UNSIGNED_SHORT, mIndices + offset);
    }
#endif
}

void GLMesh::Draw()
{
    DrawSubset(0, mIndexCount);
}

void GLMesh::BuildVBO()
{
    if (mVertexBufferId)
    {
        GLuint id = (GLuint)mVertexBufferId;
        glDeleteBuffers(1, &id);
        mVertexBufferId = 0;
    }
    if (mIndexBufferId)
    {
        GLuint id = (GLuint)mIndexBufferId;
        glDeleteBuffers(1, &id);
        mIndexBufferId = 0;
    }
#ifdef USE_VAO
    if (mVao)
    {
        GLuint id = (GLuint)mVao;
        glDeleteVertexArrays(1, &id);
        mVao = 0;
    }
#endif

	GLuint vb = 0;
	glGenBuffers(1, &vb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, mVertexSize * mVertexCount, mVertices, GL_STATIC_DRAW);

	GLuint ib = 0;
	glGenBuffers(1, &ib);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * mIndexCount, mIndices, GL_STATIC_DRAW);
#ifdef USE_VAO
	GLuint vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
#endif
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	for (size_t i = 0; i < MAX_GL_ATTRIBUTES; ++i)
	{
		GLAttribute* attr = mAttributes[i];
		if (attr)
		{
			glVertexAttribPointer(i, attr->mComponets, GL_FLOAT, GL_FALSE, mVertexSize, (const void*)attr->mOffset);
			glEnableVertexAttribArray(i);
		}
		else
		{
			glDisableVertexAttribArray(i);
		}
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	mVertexBufferId = vb;
	mIndexBufferId = ib;
#ifdef USE_VAO
    mVao = vao;
	glBindVertexArray(0);
#endif
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//if (0)
	//{
	//	delete[] mVertices;
	//	mVertices = NULL;
	//	delete[] mIndices;
	//	mIndices = NULL;
	//}
}

void GLMesh::SetFloat2(int index, int at, float x, float y)
{
    GLAttribute* attrPos = mAttributes[index];
    float* v = (float*)(mVertices + attrPos->mOffset + at * mVertexSize);
    v[0] = x;
    v[1] = y;
}

void GLMesh::SetVector2(int index, int at, const Vector2& value)
{
    GLAttribute* attrPos = mAttributes[index];
    float* v = (float*)(mVertices + attrPos->mOffset + at * mVertexSize);
    v[0] = value.x;
    v[1] = value.y;
}

void GLMesh::SetFloat3(int index, int at, float x, float y, float z)
{
    GLAttribute* attrPos = mAttributes[index];
    float* v = (float*)(mVertices + attrPos->mOffset + at * mVertexSize);
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

void GLMesh::SetVector3(int index, int at, const Vector3& value)
{
    GLAttribute* attrPos = mAttributes[index];
    float* v = (float*)(mVertices + attrPos->mOffset + at * mVertexSize);
    v[0] = value.x;
    v[1] = value.y;
    v[2] = value.z;
}

void GLMesh::SetFloat4(int index, int at, float x, float y, float z, float w)
{
    GLAttribute* attrPos = mAttributes[index];
    float* v = (float*)(mVertices + attrPos->mOffset + at * mVertexSize);
    v[0] = x;
    v[1] = y;
    v[2] = z;
    v[3] = w;
}

void GLMesh::Clear()
{
    mVertexCount = 0;
    mIndexCount = 0;

	if (mVertexBufferId)
	{
		GLuint id = (GLuint)mVertexBufferId;
		glDeleteBuffers(1, &id);
		mVertexBufferId = 0;
	}
	if (mIndexBufferId)
	{
		GLuint id = (GLuint)mIndexBufferId;
		glDeleteBuffers(1, &id);
		mIndexBufferId = 0;
	}
#ifdef USE_VAO
	if (mVao)
	{
		GLuint id = (GLuint)mVao;
		glDeleteVertexArrays(1, &id);
		mVao = 0;
	}
#endif

}

void GLMesh::AddRect(const AABB2& rect, const AABB2& texcoord, const Color& color)
{
    unsigned short n = (unsigned short)mVertexCount;
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    if (attrPos)
    {
        SetFloat2(GLAttributeTypePosition, n + 0, rect.xMin, rect.yMin);
        SetFloat2(GLAttributeTypePosition, n + 1, rect.xMax, rect.yMin);
        SetFloat2(GLAttributeTypePosition, n + 2, rect.xMax, rect.yMax);
        SetFloat2(GLAttributeTypePosition, n + 3, rect.xMin, rect.yMax);
    }

    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    if (attrTex)
    {
        SetFloat2(GLAttributeTypeTexcoord, n + 0, texcoord.xMin, texcoord.yMin);
        SetFloat2(GLAttributeTypeTexcoord, n + 1, texcoord.xMax, texcoord.yMin);
        SetFloat2(GLAttributeTypeTexcoord, n + 2, texcoord.xMax, texcoord.yMax);
        SetFloat2(GLAttributeTypeTexcoord, n + 3, texcoord.xMin, texcoord.yMax);
    }

    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
    if (attrColor)
    {
        SetFloat4(GLAttributeTypeColor, n + 0, color.r, color.g, color.b, color.a);
        SetFloat4(GLAttributeTypeColor, n + 1, color.r, color.g, color.b, color.a);
        SetFloat4(GLAttributeTypeColor, n + 2, color.r, color.g, color.b, color.a);
        SetFloat4(GLAttributeTypeColor, n + 3, color.r, color.g, color.b, color.a);
    }
    
    mVertexCount += 4;

    SetIndex(n + 0);
    SetIndex(n + 1);
    SetIndex(n + 2);
    SetIndex(n + 0);
    SetIndex(n + 2);
    SetIndex(n + 3);
}

void GLMesh::AddBlendRect(const AABB2& rect, const AABB2& texcoordSrc, const AABB2& texcoordDst)
{
    int n = (int)mVertexCount;
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    if (attrPos)
    {
        SetFloat2(GLAttributeTypePosition, n + 0, rect.xMin, rect.yMin);
        SetFloat2(GLAttributeTypePosition, n + 1, rect.xMax, rect.yMin);
        SetFloat2(GLAttributeTypePosition, n + 2, rect.xMax, rect.yMax);
        SetFloat2(GLAttributeTypePosition, n + 3, rect.xMin, rect.yMax);
    }

    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    if (attrTex)
    {
        SetFloat4(GLAttributeTypeTexcoord, n + 0, texcoordSrc.xMin, texcoordSrc.yMin, texcoordDst.xMin, texcoordDst.yMin);
        SetFloat4(GLAttributeTypeTexcoord, n + 1, texcoordSrc.xMax, texcoordSrc.yMin, texcoordDst.xMax, texcoordDst.yMin);
        SetFloat4(GLAttributeTypeTexcoord, n + 2, texcoordSrc.xMax, texcoordSrc.yMax, texcoordDst.xMax, texcoordDst.yMax);
        SetFloat4(GLAttributeTypeTexcoord, n + 3, texcoordSrc.xMin, texcoordSrc.yMax, texcoordDst.xMin, texcoordDst.yMax);
    }

    mVertexCount += 4;

    SetIndex(n + 0);
    SetIndex(n + 1);
    SetIndex(n + 2);
    SetIndex(n + 0);
    SetIndex(n + 2);
    SetIndex(n + 3);
}

void GLMesh::AddCircle(const Vector2& center, float radius, const Color& c0, const Color& c1)
{
    if (radius <= 0.0f)
    {
        return;
    }

    unsigned short n = (unsigned short)mVertexCount;

    const float d = 0.2f;
    // angle = PI * 2 / samples
    // d = r - r * cosf(angle/2)
    int samples = (int)ceilf(PI / acosf((radius - d) / radius));
    if (samples < 3)
    {
        samples = 3;
    }

    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];

    if (attrPos)
    {
        SetFloat2(GLAttributeTypePosition, n + 0, center.x, center.y);
    }
    
    if (attrTex)
    {
        SetFloat2(GLAttributeTypeTexcoord, n + 0, 0.0f, 0.0f);
    }
    
    if (attrColor)
    {
        SetFloat4(GLAttributeTypeColor, n + 0, c0.r, c0.g, c0.b, c0.a);
    }

    for (int i = 0; i < samples; ++i)
    {
        float angle = i * PI * 2 / samples;

        if (attrPos)
        {
            SetFloat2(GLAttributeTypePosition, n + 1 + i, center.x + radius * cosf(angle), center.y + radius * sinf(angle));
        }

        if (attrTex)
        {
            SetFloat2(GLAttributeTypeTexcoord, n + 1 + i, 0.0f, 0.0f);
        }

        if (attrColor)
        {
            SetFloat4(GLAttributeTypeColor, n + 1 + i, c1.r, c1.g, c1.b, c1.a);
        }

        SetIndex(n + 0);
        SetIndex(n + 1 + i);
        SetIndex(n + 1 + ((i + 1) % samples));
    }
    mVertexCount += samples + 1;
}

void GLMesh::AddLineV(const Vector3& v0, const Vector3& v1, const Color& color)
{
    float x0 = v0.x;
    float y0 = v0.y;
    float w0 = v0.z;
    float x1 = v1.x;
    float y1 = v1.y;
    float w1 = v1.z;

    unsigned short n = (unsigned short)mVertexCount;
    float vx = x1 - x0;
    float vy = y1 - y0;
    float upx = -vy;
    float upy = vx;
    float len = sqrt(upx * upx + upy * upy);
    upx /= len;
    upy /= len;

    float r0 = w0 * 0.5f;
    float r1 = w1 * 0.5f;

    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];

    if (attrPos)
    {
        SetFloat2(GLAttributeTypePosition, n + 0, x0 - upx * r0, y0 - upy * r0);
        SetFloat2(GLAttributeTypePosition, n + 1, x1 - upx * r1, y1 - upy * r1);
        SetFloat2(GLAttributeTypePosition, n + 2, x1 + upx * r1, y1 + upy * r1);
        SetFloat2(GLAttributeTypePosition, n + 3, x0 + upx * r0, y0 + upy * r0);
    }

    if (attrTex)
    {
        SetFloat2(GLAttributeTypeTexcoord, n + 0, 0.0f, 0.0f);
        SetFloat2(GLAttributeTypeTexcoord, n + 1, 0.0f, 0.0f);
        SetFloat2(GLAttributeTypeTexcoord, n + 2, 0.0f, 0.0f);
        SetFloat2(GLAttributeTypeTexcoord, n + 3, 0.0f, 0.0f);
    }

    if (attrColor)
    {
        SetFloat4(GLAttributeTypeColor, n + 0, color.r, color.g, color.b, color.a);
        SetFloat4(GLAttributeTypeColor, n + 1, color.r, color.g, color.b, color.a);
        SetFloat4(GLAttributeTypeColor, n + 2, color.r, color.g, color.b, color.a);
        SetFloat4(GLAttributeTypeColor, n + 3, color.r, color.g, color.b, color.a);
    }
    mVertexCount += 4;

    SetIndex(n + 0);
    SetIndex(n + 1);
    SetIndex(n + 2);
    SetIndex(n + 0);
    SetIndex(n + 2);
    SetIndex(n + 3);
}

void GLMesh::AddFreeLine(const Vector3* vertices, int num, const Color& color)
{
    AddCircle(Vector2(vertices[0].x, vertices[0].y), vertices[0].z * 0.5f, color, color);
    for (int i = 1; i < num; ++i)
    {
        const Vector3& v0 = vertices[i - 1];
        const Vector3& v1 = vertices[i];

        AddLineV(v0, v1, color);
        AddCircle(Vector2(v1.x, v1.y), v1.z * 0.5f, color, color);
    }
}

float GLMesh::AddTextureLine(const Vector3* vertices, int num, const Vector2& size, float step, float offset
                              , GLTexture* /*texture*/, const AABB2& texcoord, const Color& color)
{
    std::vector<float> samples;
    offset = GetPolylineSamples((const float*)vertices, num, 3, step, offset, samples);

    int n = (int)samples.size() / 3;
    AABB2 rc;
    Vector2 r;
    for (int i = 0; i < n; ++i)
    {
        float x = samples[i * 3];
        float y = samples[i * 3 + 1];
        r = size * (samples[i * 3 + 2] * 0.5f);

        rc.xMin = x - r.x;
        rc.yMin = y - r.y;
        rc.xMax = x + r.x;
        rc.yMax = y + r.y;

        AddRect(rc, texcoord, color);
    }
    return offset;
}

void GLMesh::AddPolygon(const Vector2* vertices, int num, const Color& color)
{
#ifdef __glu_h__
    TriangulatePolygonList pl;
    pl.positions = vertices;
    pl.count = num;
    pl.texcoords = NULL;
    std::vector<float> vs;
    std::vector<unsigned short> is;
    Triangulation(&pl, 1, vs, is);
    if (is.size() == 0)
    {
        return;
    }

    unsigned short n = (unsigned short)mVertexCount;
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
    
    for (size_t i = 0; i < vs.size() / 2; ++i)
    {
        float x = vs[i * 2];
        float y = vs[i * 2 + 1];

        if (attrPos)
        {
            SetFloat2(GLAttributeTypePosition, n + i, x, y);
        }

        if (attrTex)
        {
            SetFloat2(GLAttributeTypeTexcoord, n + i, 0.0f, 0.0f);
        }

        if (attrColor)
        {
            SetFloat4(GLAttributeTypeColor, n + i, color.r, color.g, color.b, color.a);
        }
    }
    mVertexCount += vs.size() / 2;
    for (size_t i = 0; i < is.size(); ++i)
    {
        SetIndex(n + is[i]);
    }

#else

#ifndef NO_TESS2
    std::vector<float> vs;
    std::vector<unsigned short> is;
    ToTri((float*)vertices, num, vs, is);
    if (vs.size() == 0)
    {
        return;
    }

    unsigned short n = (unsigned short)mVertexCount;
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];

    for (size_t i = 0; i < vs.size() / 2; ++i)
    {
        float x = vs[i * 2];
        float y = vs[i * 2 + 1];

        if (attrPos)
        {
            SetFloat2(GLAttributeTypePosition, n + i, x, y);
        }

        if (attrTex)
        {
            SetFloat2(GLAttributeTypeTexcoord, n + i, 0.0f, 0.0f);
        }

        if (attrColor)
        {
            SetFloat4(GLAttributeTypeColor, n + i, color.r, color.g, color.b, color.a);
        }
    }
    mVertexCount += vs.size() / 2;
    for (size_t i = 0; i < is.size(); ++i)
    {
        SetIndex(n + is[i]);
    }
#endif
#endif
}

void GLMesh::AddCube(const Vector3& c, const Vector3& s, const Color& color)
{
    unsigned short n = (unsigned short)mVertexCount;
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    if (attrPos)
    {
        Vector3 hs = s * 0.5f;
        int i = n;
        // -x
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y - hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y - hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y + hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y + hs.y, c.z - hs.z);
        // +x
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y - hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y - hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y + hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y + hs.y, c.z + hs.z);
        // -y
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y - hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y - hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y - hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y - hs.y, c.z - hs.z);
        // +y
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y + hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y + hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y + hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y + hs.y, c.z - hs.z);
        // -z
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y - hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y - hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y + hs.y, c.z - hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y + hs.y, c.z - hs.z);
        // +z
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y - hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y - hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x + hs.x, c.y + hs.y, c.z + hs.z);
        SetFloat3(GLAttributeTypePosition, i++, c.x - hs.x, c.y + hs.y, c.z + hs.z);
    }

    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    if (attrTex)
    {
        int i = n;
        for (int j = 0; j < 6; ++j)
        {
            SetFloat2(GLAttributeTypeTexcoord, i++, 0, 0);
            SetFloat2(GLAttributeTypeTexcoord, i++, 1, 0);
            SetFloat2(GLAttributeTypeTexcoord, i++, 1, 1);
            SetFloat2(GLAttributeTypeTexcoord, i++, 0, 1);
        }
    }

    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
    if (attrColor)
    {
        for (int i = 0; i < 24; ++i)
        {
            SetFloat4(GLAttributeTypeColor, n + i, color.r, color.g, color.b, color.a);
        }
    }

    GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];
    if (attrNormal)
    {
        int i = n;
        //-x
        SetFloat3(GLAttributeTypeNormal, i++, -1, 0, 0);
        SetFloat3(GLAttributeTypeNormal, i++, -1, 0, 0);
        SetFloat3(GLAttributeTypeNormal, i++, -1, 0, 0);
        SetFloat3(GLAttributeTypeNormal, i++, -1, 0, 0);
        //+x
        SetFloat3(GLAttributeTypeNormal, i++, 1, 0, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 1, 0, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 1, 0, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 1, 0, 0);
        //-y
        SetFloat3(GLAttributeTypeNormal, i++, 0, -1, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 0, -1, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 0, -1, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 0, -1, 0);
        //+y
        SetFloat3(GLAttributeTypeNormal, i++, 0, 1, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 1, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 1, 0);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 1, 0);
        //-z
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, -1);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, -1);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, -1);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, -1);
        //+z
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, 1);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, 1);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, 1);
        SetFloat3(GLAttributeTypeNormal, i++, 0, 0, 1);
    }

    mVertexCount += 24;

    int i = n;
    for (int j = 0; j < 6; ++j)
    {
        SetIndex(i + 0);
        SetIndex(i + 1);
        SetIndex(i + 2);
        SetIndex(i + 0);
        SetIndex(i + 2);
        SetIndex(i + 3);
        i += 4;
    }
}

void GLMesh::AddSphere(const Vector3& center, float radius, const Color& color, int xDivide, int yDivide)
{
    unsigned short n = (unsigned short)mVertexCount;
    int v = n;

    for (int j = 0; j <= yDivide; ++j)
    {
        float b = PI * (1.0f * j / yDivide - 0.5f);
        float y = radius * (float)sin(b) + center.y;
        float r = radius * (float)cos(b);
        for (int i = 0; i <= xDivide; ++i)
        {
            float a = PI * 2.0f * i / xDivide;
            float x = r * (float)cos(a) + center.x;
            float z = -r * (float)sin(a) + center.z;

            GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
            if (attrPos)
            {
                SetFloat3(GLAttributeTypePosition, v, x, y, z);
            }

            GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
            if (attrTex)
            {
                SetFloat2(GLAttributeTypeTexcoord, v, i / (float)xDivide, j / (float)yDivide);
            }

            GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
            if (attrColor)
            {
                SetFloat4(GLAttributeTypeColor, v, color.r, color.g, color.b, color.a);
            }

            GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];
            if (attrNormal)
            {
                Vector3 normal(x, y, z);
                normal -= center;
                normal.Normalise();
                SetFloat3(GLAttributeTypeNormal, v, normal.x, normal.y, normal.z);
            }

            if (i < xDivide && j < yDivide)
            {
                int i0 = v;
                int i1 = v + 1;
                int i2 = v + xDivide + 2;
                int i3 = v + xDivide + 1;
                SetIndex(i0);
                SetIndex(i1);
                SetIndex(i2);
                SetIndex(i0);
                SetIndex(i2);
                SetIndex(i3);
            }

            ++v;
            ++mVertexCount;
        }
    }
}

void GLMesh::AddCone(const Vector3& center, float radius, float height, const Color& color, int xDivide, int yDivide)
{
    unsigned short n = (unsigned short)mVertexCount;
    int v = n;

    for (int j = 0; j <= yDivide; ++j)
    {
        float y = height * (j / (float)yDivide) + center.y;
        float r = radius * (yDivide - j) / (float)yDivide;
        for (int i = 0; i <= xDivide; ++i)
        {
            float a = PI * 2.0f * i / xDivide;
            float x = r * (float)cos(a) + center.x;
            float z = -r * (float)sin(a) + center.z;

            GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
            if (attrPos)
            {
                SetFloat3(GLAttributeTypePosition, v, x, y, z);
            }

            GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
            if (attrTex)
            {
                SetFloat2(GLAttributeTypeTexcoord, v, i / (float)xDivide, j / (float)yDivide);
            }

            GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
            if (attrColor)
            {
                SetFloat4(GLAttributeTypeColor, v, color.r, color.g, color.b, color.a);
            }

            GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];
            if (attrNormal)
            {
                Vector3 normal(x, y, z);
                normal -= center;
                normal.Normalise();
                SetFloat3(GLAttributeTypeNormal, v, normal.x, normal.y, normal.z);
            }

            if (i < xDivide && j < yDivide)
            {
                int i0 = v;
                int i1 = v + 1;
                int i2 = v + xDivide + 2;
                int i3 = v + xDivide + 1;
                SetIndex(i0);
                SetIndex(i1);
                SetIndex(i2);
                SetIndex(i0);
                SetIndex(i2);
                SetIndex(i3);
            }

            ++v;
            ++mVertexCount;
        }
    }

    AddCircleCap(center, radius, color, xDivide, false);
}

void GLMesh::AddCylinder(const Vector3& center, float radius, float height, const Color& color, int xDivide, int yDivide)
{
    unsigned short n = (unsigned short)mVertexCount;
    int v = n;

    for (int j = 0; j <= yDivide; ++j)
    {
        float y = height * (j / (float)yDivide) + center.y;
        float r = radius;
        for (int i = 0; i <= xDivide; ++i)
        {
            float a = PI * 2.0f * i / xDivide;
            float x = r * (float)cos(a) + center.x;
            float z = -r * (float)sin(a) + center.z;

            GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
            if (attrPos)
            {
                SetFloat3(GLAttributeTypePosition, v, x, y, z);
            }

            GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
            if (attrTex)
            {
                SetFloat2(GLAttributeTypeTexcoord, v, i / (float)xDivide, j / (float)yDivide);
            }

            GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
            if (attrColor)
            {
                SetFloat4(GLAttributeTypeColor, v, color.r, color.g, color.b, color.a);
            }

            GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];
            if (attrNormal)
            {
                Vector3 normal(x, y, z);
                normal -= center;
                normal.Normalise();
                SetFloat3(GLAttributeTypeNormal, v, normal.x, normal.y, normal.z);
            }

            if (i < xDivide && j < yDivide)
            {
                int i0 = v;
                int i1 = v + 1;
                int i2 = v + xDivide + 2;
                int i3 = v + xDivide + 1;
                SetIndex(i0);
                SetIndex(i1);
                SetIndex(i2);
                SetIndex(i0);
                SetIndex(i2);
                SetIndex(i3);
            }

            ++v;
            ++mVertexCount;
        }
    }
    
    AddCircleCap(Vector3(center.x, center.y + height, center.z), radius, color, xDivide, true);
    AddCircleCap(Vector3(center.x, center.y, center.z), radius, color, xDivide, false);
}

void GLMesh::AddCircleCap(const Vector3& center, float radius, const Color& color, int samples, bool isUp)
{
    unsigned short n = (unsigned short)mVertexCount;
    float ny = isUp ? 1.0f : -1.0f;

    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
    GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];

    if (attrPos)
    {
        SetFloat3(GLAttributeTypePosition, n + 0, center.x, center.y, center.z);
    }

    if (attrTex)
    {
        SetFloat2(GLAttributeTypeTexcoord, n + 0, 0.0f, 0.0f);
    }

    if (attrColor)
    {
        SetFloat4(GLAttributeTypeColor, n + 0, color.r, color.g, color.b, color.a);
    }

    if (attrNormal)
    {
        SetFloat3(GLAttributeTypeNormal, n + 0, 0, ny, 0);
    }

    for (int i = 0; i < samples; ++i)
    {
        float angle = i * PI * 2 / samples;

        if (attrPos)
        {
            SetFloat3(GLAttributeTypePosition, n + 1 + i, center.x + radius * cosf(angle), center.y, center.z + radius * sinf(angle));
        }

        if (attrTex)
        {
            SetFloat2(GLAttributeTypeTexcoord, n + 1 + i, 0.0f, 0.0f);
        }

        if (attrColor)
        {
            SetFloat4(GLAttributeTypeColor, n + 1 + i, color.r, color.g, color.b, color.a);
        }

        if (attrNormal)
        {
            SetFloat3(GLAttributeTypeNormal, n + 1 + i, 0, ny, 0);
        }

        if (isUp)
        {
            SetIndex(n + 0);
            SetIndex(n + 1 + ((i + 1) % samples));
            SetIndex(n + 1 + i);
        }
        else
        {
            SetIndex(n + 0);
            SetIndex(n + 1 + i);
            SetIndex(n + 1 + ((i + 1) % samples));
        }
    }
    mVertexCount += samples + 1;
}

void GLMesh::AddFrameIndicator(const Vector3& center, const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis, float length, int samples)
{
    float axisLength = length * 0.8f;
    float arrowLength = length - axisLength;
    float arrowRadius = arrowLength * 0.25f;
    float axisRadius = arrowRadius * 0.5f;
    Color xColor(1, 0, 0);
    Color yColor(0, 1, 0);
    Color zColor(0, 0, 1);
    
    AddArrow(center, center + xAxis * length, arrowRadius, arrowLength, axisRadius, xColor, samples);
    AddArrow(center, center + yAxis * length, arrowRadius, arrowLength, axisRadius, yColor, samples);
    AddArrow(center, center + zAxis * length, arrowRadius, arrowLength, axisRadius, zColor, samples);
    AddSphere(center, arrowRadius, Color(1, 1, 1), samples, samples);
}

void GLMesh::AddArrow(const Vector3& from, const Vector3& to, float arrowRadius, float arrowLength, float axisRadius, const Color& color, int samples)
{
    Vector3 axis = to - from;
    float length = axis.Normalise();
    if (!(length > 0.0))
    {
        return;
    }
    Vector3 yAxis(0, 1, 0);
    Vector3 arrowFrom = from;
    if (arrowLength > length)
    {
        arrowLength = length;
    }
    float axisLength = length - arrowLength;
    if (axisLength > 0)
    {
        arrowFrom = from + (yAxis * axisLength);
    }
    
    int n = mVertexCount;
    AddCone(arrowFrom, arrowRadius, arrowLength, color, samples, 1);
    if (axisLength > 0.0f)
    {
        AddCylinder(from, axisRadius, axisLength, color, samples, 1);
    }

    
    if (axis != yAxis)
    {
        Vector3 xAxis = axis.Cross(yAxis);
        xAxis.Normalise();
        Vector3 zAxis = xAxis.Cross(axis);
        zAxis.Normalise();
        Matrix4 mat(
            xAxis.x, axis.x, zAxis.x, 0.0f,
            xAxis.y, axis.y, zAxis.y, 0.0f,
            xAxis.z, axis.z, zAxis.z, 0.0f,
               0.0f,   0.0f,    0.0f, 1.0f
            );
        mat = Matrix4::BuildTranslate(from.x, from.y, from.z) * mat * Matrix4::BuildTranslate(-from.x, -from.y, -from.z);
        ApplyTransform(mat, n, mVertexCount - 1);
    }
}

void GLMesh::AddGrid(const Vector3& center, const Vector3& xDir, const Vector3& yDir, const Vector2& size, const Color& color, int xSegments, int ySegments)
{
    unsigned short n = (unsigned short)mVertexCount;
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTex = mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrColor = mAttributes[GLAttributeTypeColor];
    GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];

    Vector2 halfSize = size * 0.5f;
    Vector3 xBegin = xDir * (halfSize.x * -0.5f);
    Vector3 xEnd = -xBegin;
    for (int i = 0; i <= ySegments; ++i)
    {
        Vector3 yOffset = yDir * (halfSize.y * (-0.5f + i / (float)ySegments));
        Vector3 p0 = center + xBegin + yOffset;
        Vector3 p1 = center + xEnd + yOffset;

        if (attrPos)
        {
            SetVector3(GLAttributeTypePosition, n, p0);
            SetVector3(GLAttributeTypePosition, n + 1, p1);
        }
        if (attrTex)
        {
            SetFloat2(GLAttributeTypeTexcoord, n, 0.0f, 0.0f);
            SetFloat2(GLAttributeTypeTexcoord, n + 1, 0.0f, 0.0f);
        }

        if (attrColor)
        {
            SetFloat4(GLAttributeTypeColor, n, color.r, color.g, color.b, color.a);
            SetFloat4(GLAttributeTypeColor, n + 1, color.r, color.g, color.b, color.a);
        }

        if (attrNormal)
        {
            SetFloat3(GLAttributeTypeNormal, n, 0, 0, 1);
            SetFloat3(GLAttributeTypeNormal, n + 1, 0, 0, 1);
        }

        SetIndex(n + 0);
        SetIndex(n + 1);
        n += 2;
    }
    Vector3 yBegin = yDir * (halfSize.y * -0.5f);
    Vector3 yEnd = -yBegin;
    for (int i = 0; i <= xSegments; ++i)
    {
        Vector3 xOffset = xDir * (halfSize.x * (-0.5f + i / (float)xSegments));
        Vector3 p0 = center + yBegin + xOffset;
        Vector3 p1 = center + yEnd + xOffset;

        if (attrPos)
        {
            SetVector3(GLAttributeTypePosition, n, p0);
            SetVector3(GLAttributeTypePosition, n + 1, p1);
        }
        if (attrTex)
        {
            SetFloat2(GLAttributeTypeTexcoord, n, 0.0f, 0.0f);
            SetFloat2(GLAttributeTypeTexcoord, n + 1, 0.0f, 0.0f);
        }

        if (attrColor)
        {
            SetFloat4(GLAttributeTypeColor, n, color.r, color.g, color.b, color.a);
            SetFloat4(GLAttributeTypeColor, n + 1, color.r, color.g, color.b, color.a);
        }

        if (attrNormal)
        {
            SetFloat3(GLAttributeTypeNormal, n, 0, 0, 1);
            SetFloat3(GLAttributeTypeNormal, n + 1, 0, 0, 1);
        }

        SetIndex(n + 0);
        SetIndex(n + 1);
        n += 2;
    }

    mVertexCount += (xSegments + 1 + ySegments + 1) * 2;
}

void GLMesh::ApplyTransform(const Matrix4& mat, int vertexFrom, int vertexTo)
{
    GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
    GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];
    Matrix4 nm = mat;
    nm.m03 = nm.m13 = nm.m23 = 0.0f;

    for (int i = vertexFrom; i <= vertexTo; ++i)
    {
        if (attrPos)
        {
            float* v = (float*)this->GetVertex(i);
            v += attrPos->mOffset / sizeof(float);

            Vector3 p(v[0], v[1], v[2]);
            p = mat * p;
            SetVector3(GLAttributeTypePosition, i, p);
        }

        if (attrNormal)
        {
            float* v = (float*)this->GetVertex(i);
            v += attrNormal->mOffset / sizeof(float);

            Vector3 p(v[0], v[1], v[2]);
            p = nm * p;
            SetVector3(GLAttributeTypeNormal, i, p);
        }
    }
}

static bool ParseObjFile(const char* path, std::vector<Vector3>& position, std::vector<Vector2>& texcord, std::vector<Vector3>& normal, std::vector<Vector3i>& faces)
{
    int size = 0;
    char* buf = LoadFile(path, size);
    if (!buf)
    {
        return false;
    }

    int i = 0;
    char c = buf[i++];
    while (c)
    {
        if (c == 'v')
        {
            if (i + 1 < size && buf[i + 1] == 'n')
            {
                i += 2;
                // vn
                float x = 0;
                float y = 0;
                float z = 0;
                sscanf(buf + i, " %f %f %f", &x, &y, &z);
                normal.push_back(Vector3(x, y, z));
            }
            else if (i + 1 < size && buf[i + 1] == 't')
            {
                i += 2;
                // vt
                float x = 0;
                float y = 0;
                sscanf(buf + i, " %f %f", &x, &y);
                texcord.push_back(Vector2(x, y));
            }
            else
            {
                ++i;
                // v
                float x = 0;
                float y = 0;
                float z = 0;
                sscanf(buf + i, " %f %f %f", &x, &y, &z);
                position.push_back(Vector3(x, y, z));
            }
        }
        else if (c == 'f')
        {
            // f
            ++i;
            for (int idx = 0; idx < 3; ++idx)
            {
                int x = 0;
                int y = 0;
                int z = 0;
                ++i;
                sscanf(buf + i, " %d", &x);

                int j = i;
                int slashNum = 0;
                while (j < size && buf[j] != ' ' && buf[j] != '\r' && buf[j] != '\n')
                {
                    if (buf[j] == '/')
                    {
                        if (j + 1 < size)
                        {
                            if (slashNum == 0)
                            {
                                if (buf[j + 1] == '/')
                                {
                                    ++j;
                                }
                                else if (buf[j + 1] >= '0' && buf[j + 1] <= '9')
                                {
                                    sscanf(buf + j + 1, "%d", &y);
                                }
                            }
                            else if (slashNum == 1)
                            {
                                sscanf(buf + j + 1, "%d", &z);
                            }
                        }
                        ++slashNum;
                    }
                    ++j;
                }
                faces.push_back(Vector3i(x - 1, y - 1, z - 1));
                while (buf[j] != ' ' && buf[j] != '\r' && buf[j] != '\n')
                {
                    ++j;
                }
                i = j;
            }
        }
        else
        {
            // not supported, skip to next line
        }

        while (i < size && buf[i] != '\n')
        {
            ++i;
        }
        c = buf[++i];
    }
    delete[] buf;
    return true;
}

GLMesh* GLMesh::LoadObj(const char* path)
{
    std::vector<Vector3> position;
    std::vector<Vector2> texcord;
    std::vector<Vector3> normal;
    std::vector<Vector3i> faces;

    if (!ParseObjFile(path, position, texcord, normal, faces))
    {
        return NULL;
    }

    std::vector<AttributeDesc> attrs;
    if (position.size() > 0)
    {
        attrs.push_back(AttributeDesc(GLAttributeTypePosition, 3));
    }
    if (texcord.size() > 0)
    {
        attrs.push_back(AttributeDesc(GLAttributeTypeTexcoord, 2));
    }
    if (normal.size() > 0)
    {
        attrs.push_back(AttributeDesc(GLAttributeTypeNormal, 3));
    }
    attrs.push_back(AttributeDesc(GLAttributeTypeColor, 4));
    GLMesh* mesh = new GLMesh(faces.size(), faces.size(), &attrs.front(), attrs.size());

    GLAttribute* attrPos = mesh->mAttributes[GLAttributeTypePosition];
    GLAttribute* attrTexcoord = mesh->mAttributes[GLAttributeTypeTexcoord];
    GLAttribute* attrNormal = mesh->mAttributes[GLAttributeTypeNormal];
    GLAttribute* attrColor = mesh->mAttributes[GLAttributeTypeColor];
    int v = mesh->mVertexCount;

    for (size_t i = 0; i < faces.size(); ++i)
    {
        const Vector3i& idx = faces[i];
        if (attrPos && position.size() > 0 && idx.x >= 0)
        {
            const Vector3& p = position[idx.x];
            mesh->SetVector3(GLAttributeTypePosition, v, p);
        }

        if (attrTexcoord && texcord.size() > 0 && idx.y >= 0)
        {
            const Vector2& p = texcord[idx.y];
            mesh->SetVector2(GLAttributeTypeTexcoord, v, p);
        }
        
        if (attrColor)
        {
            mesh->SetFloat4(GLAttributeTypeColor, v, 1.0f, 1.0f, 1.0f, 1.0f);
        }

        if (attrNormal && normal.size() > 0 && idx.z >= 0)
        {
            const Vector3& p = normal[idx.z];
            mesh->SetVector3(GLAttributeTypeNormal, v, p);
        }

        mesh->SetIndex(v);
        ++v;
    }
    mesh->mVertexCount += v;
    mesh->BuildVBO();
    return mesh;
}

bool GLMesh::SaveObj(const char* path)
{
	std::ofstream out(path);
	if (!out)
	{
		return false;
	}
	GLAttribute* attrPos = mAttributes[GLAttributeTypePosition];
	if (attrPos)
	{
		int numComps = attrPos->GetComponent();
		
		for (int i = 0; i < mVertexCount; ++i)
		{
			float* p = (float*)this->GetVertex(i);
			p += attrPos->mOffset / sizeof(float);
			out << "v " << p[0] << " " << p[1] << " ";
			if (numComps > 2)
			{
				out << p[2] << " ";
			}
			out << "\n";
		}
	}

    GLAttribute* attrTexcoord = mAttributes[GLAttributeTypeTexcoord];
    if (attrTexcoord)
    {
        for (int i = 0; i < mVertexCount; ++i)
        {
            float* p = (float*)this->GetVertex(i);
            p += attrTexcoord->mOffset / sizeof(float);
            out << "vt " << p[0] << " " << p[1] << "\n";
        }
    }

	GLAttribute* attrNormal = mAttributes[GLAttributeTypeNormal];
	if (attrNormal)
	{
		for (int i = 0; i < mVertexCount; ++i)
		{
			float* p = (float*)this->GetVertex(i);
			p += attrNormal->mOffset / sizeof(float);
			out << "vn " << p[0] << " " << p[1] << " " << p[2] << "\n";
		}
	}

	for (int i = 0; i < mIndexCount / 3; ++i)
	{
        out << "f";
        for (int j = 0; j < 3; ++j)
        {
            int idx = *GetIndex(i * 3 + j) + 1;
            if (attrNormal)
            {
                if (attrTexcoord)
                {
                    // v/vt/vn
                    out << " " << idx << "/" << idx << "/" << idx;
                }
                else
                {
                    // v//vn
                    out << " " << idx << "//" << idx;
                }
            }
            else
            {
                if (attrTexcoord)
                {
                    // v/vt
                    out << " " << idx << "/" << idx;
                }
                else
                {
                    // v/vt
                    out << " " << idx;
                }
            }
        }
        out << "\n";
	}
	
	out.close();
	return true;
}

GLRenderer* GLRenderer::sInstance = NULL;

GLRenderer::GLRenderer()
{
#if defined(USE_OPENGL3)
    const char* vsDefault =
        "#version 330\n"
        "in vec3 position;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* psDefault =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   FragColor = vec4(1,1,1,1);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vsDefault =
        "#version 300 es\n"
        "in vec3 position;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* psDefault =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   FragColor = vec4(1,1,1,1);\n"
        "}\n";
#else
    const char* vsDefault =
        "attribute vec3 position;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* psDefault =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "void main() {\n"
        "   gl_FragColor = vec4(1,1,1,1);\n"
        "}\n";
#endif


#ifdef GLEWAPI
	glewInit();
#endif
#ifdef __gl3w_h_
    gl3wInit();
#endif

    LOGE("Version %s\n", glGetString(GL_VERSION));
    LOGE("Vendor %s\n", glGetString(GL_VENDOR));
    LOGE("Renderer %s\n", glGetString(GL_RENDERER));
    LOGE("GLSL Version %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
//    LOGE("Extensions\n");
//    LOG_LONG((const char*)glGetString(GL_EXTENSIONS));
//    LOGE("\n");

    int data = 0xFFFFFFFF;
    mDefaultTexture = new GLTexture(1, 1, &data, 4, false, false, false, false, false, false, 0);
    mDefaultTarget = new GLRenderTarget(0);
    int maxSize = 4096;
    GLint maxTextureSize = maxSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    if ( maxTextureSize < maxSize )
    {
        maxSize = maxTextureSize;
    }
    mTempTarget = new GLRenderTarget(maxSize, maxSize, false, false, NULL, false, false, 4, 0);

    std::map<std::string, int> attrLocs;
    attrLocs["position"] = GLMesh::GLAttributeTypePosition;
    mDefaultProgram = new GLShaderProgram(vsDefault, psDefault, attrLocs, __LINE__, __FUNCTION__);
    mPrograms["default"] = mDefaultProgram;

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
#ifndef __APPLE__
    glClearDepthf(1.0f);
#endif
	glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
#ifdef GL_MULTISAMPLE
    glEnable(GL_MULTISAMPLE);
#endif
    sInstance = this;

    GLMesh::AttributeDesc attr[] = {
    		GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 2),
    		GLMesh::AttributeDesc(GLMesh::GLAttributeTypeTexcoord, 2)
    };
    mBlitMesh = new GLMesh(4, 6, attr, 2);
    mBlitMesh->AddRect(AABB2(0, 0, 0, 0), AABB2(0, 0, 1, 1), Color(1, 1, 1, 1));
    mBlitEffect = new BlitEffect(this);

    mShapeEffect = new ShapeEffect(this);
    mMixGreyscaleEffect = new MixGreyscaleEffect(this);
    mColorOverrideEffect = new ColorOverrideEffect(this);
#if defined(USE_OPENGL3)
    mFxaaEffect = new FXAAEffect(this);
    mSmaaEffect = new SmaaEffect(GL_RENDERER_SHADER_DIR);
    mDetectHoleEffect = new DetectHoleEffect(this);
    mFillHoleEffect = new FillHoleEffect(this);
#else
    mFxaaEffect = NULL;
    mSmaaEffect = NULL;
    mDetectHoleEffect = NULL;
    mFillHoleEffect = NULL;
#endif
    
    const char* mixStr =		
        "   src *= shapeColor;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * (src.a / a) + dst.rgb * (dsta / a), a);\n";
    const char* behindStr =
        "   src *= shapeColor;\n"
        "   vec4 tmp = src;\n"
        "   src = dst;\n"
        "   dst = tmp;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * src.a / a + dst.rgb * dsta / a, a);\n";
    const char* eraseStr =
        "   float dsta = max(0.0, dst.a - src.a);\n"
        "   FragColor = vec4(dst.rgb, dsta);\n";
    const char* invertStr =
        "   src.rgb = vec3(1.0,1.0,1.0) - src.rgb;\n"
        "   src *= shapeColor;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * (src.a / a) + dst.rgb * (dsta / a), a);\n";

    const char* maskStr =
        "   float a = src.a * shapeColor.a;\n"
        "   FragColor = vec4(dst.rgb, dst.a * a);\n";

    const char* cutStr =            
        "   float a = src.a * shapeColor.a;\n"
        "   FragColor = vec4(dst.rgb, dst.a * (1.0 - a));\n";

    const char* maskMixStr =
        "   src.a = src.r * shapeColor.a;\n"
        "   src.rgb = shapeColor.rgb;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4((src.rgb * (src.a / a)) + (dst.rgb * (dsta / a)), a);\n";

    const char* maskBehindStr =
        "   src.a = src.r * shapeColor.a;\n"
        "   src.rgb = shapeColor.rgb;\n"
        "   vec4 tmp = src;\n"
        "   src = dst;\n"
        "   dst = tmp;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * src.a / a + dst.rgb * dsta / a, a);\n";

    const char* maskEraseStr =
        "   src.a = src.r * shapeColor.a;\n"
        "   float dsta = max(0.0, dst.a - max(0.0,src.a));\n"
        "   FragColor = vec4(dst.rgb, dsta);\n";

    const char* maskAddStr =
        "   src.a = src.r * shapeColor.a;\n"
        "   src.rgb = min(dst.rgb + shapeColor.rgb, vec3(1.0,1.0,1.0));\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * src.a / a + dst.rgb * dsta / a, a);\n";

    const char* maskMultiplyStr =
        "   src.a = src.r * shapeColor.a;\n"
        "   src.rgb = dst.rgb * src.rgb * shapeColor.rgb;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * src.a / a + dst.rgb * dsta / a, a);\n";

    const char* maskInverseMixStr =
        "   src.a = (1.0 - src.r) * shapeColor.a;\n"
        "   src.rgb = shapeColor.rgb;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * src.a / a + dst.rgb * dsta / a, a);\n";

    const char* maskMaskStr =
        "   float a = src.r * shapeColor.a;\n"
        "   FragColor = vec4(dst.rgb, dst.a * a);\n";

    const char* maskCutStr =
        "   float a = src.r * shapeColor.a;\n"
        "   FragColor = vec4(dst.rgb, dst.a * (1.0 - a));\n";

    const char* replaceAlhpaStr =
        "   float a = src.a * shapeColor.a;\n"
        "   FragColor = vec4(dst.rgb, a);\n";


    mMixEffect = new BlendEffect(this, mixStr);
    mBehindEffect = new BlendEffect(this, behindStr);
    mEraseEffect = new BlendEffect(this, eraseStr);
    mInvertEffect = new BlendEffect(this, invertStr);
    mMaskEffect = new BlendEffect(this, maskStr);
    mCutEffect = new BlendEffect(this, cutStr);

    mMaskBlurEffect = new MaskBlurEffect(this, "");

    mMaskMixEffect = new BlendEffect(this, maskMixStr);
    mMaskBehindEffect = new BlendEffect(this, maskBehindStr);
    mMaskEraseEffect = new BlendEffect(this, maskEraseStr);    
    mMaskAddEffect = new BlendEffect(this, maskAddStr);
    mMaskMultiplyEffect = new BlendEffect(this, maskMultiplyStr);
    mMaskInverseMixEffect = new BlendEffect(this, maskInverseMixStr);
    mMaskMaskEffect = new BlendEffect(this, maskMaskStr);
    mMaskCutEffect = new BlendEffect(this, maskCutStr);

    mReplaceAlhpaEffect = new BlendEffect(this, replaceAlhpaStr);

    mMaskExpandMixEffect = new MaskExpandEffect(this, maskMixStr);
    mMaskExpandBehindEffect = new MaskExpandEffect(this, maskBehindStr);
    mMaskExpandEraseEffect = new MaskExpandEffect(this, maskEraseStr);
    mMaskExpandAddEffect = new MaskExpandEffect(this, maskAddStr);
    mMaskExpandMultiplyEffect = new MaskExpandEffect(this, maskMultiplyStr);

    mSoftenMaskMixEffect = new SoftenMaskEffect(this, maskMixStr);
    mSoftenMaskBehindEffect = new SoftenMaskEffect(this, maskBehindStr);
    mSoftenMaskEraseEffect = new SoftenMaskEffect(this, maskEraseStr);
    mSoftenMaskAddEffect = new SoftenMaskEffect(this, maskAddStr);
    mSoftenMaskMultiplyEffect = new SoftenMaskEffect(this, maskMultiplyStr);

	mPaletteEffect = new PaletteEffect(this);
    mPaletteMixEffect = new PaletteMixEffect(this);
    mPaletteAddEffect = new PaletteAddEffect(this);



    mCanvas2d = new Canvas2d(this);
    mCanvas3d = new Canvas3d(this);
    int fontAtlasSize = 1024;
    mFont = new Font(24, 1024);
    mFontTexture = new GLTexture(fontAtlasSize, fontAtlasSize, NULL, 4, true, true, false, false, false, false, 0);
}

GLRenderer::~GLRenderer()
{
    delete mCanvas2d;
    delete mCanvas3d;
    delete mFont;
    delete mFontTexture;

    delete mDetectHoleEffect;
    delete mFillHoleEffect;
	delete mPaletteEffect;
    delete mPaletteMixEffect;
    delete mPaletteAddEffect;
    delete mSoftenMaskMixEffect;
    delete mSoftenMaskBehindEffect;
    delete mSoftenMaskEraseEffect;
    delete mSoftenMaskAddEffect;
    delete mSoftenMaskMultiplyEffect;
    delete mMaskExpandMixEffect;
    delete mMaskExpandBehindEffect;
    delete mMaskExpandEraseEffect;
    delete mMaskExpandAddEffect;
    delete mMaskExpandMultiplyEffect;
    delete mMixEffect;
    delete mMaskMixEffect;
    delete mBehindEffect;
    delete mInvertEffect;
    delete mCutEffect;
    delete mMaskBehindEffect;
    delete mEraseEffect;
    delete mMaskEraseEffect;
    delete mMaskBlurEffect;
    delete mMaskAddEffect;
    delete mMaskMultiplyEffect;
    delete mMaskInverseMixEffect;
    delete mMaskMaskEffect;
    delete mMaskCutEffect;
    delete mReplaceAlhpaEffect;
    delete mColorOverrideEffect;
    delete mMixGreyscaleEffect;
    delete mShapeEffect;
#ifdef __glu_h__
    delete mFxaaEffect;
    delete mSmaaEffect;
#endif
    delete mBlitEffect;
    delete mBlitMesh;

	for (ProgramCollection::iterator it = mPrograms.begin(); it != mPrograms.end(); ++it)
	{
		delete it->second;
	}
    delete mTempTarget;
    delete mDefaultTarget;
    sInstance = NULL;
}

void GLRenderer::Reset()
{
#ifdef USE_VAO
    glBindVertexArray(0);
#endif
    for (int i = 0; i < 8; ++i)
    {
        glDisableVertexAttribArray(i);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

GLTexture* GLRenderer::CreateTexture(const char* path)
{
    int w = 0;
    int h = 0;
    int pixelSize = 0;
    unsigned char* data = LoadImg(path, w, h, pixelSize);

    GLTexture* result = NULL;
    if (data)
    {
        //switch (pixelSize)
        //{
        //case 4:
        //    result = new GLTexture(w, h, data, 4, true, true, false, true, false, false, 0);
        //    break;
        //case 3:
        //    result = new GLTexture(w, h, data, 3, true, true, false, true, false, false, 0);
        //    break;
        //case 1:
        //    result = new GLTexture(w, h, data, 1, true, true, false, true, false, false, 0);
        //    break;
        //default:
        //    break;
        //}
        int s = w * 4;
        unsigned char* row = new unsigned char[s];
        for (int y = 0; y < h / 2; ++y)
        {
            memcpy(row, data + y * s, s);
            memcpy(data + y * s, data + (h - 1 - y) * s, s);
            memcpy(data + (h - 1 - y) * s, row, s);
        }
        result = new GLTexture(w, h, data, 4, true, true, false, true, false, false, 0);
        ReleaseImg(data);
        delete[] row;
    }
    return result;

}

GLTexture* GLRenderer::CreateTexture(int width, int height, void* data, int channels, bool minSmooth, bool magSmooth, bool repeat, bool useMipmap)
{
    return new GLTexture(width, height, data, channels, minSmooth, magSmooth, repeat, useMipmap, false, false, 0);
}

GLRenderTarget* GLRenderer::CreateTarget(int width, int height, bool hasDepth, bool enableAA, void* data, bool minSmooth, bool magSmooth, int channels)
{
    return new GLRenderTarget(width, height, hasDepth, enableAA, data, minSmooth, magSmooth, channels, 1);
}

GLRenderTarget* GLRenderer::CreateTarget(int width, int height, bool hasDepth, bool enableAA)
{
    return new GLRenderTarget(width, height, hasDepth, enableAA, NULL, false, false, 4, 1);
}

GLRenderTarget* GLRenderer::CreateTarget(int width, int height, void* data)
{
    return new GLRenderTarget(width, height, false, false, data, false, false, 4, 1);
}

GLRenderTarget* GLRenderer::CreateGBuffer(int width, int height, int numTextures)
{
    return new GLRenderTarget(width, height, true, false, 0, false, false, 4, numTextures);
}

GLRenderTarget* GLRenderer::CreateTarget(const char* path)
{
    int w = 0;
    int h = 0;
    int pixelSize = 0;
    unsigned char* data = LoadImg(path, w, h, pixelSize);

    GLRenderTarget* result = NULL;
    if (data)
    {
        int s = w * 4;
        unsigned char* row = new unsigned char[s];
        for (int y = 0; y < h / 2; ++y)
        {
            memcpy(row, data + y * s, s);
            memcpy(data + y * s, data + (h - 1 - y) * s, s);
            memcpy(data + (h - 1 - y) * s, row, s);
        }

        result = new GLRenderTarget(w, h, false, false, data, false, false, 4, 1);
        ReleaseImg(data);
        delete[] row;
    }
    return result;
}

GLRenderTarget* GLRenderer::CreateTarget(int fboId)
{
    return new GLRenderTarget(fboId);
}

GLRenderTarget* GLRenderer::SetDefaultTarget(GLRenderTarget* target)
{
    GLRenderTarget* old = mDefaultTarget;
    mDefaultTarget = target;
    return old;
}

GLShaderProgram* GLRenderer::CreateShaderFromFile(const char* vsPath, const char* fsPath, std::map<std::string, int>& attrLocs)
{
    GLShaderProgram* result = NULL;
    char* vs = NULL;
    char* fs = NULL;

    do 
    {
        int size = 0;
        vs = LoadFile(vsPath, size);
        if (!vs)
        {
            break;
        }

        fs = LoadFile(fsPath, size);
        if (!fs)
        {
            break;
        }

        result = new GLShaderProgram(vs, fs, attrLocs, __LINE__, vsPath);
    } while (0);

    delete[] vs;
    delete[] fs;
    return result;
}

GLMesh* GLRenderer::CreateCube(const Vector3& center, const Vector3& size, const Color& color)
{
     GLMesh::AttributeDesc attrs[] = {
         GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 3),
         GLMesh::AttributeDesc(GLMesh::GLAttributeTypeColor, 4),
         GLMesh::AttributeDesc(GLMesh::GLAttributeTypeTexcoord, 2),
         GLMesh::AttributeDesc(GLMesh::GLAttributeTypeNormal, 3)
     };
     GLMesh* cube = new GLMesh(24, 36, attrs, sizeof(attrs) /sizeof(GLMesh::AttributeDesc));
     cube->AddCube(center, size, color);
     cube->BuildVBO();
     return cube;
}

GLMesh* GLRenderer::CreateSphere(const Vector3& center, float radius, const Color& color, int xDivide, int yDivide)
{
    GLMesh::AttributeDesc attrs[] = {
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 3),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeColor, 4),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeTexcoord, 2),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeNormal, 3)
    };
    int numVs = (xDivide + 1) * (yDivide + 1);
    int numIs = xDivide * yDivide * 6;
    GLMesh* mesh = new GLMesh(numVs, numIs, attrs, sizeof(attrs) / sizeof(GLMesh::AttributeDesc));
    mesh->AddSphere(center, radius, color, xDivide, yDivide);
    mesh->BuildVBO();
    return mesh;
}

GLMesh* GLRenderer::CreateFrameIndicator(const Vector3& center, const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis, float length, int samples)
{
    GLMesh::AttributeDesc attrs[] = {
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 3),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeColor, 4),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeTexcoord, 2),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeNormal, 3)
    };
    int coneVs = (samples + 1) * 2;
    int axisVs = coneVs * 2 + (samples + 1) * 3;
    int numVs = axisVs * 3 + (samples + 1) * (samples + 1);
    int axisIs = samples * 6 * 2 + samples * 3 * 3;
    int numIs = axisIs * 3 + samples * samples * 6;
    GLMesh* mesh = new GLMesh(numVs, numIs, attrs, sizeof(attrs) / sizeof(GLMesh::AttributeDesc));
    mesh->AddFrameIndicator(center, xAxis, yAxis, zAxis, length, samples);
    mesh->BuildVBO();
    return mesh;
}

GLMesh* GLRenderer::CreateScreenAlignedQuad()
{
    GLMesh::AttributeDesc attr(GLMesh::GLAttributeTypePosition, 2);
	GLMesh* screenQuad = new GLMesh(4, 6, &attr, 1);
    screenQuad->AddRect(AABB2(-1, -1, 1, 1), AABB2(0, 0, 1, 1), Color(1, 1, 1, 1));
    screenQuad->BuildVBO();
	return screenQuad;
}

GLMesh* GLRenderer::CreateGrid(const Vector3& center, const Vector3& xDir, const Vector3& yDir, const Vector2& size, const Color& color, int xSegments, int ySegments)
{
     GLMesh::AttributeDesc attrs[] = {
         GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 3),
         GLMesh::AttributeDesc(GLMesh::GLAttributeTypeColor, 4)
     };
     int xn = xSegments + 1;
     int yn = ySegments + 1;
     int n = xn * 2 + yn * 2;
     GLMesh* mesh = new GLMesh(n, n, attrs, sizeof(attrs) /sizeof(GLMesh::AttributeDesc), GLMesh::GLPrimitiveTypeLineList);
     mesh->AddGrid(center, xDir, yDir, size, color, xSegments, ySegments);
     mesh->BuildVBO();
     return mesh;
}

AABB2 GLRenderer::GetTextSize(const char* text, float fontHeight)
{
    AABB2 box;
    std::vector<float> vs;
    mFont->RenderText(text, 0, 0, fontHeight, vs);
    for (size_t i = 0; i < vs.size() / 4; ++i)
    {
        float* v = &vs[i * 4];
        box.Union(Vector2(v[0], v[1]));
        box.Union(Vector2(v[2], v[3]));
    }
    return box;
}

GLMesh* GLRenderer::CreateText(const char* text, float xPos, float yPos, float fontHeight, const Color& color)
{
    std::vector<float> vs;
    mFont->RenderText(text, xPos, yPos, fontHeight, vs);

    GLMesh::AttributeDesc attrs[] = {
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 2),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeColor, 4),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeTexcoord, 2)
    };
    GLMesh* mesh = new GLMesh(vs.size() / 4, vs.size() / 4 * 6, attrs, sizeof(attrs) / sizeof(GLMesh::AttributeDesc));
    for (size_t i = 0; i < vs.size() / 4; ++i)
    {
        float* v = &vs[i * 4];

        mesh->SetFloat2(GLMesh::GLAttributeTypePosition, i, v[0], v[1]);
        mesh->SetFloat2(GLMesh::GLAttributeTypeTexcoord, i, v[2], v[3]);
        mesh->SetFloat4(GLMesh::GLAttributeTypeColor, i, color.r, color.g, color.b, color.a);
        if (i % 4 == 0)
        {
            mesh->SetIndex(i + 0);
            mesh->SetIndex(i + 1);
            mesh->SetIndex(i + 2);
            mesh->SetIndex(i + 0);
            mesh->SetIndex(i + 2);
            mesh->SetIndex(i + 3);
        }
    }
    mesh->SetVertexCount(vs.size() / 4);
    mesh->BuildVBO();
    std::vector<Font::AtlasRect> rcs;
    mFont->GetAtlasUpdateRects(rcs);
    if (rcs.size() > 0)
    {
        mFontTexture->WriteTexture(mFont->GetAtlasBitmap(), mFont->GetAtlasSize(), mFont->GetAtlasSize(), false);
    }
    return mesh;
}

BlitEffect::BlitEffect(GLRenderer* /*renderer*/)
{
    CreateDirectBlitShader();
    CreateSmoothBlitShader();
}

BlitEffect::~BlitEffect()
{
    delete mProgramDirect;
    delete mProgramSmooth;
}

void BlitEffect::CreateDirectBlitShader()
{
#if defined(USE_OPENGL3)
    static const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord.xy;\n"
        "   gl_Position = mvp * vec4(position.xy, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = texture(tex0, texcoordOut.xy);"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord.xy;\n"
        "   gl_Position = mvp * vec4(position.xy, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = texture(tex0, texcoordOut.xy);"
        "}\n";
#else
    static const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord.xy;\n"
        "   gl_Position = mvp * vec4(position.xy, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   gl_FragColor = texture2D(tex0, texcoordOut.xy);"
        "}\n";
#endif

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    mProgramDirect = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mMvpLocDirect = mProgramDirect->GetUniformLocation("mvp");
    mTex0LocDirect = mProgramDirect->GetUniformLocation("tex0");
}

void BlitEffect::CreateSmoothBlitShader()
{
#if defined(USE_OPENGL3)
    static const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord.xy;\n"
        "   gl_Position = mvp * vec4(position.xy, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec2 invSize;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   vec4 c = vec4(0.0);\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(invSize.x, 0.0)) * 0.25;\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(-invSize.x, 0.0)) * 0.25;\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(0.0, invSize.y)) * 0.25;\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(0.0, -invSize.y)) * 0.25;\n"
        "   FragColor = c;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord.xy;\n"
        "   gl_Position = mvp * vec4(position.xy, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec2 invSize;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   vec4 c = vec4(0.0);\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(invSize.x, 0.0)) * 0.25;\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(-invSize.x, 0.0)) * 0.25;\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(0.0, invSize.y)) * 0.25;\n"
        "   c += texture(tex0, texcoordOut.xy + vec2(0.0, -invSize.y)) * 0.25;\n"
        "   FragColor = c;\n"
        "}\n";
#else
    static const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord.xy;\n"
        "   gl_Position = mvp * vec4(position.xy, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "uniform vec2 invSize;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   vec4 c = vec4(0.0);\n"
        "   c += texture2D(tex0, texcoordOut.xy + vec2(invSize.x, 0.0)) * 0.25;\n"
        "   c += texture2D(tex0, texcoordOut.xy + vec2(-invSize.x, 0.0)) * 0.25;\n"
        "   c += texture2D(tex0, texcoordOut.xy + vec2(0.0, invSize.y)) * 0.25;\n"
        "   c += texture2D(tex0, texcoordOut.xy + vec2(0.0, -invSize.y)) * 0.25;\n"
        "   gl_FragColor = c;\n"
        "}\n";
#endif


    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    mProgramSmooth = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mMvpLocSmooth = mProgramSmooth->GetUniformLocation("mvp");
    mTex0LocSmooth = mProgramSmooth->GetUniformLocation("tex0");
    mInvSizeLocSmooth = mProgramSmooth->GetUniformLocation("invSize");
}

void BlitEffect::Blit(GLRenderTarget* dst, int dstX, int dstY, int dstW, int dstH, GLRenderTarget* src, int srcX, int srcY, int srcW, int srcH, bool smooth)
{
	if (src->GetTexture() == NULL)
	{
#ifdef HAVE_BLIT
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst->GetId());
		glBindFramebuffer(GL_READ_FRAMEBUFFER, src->GetId());
		glBlitFramebuffer(srcX, srcY, srcX + srcW, srcY + srcH, dstX, dstY, dstX + dstW, dstY + dstH, GL_COLOR_BUFFER_BIT, smooth ? GL_LINEAR : GL_NEAREST);
#endif
		return;
	}

    AABB2 dstRc((float)dstX, (float)dstY, (float)dstX + dstW, (float)dstY + dstH);
    float sw = (float)src->GetWidth();
    float sh = (float)src->GetHeight();
    AABB2 srcRc(srcX/sw, srcY/sh, (srcX + srcW)/sw, (srcY + srcH)/sh);

    GLMesh* mesh = GLRenderer::GetInstance()->GetBlitMesh();
    mesh->SetFloat2(GLMesh::GLAttributeTypePosition, 0, dstRc.xMin, dstRc.yMin);
    mesh->SetFloat2(GLMesh::GLAttributeTypePosition, 1, dstRc.xMax, dstRc.yMin);
    mesh->SetFloat2(GLMesh::GLAttributeTypePosition, 2, dstRc.xMax, dstRc.yMax);
    mesh->SetFloat2(GLMesh::GLAttributeTypePosition, 3, dstRc.xMin, dstRc.yMax);

    mesh->SetFloat2(GLMesh::GLAttributeTypeTexcoord, 0, srcRc.xMin, srcRc.yMin);
    mesh->SetFloat2(GLMesh::GLAttributeTypeTexcoord, 1, srcRc.xMax, srcRc.yMin);
    mesh->SetFloat2(GLMesh::GLAttributeTypeTexcoord, 2, srcRc.xMax, srcRc.yMax);
    mesh->SetFloat2(GLMesh::GLAttributeTypeTexcoord, 3, srcRc.xMin, srcRc.yMax);

    const Matrix4& mvp = Matrix4::BuildOrtho(0, (float)dst->GetWidth(), 0, (float)dst->GetHeight(), -100, 100);
    if (smooth)
    {
        mProgramSmooth->Bind();
        mProgramSmooth->SetMatrix(mMvpLocSmooth, &mvp.m00);
        mProgramSmooth->SetTexture(mTex0LocSmooth, 0, src->GetTexture());
        mProgramSmooth->SetFloat2(mInvSizeLocSmooth, 0.5f / src->GetWidth(), 0.5f / src->GetHeight());
    }
    else
    {
        mProgramDirect->Bind();
        mProgramDirect->SetMatrix(mMvpLocDirect, &mvp.m00);
        mProgramDirect->SetTexture(mTex0LocDirect, 0, src->GetTexture());
    }

    glDisable(GL_BLEND);
    dst->Bind();
    mesh->Draw();
}

SuperSamplingEffect::SuperSamplingEffect(GLRenderer* renderer)
{
    static const char* vsShape =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#define MAX_SAMPLES 64"
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "uniform vec2 pixelSize;\n"
        "uniform vec2 samples[MAX_SAMPLES];\n"
        "uniform int numSamples;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   vec4 c = vec4(0,0,0,0);\n"
        "   for (int i = 0; i < numSamples; ++i) {\n"
        "      c += texture2D(tex0, texcoordOut.xy + pixelSize * samples[i].xy) * samples[i].z;\n"
        "   }\n"
        "   gl_FragColor = c;\n"
        "}\n";

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mTex0Loc = mProgram->GetUniformLocation("tex0");
    mPixelSizeLoc = mProgram->GetUniformLocation("pixelSize");
    mSamplesLoc = mProgram->GetUniformLocation("samples");
    mNumSamplesLoc = mProgram->GetUniformLocation("numSamples");

    mProgram->Bind();
    SetMvp(Matrix4());
    SetTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

SuperSamplingEffect::~SuperSamplingEffect()
{
    delete mProgram;
}

void SuperSamplingEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void SuperSamplingEffect::SetTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTex0Loc, 0, texture);
}

void SuperSamplingEffect::SetPixelSize(float w, float h)
{
    mProgram->SetFloat2(mPixelSizeLoc, w, h);
}

void SuperSamplingEffect::SetSamples(int samples)
{
    if (samples > 8)
    {
        samples = 8;
    }
    float* s = new float[samples * samples * 3];
    float h = samples * 0.5f;
    for (int j = 0; j < samples; ++j)
    {
        for (int i = 0; i < samples; ++i)
        {
            float *p = s + ((j * samples) + i) * 3;
            p[0] = (i + 0.5f - h) / h;
            p[1] = (j + 0.5f - h) / h;
            p[2] = 1.0f / (samples * samples);
        }
    }

    mProgram->SetFloat3Array(mSamplesLoc, s, samples * samples);
    mProgram->SetInt(mNumSamplesLoc, samples * samples);
}

void SuperSamplingEffect::Bind()
{
    mProgram->Bind();
}

void SuperSamplingEffect::Unbind()
{
    mProgram->Unbind();
}

ShapeEffect::ShapeEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    static const char* vsShape =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   FragColor = texture(tex0, texcoordOut.xy) * colorOut * shapeColor;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vsShape =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   FragColor = texture(tex0, texcoordOut.xy) * colorOut * shapeColor;\n"
        "}\n";
#else
    static const char* vsShape =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "varying vec2 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   gl_FragColor = texture2D(tex0, texcoordOut.xy) * colorOut * shapeColor;\n"
        "}\n";
#endif


    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);
    
    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTex0Loc = mProgram->GetUniformLocation("tex0");

    mProgram->Bind();
    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
    SetTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

ShapeEffect::~ShapeEffect()
{
    delete mProgram;
}

void ShapeEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void ShapeEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void ShapeEffect::SetTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTex0Loc, 0, texture);
}

void ShapeEffect::Bind()
{
    mProgram->Bind();
}

void ShapeEffect::Unbind()
{
    mProgram->Unbind();
}

MixGreyscaleEffect::MixGreyscaleEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    static const char* vsShape =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   FragColor = vec4(texture(tex0, texcoordOut.xy).rrr,1.0) * colorOut * shapeColor;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vsShape =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   FragColor = vec4(texture(tex0, texcoordOut.xy).rrr,1.0) * colorOut * shapeColor;\n"
        "}\n";
#else
    static const char* vsShape =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "varying vec2 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   gl_FragColor = vec4(texture2D(tex0, texcoordOut.xy).rrr,1.0) * colorOut * shapeColor;\n"
        "}\n";
#endif


    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTex0Loc = mProgram->GetUniformLocation("tex0");

    mProgram->Bind();
    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
    SetTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

MixGreyscaleEffect::~MixGreyscaleEffect()
{
    delete mProgram;
}

void MixGreyscaleEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void MixGreyscaleEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void MixGreyscaleEffect::SetTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTex0Loc, 0, texture);
}

void MixGreyscaleEffect::Bind()
{
    mProgram->Bind();
}

void MixGreyscaleEffect::Unbind()
{
    mProgram->Unbind();
}

PaletteEffect::PaletteEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    static const char* vsShape =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform sampler2D texPalette;\n"
        "uniform vec4 shapeColor;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   float index = texture(texSrc, texcoordOut.xy).x + 0.5/255.0;\n"
        "   vec4 src = texture(texPalette, vec2(index,0.5)) * shapeColor;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * (src.a / a) + dst.rgb * (dsta / a), a);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vsShape =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform sampler2D texPalette;\n"
        "uniform vec4 shapeColor;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   float index = texture(texSrc, texcoordOut.xy).x + 0.5/255.0;\n"
        "   vec4 src = texture(texPalette, vec2(index,0.5)) * shapeColor;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   FragColor = vec4(src.rgb * (src.a / a) + dst.rgb * (dsta / a), a);\n"
        "}\n";
#else
    static const char* vsShape =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform sampler2D texPalette;\n"
        "uniform vec4 shapeColor;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 dst = texture2D(texDst, texcoordOut.zw);\n"
        "   float index = texture2D(texSrc, texcoordOut.xy).x + 0.5/255.0;\n"
        "   vec4 src = texture2D(texPalette, vec2(index,0.5)) * shapeColor;\n"
        "   float dsta = dst.a * (1.0 - src.a);\n"
        "   float a = src.a + dsta;\n"
        "   gl_FragColor = vec4(src.rgb * (src.a / a) + dst.rgb * (dsta / a), a);\n"
        "}\n";
#endif


	std::map<std::string, int> bs;
	bs["position"] = GLMesh::GLAttributeTypePosition;
	bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
	bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);

    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mMvpLoc = mProgram->GetUniformLocation("mvp");
	mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexPaletteLoc = mProgram->GetUniformLocation("texPalette");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
    mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");

	mProgram->Bind();
	SetMvp(Matrix4());
    SetDstTexture(renderer->GetDefaultTexture());
    SetSrcTexture(renderer->GetDefaultTexture());
    SetPaletteTexture(renderer->GetDefaultTexture());
	mProgram->Unbind();
}

PaletteEffect::~PaletteEffect()
{
	delete mProgram;
}

void PaletteEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void PaletteEffect::SetMv(const Matrix4& mv)
{
    mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void PaletteEffect::SetMvp(const Matrix4& mvp)
{
	mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void PaletteEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 2, texture);    
    mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void PaletteEffect::SetSrcTexture(GLTexture* texture)
{
	mProgram->SetTexture(mTexSrcLoc, 0, texture);
}

void PaletteEffect::SetPaletteTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexPaletteLoc, 1, texture);
}

void PaletteEffect::Bind()
{
	mProgram->Bind();
}

void PaletteEffect::Unbind()
{
	mProgram->Unbind();
}


PaletteMixEffect::PaletteMixEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    static const char* vsShape =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   float index = texture(texSrc, texcoordOut.xy).x;\n"
        "   index = index > 0.0 ? index : dst.r;\n"
        "   FragColor = vec4(index,0.0,0.0,1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vsShape =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   float index = texture(texSrc, texcoordOut.xy).x;\n"
        "   index = index > 0.0 ? index : dst.r;\n"
        "   FragColor = vec4(index,0.0,0.0,1.0);\n"
        "}\n";
#else
    static const char* vsShape =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   vec4 dst = texture2D(texDst, texcoordOut.zw);\n"
        "   float index = texture2D(texSrc, texcoordOut.xy).x;\n"
        "   index = index > 0.0 ? index : dst.r;\n"
        "   gl_FragColor = vec4(index,0.0,0.0,1.0);\n"
        "}\n";
#endif


    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);

    mMvLoc = mProgram->GetUniformLocation("mv");
    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
    mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");

    mProgram->Bind();
    SetMvp(Matrix4());
    SetDstTexture(renderer->GetDefaultTexture());
    SetSrcTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

PaletteMixEffect::~PaletteMixEffect()
{
    delete mProgram;
}

void PaletteMixEffect::SetMv(const Matrix4& mv)
{
    mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void PaletteMixEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void PaletteMixEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 2, texture);    
    mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void PaletteMixEffect::SetSrcTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexSrcLoc, 0, texture);
}

void PaletteMixEffect::Bind()
{
    mProgram->Bind();
}

void PaletteMixEffect::Unbind()
{
    mProgram->Unbind();
}

PaletteAddEffect::PaletteAddEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    static const char* vsShape =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   float dstIndex = texture(texDst, texcoordOut.zw).r;\n"
        "   float index = texture(texSrc, texcoordOut.xy).x + dstIndex;\n"
        "   if (dstIndex == 0.0) index = 0.0;\n"
        "   FragColor = vec4(index,0.0,0.0,1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vsShape =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   float dstIndex = texture(texDst, texcoordOut.zw).r;\n"
        "   float index = texture(texSrc, texcoordOut.xy).x + dstIndex;\n"
        "   if (dstIndex == 0.0) index = 0.0;\n"
        "   FragColor = vec4(index,0.0,0.0,1.0);\n"
        "}\n";
#else
    static const char* vsShape =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   float dstIndex = texture2D(texDst, texcoordOut.zw).r;\n"
        "   float index = texture2D(texSrc, texcoordOut.xy).x + dstIndex;\n"
        "   if (dstIndex == 0.0) index = 0.0;\n"
        "   gl_FragColor = vec4(index,0.0,0.0,1.0);\n"
        "}\n";
#endif


    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);

    mMvLoc = mProgram->GetUniformLocation("mv");
    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
    mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");

    mProgram->Bind();
    SetMvp(Matrix4());
    SetDstTexture(renderer->GetDefaultTexture());
    SetSrcTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

PaletteAddEffect::~PaletteAddEffect()
{
    delete mProgram;
}

void PaletteAddEffect::SetMv(const Matrix4& mv)
{
    mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void PaletteAddEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void PaletteAddEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 2, texture);    
    mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void PaletteAddEffect::SetSrcTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexSrcLoc, 0, texture);
}

void PaletteAddEffect::Bind()
{
    mProgram->Bind();
}

void PaletteAddEffect::Unbind()
{
    mProgram->Unbind();
}

DetectHoleEffect::DetectHoleEffect(GLRenderer* /*renderer*/)
{
    CreatePatternDetectionProgram();
    CreatePatternExpandProgram();

    GLMesh::AttributeDesc attr(GLMesh::GLAttributeTypePosition, 2);
    GLMesh* screenQuad = new GLMesh(4, 6, &attr, 1);
    screenQuad->AddRect(AABB2(-1, -1, 1, 1), AABB2(0, 0, 1, 1), Color(1, 1, 1, 1));
    screenQuad->BuildVBO();
    mQuad = screenQuad;
}

DetectHoleEffect::~DetectHoleEffect()
{
    delete mQuad;
    delete mProgram;
    delete mExpandProgram;
}

void DetectHoleEffect::CreatePatternExpandProgram()
{
#if defined(USE_OPENGL3)
    static const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texHole;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   float srcIndex = texture(texSrc, texcoordOut.xy).r;\n"
        "   if (srcIndex == 0.0) {\n"
        "      float isCloseToPattern = texture(texHole, texcoordOut.xy).r;\n"
        "      if (isCloseToPattern == 0.0) {\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(1,0)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(-1,0)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(0,1)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(0,-1)).r;\n"
        "      }\n"
        "      if (isCloseToPattern > 0.0) {\n"
        "          FragColor = vec4(1.0,0.0,0.0,1.0);\n"
        "          return;\n"
        "      }\n"
        "   }\n"
        "   FragColor = vec4(0.0,0.0,0.0,1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texHole;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   float srcIndex = texture(texSrc, texcoordOut.xy).r;\n"
        "   if (srcIndex == 0.0) {\n"
        "      float isCloseToPattern = texture(texHole, texcoordOut.xy).r;\n"
        "      if (isCloseToPattern == 0.0) {\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(1,0)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(-1,0)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(0,1)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(0,-1)).r;\n"
        "      }\n"
        "      if (isCloseToPattern > 0.0) {\n"
        "          FragColor = vec4(1.0,0.0,0.0,1.0);\n"
        "          return;\n"
        "      }\n"
        "   }\n"
        "   FragColor = vec4(0.0,0.0,0.0,1.0);\n"
        "}\n";
#else
    static const char* vs =
        "attribute vec2 position;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texHole;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   float srcIndex = texture2D(texSrc, texcoordOut.xy).r;\n"
        "   if (srcIndex == 0.0) {\n"
        "      float isCloseToPattern = texture2D(texHole, texcoordOut.xy).r;\n"
        "      if (isCloseToPattern == 0.0) {\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(1,0)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(-1,0)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(0,1)).r;\n"
        "         isCloseToPattern += textureOffset(texHole, texcoordOut.xy, ivec2(0,-1)).r;\n"
        "      }\n"
        "      if (isCloseToPattern > 0.0) {\n"
        "          gl_FragColor = vec4(1.0,0.0,0.0,1.0);\n"
        "          return;\n"
        "      }\n"
        "   }\n"
        "   gl_FragColor = vec4(0.0,0.0,0.0,1.0);\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mExpandProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mExpandTexSrcLoc = mExpandProgram->GetUniformLocation("texSrc");
    mExpandTexHoleLoc = mExpandProgram->GetUniformLocation("texHole");
}

void DetectHoleEffect::CreatePatternDetectionProgram()
{
#if defined(USE_OPENGL3)
    static const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps1 =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = vec4(0.0,0.0,0.0,1.0);\n"
        "   float srcIndex = texture(texSrc, texcoordOut.xy).r;\n"
        "   if (srcIndex == 0.0) {\n"
        "      bool matchAll;\n";
        //"      bool p%d_%d = textureOffset(texSrc,texcoordOut.xy,ivec2(%d,%d)).r > 0.0;\n"
        //"      matchAll = p%d_%d && p%d_%d;\n"
        //"      if (matchAll) {\n"
        //"          FragColor = vec4(1.0,0.0,0.0,1.0);\n"
        //"          return;\n"
        //"      }\n"
    static const char* ps2 =
        "   }\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps1 =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = vec4(0.0,0.0,0.0,1.0);\n"
        "   float srcIndex = texture(texSrc, texcoordOut.xy).r;\n"
        "   if (srcIndex == 0.0) {\n"
        "      bool matchAll;\n";
        //"      bool p%d_%d = textureOffset(texSrc,texcoordOut.xy,ivec2(%d,%d)).r > 0.0;\n"
        //"      matchAll = p%d_%d && p%d_%d;\n"
        //"      if (matchAll) {\n"
        //"          FragColor = vec4(1.0,0.0,0.0,1.0);\n"
        //"          return;\n"
        //"      }\n"
    static const char* ps2 =
        "   }\n"
        "}\n";
#else
    static const char* vs =
        "attribute vec2 position;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps1 =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   gl_FragColor = vec4(0.0,0.0,0.0,1.0);\n"
        "   float srcIndex = texture2D(texSrc, texcoordOut.xy).r;\n"
        "   if (srcIndex == 0.0) {\n"
        "      bool matchAll;\n";
        //"      bool p%d_%d = textureOffset(texSrc,texcoordOut.xy,ivec2(%d,%d)).r > 0.0;\n"
        //"      matchAll = p%d_%d && p%d_%d;\n"
        //"      if (matchAll) {\n"
        //"          FragColor = vec4(1.0,0.0,0.0,1.0);\n"
        //"          return;\n"
        //"      }\n"
    static const char* ps2 =
        "   }\n"
        "}\n";
#endif
    std::vector<std::vector<Position> > patterns;
    int patterSize = 3;
    LoadPatterns(patterns, patterSize);

    // code generation by patterns
    std::string psBuf;
    std::set<Position> visitedPositions;
    char strBuf[100] = {0};
    std::string patternBuf;

    psBuf.append(ps1);
    for (size_t i = 0; i < patterns.size(); ++i)
    {
        std::vector<Position>& pattern = patterns[i];
        patternBuf = "      matchAll = ";

        for (size_t j = 0; j < pattern.size(); ++j)
        {
            const Position& offset = pattern[j];
            int x = offset.first;
            int y = offset.second;
            int ix = x + patterSize;
            int iy = y + patterSize;
            if (visitedPositions.find(offset) == visitedPositions.end())
            {
                // add texture access
#if defined(USE_OPENGL3) || defined(USE_OPENGL3_ES)
                sprintf(strBuf, "      bool p%d_%d = textureOffset(texSrc,texcoordOut.xy,ivec2(%d,%d)).r > 0.0;\n", ix, iy, x, y);
#else
                sprintf(strBuf, "      bool p%d_%d = textureOffset(texSrc,texcoordOut.xy,ivec2(%d,%d)).r > 0.0;\n", ix, iy, x, y);
#endif
                psBuf.append(strBuf);
                visitedPositions.insert(offset);
            }
            // check position
            if (j > 0)
            {
                patternBuf.append(" && ");
            }
            sprintf(strBuf, "p%d_%d", ix, iy); 
            patternBuf.append(strBuf);
        }
        patternBuf.append(";\n");
        psBuf.append(patternBuf);
        psBuf.append("      if (matchAll) {\n");
#if defined(USE_OPENGL3) || defined(USE_OPENGL3_ES)
        psBuf.append("          FragColor = vec4(1.0,0.0,0.0,1.0);\n");
#else
        psBuf.append("          gl_FragColor = vec4(1.0,0.0,0.0,1.0);\n");
#endif
        psBuf.append("          return;\n");
        psBuf.append("      }\n");
    }
    psBuf.append(ps2);

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vs, psBuf.c_str(), bs, __LINE__, __FUNCTION__);
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
}

void DetectHoleEffect::AddPatternImage(std::vector<std::vector<Position> >& patterns, int patternSize, int* patternImg)
{
    int c = patternSize / 2;
    std::vector<Position> pattern;
    for (int y = 0; y < patternSize; ++y)
    {
        for (int x = 0; x < patternSize; ++x)
        {
            if (patternImg[y * patternSize + x])
            {
                pattern.push_back(Position(x - c,y - c));
            }
        }
    }
    patterns.push_back(pattern);
}

void DetectHoleEffect::LoadPatterns(std::vector<std::vector<Position> >& patterns, int& patternSize)
{
    patternSize = 5;
    
    // 0
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    // 00
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 0, 1, 1, 0,
            0, 1, 0, 0, 1,
            0, 0, 1, 1, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }
    // 0
    // 0
    {
        int patternImg[] = {
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }
    //   0
    // 0 0
    {
        int patternImg[] = {
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            1, 0, 0, 1, 0,
            0, 1, 1, 0, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }
    //   0
    //   0 0
    {
        int patternImg[] = {
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 0, 1,
            0, 0, 1, 1, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }
    //   0 0
    //   0
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 0, 1, 1, 0,
            0, 1, 0, 0, 1,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }
    // 0 0
    //   0
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    // 0 0
    // 0 0
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 0, 1, 1, 0,
            0, 1, 0, 0, 1,
            0, 1, 0, 0, 1,
            0, 0, 1, 1, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    {
        int patternImg[] = {
            0, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            1, 0, 0, 1, 0,
            0, 1, 1, 0, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    // 000
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    {
        int patternImg[] = {
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    // 0 0 0
    // 0 0 0
    {
        int patternImg[] = {
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            0, 1, 1, 1, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    {
        int patternImg[] = {
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    // 0 0
    // 0 0
    // 0 0
    {
        int patternImg[] = {
            0, 0, 1, 1, 0,
            0, 1, 0, 0, 1,
            0, 1, 0, 0, 1,
            0, 1, 0, 0, 1,
            0, 0, 1, 1, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }

    {
        int patternImg[] = {
            0, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            1, 0, 0, 1, 0,
            1, 0, 0, 1, 0,
            0, 1, 1, 0, 0
        };
        AddPatternImage(patterns, patternSize, patternImg);
    }
}

void DetectHoleEffect::Render(GLRenderTarget* dst, GLTexture* src, GLRenderTarget* temp)
{
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    mProgram->Bind();
    mProgram->SetTexture(mTexSrcLoc, 0, src);
    temp->Bind();
    mQuad->Draw();

    mExpandProgram->Bind();
    mExpandProgram->SetTexture(mExpandTexSrcLoc, 0, src);
    mExpandProgram->SetTexture(mExpandTexHoleLoc, 1, temp->GetTexture());
    dst->Bind();
    mQuad->Draw();
}

FillHoleEffect::FillHoleEffect(GLRenderer* /*renderer*/)
{
#if defined(USE_OPENGL3)
    static const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps1 =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texHole;\n"
        "uniform sampler2D texColor;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   bool isHole = texture(texHole, texcoordOut.xy).r == 1.0;\n"
        "   if (isHole) {\n"
        "      vec4 c;\n";
    static const char* psSample =
        "      c = textureOffset(texColor,texcoordOut.xy,ivec2(%d,%d));\n"
        "      if (c.r > 0.0) {\n"
        "         FragColor = c;\n"
        "         return;\n"
        "      }\n";
    static const char* ps2 =
        "   }\n"
        "   discard;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps1 =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texHole;\n"
        "uniform sampler2D texColor;\n"
        "in vec4 texcoordOut;\n"
        "void main() {\n"
        "   bool isHole = texture(texHole, texcoordOut.xy).r == 1.0;\n"
        "   if (isHole) {\n"
        "      vec4 c;\n";
    static const char* psSample =
        "      c = textureOffset(texColor,texcoordOut.xy,ivec2(%d,%d));\n"
        "      if (c.r > 0.0) {\n"
        "         FragColor = c;\n"
        "         return;\n"
        "      }\n";
    static const char* ps2 =
        "   }\n"
        "   discard;\n"
        "}\n";
#else
    static const char* vs =
        "attribute vec2 position;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps1 =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texHole;\n"
        "uniform sampler2D texColor;\n"
        "varying vec4 texcoordOut;\n"
        "void main() {\n"
        "   bool isHole = texture2D(texHole, texcoordOut.xy).r == 1.0;\n"
        "   if (isHole) {\n"
        "      vec4 c;\n";
    static const char* psSample =
        "      c = textureOffset(texColor,texcoordOut.xy,ivec2(%d,%d));\n"
        "      if (c.r > 0.0) {\n"
        "         gl_FragColor = c;\n"
        "         return;\n"
        "      }\n";
    static const char* ps2 =
        "   }\n"
        "   discard;\n"
        "}\n";
#endif
    std::string psBuf;
    psBuf.append(ps1);

    char str[200] = {0};
    
    sprintf(str, psSample, 1, 0); psBuf.append(str);
    sprintf(str, psSample, -1, 0); psBuf.append(str);
    sprintf(str, psSample, 0, 1); psBuf.append(str);
    sprintf(str, psSample, 0, -1); psBuf.append(str);
    
    sprintf(str, psSample, 1, 1); psBuf.append(str);
    sprintf(str, psSample, 1, -1); psBuf.append(str);
    sprintf(str, psSample, -1, 1); psBuf.append(str);
    sprintf(str, psSample, -1, -1); psBuf.append(str);

    sprintf(str, psSample, 2, 0); psBuf.append(str);
    sprintf(str, psSample, -2, 0); psBuf.append(str);
    sprintf(str, psSample, 0, 2); psBuf.append(str);
    sprintf(str, psSample, 0, -2); psBuf.append(str);

    sprintf(str, psSample, 2, 1); psBuf.append(str);
    sprintf(str, psSample, 2, -1); psBuf.append(str);
    sprintf(str, psSample, -2, 1); psBuf.append(str);
    sprintf(str, psSample, -2, -1); psBuf.append(str);
    sprintf(str, psSample, 1, 2); psBuf.append(str);
    sprintf(str, psSample, -1, 2); psBuf.append(str);
    sprintf(str, psSample, 1, -2); psBuf.append(str);
    sprintf(str, psSample, -1, -2); psBuf.append(str);

    sprintf(str, psSample, 2, 2); psBuf.append(str);
    sprintf(str, psSample, 2, -2); psBuf.append(str);
    sprintf(str, psSample, -2, 2); psBuf.append(str);
    sprintf(str, psSample, -2, -2); psBuf.append(str);

    psBuf.append(ps2);

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vs, psBuf.c_str(), bs, __LINE__, __FUNCTION__);

    mTexHoleLoc = mProgram->GetUniformLocation("texHole");
    mTexColorLoc = mProgram->GetUniformLocation("texColor");

    GLMesh::AttributeDesc attr(GLMesh::GLAttributeTypePosition, 2);
    GLMesh* screenQuad = new GLMesh(4, 6, &attr, 1);
    screenQuad->AddRect(AABB2(-1, -1, 1, 1), AABB2(0, 0, 1, 1), Color(1, 1, 1, 1));
    screenQuad->BuildVBO();
    mQuad = screenQuad;
}

FillHoleEffect::~FillHoleEffect()
{
    delete mQuad;
    delete mProgram;
}

void FillHoleEffect::Render(GLRenderTarget* dst, GLTexture* holeTexture, GLTexture* colorTexture)
{
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    mProgram->Bind();
    mProgram->SetTexture(mTexHoleLoc, 0, holeTexture);
    mProgram->SetTexture(mTexColorLoc, 1, colorTexture);

    dst->Bind();
    mQuad->Draw();
}

FreelineEffect::FreelineEffect(GLRenderer* /*renderer*/)
{
#ifdef USE_OPENGL3
    static const char* vsShape =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 posOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   posOut = position;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform vec3 points[2];\n"
        "uniform int numPoints;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 posOut;\n"
        "in vec4 colorOut;\n"
        "float getAlpha(vec2 pos) {\n"
        "   float minDist = 100000.0;\n"
        "   for(int i = 1; i < numPoints; ++i) {\n"
        "       vec2 p0 = points[i - 1].xy;\n"
        "       vec2 p1 = points[i].xy;\n"
        "       float r0 = points[i - 1].z;\n"
        "       float r1 = points[i].z;\n"
        "       float dist = 0.0;\n"
        "       if (p0 == p1) {\n"
        "           dist = distance(pos, p0) - max(r0,r1);\n"
        "       } else {\n"
        "           vec2 p0p = pos - p0;\n"
        "           vec2 p0p1 = p1 - p0;\n"
        "           vec2 dir = normalize(p0p1);\n"
        "           float dt = dot(p0p, dir);\n"
        "           float t = dt / length(p0p1);\n"
        "           if (t <= 0.0) {\n"
        "              dist = distance(pos,p0) - r0;\n"
        "           } else if (t >= 1.0) {\n"
        "              dist = distance(pos,p1) - r1;\n"
        "           } else {\n"
        "              float rt = mix(r0,r1,t);\n"
        "              dist = distance(pos, dir * dt + p0) - rt;\n"
        "           }\n"
        "       }\n"
        "       if (dist <= 0.0) {\n"
        "           return 1.0;\n"
        "       }\n"
        "       minDist = min(minDist, dist);\n"
        "   }\n"
        "   float smoothRadius = 1.0;\n"
        "   return 1.0 - min(1.0,minDist/smoothRadius);\n"
        "}\n"
        "void main() {\n"
        "   float a = getAlpha(posOut);\n"
        "   FragColor = vec4(0.0,0.0,0.0,1.0) * vec4(1.0, 1.0, 1.0, a);\n"
        "}\n";
#else
    static const char* vsShape =
        "attribute vec2 position;\n"
    	"attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 posOut;\n"
    	"varying vec4 colorOut;\n"
        "void main() {\n"
        "   posOut = position;\n"
    	"   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* psShape =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform vec3 points[2];\n"
        "uniform int numPoints;\n"
        "uniform vec4 shapeColor;\n"
        "varying vec2 posOut;\n"
    	"varying vec4 colorOut;\n"
        "float getAlpha(vec2 pos) {\n"
        "   float minDist = 100000.0;\n"
        "   for(int i = 1; i < numPoints; ++i) {\n"
        "       vec2 p0 = points[i - 1].xy;\n"
        "       vec2 p1 = points[i].xy;\n"
        "       float r0 = points[i - 1].z;\n"
        "       float r1 = points[i].z;\n"
        "       float dist = 0.0;\n"
        "       if (p0 == p1) {\n"
        "           dist = distance(pos, p0) - max(r0,r1);\n"
        "       } else {\n"
        "           vec2 p0p = pos - p0;\n"
        "           vec2 p0p1 = p1 - p0;\n"
        "           vec2 dir = normalize(p0p1);\n"
        "           float dt = dot(p0p, dir);\n"
        "           float t = dt / length(p0p1);\n"
        "           if (t <= 0.0) {\n"
        "              dist = distance(pos,p0) - r0;\n"
        "           } else if (t >= 1.0) {\n"
        "              dist = distance(pos,p1) - r1;\n"
        "           } else {\n"
        "              float rt = mix(r0,r1,t);\n"
        "              dist = distance(pos, dir * dt + p0) - rt;\n"
        "           }\n"
        "       }\n"
        "       if (dist <= 0.0) {\n"
        "           return 1.0;\n"
        "       }\n"
        "       minDist = min(minDist, dist);\n"
        "   }\n"
        "   float smoothRadius = 1.0;\n"
        "   return 1.0 - min(1.0,minDist/smoothRadius);\n"
        "}\n"
        "void main() {\n"
        "   float a = getAlpha(posOut);\n"
        "   gl_FragColor = vec4(0.0,0.0,0.0,1.0) * vec4(1.0, 1.0, 1.0, a);\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vsShape, psShape, bs, __LINE__, __FUNCTION__);
    
    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mPointsLoc = mProgram->GetUniformLocation("points");
    mNumPointsLoc = mProgram->GetUniformLocation("numPoints");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");

    mProgram->Bind();
    SetMvp(Matrix4());
    SetPoints(NULL, 0);
    SetColor(Color(1,1,1,1));
    mProgram->Unbind();
}

FreelineEffect::~FreelineEffect()
{
    delete mProgram;
}

void FreelineEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void FreelineEffect::Bind()
{
    mProgram->Bind();
}

void FreelineEffect::Unbind()
{
    mProgram->Unbind();
}

void FreelineEffect::SetPoints(Vector3* points, int numPoints)
{
    assert(numPoints <= 64);
    mProgram->SetInt(mNumPointsLoc, numPoints);
    if (numPoints > 0)
    {
        mProgram->SetFloat3Array(mPointsLoc, (float*)points, numPoints);
    }
}

void FreelineEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

ColorOverrideEffect::ColorOverrideEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    static const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   FragColor = vec4(shapeColor.rgb, texture(tex0, texcoordOut.xy).a * colorOut.a * shapeColor.a);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    static const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "out vec2 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "in vec2 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   FragColor = vec4(shapeColor.rgb, texture(tex0, texcoordOut.xy).a * colorOut.a * shapeColor.a);\n"
        "}\n";
#else
    static const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    static const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "uniform vec4 shapeColor;\n"
        "varying vec2 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   gl_FragColor = vec4(shapeColor.rgb, texture2D(tex0, texcoordOut.xy).a * colorOut.a * shapeColor.a);\n"
        "}\n";
#endif

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTex0Loc = mProgram->GetUniformLocation("tex0");

    mProgram->Bind();
    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
    SetTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

ColorOverrideEffect::~ColorOverrideEffect()
{
    delete mProgram;
}

void ColorOverrideEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void ColorOverrideEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void ColorOverrideEffect::SetTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTex0Loc, 0, texture);
}

void ColorOverrideEffect::Bind()
{
    mProgram->Bind();
}

void ColorOverrideEffect::Unbind()
{
    mProgram->Unbind();
}

BlendEffect::BlendEffect(GLRenderer* renderer, const char* blendStr)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = texture(texSrc, texcoordOut.xy) * colorOut;\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = texture(texSrc, texcoordOut.xy) * colorOut;\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n";
#else
    const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#define FragColor gl_FragColor\n"
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = texture2D(texSrc, texcoordOut.xy) * colorOut;\n"
        "   vec4 dst = texture2D(texDst, texcoordOut.zw);\n";
#endif


    std::string psBuf(ps);
    psBuf.append(blendStr);
    psBuf.append("}\n");

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vs, psBuf.c_str(), bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
	mMvLoc = mProgram->GetUniformLocation("mv");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
	mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");

    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
	SetMv(Matrix4());
    SetSrcTexture(renderer->GetDefaultTexture());
    SetDstTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

BlendEffect::~BlendEffect()
{
    delete mProgram;
}

void BlendEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void BlendEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void BlendEffect::SetMv(const Matrix4& mv)
{
	mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void BlendEffect::SetSrcTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexSrcLoc, 0, texture);
}

void BlendEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 1, texture);
	mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void BlendEffect::Bind()
{
    mProgram->Bind();
}

void BlendEffect::Unbind()
{
    mProgram->Unbind();
}

MaskExpandEffect::MaskExpandEffect(GLRenderer* renderer, const char* blendStr)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 expandTexcoordStep;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = texcoordOut.xy + vec2(expandTexcoordStep.x * float(i), expandTexcoordStep.y * float(j));\n"
        "           src = max(src,texture(texSrc, uv));\n"
        "      }\n"
        "   }\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 expandTexcoordStep;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = texcoordOut.xy + vec2(expandTexcoordStep.x * float(i), expandTexcoordStep.y * float(j));\n"
        "           src = max(src,texture(texSrc, uv));\n"
        "      }\n"
        "   }\n";
#else
    const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#define FragColor gl_FragColor\n"
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 expandTexcoordStep;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 dst = texture2D(texDst, texcoordOut.zw);\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = texcoordOut.xy + vec2(expandTexcoordStep.x * float(i), expandTexcoordStep.y * float(j));\n"
        "           src = max(src,texture2D(texSrc, uv));\n"
        "      }\n"
        "   }\n";
#endif

    std::string psBuf(ps);
    psBuf.append(blendStr);
    psBuf.append("}\n");

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vs, psBuf.c_str(), bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
    mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");
    mExpandLoc = mProgram->GetUniformLocation("expand");
    mExpandTexcoordStep = mProgram->GetUniformLocation("expandTexcoordStep");

    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
    SetMv(Matrix4());
    SetSrcTexture(renderer->GetDefaultTexture());
    SetDstTexture(renderer->GetDefaultTexture());
    SetExpand(1);
    mProgram->Unbind();
}

MaskExpandEffect::~MaskExpandEffect()
{
    delete mProgram;
}

void MaskExpandEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void MaskExpandEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void MaskExpandEffect::SetMv(const Matrix4& mv)
{
    mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void MaskExpandEffect::SetSrcTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexSrcLoc, 0, texture);
    mProgram->SetFloat2(mExpandTexcoordStep, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void MaskExpandEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 1, texture);
    mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void MaskExpandEffect::SetExpand(int expand)
{
    mProgram->SetInt(mExpandLoc, expand);
}

void MaskExpandEffect::Bind()
{
    mProgram->Bind();
}

void MaskExpandEffect::Unbind()
{
    mProgram->Unbind();
}

MaskBlurEffect::MaskBlurEffect(GLRenderer* renderer, const char* /*blendStr*/)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 dstSizeInv;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   dst.rgb *= dst.a;\n"
        "   vec4 c = texture(texSrc, texcoordOut.xy) * colorOut;\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = texcoordOut.zw + vec2(dstSizeInv.x * float(i), dstSizeInv.y * float(j));\n"
        "           vec4 v = texture(texDst, uv);\n"
        "           v.rgb *= v.a;\n"
        "           src += v;\n"
        "      }\n"
        "   }\n"
        "   float n = float(expand) * 2.0 + 1.0;\n"
        "   n = 1.0 / (n * n);\n"
        "   src *= n;\n"
        "   float a = c.r * shapeColor.a;\n"
        "   dst = src * a + dst * (1.0 - a);\n"
        "   dst.rgb *= dst.a > 0.0 ? 1.0 / dst.a : 0.0;\n"
        "   FragColor = dst;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 dstSizeInv;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   dst.rgb *= dst.a;\n"
        "   vec4 c = texture(texSrc, texcoordOut.xy) * colorOut;\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = texcoordOut.zw + vec2(dstSizeInv.x * float(i), dstSizeInv.y * float(j));\n"
        "           vec4 v = texture(texDst, uv);\n"
        "           v.rgb *= v.a;\n"
        "           src += v;\n"
        "      }\n"
        "   }\n"
        "   float n = float(expand) * 2.0 + 1.0;\n"
        "   n = 1.0 / (n * n);\n"
        "   src *= n;\n"
        "   float a = c.r * shapeColor.a;\n"
        "   dst = src * a + dst * (1.0 - a);\n"
        "   dst.rgb *= dst.a > 0.0 ? 1.0 / dst.a : 0.0;\n"
        "   FragColor = dst;\n"
        "}\n";
#else
    const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   vec4 dst = texture2D(texDst, texcoordOut.zw);\n"
        "   dst.rgb *= dst.a;\n"
        "   vec4 c = texture2D(texSrc, texcoordOut.xy) * colorOut;\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = texcoordOut.zw + vec2(dstSizeInv.x * float(i), dstSizeInv.y * float(j));\n"
        "           vec4 v = texture2D(texDst, uv);\n"
        "           v.rgb *= v.a;\n"
        "           src += v;\n"
        "      }\n"
        "   }\n"
        "   float n = float(expand) * 2.0 + 1.0;\n"
        "   n = 1.0 / (n * n);\n"
        "   src *= n;\n"
        "   float a = c.r * shapeColor.a;\n"
        "   dst = src * a + dst * (1.0 - a);\n"
        "   dst.rgb *= dst.a > 0.0 ? 1.0 / dst.a : 0.0;\n"
        "   gl_FragColor = dst;\n"
        "}\n";
#endif


    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
    mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");
    mExpandLoc = mProgram->GetUniformLocation("expand");
    mExpandTexcoordStep = mProgram->GetUniformLocation("expandTexcoordStep");

    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
    SetMv(Matrix4());
    SetSrcTexture(renderer->GetDefaultTexture());
    SetDstTexture(renderer->GetDefaultTexture());
    SetRadius(50);
    mProgram->Unbind();
}

MaskBlurEffect::~MaskBlurEffect()
{
    delete mProgram;
}

void MaskBlurEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void MaskBlurEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void MaskBlurEffect::SetMv(const Matrix4& mv)
{
    mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void MaskBlurEffect::SetSrcTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexSrcLoc, 0, texture);
    mProgram->SetFloat2(mExpandTexcoordStep, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void MaskBlurEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 1, texture);
    mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void MaskBlurEffect::SetRadius(int r)
{
    mProgram->SetInt(mExpandLoc, r);
}

void MaskBlurEffect::Bind()
{
    mProgram->Bind();
}

void MaskBlurEffect::Unbind()
{
    mProgram->Unbind();
}

SoftenMaskEffect::SoftenMaskEffect(GLRenderer* renderer, const char* blendStr)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 expandTexcoordStep;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   float total = 0.0;\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = vec2(float(i), float(j));\n"
        "           float f = min(1.0 - length(uv) / float(expand), 1.0);\n"
        "           src += texture(texSrc, texcoordOut.xy + uv * expandTexcoordStep) * f;\n"
        "           total += f;\n"
        "      }\n"
        "   }\n"
        "   src /= total;\n"
        ;
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "in vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "out vec4 texcoordOut;\n"
        "out vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 expandTexcoordStep;\n"
        "in vec4 texcoordOut;\n"
        "in vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   vec4 dst = texture(texDst, texcoordOut.zw);\n"
        "   float total = 0.0;\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = vec2(float(i), float(j));\n"
        "           float f = min(1.0 - length(uv) / float(expand), 1.0);\n"
        "           src += texture(texSrc, texcoordOut.xy + uv * expandTexcoordStep) * f;\n"
        "           total += f;\n"
        "      }\n"
        "   }\n"
        "   src /= total;\n"
        ;
#else
    const char* vs =
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec4 color;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform vec2 dstSizeInv;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   texcoordOut.xy = texcoord.xy;\n"
        "	vec4 dstPos = mv * vec4(position, 0.0, 1.0);\n"
        "   texcoordOut.zw = dstPos.xy * dstSizeInv;\n"
        "   colorOut = color;\n"
        "   gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#define FragColor gl_FragColor\n"
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texSrc;\n"
        "uniform sampler2D texDst;\n"
        "uniform vec4 shapeColor;\n"
        "uniform int expand;\n"
        "uniform vec2 expandTexcoordStep;\n"
        "varying vec4 texcoordOut;\n"
        "varying vec4 colorOut;\n"
        "void main() {\n"
        "   vec4 src = vec4(0.0, 0.0, 0.0, 0.0);\n"
        "   vec4 dst = texture2D(texDst, texcoordOut.zw);\n"
        "   float total = 0.0;\n"
        "   for (int i = -expand; i <= expand; ++i) {\n"
        "      for (int j = -expand; j <= expand; ++j) {\n"
        "           vec2 uv = vec2(float(i), float(j));\n"
        "           float f = min(1.0 - length(uv) / float(expand), 1.0);\n"
        "           src += texture2D(texSrc, texcoordOut.xy + uv * expandTexcoordStep) * f;\n"
        "           total += f;\n"
        "      }\n"
        "   }\n"
        "   src /= total;\n"
        ;
#endif


    std::string psBuf(ps);
    psBuf.append(blendStr);
    psBuf.append("}\n");

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vs, psBuf.c_str(), bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mShapeColorLoc = mProgram->GetUniformLocation("shapeColor");
    mTexSrcLoc = mProgram->GetUniformLocation("texSrc");
    mTexDstLoc = mProgram->GetUniformLocation("texDst");
    mDstSizeInvLoc = mProgram->GetUniformLocation("dstSizeInv");
    mExpandLoc = mProgram->GetUniformLocation("expand");
    mExpandTexcoordStep = mProgram->GetUniformLocation("expandTexcoordStep");

    SetColor(Color(1, 1, 1, 1));
    SetMvp(Matrix4());
    SetMv(Matrix4());
    SetSrcTexture(renderer->GetDefaultTexture());
    SetDstTexture(renderer->GetDefaultTexture());
    SetRadius(30);
    mProgram->Unbind();
}

SoftenMaskEffect::~SoftenMaskEffect()
{
    delete mProgram;
}

void SoftenMaskEffect::SetColor(const Color& color)
{
    mProgram->SetFloat4(mShapeColorLoc, color.r, color.g, color.b, color.a);
}

void SoftenMaskEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void SoftenMaskEffect::SetMv(const Matrix4& mv)
{
    mProgram->SetMatrix(mMvLoc, &mv.m00);
}

void SoftenMaskEffect::SetSrcTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexSrcLoc, 0, texture);
    mProgram->SetFloat2(mExpandTexcoordStep, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void SoftenMaskEffect::SetDstTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexDstLoc, 1, texture);
    mProgram->SetFloat2(mDstSizeInvLoc, 1.0f / (float)texture->GetWidth(), 1.0f / (float)texture->GetHeight());
}

void SoftenMaskEffect::SetRadius(int r)
{
    mProgram->SetInt(mExpandLoc, r);
}

void SoftenMaskEffect::Bind()
{
    mProgram->Bind();
}

void SoftenMaskEffect::Unbind()
{
    mProgram->Unbind();
}

FXAAEffect::FXAAEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "#define FXAA_SUBPIX_SHIFT 0.0\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "uniform vec2 rcpFrame; //=vec2(1.0/rt_w, 1.0/rt_h);\n"
        "out vec4 posPos;\n"
        "out vec2 texcoordOut;\n"
        "void main(void)\n"
        "{\n"
        "  gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "  texcoordOut = texcoord;\n"
        "  posPos.xy = texcoordOut.xy;\n"
        "  posPos.zw = texcoordOut.xy - (rcpFrame * (0.5 + FXAA_SUBPIX_SHIFT));\n"
        "}\n"
        ;

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex;\n"
        "uniform vec2 rcpFrame; //=vec2(1.0/rt_w, 1.0/rt_h);\n"
        "in vec4 posPos;\n"
        "in vec2 texcoordOut;\n"
        "#define FXAA_SPAN_MAX 8.0\n"
        "#define FXAA_REDUCE_MUL 0.0\n"
        "#define FxaaInt2 ivec2\n"
        "#define FxaaFloat2 vec2\n"
        "#define FxaaTexLod0(t, p) texture(t, p, 0.0)\n"
        "#define FxaaTexOff(t, p, o, r) textureOffset(t, p, o)\n"
        "#define FXAA_REDUCE_MIN   (1.0/128.0)\n"
        "void main()\n"
        "{\n"
        "    vec4 rgbNW = FxaaTexLod0(tex, posPos.zw);\n"
        "    vec4 rgbNE = FxaaTexOff(tex, posPos.zw, FxaaInt2(1,0), rcpFrame.xy);\n"
        "    vec4 rgbSW = FxaaTexOff(tex, posPos.zw, FxaaInt2(0,1), rcpFrame.xy);\n"
        "    vec4 rgbSE = FxaaTexOff(tex, posPos.zw, FxaaInt2(1,1), rcpFrame.xy);\n"
        "    vec4 rgbM  = FxaaTexLod0(tex, posPos.xy);\n"
        "    vec3 luma = vec3(0.299, 0.587, 0.114);\n"
        "    float lumaNW = dot(rgbNW.xyz, luma);\n"
        "    float lumaNE = dot(rgbNE.xyz, luma);\n"
        "    float lumaSW = dot(rgbSW.xyz, luma);\n"
        "    float lumaSE = dot(rgbSE.xyz, luma);\n"
        "    float lumaM  = dot(rgbM.xyz,  luma);\n"
        "    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));\n"
        "    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));\n"
        "    vec2 dir;\n"
        "    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));\n"
        "    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));\n"
        "    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),FXAA_REDUCE_MIN);\n"
        "    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);\n"
        "    dir = min(FxaaFloat2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),max(FxaaFloat2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * rcpFrame.xy;\n"
        "    vec3 rgbA = (1.0/2.0) * (FxaaTexLod0(tex, posPos.xy + dir * (1.0/3.0 - 0.5)).xyz + FxaaTexLod0(tex, posPos.xy + dir * (2.0/3.0 - 0.5)).xyz);\n"
        "    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (FxaaTexLod0(tex, posPos.xy + dir * (0.0/3.0 - 0.5)).xyz + FxaaTexLod0(tex, posPos.xy + dir * (3.0/3.0 - 0.5)).xyz);\n"
        "    float lumaB = dot(rgbB, luma);\n"
        "    FragColor = vec4(((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB, 1.0);\n"
        "}"
        ;
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "#define FXAA_SUBPIX_SHIFT 0.0\n"
        "in vec2 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "uniform vec2 rcpFrame; //=vec2(1.0/rt_w, 1.0/rt_h);\n"
        "out vec4 posPos;\n"
        "out vec2 texcoordOut;\n"
        "void main(void)\n"
        "{\n"
        "  gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "  texcoordOut = texcoord;\n"
        "  posPos.xy = texcoordOut.xy;\n"
        "  posPos.zw = texcoordOut.xy - (rcpFrame * (0.5 + FXAA_SUBPIX_SHIFT));\n"
        "}\n"
        ;

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex;\n"
        "uniform vec2 rcpFrame; //=vec2(1.0/rt_w, 1.0/rt_h);\n"
        "in vec4 posPos;\n"
        "in vec2 texcoordOut;\n"
        "#define FXAA_SPAN_MAX 8.0\n"
        "#define FXAA_REDUCE_MUL 0.0\n"
        "#define FxaaInt2 ivec2\n"
        "#define FxaaFloat2 vec2\n"
        "#define FxaaTexLod0(t, p) texture(t, p, 0.0)\n"
        "#define FxaaTexOff(t, p, o, r) textureOffset(t, p, o)\n"
        "#define FXAA_REDUCE_MIN   (1.0/128.0)\n"
        "void main()\n"
        "{\n"
        "    vec4 rgbNW = FxaaTexLod0(tex, posPos.zw);\n"
        "    vec4 rgbNE = FxaaTexOff(tex, posPos.zw, FxaaInt2(1,0), rcpFrame.xy);\n"
        "    vec4 rgbSW = FxaaTexOff(tex, posPos.zw, FxaaInt2(0,1), rcpFrame.xy);\n"
        "    vec4 rgbSE = FxaaTexOff(tex, posPos.zw, FxaaInt2(1,1), rcpFrame.xy);\n"
        "    vec4 rgbM  = FxaaTexLod0(tex, posPos.xy);\n"
        "    vec3 luma = vec3(0.299, 0.587, 0.114);\n"
        "    float lumaNW = dot(rgbNW.xyz, luma);\n"
        "    float lumaNE = dot(rgbNE.xyz, luma);\n"
        "    float lumaSW = dot(rgbSW.xyz, luma);\n"
        "    float lumaSE = dot(rgbSE.xyz, luma);\n"
        "    float lumaM  = dot(rgbM.xyz,  luma);\n"
        "    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));\n"
        "    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));\n"
        "    vec2 dir;\n"
        "    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));\n"
        "    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));\n"
        "    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),FXAA_REDUCE_MIN);\n"
        "    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);\n"
        "    dir = min(FxaaFloat2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),max(FxaaFloat2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * rcpFrame.xy;\n"
        "    vec3 rgbA = (1.0/2.0) * (FxaaTexLod0(tex, posPos.xy + dir * (1.0/3.0 - 0.5)).xyz + FxaaTexLod0(tex, posPos.xy + dir * (2.0/3.0 - 0.5)).xyz);\n"
        "    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (FxaaTexLod0(tex, posPos.xy + dir * (0.0/3.0 - 0.5)).xyz + FxaaTexLod0(tex, posPos.xy + dir * (3.0/3.0 - 0.5)).xyz);\n"
        "    float lumaB = dot(rgbB, luma);\n"
        "    FragColor = vec4(((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB, 1.0);\n"
        "}"
        ;
#else
    const char* vs =
        "#define FXAA_SUBPIX_SHIFT 0.0\n"
        "attribute vec2 position;\n"
        "attribute vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "uniform vec2 rcpFrame; //=vec2(1.0/rt_w, 1.0/rt_h);\n"
        "varying vec4 posPos;\n"
        "varying vec2 texcoordOut;\n"
        "void main(void)\n"
        "{\n"
        "  gl_Position = mvp * vec4(position, 0.0, 1.0);\n"
        "  texcoordOut = texcoord;\n"
        "  posPos.xy = texcoordOut.xy;\n"
        "  posPos.zw = texcoordOut.xy - (rcpFrame * (0.5 + FXAA_SUBPIX_SHIFT));\n"
        "}\n"
        ;

    const char* ps =
        //"#extension GL_EXT_gpu_shader4 : enable // For NVIDIA cards.\n"
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex;\n"
        "uniform vec2 rcpFrame; //=vec2(1.0/rt_w, 1.0/rt_h);\n"
        "varying vec4 posPos;\n"
        "varying vec2 texcoordOut;\n"
        "#define FXAA_SPAN_MAX 8.0\n"
        "#define FXAA_REDUCE_MUL 0.0\n"
        "#define FxaaInt2 ivec2\n"
        "#define FxaaFloat2 vec2\n"
        "#ifdef GL_ES\n"
        "#define FxaaTexLod0(t, p) texture2D(t, p)\n"
        "#define FxaaTexOff(t, p, o, r) texture2D(t, p + (o * r))\n"
        "#else\n"
        "#define FxaaTexLod0(t, p) texture2D(t, p, 0.0)\n"
        "#define FxaaTexOff(t, p, o, r) texture2D(t, p + (o * r), 0.0)\n"
        "#endif\n"
        "#define FXAA_REDUCE_MIN   (1.0/128.0)\n"
        "void main()\n"
        "{\n"
        "    vec4 rgbNW = FxaaTexLod0(tex, posPos.zw);\n"
        "    vec4 rgbNE = FxaaTexOff(tex, posPos.zw, FxaaInt2(1,0), rcpFrame.xy);\n"
        "    vec4 rgbSW = FxaaTexOff(tex, posPos.zw, FxaaInt2(0,1), rcpFrame.xy);\n"
        "    vec4 rgbSE = FxaaTexOff(tex, posPos.zw, FxaaInt2(1,1), rcpFrame.xy);\n"
        "    vec4 rgbM  = FxaaTexLod0(tex, posPos.xy);\n"
        "    vec3 luma = vec3(0.299, 0.587, 0.114);\n"
        "    float lumaNW = dot(rgbNW.xyz, luma);\n"
        "    float lumaNE = dot(rgbNE.xyz, luma);\n"
        "    float lumaSW = dot(rgbSW.xyz, luma);\n"
        "    float lumaSE = dot(rgbSE.xyz, luma);\n"
        "    float lumaM  = dot(rgbM.xyz,  luma);\n"
        "    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));\n"
        "    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));\n"
        "    vec2 dir;\n"
        "    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));\n"
        "    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));\n"
        "    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),FXAA_REDUCE_MIN);\n"
        "    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);\n"
        "    dir = min(FxaaFloat2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),max(FxaaFloat2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * rcpFrame.xy;\n"
        "    vec3 rgbA = (1.0/2.0) * (FxaaTexLod0(tex, posPos.xy + dir * (1.0/3.0 - 0.5)).xyz + FxaaTexLod0(tex, posPos.xy + dir * (2.0/3.0 - 0.5)).xyz);\n"
        "    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (FxaaTexLod0(tex, posPos.xy + dir * (0.0/3.0 - 0.5)).xyz + FxaaTexLod0(tex, posPos.xy + dir * (3.0/3.0 - 0.5)).xyz);\n"
        "    float lumaB = dot(rgbB, luma);\n"
        "    gl_FragColor = vec4(((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB, 1.0);\n"
        "}"
        ;
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mTexLoc = mProgram->GetUniformLocation("tex");
    mRcpFrameLoc = mProgram->GetUniformLocation("rcpFrame");

    SetMvp(Matrix4());
    SetTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

FXAAEffect::~FXAAEffect()
{
    delete mProgram;
}

void FXAAEffect::SetMvp(const Matrix4& mvp)
{
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
}

void FXAAEffect::SetTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTexLoc, 0, texture);
    mProgram->SetFloat2(mRcpFrameLoc, 1.0f / texture->GetWidth(), 1.0f / texture->GetHeight());
}

void FXAAEffect::Bind()
{
    mProgram->Bind();
}

void FXAAEffect::Unbind()
{
    mProgram->Unbind();
}

Fxaa3Effect::Fxaa3Effect(const char* dir)
{
    GLRenderer* renderer = GLRenderer::GetInstance();
    std::string path(dir);
    std::string vsPath = path + "/fxaa3.vs";
    std::string fsPath = path + "/fxaa3.fs";

    std::map<std::string, int> bs;
    bs["pos"] = GLMesh::GLAttributeTypePosition;
    mProgram0 = renderer->CreateShaderFromFile(vsPath.c_str(), fsPath.c_str(), bs);

    mColorTexLoc0 = mProgram0->GetUniformLocation("colorTex");
    mScreenSizeLoc0 = mProgram0->GetUniformLocation("screenSize");

    GLMesh::AttributeDesc attr(GLMesh::GLAttributeTypePosition, 2);
    GLMesh* screenQuad = new GLMesh(4, 6, &attr, 1);
    screenQuad->AddRect(AABB2(-1, -1, 1, 1), AABB2(0, 0, 1, 1), Color(1, 1, 1, 1));
    screenQuad->BuildVBO();
    mQuad = screenQuad;
}

Fxaa3Effect::~Fxaa3Effect()
{
    delete mQuad;
    delete mProgram0;
}

void Fxaa3Effect::Render(GLRenderTarget* dst, GLTexture* src)
{
    float w = (float)src->GetWidth();
    float h = (float)src->GetHeight();
    float screenSize[4] = { 1.0f / w, 1.0f / h, w, h };

    mProgram0->Bind();
    mProgram0->SetFloat4(mScreenSizeLoc0, screenSize);
    mProgram0->SetTexture(mColorTexLoc0, 0, src);
    dst->Clear(Color(0, 0, 0, 0));
    dst->Bind();
    mQuad->Draw();
}


SmaaEffect::SmaaEffect(const char* dir)
{
    GLRenderer* renderer = GLRenderer::GetInstance();
    std::string path(dir);
    std::string areaTexPath = path + "/smaaTexArea.png";
    std::string searchTexPath = path + "/smaaTexSearch.png";
    std::string vs0Path = path + "/smaaEdgeDetection.vs";
    std::string fs0Path = path + "/smaaEdgeDetection.fs";
    std::string vs1Path = path + "/smaaBlendingWeightCalculation.vs";
    std::string fs1Path = path + "/smaaBlendingWeightCalculation.fs";
    std::string vs2Path = path + "/smaaNeighborhoodBlending.vs";
    std::string fs2Path = path + "/smaaNeighborhoodBlending.fs";

    mAreaTexture = renderer->CreateTexture(areaTexPath.c_str());
    mSearchTexture = renderer->CreateTexture(searchTexPath.c_str());

    std::map<std::string, int> bs;
    bs["pos"] = GLMesh::GLAttributeTypePosition;
    mProgram0 = renderer->CreateShaderFromFile(vs0Path.c_str(), fs0Path.c_str(), bs);
    mProgram1 = renderer->CreateShaderFromFile(vs1Path.c_str(), fs1Path.c_str(), bs);
    mProgram2 = renderer->CreateShaderFromFile(vs2Path.c_str(), fs2Path.c_str(), bs);

    mColorTexLoc0 = mProgram0->GetUniformLocation("colorTex");
    mScreenSizeLoc0 = mProgram0->GetUniformLocation("screenSize");

    mScreenSizeLoc1 = mProgram1->GetUniformLocation("screenSize");
    mEdgeTexLoc1 = mProgram1->GetUniformLocation("edgesTex");
    mAreaTexLoc1 = mProgram1->GetUniformLocation("areaTex");
    mSearchTexLoc1 = mProgram1->GetUniformLocation("searchTex");

    mScreenSizeLoc2 = mProgram2->GetUniformLocation("screenSize");
    mBlendTexLoc2 = mProgram2->GetUniformLocation("blendTex");
    mColorTexLoc2 = mProgram2->GetUniformLocation("colorTex");

    GLMesh::AttributeDesc attr(GLMesh::GLAttributeTypePosition, 2);
    GLMesh* screenQuad = new GLMesh(4, 6, &attr, 1);
    screenQuad->AddRect(AABB2(-1, -1, 1, 1), AABB2(0, 0, 1, 1), Color(1, 1, 1, 1));
    screenQuad->BuildVBO();
    mQuad = screenQuad;

    std::string vsAlphaToMaskPath = path + "/alphaToMask.vs";
    std::string fsAlphaToMaskPath = path + "/alphaToMask.fs";
    mProgramAlphaToMask = renderer->CreateShaderFromFile(vsAlphaToMaskPath.c_str(), fsAlphaToMaskPath.c_str(), bs);
    mColorTexLoc3 = mProgramAlphaToMask->GetUniformLocation("colorTex");

    std::string vsCombinePath = path + "/combineColorAlpha.vs";
    std::string fsCombinePath = path + "/combineColorAlpha.fs";
    mProgramCombine = renderer->CreateShaderFromFile(vsCombinePath.c_str(), fsCombinePath.c_str(), bs);
    mColorTexLoc4 = mProgramCombine->GetUniformLocation("colorTex");
    mAlphaTexLoc4 = mProgramCombine->GetUniformLocation("alphaTex");
    mScreenSizeLoc4 = mProgram2->GetUniformLocation("screenSize");

    std::string vsExpandPath = path + "/expand.vs";
    std::string fsExpandPath = path + "/expand.fs";
    mProgramExpand = renderer->CreateShaderFromFile(vsExpandPath.c_str(), fsExpandPath.c_str(), bs);
    mColorTexLoc5 = mProgramExpand->GetUniformLocation("colorTex");
    mScreenSizeLoc5 = mProgramExpand->GetUniformLocation("screenSize");
}

SmaaEffect::~SmaaEffect(void)
{
    delete mProgramExpand;
    delete mProgramCombine;
    delete mProgramAlphaToMask;

    delete mQuad;
    delete mProgram0;
    delete mProgram1;
    delete mProgram2;
    delete mAreaTexture;
    delete mSearchTexture;
}

void SmaaEffect::RenderRGB(GLRenderTarget* dst, GLTexture* src, GLRenderTarget* temp0, GLRenderTarget* temp1)
{
    float w = (float)src->GetWidth();
    float h = (float)src->GetHeight();
    float screenSize[4] = { 1.0f / w, 1.0f / h, w, h };

    mProgram0->Bind();
    mProgram0->SetFloat4(mScreenSizeLoc0, screenSize);
    mProgram0->SetTexture(mColorTexLoc0, 0, src);
    temp0->Bind();
    temp0->Clear(Color(0,0,0,0));
    mQuad->Draw();

    mProgram1->Bind();
    mProgram1->SetFloat4(mScreenSizeLoc1, screenSize);
    mProgram1->SetTexture(mEdgeTexLoc1, 1, temp0->GetTexture());
    mProgram1->SetTexture(mAreaTexLoc1, 2, mAreaTexture);
    mProgram1->SetTexture(mSearchTexLoc1, 3, mSearchTexture);
    temp1->Bind();
    temp1->Clear(Color(0,0,0,0));
    mQuad->Draw();

    mProgram2->Bind();
    mProgram2->SetFloat4(mScreenSizeLoc2, screenSize);
    mProgram2->SetTexture(mBlendTexLoc2, 4, temp1->GetTexture());
    mProgram2->SetTexture(mColorTexLoc2, 0, src);
    dst->Bind();
    dst->Clear(Color(0,0,0,0));
    mQuad->Draw();

    mProgram2->Unbind();
}

void SmaaEffect::RenderRGBA(GLRenderTarget* dst, GLTexture* src, GLRenderTarget* temp0, GLRenderTarget* temp1, GLRenderTarget* temp2, GLRenderTarget* temp3, GLRenderTarget* temp4)
{
//#define DEBUG_SMAA
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    float w = (float)src->GetWidth();
    float h = (float)src->GetHeight();
    float screenSize[4] = { 1.0f / w, 1.0f / h, w, h };
    // dst-temp3-temp0
    //            -temp1
    //            -temp2-src
    //    -temp4-temp0
    //            -temp1
    //            -temp2   -src
#ifdef DEBUG_SMAA
    src->Save("d:/0_src.png");
#endif
    // expand src to eliminate bleeding
    mProgramExpand->Bind();
    mProgramExpand->SetTexture(mColorTexLoc5, 0, src);
    mProgramExpand->SetFloat4(mScreenSizeLoc5, screenSize);
    temp2->Bind();
    mQuad->Draw();
#ifdef DEBUG_SMAA
    temp2->Save("d:/1_expand.png");
#endif
    // RenderRGB(colorMaskAA,src,temp0,temp1)
    RenderRGB(temp3, temp2->GetTexture(), temp0, temp1);
#ifdef DEBUG_SMAA
    temp3->Save("d:/4_color.png");
#endif
    // alphaMask = vec4(src.aaa,1.0)
    mProgramAlphaToMask->Bind();
    mProgramAlphaToMask->SetTexture(mColorTexLoc3, 0, src);
    temp2->Bind();
    mQuad->Draw();
#ifdef DEBUG_SMAA
    temp2->Save("d:/2_alphaToMask.png");
#endif
    // RenderRGB(alphaMaskAA,alphaMask,temp0,temp1)
    RenderRGB(temp4, temp2->GetTexture(), temp0, temp1);
#ifdef DEBUG_SMAA
    temp4->Save("d:/3_alpha.png");
#endif
    // dst = vec4(findNearestSolidColor(uv,src.a),alphaMaskAA.r)
    mProgramCombine->Bind();
    mProgramCombine->SetTexture(mColorTexLoc4, 0, temp3->GetTexture());
    mProgramCombine->SetTexture(mAlphaTexLoc4, 1, temp4->GetTexture());
    dst->Bind();
    mQuad->Draw();
#ifdef DEBUG_SMAA
    dst->Save("d:/5_final.png");
#endif
}


PresentEffect::PresentEffect(GLRenderer* renderer)
{
#if defined(USE_OPENGL3)
    const char* vsPresent =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psPresent =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = texture(tex0, texcoordOut);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vsPresent =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psPresent =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex0;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = texture(tex0, texcoordOut);\n"
        "}\n";
#else
    const char* vsPresent =
        "attribute vec2 position;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psPresent =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D tex0;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   gl_FragColor = texture2D(tex0, texcoordOut);\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vsPresent, psPresent, bs, __LINE__, __FUNCTION__);

    mTex0Loc = mProgram->GetUniformLocation("tex0");

    mProgram->Bind();
    SetTexture(renderer->GetDefaultTexture());
    mProgram->Unbind();
}

PresentEffect::~PresentEffect()
{
    delete mProgram;
}

void PresentEffect::SetTexture(GLTexture* texture)
{
    mProgram->SetTexture(mTex0Loc, 0, texture);
}

void PresentEffect::Bind()
{
    mProgram->Bind();
}

void PresentEffect::Unbind()
{
    mProgram->Unbind();
}

FloatTestEffect::FloatTestEffect()
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform vec2 resolution;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   float y = (texcoordOut.y)*26.0;\n"
        "   float x = 1.0-(texcoordOut.x);\n"
        "   float b = fract(pow(2.0,floor(y))+x);\n"
        "   if(fract(y) >= 0.9)\n"
        "      b = 0.0;\n"
        "   FragColor = vec4(b,b,b,1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform vec2 resolution;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   float y = (texcoordOut.y)*26.0;\n"
        "   float x = 1.0-(texcoordOut.x);\n"
        "   float b = fract(pow(2.0,floor(y))+x);\n"
        "   if(fract(y) >= 0.9)\n"
        "      b = 0.0;\n"
        "   FragColor = vec4(b,b,b,1.0);\n"
        "}\n";
#else
	const char* vs =
		"attribute vec2 position;\n"
		"varying vec2 texcoordOut;\n"
		"void main() {\n"
		"   texcoordOut = position.xy * 0.5 + 0.5;\n"
		"   gl_Position = vec4(position, 0.0, 1.0);\n"
		"}\n";

	const char* ps =
		"#ifdef GL_ES\n"
		"precision highp float;\n"
		"#endif\n"
		"uniform vec2 resolution;\n"
		"varying vec2 texcoordOut;\n"
		"void main() {\n"
		"   float y = (texcoordOut.y)*26.0;\n"
		"   float x = 1.0-(texcoordOut.x);\n"
		"   float b = fract(pow(2.0,floor(y))+x);\n"
		"   if(fract(y) >= 0.9)\n"
		"      b = 0.0;\n"
		"   gl_FragColor = vec4(b,b,b,1.0);\n"
		"}\n";
#endif
    std::map<std::string, int> attrLocs;
	attrLocs["position"] = 0;
    mProgram = new GLShaderProgram(vs, ps, attrLocs, __LINE__, __FUNCTION__);
	mQuad = GLRenderer::GetInstance()->CreateScreenAlignedQuad();
}

FloatTestEffect::~FloatTestEffect()
{
	delete mProgram;
	delete mQuad;
}

void FloatTestEffect::Render()
{
	mProgram->Bind();
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	mQuad->Draw();
}

Transform2 Transform2::IDENTITY = Transform2();

Transform2::Transform2()
    :mTranslate(0, 0)
    ,mScale(1.0f, 1.0f)
    ,mRotate(0)
{

}

Transform2::Transform2(const Matrix3& mat)
{
    mTranslate = mat.GetTranslate();
    Vector2 xAxis = mat.GetXAxis();
    float xLength = xAxis.Length();
    mScale.x = xLength;
    mScale.y = mat.GetYAxis().Length();
    mRotate = acos(xAxis.x / xLength);
    if (xAxis.y < 0)
    {
        mRotate = PI * 2 - mRotate;
    }
}

Transform2::~Transform2()
{

}

void Transform2::ResetTransform()
{
    mTranslate.Set(0, 0);
    mScale.Set(1.0f, 1.0f);
    mRotate = 0;
}

void Transform2::SetTranslate(float x, float y)
{
    mTranslate.x = x;
    mTranslate.y = y;
}

void Transform2::SetScale(float x, float y)
{
    mScale.x = x;
    mScale.y = y;
}

void Transform2::SetRotate(float radians)
{
    mRotate = radians;
}

void Transform2::ModTranslate(float x, float y)
{
    mTranslate.x += x;
    mTranslate.y += y;
}

void Transform2::ModScale(float x, float y)
{
    mScale.x *= x;
    mScale.y *= y;
}

void Transform2::ModRotate(float radians)
{
    mRotate += radians;
}

void Transform2::Merge(const Transform2& t)
{
    Matrix3 m0;
    GetMatrix(m0);
    Matrix3 m1;
    t.GetMatrix(m1);
    m0 = m1 * m0;
    mTranslate = m0.GetTranslate();
    const Vector2& xAxis = m0.GetXAxis();
    const Vector2& yAxis = m0.GetYAxis();
    mScale.x = xAxis.Length();
    mScale.y = yAxis.Length();
    mRotate = mRotate + t.mRotate;
}

void Transform2::GetMatrix(Matrix3& matrix) const
{
    matrix.Identity();
    matrix.Scale(mScale.x, mScale.y);
    matrix.Rotate(mRotate);
    matrix.Translate(mTranslate.x, mTranslate.y);
}

bool Transform2::operator == (const Transform2& t)
{
    return mTranslate == t.mTranslate && mScale == t.mScale && mRotate == t.mRotate;
}

Camera2::Camera2()
{

}

Camera2::~Camera2()
{

}

const Transform2& Camera2::GetTransform()
{
    return mTransform;
}

void Camera2::Reset()
{
    mTransform.ResetTransform();
}

void Camera2::Set(const Vector2 position, float scale, float rotate)
{
    mTransform.SetTranslate(position.x, position.y);
    mTransform.SetScale(scale, scale);
    mTransform.SetRotate(rotate);
}

void Camera2::SetTransform(const Transform2& tran)
{
    mTransform = tran;
}

Vector2 Camera2::WorldToView(const Vector2& position)
{
    Matrix3 mat;
    mTransform.GetMatrix(mat);
    Vector2 result = position;
    mat.Transform(result);
    return result;
}

Vector2 Camera2::ViewToWorld(const Vector2& position)
{
    Matrix3 mat;
    mTransform.GetMatrix(mat);
    mat = mat.Inverse();
    Vector2 result = position;
    mat.Transform(result);
    return result;
}

void Camera2::OnPanBegin(float x, float y)
{   
    mPanAnchor.Set(x, y);
}

void Camera2::OnPanMove(float x, float y)
{
    mTransform.ModTranslate(x - mPanAnchor.x, y - mPanAnchor.y);
    mPanAnchor.Set(x, y);
}

void Camera2::OnPanEnd(float /*x*/, float /*y*/)
{
}

void Camera2::OnZoomBegin(float x, float y)
{
    mZoomScreenAnchor.Set(x, y);
    mZoomWorldAnchor = ViewToWorld(mZoomScreenAnchor);
    mZoomScale = mTransform.GetScale().x;
}

void Camera2::OnZoomMove(float x, float /*y*/)
{
    float s = (x - mZoomScreenAnchor.x) * 0.01f;
    float scale = mZoomScale * (1.0f + s);
    if (scale < 0.01f)
    {
        scale = 0.01f;
    }
    mTransform.SetScale(scale, scale);
    Vector2 anchorOnScreen = WorldToView(mZoomWorldAnchor);
    Vector2 t = mZoomScreenAnchor - anchorOnScreen;
    mTransform.ModTranslate(t.x, t.y);
}

void Camera2::OnZoomEnd(float /*x*/, float /*y*/)
{
}

void Camera2::OnPinchZoomBegin(float x, float y)
{
    mZoomScreenAnchor.Set(x, y);
    mZoomWorldAnchor = ViewToWorld(mZoomScreenAnchor);
    mZoomScale = mTransform.GetScale().x;
}

void Camera2::OnPinchZoomMove(float s)
{
    float scale = mZoomScale * s;
    if (scale < 0.01f)
    {
        scale = 0.01f;
    }
    mTransform.SetScale(scale, scale);
    Vector2 anchorOnScreen = WorldToView(mZoomWorldAnchor);
    Vector2 t = mZoomScreenAnchor - anchorOnScreen;
    mTransform.ModTranslate(t.x, t.y);
}

void Camera2::OnPinchZoomEnd(float /*s*/)
{
}

void Camera2::OnRotateBegin(float x, float y)
{
    mRoateScreenAnchor.Set(x, y);
    mRoateWorldAnchor = ViewToWorld(mRoateScreenAnchor);
    mRotateBegin = false;
    mRotateRefAngle = mTransform.GetRotate();
}

void Camera2::OnRotateMove(float x, float y)
{
    static float rotateLockRadius = 10.0f;
    Vector2 sp(x, y);
    if (sp.DistanceTo(mRoateScreenAnchor) > rotateLockRadius)
    {
        if (mRotateBegin)
        {
            float angle = sp.GetAngle(mRoateScreenAnchor);
			mTransform.SetRotate(mRotateRefAngle + angle - mRotateStartAngle);

            Vector2 anchorOnScreen = WorldToView(mRoateWorldAnchor);
            Vector2 t = mRoateScreenAnchor - anchorOnScreen;
            mTransform.ModTranslate(t.x, t.y);
        }
        else
        {
            mRotateBegin = true;
            mRotateStartAngle = sp.GetAngle(mRoateScreenAnchor);
        }
    }
}

void Camera2::OnRotateEnd(float /*x*/, float /*y*/)
{
	mRotateBegin = false;
}

void Camera2::OnPinchBegin(const Vector2& p0, const Vector2& p1)
{
    mPinchPoints[0] = p0;
    mPinchPoints[1] = p1;

    Vector2 center = (p0 + p1) * 0.5f;
    mPanAnchor = center;

    mPinchState = PinchStateEvaluate;
    mInitalPinchDistance = p0.DistanceTo(p1);
    mTotalPinchDistance += mInitalPinchDistance;
    mPinchUpdates = 1;

    mZoomScreenAnchor.Set(center.x, center.y);
    mZoomWorldAnchor = ViewToWorld(mZoomScreenAnchor);
    mZoomScale = mTransform.GetScale().x;
    mZoomInitLength = p0.DistanceTo(p1);
}

void Camera2::OnPinchMove(const Vector2& p0, const Vector2& p1)
{
    Vector2 center = (p0 + p1) * 0.5f;

    if (mPinchState == PinchStateEvaluate)
    {
//        mTotalPinchDistance += p0.DistanceTo(p1);
//        ++mPinchUpdates;

//        const float DRAG_START_DISTANCE = 15.0f;
//        const float DRAG_TOLERANCE = 10.0f;
//        if (center.DistanceTo(mPanAnchor) > DRAG_START_DISTANCE)
//        {
//            float dist = mTotalPinchDistance / mPinchUpdates - mInitalPinchDistance;
//            if (dist < 0)
//            {
//                dist = -dist;
//            }
//            if (dist < DRAG_TOLERANCE)
//            {
//                mPanAnchor = center;
//                mPinchState = PinchStateMove;
//            }
//            else
//            {
//                mPinchState = PinchStateScale;
//                mZoomScreenAnchor.Set(center.x, center.y);
//                mZoomWorldAnchor = ViewToWorld(mZoomScreenAnchor);
//                mZoomScale = mTransform.GetScale().x;
//                mZoomInitLength = p0.DistanceTo(p1);
//            }
//        }
        mPanAnchor = center;
        mPinchState = PinchStateMove;
    }

    if (mPinchState == PinchStateMove)
    {
        // apply panning
        mTransform.ModTranslate(center.x - mPanAnchor.x, center.y - mPanAnchor.y);
        mPanAnchor = center;
    }
    else if (mPinchState == PinchStateScale)
    {
        // apply zooming
        float lastLength = mZoomInitLength;
        float length = p0.DistanceTo(p1);
        float scale = mZoomScale * (length / lastLength);
        if (scale < 0.01f)
        {
            scale = 0.01f;
        }
        mTransform.SetScale(scale, scale);
        Vector2 anchorOnScreen = WorldToView(mZoomWorldAnchor);
        Vector2 t = mZoomScreenAnchor - anchorOnScreen;
        mTransform.ModTranslate(t.x, t.y);
    }
    else if (mPinchState == PinchStateRotate)
    {

    }

    mPinchPoints[0] = p0;
    mPinchPoints[1] = p1;
}

void Camera2::OnPinchEnd(const Vector2& p0, const Vector2& p1)
{
    Vector2 center = (p0 + p1) * 0.5f;

    if (mPinchState == PinchStateMove)
    {
        // apply panning
        mTransform.ModTranslate(center.x - mPanAnchor.x, center.y - mPanAnchor.y);
        mPanAnchor = center;
    }
    else if (mPinchState == PinchStateScale)
    {
        // apply zooming
        float lastLength = mZoomInitLength;
        float length = p0.DistanceTo(p1);
        float scale = mZoomScale * (length / lastLength);
        if (scale < 0.01f)
        {
            scale = 0.01f;
        }
        mTransform.SetScale(scale, scale);
        Vector2 anchorOnScreen = WorldToView(mZoomWorldAnchor);
        Vector2 t = mZoomScreenAnchor - anchorOnScreen;
        mTransform.ModTranslate(t.x, t.y);
    }
    else if (mPinchState == PinchStateRotate)
    {

    }

    mPinchState = PinchStateEvaluate;
}

Transform3::Transform3()
    :mXAxis(1, 0, 0)
    ,mYAxis(0, 1, 0)
    ,mZAxis(0, 0, 1)
    ,mPosition(0, 0, 0)
{

}

Transform3::~Transform3()
{

}

Matrix4 Transform3::ToMatrix()
{
    return Matrix4(
        mXAxis.x, mYAxis.x, mZAxis.x, mPosition.x,
        mXAxis.y, mYAxis.y, mZAxis.y, mPosition.y,
        mXAxis.z, mYAxis.z, mZAxis.z, mPosition.z,
               0,        0,        0,           1);
}

void Transform3::BuildFrom(const Matrix4& mat)
{
    mXAxis = mat.GetXAxis();
    mYAxis = mat.GetYAxis();
    mZAxis = mat.GetZAxis();
    mPosition = mat.GetTranslate();
}

void Transform3::LookAt(const Vector3& eye, const Vector3& look, const Vector3& up)
{
    mZAxis = eye - look;
    mZAxis.Normalise();
    mXAxis = up.Cross(mZAxis);
    mXAxis.Normalise();
    mYAxis = mZAxis.Cross(mXAxis);
    mYAxis.Normalise();
    mPosition = eye;
}

Camera3::Camera3()
{
    mProjection.type = ProjectionTypeNone;
}

Camera3::~Camera3()
{

}

void Camera3::SetPerspective(float fovy, float aspect, float nearZ, float farZ)
{
    mProjection.type = ProjectionTypePerspective;
    mProjection.fovy = fovy;
    mProjection.aspect = aspect;
    mProjection.nearZ = nearZ;
    mProjection.farZ = farZ;
    mProjectMatrix = Matrix4::BuildPerspective(fovy, aspect, nearZ, farZ);
}

void Camera3::SetOrtho(float left, float right, float bottom, float top, float nearZ, float farZ)
{
    mProjection.type = ProjectionTypeOrtho;
    mProjection.left = left;
    mProjection.right = right;
    mProjection.bottom = bottom;
    mProjection.top = top;
    mProjection.nearZ = nearZ;
    mProjection.farZ = farZ;
    mProjectMatrix = Matrix4::BuildOrtho(left, right, bottom, top, nearZ, farZ);
}

Ray3 Camera3::GetPickingRay(const Vector2& screenPos)
{
    if (mProjection.type == ProjectionTypePerspective)
    {
        Matrix4 frame = mViewMatrix.Inverse();
        Vector3 eyePos = frame.GetTranslate();
        Vector3 eyeDir = -frame.GetZAxis();
        Vector3 projectPoint = eyePos;
        projectPoint += eyeDir * mProjection.nearZ;

        float top = tanf(mProjection.fovy * 0.5f * PI / 180.0f) *  mProjection.nearZ;
        float right = top * mProjection.aspect;
        float h = right * 2;
        float w = top * 2;
        projectPoint += frame.GetXAxis() * (screenPos.x - 0.5f) * w;
        projectPoint += frame.GetYAxis() * (screenPos.y - 0.5f) * h;

        Vector3 rayDir = projectPoint - eyePos;
        rayDir.Normalise();
        return Ray3(eyePos, rayDir);
    }
    else if (mProjection.type == ProjectionTypeOrtho)
    {
        Vector3 nearP = Vector3(screenPos.x * 2.0f - 1.0f, screenPos.x * 2.0f - 1.0f, 0);
        nearP = (mProjectMatrix * mViewMatrix).Inverse() * nearP;
        Vector3 farP = Vector3(screenPos.x * 2.0f - 1.0f, screenPos.x * 2.0f - 1.0f, 1);
        farP = (mProjectMatrix * mViewMatrix).Inverse() * farP;

        Vector3 rayDir = farP - nearP;
        rayDir.Normalise();
        return Ray3(nearP, rayDir);
    }
    else
    {
        // should not be called
        return Ray3(Vector3(0, 0, 0), Vector3(0, 0, 1));
    }
}

Camera3Free::Camera3Free()
    :mRight(1, 0, 0)
    , mUp(0, 1, 0)
    , mBack(0, 0, 1)
{

}

Camera3Free::~Camera3Free()
{

}

void Camera3Free::Reset()
{
    mRight.Set(1, 0, 0);
    mUp.Set(0, 1, 0);
    mBack.Set(0, 0, 1);
    mViewMatrix.Identity();
}

void Camera3Free::LookAt(const Vector3& eye, const Vector3& look, const Vector3& up)
{
    mBack = eye - look;
    mBack.Normalise();
    mRight = up.Cross(mBack);
    mRight.Normalise();
    mUp = mBack.Cross(mRight);
    mUp.Normalise();
    mPosition = eye;
    UpdateViewMatrix();
}

void Camera3Free::UpdateViewMatrix()
{
    mViewMatrix.m00 = mRight.x; mViewMatrix.m01 = mRight.y; mViewMatrix.m02 = mRight.z; mViewMatrix.m03 = -mRight.Dot(mPosition);
    mViewMatrix.m10 = mUp.x; mViewMatrix.m11 = mUp.y; mViewMatrix.m12 = mUp.z; mViewMatrix.m13 = -mUp.Dot(mPosition);
    mViewMatrix.m20 = mBack.x; mViewMatrix.m21 = mBack.y; mViewMatrix.m22 = mBack.z; mViewMatrix.m23 = -mBack.Dot(mPosition);
    mViewMatrix.m30 = 0.0f;    mViewMatrix.m31 = 0.0f;    mViewMatrix.m32 = 0.0f;    mViewMatrix.m33 = 1.0f;
}

void Camera3Free::MoveForward(float delta)
{
    mPosition += mBack * -delta;
    UpdateViewMatrix();
}

void Camera3Free::MoveRight(float delta)
{
    mPosition += mRight * delta;
    UpdateViewMatrix();
}

void Camera3Free::MoveUp(float delta)
{
    mPosition += mUp * delta;
    UpdateViewMatrix();
}

void Camera3Free::Rotate(const Vector3& delta)
{
    Matrix4 mat(
        mRight.x, mUp.x, mBack.x, 0.0f,
        mRight.y, mUp.y, mBack.y, 0.0f,
        mRight.z, mUp.z, mBack.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
        );
    mat = Matrix4::BuildRotate(delta.x, mRight.x, mRight.y, mRight.z)
        * Matrix4::BuildRotate(delta.y, mUp.x, mUp.y, mUp.z)
        * Matrix4::BuildRotate(delta.z, mBack.x, mBack.y, mBack.z)
        * mat;
    mRight = mat.GetXAxis();
    mRight.Normalise();
    mUp = mat.GetYAxis();
    mUp.Normalise();
    mBack = mat.GetZAxis();
    mBack.Normalise();

    UpdateViewMatrix();
}


Camera3Flight::Camera3Flight(const Vector3& right, const Vector3& up)
    :mRight(right)
    ,mUp(up)
{
    mRight.Normalise();
    mUp.Normalise();
    mBack = mRight.Cross(mUp);
    mBack.Normalise();
    mWorldUp = mUp;
}

Camera3Flight::~Camera3Flight()
{

}

void Camera3Flight::MoveForward(float delta)
{
    mPosition += mBack * -delta;
    UpdateViewMatrix();
}

void Camera3Flight::MoveRight(float delta)
{
    mPosition += mRight * delta;
    UpdateViewMatrix();
}

void Camera3Flight::MoveUp(float delta)
{
    mPosition += mUp * delta;
    UpdateViewMatrix();
}

void Camera3Flight::TurnUp(float delta)
{
    Matrix4 mat(
        mRight.x, mUp.x, mBack.x, 0.0f,
        mRight.y, mUp.y, mBack.y, 0.0f,
        mRight.z, mUp.z, mBack.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
        );
    mat = Matrix4::BuildRotate(delta, mRight.x, mRight.y, mRight.z)
        * mat;
    mRight = mat.GetXAxis();
    mRight.Normalise();
    mUp = mat.GetYAxis();
    mUp.Normalise();
    mBack = mat.GetZAxis();
    mBack.Normalise();

    UpdateViewMatrix();
}

void Camera3Flight::TurnRight(float delta)
{
    Matrix4 mat(
        mRight.x, mUp.x, mBack.x, 0.0f,
        mRight.y, mUp.y, mBack.y, 0.0f,
        mRight.z, mUp.z, mBack.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
        );
    mat = Matrix4::BuildRotate(delta, mWorldUp.x, mWorldUp.y, mWorldUp.z)
        * mat;
    mRight = mat.GetXAxis();
    mRight.Normalise();
    mUp = mat.GetYAxis();
    mUp.Normalise();
    mBack = mat.GetZAxis();
    mBack.Normalise();

    UpdateViewMatrix();
}

void Camera3Flight::SetPosition(const Vector3& position)
{
    mPosition = position;
    UpdateViewMatrix();
}

void Camera3Flight::LookAt(const Vector3& eye, const Vector3& look)
{
    mBack = eye - look;
    mBack.Normalise();
    mRight = mWorldUp.Cross(mBack);
    mRight.Normalise();
    mUp = mBack.Cross(mRight);
    mUp.Normalise();
    mPosition = eye;
    UpdateViewMatrix();
}

void Camera3Flight::UpdateViewMatrix()
{
    mViewMatrix.m00 = mRight.x; mViewMatrix.m01 = mRight.y; mViewMatrix.m02 = mRight.z; mViewMatrix.m03 = -mRight.Dot(mPosition);
    mViewMatrix.m10 = mUp.x; mViewMatrix.m11 = mUp.y; mViewMatrix.m12 = mUp.z; mViewMatrix.m13 = -mUp.Dot(mPosition);
    mViewMatrix.m20 = mBack.x; mViewMatrix.m21 = mBack.y; mViewMatrix.m22 = mBack.z; mViewMatrix.m23 = -mBack.Dot(mPosition);
    mViewMatrix.m30 = 0.0f;    mViewMatrix.m31 = 0.0f;    mViewMatrix.m32 = 0.0f;    mViewMatrix.m33 = 1.0f;
}

void Camera3Flight::SetTransform(const Transform3& t)
{
    mRight = t.GetXAxis();
    mUp = t.GetYAxis();
    mBack = t.GetZAxis();
    mPosition = t.GetPosition();
    UpdateViewMatrix();
}

const Transform3& Camera3Flight::GetTransform()
{
    mTransform.SetXAxis(mRight);
    mTransform.SetYAxis(mUp);
    mTransform.SetZAxis(mBack);
    mTransform.SetPosition(mPosition);
    return mTransform;
}

Camera3FirstPerson::Camera3FirstPerson(const Vector3& right, const Vector3& up)
    :mRight(right)
    , mUp(up)
{
    mRight.Normalise();
    mUp.Normalise();
    mBack = mRight.Cross(mUp);
    mBack.Normalise();
    mWorldUp = mUp;
}

Camera3FirstPerson::~Camera3FirstPerson()
{

}

void Camera3FirstPerson::MoveForward(float delta)
{
    Vector3 forward = mWorldUp.Cross(mRight);
    forward.Normalise();
    mPosition += forward * delta;
    UpdateViewMatrix();
}

void Camera3FirstPerson::MoveRight(float delta)
{
    mPosition += mRight * delta;
    UpdateViewMatrix();
}

void Camera3FirstPerson::MoveUp(float delta)
{
    mPosition += mWorldUp * delta;
    UpdateViewMatrix();
}

void Camera3FirstPerson::TurnUp(float delta)
{
    Matrix4 mat(
        mRight.x, mUp.x, mBack.x, 0.0f,
        mRight.y, mUp.y, mBack.y, 0.0f,
        mRight.z, mUp.z, mBack.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
        );
    mat = Matrix4::BuildRotate(delta, mRight.x, mRight.y, mRight.z)
        * mat;
    mRight = mat.GetXAxis();
    mRight.Normalise();
    mUp = mat.GetYAxis();
    mUp.Normalise();
    mBack = mat.GetZAxis();
    mBack.Normalise();

    UpdateViewMatrix();
}

void Camera3FirstPerson::TurnRight(float delta)
{
    Matrix4 mat(
        mRight.x, mUp.x, mBack.x, 0.0f,
        mRight.y, mUp.y, mBack.y, 0.0f,
        mRight.z, mUp.z, mBack.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
        );
    mat = Matrix4::BuildRotate(delta, mWorldUp.x, mWorldUp.y, mWorldUp.z)
        * mat;
    mRight = mat.GetXAxis();
    mRight.Normalise();
    mUp = mat.GetYAxis();
    mUp.Normalise();
    mBack = mat.GetZAxis();
    mBack.Normalise();

    UpdateViewMatrix();
}

void Camera3FirstPerson::SetPosition(const Vector3& position)
{
    mPosition = position;
    UpdateViewMatrix();
}

void Camera3FirstPerson::LookAt(const Vector3& eye, const Vector3& look)
{
    mBack = eye - look;
    mBack.Normalise();
    mRight = mWorldUp.Cross(mBack);
    mRight.Normalise();
    mUp = mBack.Cross(mRight);
    mUp.Normalise();
    mPosition = eye;
    UpdateViewMatrix();
}

void Camera3FirstPerson::UpdateViewMatrix()
{
    mViewMatrix.m00 = mRight.x; mViewMatrix.m01 = mRight.y; mViewMatrix.m02 = mRight.z; mViewMatrix.m03 = -mRight.Dot(mPosition);
    mViewMatrix.m10 = mUp.x; mViewMatrix.m11 = mUp.y; mViewMatrix.m12 = mUp.z; mViewMatrix.m13 = -mUp.Dot(mPosition);
    mViewMatrix.m20 = mBack.x; mViewMatrix.m21 = mBack.y; mViewMatrix.m22 = mBack.z; mViewMatrix.m23 = -mBack.Dot(mPosition);
    mViewMatrix.m30 = 0.0f;    mViewMatrix.m31 = 0.0f;    mViewMatrix.m32 = 0.0f;    mViewMatrix.m33 = 1.0f;
}

Camera3ThirdPerson::Camera3ThirdPerson(const Vector3& right, const Vector3& up)
    :mWorldRight(right)
    ,mWorldUp(up)
    ,mLookAt(0, 0, 0)
    ,mDistance(1.0f)
    ,mRotate(0.0f)
    ,mTilt(45.0f * PI / 180.0f)
{
    
}

Camera3ThirdPerson::~Camera3ThirdPerson()
{

}

void Camera3ThirdPerson::TurnUp(float delta)
{
    mTilt += delta;
    float halfPi = PI * 0.5f * 0.998f;
    if (mTilt > halfPi)
    {
        mTilt = halfPi;
    }
    else if (mTilt < -halfPi)
    {
        mTilt = -halfPi;
    }
    UpdateViewMatrix();
}

void Camera3ThirdPerson::TurnRight(float delta)
{
    mRotate += delta;
    UpdateViewMatrix();
}

void Camera3ThirdPerson::Pan(const Vector3& delta)
{
    mLookAt += delta;
    UpdateViewMatrix();
}

void Camera3ThirdPerson::Zoom(float delta)
{
    mDistance *= 1.0f + delta;
    if (mDistance < 0.001f)
    {
        mDistance = 0.001f;
    }
    UpdateViewMatrix();
}

void Camera3ThirdPerson::SetPosition(const Vector3& position)
{
    mLookAt = position;
    UpdateViewMatrix();
}

void Camera3ThirdPerson::SetDistance(float distance)
{
    mDistance = distance;
    UpdateViewMatrix();
}

void Camera3ThirdPerson::SetTilt(float radius)
{
    mTilt = radius;
    UpdateViewMatrix();
}

void Camera3ThirdPerson::SetRotate(float radius)
{
    mRotate = radius;
    UpdateViewMatrix();
}

void Camera3ThirdPerson::UpdateViewMatrix()
{
    Vector3 toEyeGround = Matrix4::BuildRotate(mRotate, mWorldUp.x, mWorldUp.y, mWorldUp.z) * mWorldRight;
    Vector3 back = toEyeGround.Cross(mWorldUp);
    Vector3 toEye = Matrix4::BuildRotate(mTilt, back.x, back.y, back.z) * toEyeGround;
    mTransform.LookAt(mLookAt + toEye * mDistance, mLookAt, mWorldUp);
    mViewMatrix = mTransform.ToMatrix().Inverse();
}

void Camera3ThirdPerson::SetTransform(const Transform3& t)
{
    Vector3 eye = t.GetPosition();
    Vector3 toCenter = -t.GetZAxis();
    toCenter.Normalise();
    Ray3 eyeRay(eye, toCenter);
    Plane3 ground(mWorldUp, mLookAt.Dot(mWorldUp));
    eyeRay.GetIntersection(ground, mLookAt);

    mDistance = mLookAt.DistanceTo(eye);
    float d = mWorldUp.Dot(-toCenter);
    mTilt = acos(d);
    if (mTilt > PI * 0.5f)
    {
        mTilt = PI * 0.5f - mTilt;
    }
    else
    {
        mTilt = PI * 0.5f - mTilt;
    }
    Vector3 eyeToUpPrj = mLookAt + mWorldUp * d * mDistance;
    Vector3 groundDir = eye - eyeToUpPrj;
    groundDir.Normalise();

    mRotate = acos(groundDir.Dot(mWorldRight));
    Vector3 back = mWorldRight.Cross(mWorldUp);
    if (groundDir.Dot(back) > 0.0f)
    {
        mRotate = PI * 2.0f - mRotate;
    }

    UpdateViewMatrix();
    mTransform = t;
}

Vector2 Camera3ThirdPerson::ScreenToGroundPoint(const Vector2& screenPos)
{
    Ray3 ray = GetPickingRay(screenPos);
    Vector3 intersection;
    Plane3 ground(mWorldUp, mLookAt.Dot(mWorldUp));
    ray.GetIntersection(ground, intersection);
    return Vector2(intersection.x, intersection.y);
}

Canvas2d::Canvas2d(GLRenderer* renderer)
    :mRenderer(renderer)
    ,mMesh(NULL)
    ,mBlendMode(BlendModeMix)
    ,mAAMode(AntiAliasingModeNone)
    ,mAATarget(NULL)
    ,mAATargetHalf(NULL)
    ,mFSAATarget(NULL)
    ,mFSAABlitTarget(NULL)
    ,mColor(1, 1, 1, 1)
    ,mSamples(2)
    ,mRadius(10)
    ,mHasMatrixChange(true)
    ,mDeferredDraw(false)
    ,mFreelineEffect(NULL)
{
    GLMesh::AttributeDesc attrs[] = {GLMesh::AttributeDesc(GLMesh::GLAttributeTypePosition, 2),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeTexcoord, 2),
        GLMesh::AttributeDesc(GLMesh::GLAttributeTypeColor, 4)};
    mMesh = new GLMesh(10000, 20000, attrs, sizeof(attrs) / sizeof(GLMesh::AttributeDesc));
    mTempTarget = mRenderer->GetTempTarget();
    mSrcTexture = mRenderer->GetDefaultTexture();
    mSrcTexture2 = NULL;
    mTarget = mRenderer->GetDefaultTarget();

    mFreelineEffect = new FreelineEffect(mRenderer);
}

Canvas2d::~Canvas2d()
{
    delete mFSAABlitTarget;
    delete mFSAATarget;
    delete mAATarget;
    delete mAATargetHalf;
    delete mMesh;
    delete mFreelineEffect;
}

void Canvas2d::SetState(const Canvas2dState& state)
{
    mTarget = state.target;
    mSrcTexture = state.texture;
    mSrcTexture2 = state.texture2;
    mColor = state.color;
    mTransform = state.transform;
    mBlendMode = state.blendMode;
    mAAMode = state.aaMode;
    mSamples = state.samples;
    mRadius = state.radius;
}

void Canvas2d::BeginBatchDraw()
{
    mDeferredDraw = true;
}

void Canvas2d::EndBatchDraw()
{
    Flush();
    mDeferredDraw = false;
}

void Canvas2d::Draw(float* positions, float* texcoords, float* colors, int vertexNum, unsigned short* indices, int indexNum)
{
    if (vertexNum + mMesh->GetVertexCount() > mMesh->GetVertexCapcity() ||
        indexNum + mMesh->GetIndexCount() > mMesh->GetIndexCapacity())
    {
        Flush();
    }

    int base = mMesh->GetVertexCount();
    for (int i = 0; i < vertexNum; ++i)
    {
        mMesh->SetFloat2(GLMesh::GLAttributeTypePosition, base + i, positions[i * 2], positions[i * 2 + 1]);
        mMesh->SetFloat2(GLMesh::GLAttributeTypeTexcoord, base + i, texcoords[i * 2], texcoords[i * 2 + 1]);
        mMesh->SetFloat4(GLMesh::GLAttributeTypeColor, base + i, colors[i * 4], colors[i * 4 + 1], colors[i * 4 + 2], colors[i * 4 + 3]);

		mBounds.Union(Vector2(positions[i * 2], positions[i * 2 + 1]));
    }
	mMesh->mVertexCount += vertexNum;
    for (int i = 0; i < indexNum; ++i)
    {
        mMesh->SetIndex((unsigned short)base + indices[i]);
    }

    if (!mDeferredDraw)
    {
        Flush();
    }
}

//#define DEBUG_BLIT

void Canvas2d::RenderTile(int /*w*/, int /*h*/, int x, int y, int tw, int th, int samples)
{
    mFSAATarget->Blit(mTarget, 0, 0, tw * samples, th * samples
        , x, y, tw, th, false);

#ifdef DEBUG_BLIT
    mFSAATarget->Save("d:/tmp/fsaa_old.png");
#endif

    mMv = Matrix4::BuildTranslate(mTransform.GetTranslate().x, mTransform.GetTranslate().y, 0)
        * Matrix4::BuildRotate(mTransform.GetRotate(), 0, 0, 1)
        * Matrix4::BuildScale(mTransform.GetScale().x, mTransform.GetScale().y, 1);

    mMvp = Matrix4::BuildOrtho(0, (float)mFSAATarget->GetWidth(), 0, (float)mFSAATarget->GetHeight(), -100.0f, 100.0f)
        * Matrix4::BuildScale((float)samples, (float)samples, 1.0f) * Matrix4::BuildTranslate((float)-x, (float)-y, 0) * mMv;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ShapeEffect* fx = mRenderer->GetShapeEffect();
    fx->Bind();
    fx->SetColor(mColor);
    fx->SetMvp(mMvp);
    fx->SetTexture(mSrcTexture);

    mFSAATarget->Bind();
    mMesh->Draw();

#ifdef DEBUG_BLIT
    mFSAATarget->Save("d:/tmp/fsaa_new.png");
#endif
	if (samples > 2)
	{
        int pass = (int)(log((double)samples) / log(2.0)) - 1;
        GLRenderTarget* ts[2] = { mFSAABlitTarget, mFSAATarget };
        int s = samples / 2;
        int i = 0;
        for (; i < pass; ++i)
        {
            ts[i % 2]->Blit(ts[(i + 1) % 2], 0, 0, tw * s, th * s, 0, 0, tw * s * 2, th * s * 2, true);
#ifdef DEBUG_BLIT
            char pathBuf[260];
            sprintf(pathBuf, "d:/tmp/pass_%03d.png", i);
            ts[i % 2]->Save(pathBuf);
#endif
            s /= 2;
        }
        mTarget->Blit(ts[(i - 1) % 2], x, y, tw, th, 0, 0, tw * s * 2, th * s * 2, true);
	}
	else
	{
		mTarget->Blit(mFSAATarget, x, y, tw, th, 0, 0, tw * samples, th * samples, true);
	}
#ifdef DEBUG_BLIT
    mTarget->Save("d:/tmp/final.png");
#endif
}

void Canvas2d::Flush()
{
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_STENCIL_TEST);

    if (mAAMode == AntiAliasingModeFSAA)
    {
        int targetSize = 2048;
		int tileSizeX = targetSize;
		int tileSizeY = targetSize;
        int w = mTarget->GetWidth();
        int h = mTarget->GetHeight();
        int samples = mSamples;

		Matrix3 mat;
		mTransform.GetMatrix(mat);
		AABB2 aabb = mat.Transform(mBounds);
		aabb.xMin = (float)floor(aabb.xMin);
		aabb.yMin = (float)floor(aabb.yMin);
		aabb.xMax = (float)ceil(aabb.xMax);
		aabb.yMax = (float)ceil(aabb.yMax);

		aabb = aabb.Intersect(AABB2(0, 0, (float)w, (float)h));
		if (aabb.Width() > 0 && aabb.Height() > 0)
		{
			if (samples > 256)
			{
				samples = 256;
			}

			int rsX = (int)aabb.Width() * samples;
			int rsY = (int)aabb.Height() * samples;

			if (targetSize > rsX)
			{
				tileSizeX = rsX;
			}
			if (targetSize > rsY)
			{
				tileSizeY = rsY;
			}

			if (tileSizeX * mSamples > targetSize)
			{
				tileSizeX = targetSize / mSamples;
			}
			if (tileSizeY * mSamples > targetSize)
			{
				tileSizeY = targetSize / mSamples;
			}

			if (mFSAATarget == NULL)
			{
				mFSAATarget = mRenderer->CreateTarget(targetSize, targetSize, false, false);
				mFSAABlitTarget = mRenderer->CreateTarget(targetSize, targetSize, false, false);
			}

			int xTiles = ((int)aabb.Width() + tileSizeX - 1) / tileSizeX;
			int yTiles = ((int)aabb.Height() + tileSizeY - 1) / tileSizeY;

			int tCount = 0;

			for (int y = 0; y < yTiles; ++y)
			{
				for (int x = 0; x < xTiles; ++x)
				{
					int tx = (int)aabb.xMin + x * tileSizeX;
					int ty = (int)aabb.yMin + y * tileSizeY;
                    int px = tx + tileSizeX > w ? w : tx + tileSizeX;
                    int py = ty + tileSizeY > h ? h : ty + tileSizeY;

					RenderTile(w, h, tx, ty, px - tx, py - ty, samples);
					++tCount;
				}
			}

            //LOGE("DrawTiles %d/%d\n", tCount, xTiles * yTiles);
		}
    }
    else
    {
		mMv = Matrix4::BuildTranslate(mTransform.GetTranslate().x, mTransform.GetTranslate().y, 0)
			* Matrix4::BuildRotate(mTransform.GetRotate(), 0, 0, 1)
			* Matrix4::BuildScale(mTransform.GetScale().x, mTransform.GetScale().y, 1);
        
        if (mAAMode == AntiAliasingModeMSAA)
        {
            if (mAATarget && (mAATarget->GetWidth() != mTarget->GetWidth() || mAATarget->GetHeight() != mTarget->GetHeight()))
            {
                delete mAATarget;
                mAATarget = NULL;
            }
            if (mAATarget == NULL)
            {
                mAATarget = mRenderer->CreateTarget(mTarget->GetWidth(), mTarget->GetHeight(), false, true);
            }
            mMvp = Matrix4::BuildOrtho(0, (float)mAATarget->GetWidth(), 0, (float)mAATarget->GetHeight(), -100, 100) * mMv;
            mAATarget->Blit(mTarget, 0, 0, mAATarget->GetWidth(), mAATarget->GetHeight()
                , 0, 0, mTarget->GetWidth(), mTarget->GetHeight(), false);
        }
        else
        {
            mMvp = Matrix4::BuildOrtho(0, (float)mTarget->GetWidth(), 0, (float)mTarget->GetHeight(), -100, 100) * mMv;
        }

        switch (mBlendMode)
        {
        case BlendModeMix:
            {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                ShapeEffect* fx = mRenderer->GetShapeEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetTexture(mSrcTexture);
            }
            break;
        case BlendModeGreyscaleMix:
            {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                MixGreyscaleEffect* fx = mRenderer->GetMixGreyscaleEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetTexture(mSrcTexture);
            }
            break;
        case BlendModeSource:
            {
                glDisable(GL_BLEND);
                ShapeEffect* fx = mRenderer->GetShapeEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetTexture(mSrcTexture);
            }
            break;
        case BlendModeAdd:
            {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                ShapeEffect* fx = mRenderer->GetShapeEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetTexture(mSrcTexture);
            }
            break;
        case BlendModeColorOverride:
            {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                ColorOverrideEffect* fx = mRenderer->GetColorOverrideEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetTexture(mSrcTexture);
            }
            break;
        case BlendModeFXAA:
            {
                glDisable(GL_BLEND);
                FXAAEffect* fx = mRenderer->GetFXAAEffect();
                fx->Bind();
                fx->SetMvp(mMvp);
                fx->SetTexture(mSrcTexture);
            }
            break;
        case BlendModeTransparentMix:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMixEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentBehind:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetBehindEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentErase:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetEraseEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());

            }
            break;
        case BlendModeInvert:
        case BlendModeTransparentInvert:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetInvertEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMask:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentCut:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetCutEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeMaskMix:
        case BlendModeTransparentMaskMix:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskMixEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskBehind:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskBehindEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskErase:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskEraseEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskAdd:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskAddEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskMultiply:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskMultiplyEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeMaskInverseMix://TODO: non-transparent to speed up
        case BlendModeTransparentMaskInverseMix:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskInverseMixEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskMask:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskMaskEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskCut:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetMaskCutEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentBlur:// @TODO: using separate filter
        case BlendModeTransparentMaskBlur:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                MaskBlurEffect* fx = mRenderer->GetMaskBlurEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
                fx->SetRadius(mRadius);
            }
            break;
        case BlendModeTransparentReplaceAlpha:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                BlendEffect* fx = mRenderer->GetReplaceAlhpaEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskExpandMix:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                MaskExpandEffect* fx = mRenderer->GetMaskExpandMixEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskExpandBehind:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                MaskExpandEffect* fx = mRenderer->GetMaskExpandBehindEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskExpandErase:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                MaskExpandEffect* fx = mRenderer->GetMaskExpandEraseEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskExpandAdd:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                MaskExpandEffect* fx = mRenderer->GetMaskExpandAddEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentMaskExpandMultiply:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                MaskExpandEffect* fx = mRenderer->GetMaskExpandMultiplyEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentSoftenMaskMix:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                SoftenMaskEffect* fx = mRenderer->GetSoftenMaskMixEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentSoftenMaskBehind:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                SoftenMaskEffect* fx = mRenderer->GetSoftenMaskBehindEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentSoftenMaskErase:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                SoftenMaskEffect* fx = mRenderer->GetSoftenMaskEraseEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentSoftenMaskAdd:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                SoftenMaskEffect* fx = mRenderer->GetSoftenMaskAddEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModeTransparentSoftenMaskMultiply:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                SoftenMaskEffect* fx = mRenderer->GetSoftenMaskMultiplyEffect();
                fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
		case BlendModePalettePresent:
			{
                mTempTarget->Blit(mTarget, 0, 0, false);
				glDisable(GL_BLEND);
				PaletteEffect* fx = mRenderer->GetPaletteEffect();
				fx->Bind();
                fx->SetColor(mColor);
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetPaletteTexture(mSrcTexture2);
                fx->SetDstTexture(mTempTarget->GetTexture());
			}
			break;
        case BlendModePaletteMix:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                PaletteMixEffect* fx = mRenderer->GetPaletteMixEffect();
                fx->Bind();
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        case BlendModePaletteAdd:
            {
                mTempTarget->Blit(mTarget, 0, 0, false);
                glDisable(GL_BLEND);
                PaletteAddEffect* fx = mRenderer->GetPaletteAddEffect();
                fx->Bind();
                fx->SetMvp(mMvp);
                fx->SetMv(mMv);
                fx->SetSrcTexture(mSrcTexture);
                fx->SetDstTexture(mTempTarget->GetTexture());
            }
            break;
        default:
            break;
        }

        if (mAAMode == AntiAliasingModeMSAA)
        {
            mAATarget->Bind();
            mMesh->Draw();
            mTarget->Blit(mAATarget, 0, 0, mTarget->GetWidth(), mTarget->GetHeight()
                ,0, 0, mAATarget->GetWidth(), mAATarget->GetHeight(), true);
        }
        else
        {
            mTarget->Bind();
            mMesh->Draw();
        }
    }
    
    mMesh->Clear();
	mBounds.SetNull();
}

void Canvas2d::DrawCircle(float x, float y, float r, const Color& color)
{
    if (r <= 0.0f)
    {
        return;
    }

    const float d = 0.2f;
    // angle = PI * 2 / samples
    // d = r - r * cosf(angle/2)
    int samples = (int)ceilf(PI / acosf((r - d) / r));
    if (samples < 32)
    {
        samples = 32;
    }

    int N = samples + 1;
    float* pos = new float[2 * N];
    float* texcoords = new float[2 * N];
    float* colors = new float[4 * N];
    unsigned short* indices = new unsigned short[samples * 3];
    unsigned short* pI = indices;

    pos[0] = x;
    pos[1] = y;

    texcoords[0] = 0;
    texcoords[1] = 0;

    colors[0] = color.r;
    colors[1] = color.g;
    colors[2] = color.b;
    colors[3] = color.a;

    for (int i = 0; i < samples; ++i)
    {
        float angle = i * PI * 2 / samples;
        float rx = x + r * cosf(angle);
        float ry = y + r * sinf(angle);
		pos[2 * (1 + i)] = rx;
		pos[2 * (1 + i) + 1] = ry;

		texcoords[2 * (1 + i) + 0] = 0;
		texcoords[2 * (1 + i) + 1] = 0;

		colors[4 * (1 + i) + 0] = color.r;
		colors[4 * (1 + i) + 1] = color.g;
		colors[4 * (1 + i) + 2] = color.b;
		colors[4 * (1 + i) + 3] = color.a;

        pI[0] = 0;
        pI[1] = (unsigned short)(1 + i);
        pI[2] = (unsigned short)(1 + ((i + 1) % samples));
		pI += 3;
    }

	Draw(pos, texcoords, colors, N, indices, samples * 3);

    delete[] indices;
    delete[] colors;
    delete[] texcoords;
    delete[] pos;
}

void Canvas2d::DrawGradientCircle(float x, float y, float r, const Color* stopColors, float* stops, int stopNum)
{
    if (r <= 0.0f)
    {
        return;
    }

    const float d = 0.2f;
    // angle = PI * 2 / samples
    // d = r - r * cosf(angle/2)
    int samples = (int)ceilf(PI / acosf((r - d) / r));
    if (samples < 3)
    {
        samples = 3;
    }

    int N = samples * stopNum + 1;
    int NI = samples * ((stopNum - 1) * 2 + 1) * 3;
    float* pos = new float[2 * N];
    float* texcoords = new float[2 * N];
    float* colors = new float[4 * N];
    unsigned short* indices = new unsigned short[NI];
    unsigned short* pI = indices;

    pos[0] = x;
    pos[1] = y;

    texcoords[0] = 0;
    texcoords[1] = 0;

    colors[0] = stopColors[0].r;
    colors[1] = stopColors[0].g;
    colors[2] = stopColors[0].b;
    colors[3] = stopColors[0].a;
    int iv = 1;
    for (int j = 0; j < stopNum; ++j)
    {
        const Color& c = stopColors[j];
        float s = stops[j];
        for (int i = 0; i < samples; ++i)
        {
            float angle = i * PI * 2 / samples;
            float rx = x + r * cosf(angle) * s;
            float ry = y + r * sinf(angle) * s;
            int ivd = iv + i;
            pos[2 * ivd + 0] = rx;
            pos[2 * ivd + 1] = ry;

            texcoords[2 * ivd + 0] = 0;
            texcoords[2 * ivd + 1] = 0;

            colors[4 * ivd + 0] = c.r;
            colors[4 * ivd + 1] = c.g;
            colors[4 * ivd + 2] = c.b;
            colors[4 * ivd + 3] = c.a;

            if (j == 0)
            {
                pI[0] = 0;
                pI[1] = (unsigned short)(1 + i);
                pI[2] = (unsigned short)(1 + ((i + 1) % samples));
                pI += 3;
            }
            else
            {
                int i0 = iv + i;
                int i1 = iv + ((i + 1) % samples);
                int i2 = i1 - samples;
                int i3 = i0 - samples;
                pI[0] = (unsigned short)(i0);
                pI[1] = (unsigned short)(i1);
                pI[2] = (unsigned short)(i2);
                pI[3] = (unsigned short)(i0);
                pI[4] = (unsigned short)(i2);
                pI[5] = (unsigned short)(i3);
                pI += 6;
            }

        }
        iv += samples;
    }

    Draw(pos, texcoords, colors, N, indices, NI);

    delete[] indices;
    delete[] colors;
    delete[] texcoords;
    delete[] pos;
}

void Canvas2d::DrawRect(float x, float y, float w, float h, const Color& color)
{
    const int N = 4;
    float pos[2 * N] = {
        x,     y, 
        x + w, y, 
        x + w, y + h, 
        x,     y + h};

    float texcoords[2 * N] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0
    };

    float colors[4 * N] = {
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a};

    unsigned short indices[6] = {
        0, 1, 2, 
        0, 2, 3};

    Draw(pos, texcoords, colors, N, indices, 6);
}

void Canvas2d::DrawRectOutline(const AABB2& rect, const Color& color, float outlineWidth, OutlineDirection direction)
{
    float hw = outlineWidth * 0.5f;
    float x0, x1, y0, y1;
    switch (direction)
    {
    case Canvas2d::OutlineDirectionOuter:
        x0 = rect.xMin - hw;
        x1 = rect.xMax + hw;
        y0 = rect.yMin - hw;
        y1 = rect.yMax + hw;
        break;
    case Canvas2d::OutlineDirectionInner:
        x0 = rect.xMin + hw;
        x1 = rect.xMax - hw;
        y0 = rect.yMin + hw;
        y1 = rect.yMax - hw;
        break;
    case Canvas2d::OutlineDirectionCenter:
    default:
        x0 = rect.xMin;
        x1 = rect.xMax;
        y0 = rect.yMin;
        y1 = rect.yMax;
        break;
    }

    DrawLine(x0 - hw, y0, outlineWidth, x1 + hw, y0, outlineWidth, color);
    DrawLine(x0 - hw, y1, outlineWidth, x1 + hw, y1, outlineWidth, color);
    DrawLine(x0, y0 + hw, outlineWidth, x0, y1 - hw, outlineWidth, color);
    DrawLine(x1, y0 + hw, outlineWidth, x1, y1 - hw, outlineWidth, color);
}

void Canvas2d::DrawLine(float x0, float y0, float w0, float x1, float y1, float w1, const Color& color)
{
    Vector2 p0(x0, y0);
    Vector2 p1(x1, y1);
    Vector2 dirL = p1 - p0;
    Vector2 dirW = dirL.GetPerpendicular();

    Vector2 vs[4] = {p0, p1, p1, p0};

    if (dirW.LengthSq() > 0)
    {
        dirW.Normalise();
        Vector2 d0 = dirW * (w0 * 0.5f);
        Vector2 d1 = dirW * (w1 * 0.5f);
        vs[0] -= d0;
        vs[1] -= d1;
        vs[2] += d1;
        vs[3] += d0;
    }

    const int N = 4;
    float pos[2 * N] = {
        vs[0].x, vs[0].y,
        vs[1].x, vs[1].y,
        vs[2].x, vs[2].y,
        vs[3].x, vs[3].y
    };

    float texcoords[2 * N] = {
        0, 0,
        0, 0,
        0, 0,
        0, 0
    };

    float colors[4 * N] = {
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a};

    unsigned short indices[6] = {
        0, 1, 2, 
        0, 2, 3
    };

    Draw(pos, texcoords, colors, N, indices, 6);
}

void Canvas2d::DrawLine(float x0, float y0, float x1, float y1, float width, const Color& color)
{
    DrawLine(x0, y0, width, x1, y1, width, color);
}

void Canvas2d::DrawLineJoint(float /*x0*/, float /*y0*/, float x1, float y1, float /*x2*/, float /*y2*/, float w, const Color& color)
{
    DrawCircle(x1, y1, w * 0.5f, color);
//    Vector2 p0(x0, y0);
//    Vector2 p1(x1, y1);
//    Vector2 p2(x2, y2);
//    Vector2 dir0 = (p1 - p0).GetPerpendicular();
//    Vector2 dir1 = (p2 - p1).GetPerpendicular();

//    Vector2 vs[4] = {p1, p1, p1, p1};

//    if (dir0.LengthSq() > 0 && dir1.LengthSq() > 0)
//    {
//        dir0.Normalise();
//        dir1.Normalise();
//        Vector2 d0 = dir0 * (w * 0.5f);
//        Vector2 d1 = dir1 * (w * 0.5f);
//        vs[0] -= d1;
//        vs[1] -= d0;
//        vs[2] += d1;
//        vs[3] += d0;
//    }

//    const int N = 4;
//    float pos[2 * N] = {
//        vs[0].x, vs[0].y,
//        vs[1].x, vs[1].y,
//        vs[2].x, vs[2].y,
//        vs[3].x, vs[3].y
//    };

//    float texcoords[2 * N] = {
//        0, 0,
//        0, 0,
//        0, 0,
//        0, 0
//    };

//    float colors[4 * N] = {
//        color.r, color.g, color.b, color.a,
//        color.r, color.g, color.b, color.a,
//        color.r, color.g, color.b, color.a,
//        color.r, color.g, color.b, color.a};

//    unsigned short indices[6] = {
//        0, 1, 2,
//        0, 2, 3};

//    Draw(pos, texcoords, colors, N, indices, 6);
}

void Canvas2d::DrawImage(float x, float y, AABB2 srcRect, const Color& color)
{
    if (!mSrcTexture || !mTarget)
    {
        return;
    }

    float w = srcRect.Width();
    float h = srcRect.Height();

    x += srcRect.xMin;
    y += srcRect.yMin;
    const int N = 4;
    float pos[2 * N] = {
        x,     y,
        x + w, y, 
        x + w, y + h, 
        x,     y + h};

    float sw = (float)mSrcTexture->GetWidth();
    float sh = (float)mSrcTexture->GetHeight();
    AABB2 st(srcRect.xMin / sw, srcRect.yMin / sh, srcRect.xMax / sw, srcRect.yMax / sh);
    float texcoords[4 * N] = {
        st.xMin, st.yMin,
        st.xMax, st.yMin,
        st.xMax, st.yMax,
        st.xMin, st.yMax
    };

    float colors[4 * N] = {
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a};

    unsigned short indices[6] = {
        0, 1, 2, 
        0, 2, 3};

    Draw(pos, texcoords, colors, N, indices, 6);
}

void Canvas2d::DrawImage(AABB2 dstRect, AABB2 srcRect, const Color& color)
{
    if (!mSrcTexture || !mTarget)
    {
        return;
    }

    const int N = 4;
    float pos[2 * N] = {
        dstRect.xMin, dstRect.yMin,
        dstRect.xMax, dstRect.yMin,
        dstRect.xMax, dstRect.yMax,
        dstRect.xMin, dstRect.yMax};

    float sw = (float)mSrcTexture->GetWidth();
    float sh = (float)mSrcTexture->GetHeight();
    AABB2 st(srcRect.xMin / sw, srcRect.yMin / sh, srcRect.xMax / sw, srcRect.yMax / sh);
    float texcoords[4 * N] = {
        st.xMin, st.yMin,
        st.xMax, st.yMin,
        st.xMax, st.yMax,
        st.xMin, st.yMax
    };

    float colors[4 * N] = {
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a};

    unsigned short indices[6] = {
        0, 1, 2,
        0, 2, 3};

    Draw(pos, texcoords, colors, N, indices, 6);
}

void Canvas2d::DrawImage(AABB2 dstRect, AABB2 srcRect, const Color& color, const Transform2& trans)
{
    if (!mSrcTexture || !mTarget)
    {
        return;
    }

    const int N = 4;
    float pos[2 * N] = {
        dstRect.xMin, dstRect.yMin,
        dstRect.xMax, dstRect.yMin,
        dstRect.xMax, dstRect.yMax,
        dstRect.xMin, dstRect.yMax};

    Matrix3 mat;
    trans.GetMatrix(mat);
    mat.Transform((Vector2*)pos, 4);

    float sw = (float)mSrcTexture->GetWidth();
    float sh = (float)mSrcTexture->GetHeight();
    AABB2 st(srcRect.xMin / sw, srcRect.yMin / sh, srcRect.xMax / sw, srcRect.yMax / sh);
    float texcoords[4 * N] = {
        st.xMin, st.yMin,
        st.xMax, st.yMin,
        st.xMax, st.yMax,
        st.xMin, st.yMax
    };

    float colors[4 * N] = {
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a,
        color.r, color.g, color.b, color.a};

    unsigned short indices[6] = {
        0, 1, 2,
        0, 2, 3};

    Draw(pos, texcoords, colors, N, indices, 6);
}

void Canvas2d::DrawPolygon(const Vector2* vertices, int num, const Color& color)
{
#ifdef __glu_h__
    TriangulatePolygonList pl;
    pl.positions = vertices;
    pl.count = num;
    pl.texcoords = NULL;
    std::vector<float> vs;
    std::vector<unsigned short> is;
    Triangulation(&pl, 1, vs, is);
    if (is.size() == 0)
    {
        return;
    }

    int N = vs.size() / 2;
    float* texcoords = new float[2 * N];
    float* colors = new float[4 * N];

    for (int i = 0; i < N; ++i)
    {
        texcoords[2 * i + 0] = 0;
        texcoords[2 * i + 1] = 0;

        colors[4 * i + 0] = color.r;
        colors[4 * i + 1] = color.g;
        colors[4 * i + 2] = color.b;
        colors[4 * i + 3] = color.a;
    }

    if (N > mMesh->GetVertexCapcity())
    {
        for (int i = 0; i < N; i += mMesh->GetVertexCapcity())
        {
            int num = N - i;
            if (num > mMesh->GetVertexCapcity())
            {
                num = mMesh->GetVertexCapcity();
            }
            Draw(&vs[i * 2], texcoords + i * 2, colors + i * 4, num, &is[i], num);
        }
    }
    else
    {
        Draw(&vs[0], texcoords, colors, N, &is[0], (int)is.size());
    }
    

    delete[] colors;
    delete[] texcoords;
#else

#ifndef NO_TESS2
    std::vector<float> vs;
    std::vector<unsigned short> is;
    ToTri((float*)vertices, num, vs, is);
    if (vs.size() == 0)
    {
        return;
    }

    int N = vs.size() / 2;
    float* texcoords = new float[2 * N];
    float* colors = new float[4 * N];

    for (int i = 0; i < N; ++i)
    {
        texcoords[2 * i + 0] = 0;
        texcoords[2 * i + 1] = 0;

        colors[4 * i + 0] = color.r;
        colors[4 * i + 1] = color.g;
        colors[4 * i + 2] = color.b;
        colors[4 * i + 3] = color.a;
    }

    if (N > mMesh->GetVertexCapcity())
    {
        for (int i = 0; i < N; i += mMesh->GetVertexCapcity())
        {
            int num = N - i;
            if (num > mMesh->GetVertexCapcity())
            {
                num = mMesh->GetVertexCapcity();
            }
            Draw(&vs[i * 2], texcoords + i * 2, colors + i * 4, num, &is[i], num);
        }
    }
    else
    {
        Draw(&vs[0], texcoords, colors, N, &is[0], (int)is.size());
    }


    delete[] colors;
    delete[] texcoords;
#endif

#endif
}

void Canvas2d::DrawFreeline(Vector3* vertices, int num)
{
    Flush();

    mMv = Matrix4::BuildTranslate(mTransform.GetTranslate().x, mTransform.GetTranslate().y, 0)
        * Matrix4::BuildRotate(mTransform.GetRotate(), 0, 0, 1)
        * Matrix4::BuildScale(mTransform.GetScale().x, mTransform.GetScale().y, 1);
    mMvp = Matrix4::BuildOrtho(0, (float)mTarget->GetWidth(), 0, (float)mTarget->GetHeight(), -100, 100) * mMv;

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    mFreelineEffect->Bind();
    mFreelineEffect->SetMvp(mMvp);
    mFreelineEffect->SetColor(mColor);
    mTarget->Bind();

    int batchSize = 1;
    for (int i = 0; i < num - 1; i++)
    {
        int n = num - i * batchSize;
        if (n > batchSize)
        {
            n = batchSize;
        }
        for (int j = 0; j < n - 1; ++j)
        {
            Vector3 p0 = vertices[i + j];
            Vector3 p1 = vertices[i + j + 1];
            //DrawLine(p0.x, p0.y, p0.z + 2.0f, p1.x, p1.y, p1.z + 2.0f, Color(1,1,1,1));
            p0.z += 10.0f;
            p1.z += 10.0f;
            mMesh->AddLineV(p0, p1, Color((float)j,1,1,1));
        }
  //  	Vector3 pts[2];
  //  	Vector3& p0 = pts[0];
  //  	p0 = vertices[i];
		//Vector3& p1 = pts[1];
		//p1 = vertices[i + 1];
		////DrawLine(p0.x, p0.y, p0.z + 2.0f, p1.x, p1.y, p1.z + 2.0f, Color(1,1,1,1));
		//p0.z += 100.0f;
		//p1.z += 100.0f;
		//mMesh->AddFreeLine(pts, 2, Color((float)i,1,1,1));
  //      mFreelineEffect->SetPoints(vertices + i, 2);
  //      mMesh->Draw();
  //      mMesh->Clear();
    }

    mBounds.SetNull();
}

void Canvas2d::DrawString(const char* text, float xPos, float yPos, float fontHeight, const Color& color)
{
    GLMesh* mesh = mRenderer->CreateText(text, xPos, yPos, fontHeight, color);
    ShapeEffect* fx = mRenderer->GetShapeEffect();
    fx->Bind();
    Matrix4 mvp = Matrix4::BuildOrtho(0, (float)mTarget->GetWidth(), 0, (float)mTarget->GetHeight(), -100, 100);
    fx->SetMvp(mvp);
    fx->SetColor(Color(1, 1, 1, 1));
    fx->SetTexture(mRenderer->GetFontTexture());
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    mTarget->Bind();
    mesh->Draw();
    delete mesh;
}

Effect3dUnlighted::Effect3dUnlighted()
    :mMvpLoc(-1)
    ,mTexDiffuseLoc(-1)
    ,mColorLoc(-1)
    ,mTexDiffuse(NULL)
    ,mColor(1,1,1,1)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec3 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "out vec2 uvOut;\n"
        "void main() {\n"
        "	uvOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform vec4 objectColor;\n"
        "in vec2 uvOut;\n"
        "void main() {\n"
        "   FragColor = texture(texDiffuse, uvOut) * objectColor;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec3 position;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "out vec2 uvOut;\n"
        "void main() {\n"
        "	uvOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform vec4 objectColor;\n"
        "in vec2 uvOut;\n"
        "void main() {\n"
        "   FragColor = texture(texDiffuse, uvOut) * objectColor;\n"
        "}\n";
#else
    const char* vs =
        "attribute vec3 position;\n"
        "attribute vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "varying vec2 uvOut;\n"
        "void main() {\n"
        "	uvOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform vec4 objectColor;\n"
        "varying vec2 uvOut;\n"
        "void main() {\n"
        "   gl_FragColor = texture2D(texDiffuse, uvOut) * objectColor;\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mTexDiffuseLoc = mProgram->GetUniformLocation("texDiffuse");
    mColorLoc = mProgram->GetUniformLocation("objectColor");

    mProgram->Unbind();
}

Effect3dUnlighted::~Effect3dUnlighted()
{
    delete mProgram;
}

void Effect3dUnlighted::Bind()
{
    Matrix4 mvp = mProjMat * mViewMat * mWorldMat;
    mProgram->Bind();
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
    mProgram->SetTexture(mTexDiffuseLoc, 0, mTexDiffuse);
    mProgram->SetFloat4(mColorLoc, mColor.r, mColor.g, mColor.b, mColor.a);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
}

void Effect3dUnlighted::SetProjectionMatrix(const Matrix4& mat)
{
    mProjMat = mat;
}

void Effect3dUnlighted::SetViewMatrix(const Matrix4& mat)
{
    mViewMat = mat;
}

void Effect3dUnlighted::SetWorldMatrix(const Matrix4& mat)
{
    mWorldMat = mat;
}

void Effect3dUnlighted::SetTextures(GLTexture** textures)
{
    mTexDiffuse = textures[0];
}

void Effect3dUnlighted::SetColor(const Color& color)
{
    mColor = color;
}

Effect3dForwardShading::Effect3dForwardShading()
    :mMvpLoc(-1)
    ,mMvLoc(-1)
    ,mTexDiffuseLoc(-1)
    ,mLightDir(-1)
    ,mLightColor(-1)
    ,mAmbientFactor(-1)
    ,mShiness(-1)
    ,mTexDiffuse(NULL)
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec3 position;\n"
        "in vec2 texcoord;\n"
        "in vec3 normal;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform mat4 normalMat;\n"
        "out vec3 positionOut;\n"
        "out vec3 normalOut;\n"
        "out vec2 uvOut;\n"
        "void main() {\n"
        "   positionOut = (mv * vec4(position, 1.0)).xyz;\n"
        "   vec3 N = (normalMat * vec4(normal, 1.0)).xyz;\n"
        "   normalOut = normalize(N);\n"
        "	uvOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 lightColor;\n"
        "uniform float ambientFactor;\n"
        "uniform float shiness;\n"
        "in vec3 positionOut;\n"
        "in vec3 normalOut;\n"
        "in vec2 uvOut;\n"
        "void main() {\n"
        "   vec3 P = positionOut;\n"
        "   vec3 N = normalOut;\n"
        "   vec3 L = normalize(-lightDir);\n"
        "   vec3 E = normalize(-P);\n"
        "   vec4 D = texture(texDiffuse, uvOut) * vec4(lightColor * max(0.0, dot(N,L) * (1.0 - ambientFactor) + ambientFactor), 1.0);\n"
        "   float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
        "   vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "   FragColor = mix(D,S,specular);\n"
        //"   FragColor = vec4(uvOut, 0.0, 1.0);\n"
        //"   FragColor = vec4(N * 0.5 + 0.5, 1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "precision highp float;\n"
        "in vec3 position;\n"
        "in vec2 texcoord;\n"
        "in vec3 normal;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform mat4 normalMat;\n"
        "out vec3 positionOut;\n"
        "out vec3 normalOut;\n"
        "out vec2 uvOut;\n"
        "void main() {\n"
        "   positionOut = (mv * vec4(position, 1.0)).xyz;\n"
        "   vec3 N = (normalMat * vec4(normal, 1.0)).xyz;\n"
        "   normalOut = normalize(N);\n"
        "	uvOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 lightColor;\n"
        "uniform float ambientFactor;\n"
        "uniform float shiness;\n"
        "in vec3 positionOut;\n"
        "in vec3 normalOut;\n"
        "in vec2 uvOut;\n"
        "void main() {\n"
        "   vec3 P = positionOut;\n"
        "   vec3 N = normalOut;\n"
        "   vec3 L = normalize(-lightDir);\n"
        "   vec3 E = normalize(-P);\n"
        "   vec4 D = texture(texDiffuse, uvOut) * vec4(lightColor * max(0.0, dot(N,L) * (1.0 - ambientFactor) + ambientFactor), 1.0);\n"
        "   float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
        "   vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "   FragColor = mix(D,S,specular);\n"
        //"   FragColor = vec4(uvOut, 0.0, 1.0);\n"
        //"   FragColor = vec4(N * 0.5 + 0.5, 1.0);\n"
        "}\n";
#else
    const char* vs =
        "attribute vec3 position;\n"
        "attribute vec2 texcoord;\n"
        "attribute vec3 normal;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform mat4 normalMat;\n"
        "varying vec3 positionOut;\n"
        "varying vec3 normalOut;\n"
        "varying vec2 uvOut;\n"
        "void main() {\n"
        "   positionOut = (mv * vec4(position, 1.0)).xyz;\n"
        "   vec3 N = (normalMat * vec4(normal, 1.0)).xyz;\n"
        "   normalOut = normalize(N);\n"
        "	uvOut = texcoord;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 lightColor;\n"
        "uniform float ambientFactor;\n"
        "uniform float shiness;\n"
        "varying vec3 positionOut;\n"
        "varying vec3 normalOut;\n"
        "varying vec2 uvOut;\n"
        "void main() {\n"
        "   vec3 P = positionOut;\n"
        "   vec3 N = normalOut;\n"
        "   vec3 L = normalize(-lightDir);\n"
        "   vec3 E = normalize(-P);\n"
        "   vec4 D = texture2D(texDiffuse, uvOut) * vec4(lightColor * max(0.0, dot(N,L) * (1.0 - ambientFactor) + ambientFactor), 1.0);\n"
        "   float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
        "   vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "   gl_FragColor = mix(D,S,specular);\n"
        //"   gl_FragColor = vec4(uvOut, 0.0, 1.0);\n"
        //"   gl_FragColor = vec4(N * 0.5 + 0.5, 1.0);\n"
        "}\n";
#endif

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["normal"] = GLMesh::GLAttributeTypeNormal;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
	mNormalMatLoc = mProgram->GetUniformLocation("normalMat");
    mTexDiffuseLoc = mProgram->GetUniformLocation("texDiffuse");
    mLightDir = mProgram->GetUniformLocation("lightDir");
    mLightColor = mProgram->GetUniformLocation("lightColor");
    mAmbientFactor = mProgram->GetUniformLocation("ambientFactor");
    mShiness = mProgram->GetUniformLocation("shiness");

    mProgram->Unbind();
}

Effect3dForwardShading::~Effect3dForwardShading()
{
    delete mProgram;
}

void Effect3dForwardShading::Bind()
{
    Matrix4 mv = mViewMat * mWorldMat;
    Matrix4 mvp = mProjMat * mv;
	Matrix4 nm = mv;
	nm.m03 = nm.m13 = nm.m23 = 0.0f;
    mProgram->Bind();
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
    mProgram->SetMatrix(mMvLoc, &mv.m00);
	mProgram->SetMatrix(mNormalMatLoc, &nm.m00);
    mProgram->SetTexture(mTexDiffuseLoc, 0, mTexDiffuse);

    Vector3 lightDirection(-1, 1, -1);
    lightDirection = nm * lightDirection;
    lightDirection.Normalise();
    mProgram->SetFloat3(mLightDir, lightDirection.x, lightDirection.y, lightDirection.z);
    mProgram->SetFloat3(mLightColor, 1, 1, 1);
    mProgram->SetFloat(mAmbientFactor, 0.3f);
    mProgram->SetFloat(mShiness, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Effect3dForwardShading::SetProjectionMatrix(const Matrix4& mat)
{
    mProjMat = mat;
}

void Effect3dForwardShading::SetViewMatrix(const Matrix4& mat)
{
    mViewMat = mat;
}

void Effect3dForwardShading::SetWorldMatrix(const Matrix4& mat)
{
    mWorldMat = mat;
}

void Effect3dForwardShading::SetTextures(GLTexture** textures)
{
    mTexDiffuse = textures[0];
}

Effect3dGPass::Effect3dGPass()
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec3 position;\n"
        "in vec4 color;\n"
        "in vec3 normal;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform mat4 mvInvT;\n"
        "out vec3 posOut;\n"
        "out vec4 colorOut;\n"
        "out vec3 normalOut;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = vec4(color.xyz,1.0);\n"
        "   normalOut = normalize((mvInvT * vec4(normal, 1.0)).xyz);\n"
        "   posOut = (mv * vec4(position, 1.0)).xyz;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "layout (location = 0) out vec4 FragColor;\n"
        "layout (location = 1) out vec4 FragNormal;\n"
        "layout (location = 2) out vec4 FragPosition;\n"
        "uniform sampler2D texDiffuse;\n"
        "in vec3 posOut;\n"
        "in vec4 colorOut;\n"
        "in vec3 normalOut;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = texture(texDiffuse, texcoordOut) * colorOut;\n"
        "   FragNormal = vec4(normalOut, 0.0);\n"
        "   FragPosition = vec4(posOut, 1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec3 position;\n"
        "in vec4 color;\n"
        "in vec3 normal;\n"
        "in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform mat4 mvInvT;\n"
        "out vec3 posOut;\n"
        "out vec4 colorOut;\n"
        "out vec3 normalOut;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = vec4(color.xyz,1.0);\n"
        "   normalOut = normalize((mvInvT * vec4(normal, 1.0)).xyz);\n"
        "   posOut = (mv * vec4(position, 1.0)).xyz;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "layout (location = 0) out vec4 FragColor;\n"
        "layout (location = 1) out vec4 FragNormal;\n"
        "layout (location = 2) out vec4 FragPosition;\n"
        "uniform sampler2D texDiffuse;\n"
        "in vec3 posOut;\n"
        "in vec4 colorOut;\n"
        "in vec3 normalOut;\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "   FragColor = texture(texDiffuse, texcoordOut) * colorOut;\n"
        "   FragNormal = vec4(normalOut, 0.0);\n"
        "   FragPosition = vec4(posOut, 1.0);\n"
        "}\n";
#else
    const char* vs =
        "attribute vec3 position;\n"
        "attribute vec4 color;\n"
        "attribute vec3 normal;\n"
        "attribute vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "uniform mat4 mv;\n"
        "uniform mat4 mvInvT;\n"
        "varying vec3 posOut;\n"
        "varying vec4 colorOut;\n"
        "varying vec3 normalOut;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = texcoord;\n"
        "   colorOut = vec4(color.xyz,1.0);\n"
		"   normalOut = normalize((mvInvT * vec4(normal, 1.0)).xyz);\n"
        "   posOut = (mv * vec4(position, 1.0)).xyz;\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "uniform sampler2D texDiffuse;\n"
        "varying vec3 posOut;\n"
        "varying vec4 colorOut;\n"
        "varying vec3 normalOut;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   gl_FragData[0] = texture2D(texDiffuse, texcoordOut) * colorOut;\n"
		"   gl_FragData[1] = vec4(normalOut, 0.0);\n"
        "   gl_FragData[2] = vec4(posOut, 1.0);\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    bs["color"] = GLMesh::GLAttributeTypeColor;
    bs["texcoord"] = GLMesh::GLAttributeTypeTexcoord;
    bs["normal"] = GLMesh::GLAttributeTypeNormal;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mNormalMatLoc = mProgram->GetUniformLocation("mvInvT");
    mTexDiffuseLoc = mProgram->GetUniformLocation("texDiffuse");

    mProgram->Unbind();
}

Effect3dGPass::~Effect3dGPass()
{
    delete mProgram;
}

void Effect3dGPass::Bind()
{
    Matrix4 mv = mViewMat * mWorldMat;
    Matrix4 mvp = mProjMat * mv;
    //Matrix4 nm = mv.Inverse();
    //nm.Transpose();
    Matrix4 nm = mv;
    nm.m03 = nm.m13 = nm.m23 = 0.0f;

    mProgram->Bind();
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
    mProgram->SetMatrix(mMvLoc, &mv.m00);
    mProgram->SetMatrix(mNormalMatLoc, &nm.m00);
    mProgram->SetTexture(mTexDiffuseLoc, 0, mTexDiffuse);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Effect3dGPass::SetProjectionMatrix(const Matrix4& mat)
{
    mProjMat = mat;
}

void Effect3dGPass::SetViewMatrix(const Matrix4& mat)
{
    mViewMat = mat;
}

void Effect3dGPass::SetWorldMatrix(const Matrix4& mat)
{
    mWorldMat = mat;
}

void Effect3dGPass::SetTextures(GLTexture** textures)
{
    mTexDiffuse = textures[0];
}

Effect3dShadowPass::Effect3dShadowPass()
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec3 position;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   FragColor = vec4(0.0, 0.0, 1.0, 1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec3 position;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   FragColor = vec4(0.0, 0.0, 1.0, 1.0);\n"
        "}\n";
#else
    const char* vs =
        "attribute vec3 position;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "   gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";

    const char* ps =
        "void main() {\n"
        "   gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");

    mProgram->Unbind();
}

Effect3dShadowPass::~Effect3dShadowPass()
{
    delete mProgram;
}

void Effect3dShadowPass::Bind()
{
    Matrix4 mv = mViewMat * mWorldMat;
    Matrix4 mvp = mProjMat * mv;

    mProgram->Bind();
    mProgram->SetMatrix(mMvpLoc, &mvp.m00);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_FRONT);
}

void Effect3dShadowPass::SetProjectionMatrix(const Matrix4& mat)
{
    mProjMat = mat;
}

void Effect3dShadowPass::SetViewMatrix(const Matrix4& mat)
{
    mViewMat = mat;
}

void Effect3dShadowPass::SetWorldMatrix(const Matrix4& mat)
{
    mWorldMat = mat;
}


Effect3dDSDiHemisphereLighting::Effect3dDSDiHemisphereLighting()
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texAO;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 skyColor;\n"
        "uniform vec3 groundColor;\n"
        "in vec2 uv;\n"
        "void main() {\n"
        "   vec3 D = texture(texDiffuse, uv).rgb;\n"
        "   float AO = texture(texAO, uv).r;\n"
        "   vec3 N = texture(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(lightDir);\n"
        "   float NL = dot(N,L);\n"
        "   float a = 0.5 + 0.5 * dot(N,L);\n"
        "   FragColor = vec4(mix(skyColor,groundColor,a) * D * AO, 1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texAO;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 skyColor;\n"
        "uniform vec3 groundColor;\n"
        "in vec2 uv;\n"
        "void main() {\n"
        "   vec3 D = texture(texDiffuse, uv).rgb;\n"
        "   float AO = texture(texAO, uv).r;\n"
        "   vec3 N = texture(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(lightDir);\n"
        "   float NL = dot(N,L);\n"
        "   float a = 0.5 + 0.5 * dot(N,L);\n"
        "   FragColor = vec4(mix(skyColor,groundColor,a) * D * AO, 1.0);\n"
        "}\n";
#else
    const char* vs =
        "attribute vec2 position;\n"
        "varying vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texAO;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 skyColor;\n"
        "uniform vec3 groundColor;\n"
        "varying vec2 uv;\n"
        "void main() {\n"
        "   vec3 D = texture2D(texDiffuse, uv).rgb;\n"
        "   float AO = texture2D(texAO, uv).r;\n"
        "   vec3 N = texture2D(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(lightDir);\n"
        "   float NL = dot(N,L);\n"
        "   float a = 0.5 + 0.5 * dot(N,L);\n"
        "   gl_FragColor = vec4(mix(skyColor,groundColor,a) * D * AO, 1.0);\n"
        "}\n";
#endif

    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mNormalMatLoc = mProgram->GetUniformLocation("mvInvT");
    mTexDiffuseLoc = mProgram->GetUniformLocation("texDiffuse");
    mTexNormalLoc = mProgram->GetUniformLocation("texNormal");
    mTexAOLoc = mProgram->GetUniformLocation("texAO");
    mLightDirLoc = mProgram->GetUniformLocation("lightDir");
    mSkyColorLoc = mProgram->GetUniformLocation("skyColor");
    mGroundColorLoc = mProgram->GetUniformLocation("groundColor");

    SetSkyDirection(Vector3(0, 1, 0));
    mSkyLightColor = Color(0.2f, 0.2f, 0.2f);
    mGroundLightColor = Color(0.1f, 0.1f, 0.1f);
    mProgram->Unbind();
}

Effect3dDSDiHemisphereLighting::~Effect3dDSDiHemisphereLighting()
{
    delete mProgram;
}

void Effect3dDSDiHemisphereLighting::Bind()
{
    Matrix4 mv = mViewMat * mWorldMat;
    Matrix4 mvp = mProjMat * mv;
    //Matrix4 nm = mv.Inverse();
    //nm.Transpose();
    Matrix4 nm = mv;
    nm.m03 = nm.m13 = nm.m23 = 0.0f;
    Vector3 viewLightDir = nm * mLightDirection;
    viewLightDir.Normalise();

    mProgram->Bind();
    mProgram->SetTexture(mTexDiffuseLoc, 0, mTexDiffuse);
    mProgram->SetTexture(mTexNormalLoc, 1, mTexNormal);
    mProgram->SetTexture(mTexAOLoc, 3, mTexAO);
    mProgram->SetFloat3(mLightDirLoc, viewLightDir.x, viewLightDir.y, viewLightDir.z);
    mProgram->SetFloat3(mSkyColorLoc, mSkyLightColor.r, mSkyLightColor.g, mSkyLightColor.b);
    mProgram->SetFloat3(mGroundColorLoc, mGroundLightColor.r, mGroundLightColor.g, mGroundLightColor.b);
    
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
}

void Effect3dDSDiHemisphereLighting::SetProjectionMatrix(const Matrix4& mat)
{
    mProjMat = mat;
}

void Effect3dDSDiHemisphereLighting::SetViewMatrix(const Matrix4& mat)
{
    mViewMat = mat;
}

void Effect3dDSDiHemisphereLighting::SetWorldMatrix(const Matrix4& mat)
{
    mWorldMat = mat;
}

void Effect3dDSDiHemisphereLighting::SetTextures(GLTexture** textures)
{
    mTexDiffuse = textures[0];
    mTexNormal = textures[1];
    mTexAO = textures[2];
}

void Effect3dDSDiHemisphereLighting::SetSkyDirection(const Vector3& dir)
{
    mLightDirection = -dir;
    mLightDirection.Normalise();
}

void Effect3dDSDiHemisphereLighting::SetSkyColor(const Color& color)
{
    mSkyLightColor = color;
}

void Effect3dDSDiHemisphereLighting::SetGroundColor(const Color& color)
{
    mGroundLightColor = color;
}


Effect3dDSDirectionalLighting::Effect3dDSDirectionalLighting()
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texShadow;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 lightColor;\n"
        "uniform mat4 shadowMat;\n"
        "in vec2 uv;\n"
        "float isLighted(vec3 pos, float cosTheta) { \n"
        "   vec4 shadowP = shadowMat * vec4(pos,1.0);\n"
        "   float bias = 0.005*tan(acos(cosTheta));\n"
        "   bias = clamp(bias, 0.0, 0.01);\n"
        "   float z = (shadowP.z - bias) / shadowP.w;\n"
        "   float result = texture(texShadow, shadowP.xy).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(-1, 0)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(1, 0)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(0,-1)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(0,1)).z < z ? 0.0 : 1.0;\n"
        "   return result / 5.0;\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = texture(texPosition, uv).xyz;\n"
        "   vec3 N = texture(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(-lightDir);\n"
        "   vec3 E = normalize(-P);\n"
        "   float NL = dot(N,L);\n"
        "   vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "   if (NL > 0.0) {\n"
        "       float lighted = isLighted(P, NL);\n"
        "       if ( lighted > 0.0) {\n"
        "           vec4 D = texture(texDiffuse, uv) * vec4(lightColor * NL, 1.0);\n"
        "	        float shiness = 10.0;\n"
        "           float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
        "           vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "           outColor = lighted * mix(D,S,specular);\n"
        "       }\n"
        "   }\n"
        "   FragColor = outColor;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texShadow;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 lightColor;\n"
        "uniform mat4 shadowMat;\n"
        "in vec2 uv;\n"
        "float isLighted(vec3 pos, float cosTheta) { \n"
        "   vec4 shadowP = shadowMat * vec4(pos,1.0);\n"
        "   float bias = 0.005*tan(acos(cosTheta));\n"
        "   bias = clamp(bias, 0.0, 0.01);\n"
        "   float z = (shadowP.z - bias) / shadowP.w;\n"
        "   float result = texture(texShadow, shadowP.xy).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(-1, 0)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(1, 0)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(0,-1)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(0,1)).z < z ? 0.0 : 1.0;\n"
        "   return result / 5.0;\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = texture(texPosition, uv).xyz;\n"
        "   vec3 N = texture(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(-lightDir);\n"
        "   vec3 E = normalize(-P);\n"
        "   float NL = dot(N,L);\n"
        "   vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "   if (NL > 0.0) {\n"
        "       float lighted = isLighted(P, NL);\n"
        "       if ( lighted > 0.0) {\n"
        "           vec4 D = texture(texDiffuse, uv) * vec4(lightColor * NL, 1.0);\n"
        "	        float shiness = 10.0;\n"
        "           float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
        "           vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "           outColor = lighted * mix(D,S,specular);\n"
        "       }\n"
        "   }\n"
        "   FragColor = outColor;\n"
        "}\n";
#else
	const char* vs =
        "attribute vec2 position;\n"
		"varying vec2 uv;\n"
		"void main() {\n"
		"	uv = position.xy * 0.5 + 0.5;\n"
		"   gl_Position = vec4(position, 0.0, 1.0);\n"
		"}\n";

    const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texShadow;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 lightColor;\n"
        "uniform mat4 shadowMat;\n"
        "varying vec2 uv;\n"
        "float isLighted(vec3 pos, float cosTheta) { \n"
        "   vec4 shadowP = shadowMat * vec4(pos,1.0);\n"
        "   float bias = 0.005*tan(acos(cosTheta));\n"
        "   bias = clamp(bias, 0.0, 0.01);\n"
        "   float z = (shadowP.z - bias) / shadowP.w;\n"
        "   float result = texture2D(texShadow, shadowP.xy).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(-1, 0)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(1, 0)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(0,-1)).z < z ? 0.0 : 1.0;\n"
        "   result += textureOffset(texShadow, shadowP.xy, ivec2(0,1)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(-2, 0)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(2, 0)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(0,-2)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(-3, 0)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(3, 0)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(0,-3)).z < z ? 0.0 : 1.0;\n"
        //"   result += textureOffset(texShadow, shadowP.xy, ivec2(0,3)).z < z ? 0.0 : 1.0;\n"
        //"   for (int i = 0; i < 4; ++i) {\n"
        //"   }\n"
        "   return result / 5.0;\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = texture2D(texPosition, uv).xyz;\n"
        "   vec3 N = texture2D(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(-lightDir);\n"
        "   vec3 E = normalize(-P);\n"
        "   float NL = dot(N,L);\n"
        "   vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "   if (NL > 0.0) {\n"
        "       float lighted = isLighted(P, NL);\n"
        "       if ( lighted > 0.0) {\n"
        "           vec4 D = texture2D(texDiffuse, uv) * vec4(lightColor * NL, 1.0);\n"
        "	        float shiness = 10.0;\n"
        "           float specular = pow(max(0.0, dot(normalize(reflect(lightDir,N)), E)), shiness);\n"
        "           vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "           outColor = lighted * mix(D,S,specular);\n"
        "       }\n"
        "   }\n"
        "   gl_FragColor = outColor;\n"
		"}\n";
#endif

	std::map<std::string, int> bs;
	bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

	mProgram->Bind();
	
	mMvpLoc = mProgram->GetUniformLocation("mvp");
	mMvLoc = mProgram->GetUniformLocation("mv");
	mNormalMatLoc = mProgram->GetUniformLocation("mvInvT");
    mShadowMatLoc = mProgram->GetUniformLocation("shadowMat");
	mTexDiffuseLoc = mProgram->GetUniformLocation("texDiffuse");
	mTexNormalLoc = mProgram->GetUniformLocation("texNormal");
	mTexPositionLoc = mProgram->GetUniformLocation("texPosition");
    mTexShadowLoc = mProgram->GetUniformLocation("texShadow");
	mLightDirLoc = mProgram->GetUniformLocation("lightDir");
	mLightColorLoc = mProgram->GetUniformLocation("lightColor");

	SetLightDirection(Vector3(0, 0, -1));
	mProgram->Unbind();
}

Effect3dDSDirectionalLighting::~Effect3dDSDirectionalLighting()
{
	delete mProgram;
}

void Effect3dDSDirectionalLighting::Bind()
{
    Matrix4 mv = mViewMat * mWorldMat;
    Matrix4 mvp = mProjMat * mv;
    //Matrix4 nm = mv.Inverse();
    //nm.Transpose();
    Matrix4 nm = mv;
    nm.m03 = nm.m13 = nm.m23 = 0.0f;
    Vector3 viewLightDir = nm * mLightDirection;
    viewLightDir.Normalise();
    
	mProgram->Bind();
	mProgram->SetTexture(mTexDiffuseLoc, 0, mTexDiffuse);
	mProgram->SetTexture(mTexNormalLoc, 1, mTexNormal);
	mProgram->SetTexture(mTexPositionLoc, 2, mTexPosition);
    mProgram->SetTexture(mTexShadowLoc, 3, mTexShadow);
    mProgram->SetFloat3(mLightDirLoc, viewLightDir.x, viewLightDir.y, viewLightDir.z);
	mProgram->SetFloat3(mLightColorLoc, mLightColor.r, mLightColor.g, mLightColor.b);
    mProgram->SetMatrix(mShadowMatLoc, &mShadowMat.m00);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
}

void Effect3dDSDirectionalLighting::SetProjectionMatrix(const Matrix4& mat)
{
	mProjMat = mat;
}

void Effect3dDSDirectionalLighting::SetViewMatrix(const Matrix4& mat)
{
	mViewMat = mat;
}

void Effect3dDSDirectionalLighting::SetWorldMatrix(const Matrix4& mat)
{
	mWorldMat = mat;
}

void Effect3dDSDirectionalLighting::SetTextures(GLTexture** textures)
{
	mTexDiffuse = textures[0];
	mTexNormal = textures[1];
	mTexPosition = textures[2];
    mTexShadow = textures[3];
}

void Effect3dDSDirectionalLighting::SetLightDirection(const Vector3& dir)
{
	mLightDirection = dir;
	mLightDirection.Normalise();
}

void Effect3dDSDirectionalLighting::SetLightColor(const Color& color)
{
	mLightColor = color;
}

void Effect3dDSDirectionalLighting::SetShadowMatrix(const Matrix4& mat)
{
    mShadowMat = mat;
}


Effect3dDSPointLighting::Effect3dDSPointLighting()
{
#if defined(USE_OPENGL3)
    const char* vs =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 330\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texShadow;\n"
        "uniform vec3 lightPos;\n"
        "uniform vec3 lightColor;\n"
        "uniform mat4 shadowMat;\n"
        "in vec2 uv;\n"
        "void main() {\n"
        "   vec3 P = texture(texPosition, uv).xyz;\n"
        "   vec3 N = texture(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(lightPos - P);\n"
        "   vec3 E = normalize(-P);\n"
        "   float NL = dot(N,L);\n"
        "   vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "           vec4 D = texture(texDiffuse, uv) * vec4(lightColor * NL, 1.0);\n"
        "	        float shiness = 10.0;\n"
        "           float specular = pow(max(0.0, dot(normalize(reflect(-L,N)), E)), shiness);\n"
        "           vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "           outColor = mix(D,S,specular);\n"
        "   FragColor = outColor;\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vs =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#version 300 es\n"
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texShadow;\n"
        "uniform vec3 lightPos;\n"
        "uniform vec3 lightColor;\n"
        "uniform mat4 shadowMat;\n"
        "in vec2 uv;\n"
        "void main() {\n"
        "   vec3 P = texture(texPosition, uv).xyz;\n"
        "   vec3 N = texture(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(lightPos - P);\n"
        "   vec3 E = normalize(-P);\n"
        "   float NL = dot(N,L);\n"
        "   vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "           vec4 D = texture(texDiffuse, uv) * vec4(lightColor * NL, 1.0);\n"
        "	        float shiness = 10.0;\n"
        "           float specular = pow(max(0.0, dot(normalize(reflect(-L,N)), E)), shiness);\n"
        "           vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "           outColor = mix(D,S,specular);\n"
        "   FragColor = outColor;\n"
        "}\n";
#else
    const char* vs =
        "attribute vec2 position;\n"
        "varying vec2 uv;\n"
        "void main() {\n"
        "	uv = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* ps =
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texDiffuse;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texShadow;\n"
        "uniform vec3 lightPos;\n"
        "uniform vec3 lightColor;\n"
        "uniform mat4 shadowMat;\n"
        "varying vec2 uv;\n"
        "void main() {\n"
        "   vec3 P = texture2D(texPosition, uv).xyz;\n"
        "   vec3 N = texture2D(texNormal, uv).xyz;\n"
        "   vec3 L = normalize(lightPos - P);\n"
        "   vec3 E = normalize(-P);\n"
        "   float NL = dot(N,L);\n"
        "   vec4 outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "           vec4 D = texture2D(texDiffuse, uv) * vec4(lightColor * NL, 1.0);\n"
        "	        float shiness = 10.0;\n"
        "           float specular = pow(max(0.0, dot(normalize(reflect(-L,N)), E)), shiness);\n"
        "           vec4 S = vec4(clamp(lightColor, 0.0, 1.0), 1.0);\n"
        "           outColor = mix(D,S,specular);\n"
        "   gl_FragColor = outColor;\n"
        "}\n";
#endif
    std::map<std::string, int> bs;
    bs["position"] = GLMesh::GLAttributeTypePosition;
    mProgram = new GLShaderProgram(vs, ps, bs, __LINE__, __FUNCTION__);

    mProgram->Bind();

    mMvpLoc = mProgram->GetUniformLocation("mvp");
    mMvLoc = mProgram->GetUniformLocation("mv");
    mNormalMatLoc = mProgram->GetUniformLocation("mvInvT");
    mShadowMatLoc = mProgram->GetUniformLocation("shadowMat");
    mTexDiffuseLoc = mProgram->GetUniformLocation("texDiffuse");
    mTexNormalLoc = mProgram->GetUniformLocation("texNormal");
    mTexPositionLoc = mProgram->GetUniformLocation("texPosition");
    mTexShadowLoc = mProgram->GetUniformLocation("texShadow");
    mLightPosLoc = mProgram->GetUniformLocation("lightPos");
    mLightColorLoc = mProgram->GetUniformLocation("lightColor");

    SetLightPosition(Vector3(0, 0, 0));
    mProgram->Unbind();
}

Effect3dDSPointLighting::~Effect3dDSPointLighting()
{
    delete mProgram;
}

void Effect3dDSPointLighting::Bind()
{
    Matrix4 mv = mViewMat * mWorldMat;
    Vector3 viewLightPos = mv * mLightPosition;

    mProgram->Bind();
    mProgram->SetTexture(mTexDiffuseLoc, 0, mTexDiffuse);
    mProgram->SetTexture(mTexNormalLoc, 1, mTexNormal);
    mProgram->SetTexture(mTexPositionLoc, 2, mTexPosition);
    mProgram->SetTexture(mTexShadowLoc, 3, mTexShadow);
    mProgram->SetFloat3(mLightPosLoc, viewLightPos.x, viewLightPos.y, viewLightPos.z);
    mProgram->SetFloat3(mLightColorLoc, mLightColor.r, mLightColor.g, mLightColor.b);
    mProgram->SetMatrix(mShadowMatLoc, &mShadowMat.m00);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
}

void Effect3dDSPointLighting::SetProjectionMatrix(const Matrix4& mat)
{
    mProjMat = mat;
}

void Effect3dDSPointLighting::SetViewMatrix(const Matrix4& mat)
{
    mViewMat = mat;
}

void Effect3dDSPointLighting::SetWorldMatrix(const Matrix4& mat)
{
    mWorldMat = mat;
}

void Effect3dDSPointLighting::SetTextures(GLTexture** textures)
{
    mTexDiffuse = textures[0];
    mTexNormal = textures[1];
    mTexPosition = textures[2];
    mTexShadow = textures[3];
}

void Effect3dDSPointLighting::SetLightPosition(const Vector3& position)
{
    mLightPosition = position;
}

void Effect3dDSPointLighting::SetLightColor(const Color& color)
{
    mLightColor = color;
}

void Effect3dDSPointLighting::SetShadowMatrix(const Matrix4& mat)
{
    mShadowMat = mat;
}

Effect3dSSAO::Effect3dSSAO(GLRenderer* renderer, bool depthOnly, int maxSamples, int randTextureSize, float radius, float falloff, float bias, float density)
	:mRenderer(renderer)
	, mDepthOnly(depthOnly)
	, mMaxSamples(maxSamples)
	, mRandTextureSize(randTextureSize)
	, mRadius(radius)
	, mFalloff(falloff)
	, mBias(bias)
	, mDensity(density)
	, mAOProgram(NULL)
	, mRandTexture(NULL)
{
	Build();
}

Effect3dSSAO::~Effect3dSSAO()
{
	delete mAOProgram;
	delete mRandTexture;
}

void Effect3dSSAO::Build()
{
#if defined(USE_OPENGL3)
    const char* vsDSAo =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psDSAo =
        //"#version 330\n"
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "out vec4 FragColor;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texRandDir;\n"
        "uniform vec2 sampleDirs[MAX_SAMPLES];\n"
        "uniform vec2 randSampleScale;\n"//=randSize/screenSize
        "uniform float radius;\n"// in view space units
        "uniform float falloff;\n"// 1 to INF
        "uniform float bias;\n"//0 to 1
        "uniform float density;\n"//0 to INF
        "uniform mat4 projMat;\n"
        "in vec2 texcoordOut;\n"
        "float getAO(vec3 N, vec3 P, vec2 tc) {\n"
        "   vec3 pos = texture(texPosition, tc).xyz;\n"
        "   vec3 tp = pos.xyz - P.xyz;\n"
        "   float l = min(1.0, length(tp)/radius);\n"
        "   return max(0.0, dot(N,normalize(tp)) - bias) * max(0.0, 1.0 - pow(l, falloff));\n"
        "}\n"
        "\n"
        "vec2 radiusToUVOffset(float radius, float zEye){\n"
        "   return vec2(1.0 / projMat[0][0], 1.0 / projMat[1][1]) * (radius / -zEye * 0.5);\n"//z is negative in view frustum
        "}\n"
        "\n"
        "vec2 rand(vec2 co){\n"
        "   float n = 3.1415926 * 2.0 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
        "   return vec2(cos(n),sin(n));\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = texture(texPosition, texcoordOut).xyz;\n"
        "   if (P.z < 0.0) {\n"//any geometry render at the pixel?
        "       vec3 N = normalize(texture(texNormal, texcoordOut).rgb);\n"
        //"       vec2 rn = normalize(texture(texRandDir, texcoordOut * randSampleScale).xy * 2.0 - 1.0);\n"
        "       vec2 rn = rand(texcoordOut);\n"
        "       float ao = 0.0;\n"
        "       vec2 baseOffset = radiusToUVOffset(radius, P.z);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (reflect(sampleDirs[i] * baseOffset, rn));\n"
        "           ao += getAO(N, P, uv);\n"
        "       }\n"
        "       ao = 1.0 - ao * density / float(MAX_SAMPLES);\n"
        "       FragColor = vec4(ao, ao, ao, 1.0);\n"
        "   } else {\n"
        "       FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "   }\n"

        "}\n";

    const char* vsAo =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psAo =
        //"#version 330\n"
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "out vec4 FragColor;\n"
        "uniform sampler2D texDepth;\n"
        "uniform sampler2D texRandDir;\n"
        "uniform mat4 projInv;\n"
        "uniform vec2 sizeInv;\n"
        "uniform float nearZ;\n"
        "uniform float farZ;\n"
        "uniform vec2 sampleDirs[MAX_SAMPLES];\n"
        "uniform vec2 randSampleScale;\n"//=randSize/screenSize
        "uniform float radius;\n"// in view space units
        "uniform float falloff;\n"// 1 to INF
        "uniform float bias;\n"//0 to 1
        "uniform float density;\n"//0 to INF
        "uniform mat4 projMat;\n"
        "in vec2 texcoordOut;\n"
        "vec3 GetPosition(vec2 uv){\n"
        "   vec4 P = projInv * vec4(vec3(uv, texture(texDepth, uv).r) * 2.0 - 1.0, 1.0);\n"
        "   P.xyz /= P.w;\n"
        "   return P.xyz;"
        "}\n"
        "vec3 GetNormal(vec3 P, vec2 uv){\n"
        "   vec2 dx = vec2(sizeInv.x * 1.0, 0.0);\n"
        "   vec2 dy = vec2(0.0, sizeInv.y * 1.0);\n"
        "   vec3 PX = GetPosition(uv + dx) - P;\n"
        "   vec3 PY = GetPosition(uv + dy) - P;\n"
        "   vec3 N = cross(PX, PY);\n"
        "   return normalize(N);\n"
        "}\n"
        "float LinearZ(vec2 uv){\n"
        "   float nearZ = 0.1;\n"
        "   float farZ = 10000.0;\n"
        "   float zn = texture(texDepth, uv).r * 2.0 - 1.0;\n"
        "   float ze = 2.0 * nearZ * farZ / (-(farZ + nearZ) + zn * (farZ - nearZ));\n"
        "   return (-ze - nearZ) / (farZ - nearZ);\n"
        "}\n"
        "\n"
        "float getAO(vec3 N, vec3 P, vec2 tc) {\n"
        "   vec3 pos = GetPosition(tc);\n"
        "   vec3 tp = pos.xyz - P.xyz;\n"
        "   float l = min(1.0, length(tp)/radius);\n"
        "   return max(0.0, dot(N,normalize(tp)) - bias) * max(0.0, 1.0 - pow(l, falloff));\n"
        "}\n"
        "\n"
        "vec2 radiusToUVOffset(float radius, float zEye){\n"
        "   return vec2(1.0 / projMat[0][0], 1.0 / projMat[1][1]) * (radius / -zEye * 0.5);\n"//z is negative in view frustum
        "}\n"
        "\n"
        "vec2 rand(vec2 co){\n"
        "   float n = 3.1415926 * 2.0 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
        "   return vec2(cos(n),sin(n));\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = GetPosition(texcoordOut);\n"
        "   if (P.z < 0.0) {\n"//any geometry render at the pixel?
        "       vec3 N = GetNormal(P, texcoordOut);\n"
        "       vec2 rn = normalize(texture(texRandDir, texcoordOut * randSampleScale).xy * 2.0 - 1.0);\n"
        //"       vec2 rn = rand(texcoordOut);\n"
        "       float ao = 0.0;\n"
        "       vec2 baseOffset = radiusToUVOffset(radius, P.z);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (reflect(sampleDirs[i] * baseOffset, rn));\n"
        "           ao += getAO(N, P, uv);\n"
        "       }\n"
        "       ao = 1.0 - ao * density / float(MAX_SAMPLES);\n"
        "       FragColor = vec4(ao, ao, ao, 1.0);\n"
        "   } else {\n"
        "       FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "   }\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vsDSAo =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psDSAo =
        //"#version 300 es\n"
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texPosition;\n"
        "uniform sampler2D texRandDir;\n"
        "uniform vec2 sampleDirs[MAX_SAMPLES];\n"
        "uniform vec2 randSampleScale;\n"//=randSize/screenSize
        "uniform float radius;\n"// in view space units
        "uniform float falloff;\n"// 1 to INF
        "uniform float bias;\n"//0 to 1
        "uniform float density;\n"//0 to INF
        "uniform mat4 projMat;\n"
        "in vec2 texcoordOut;\n"
        "float getAO(vec3 N, vec3 P, vec2 tc) {\n"
        "   vec3 pos = texture(texPosition, tc).xyz;\n"
        "   vec3 tp = pos.xyz - P.xyz;\n"
        "   float l = min(1.0, length(tp)/radius);\n"
        "   return max(0.0, dot(N,normalize(tp)) - bias) * max(0.0, 1.0 - pow(l, falloff));\n"
        "}\n"
        "\n"
        "vec2 radiusToUVOffset(float radius, float zEye){\n"
        "   return vec2(1.0 / projMat[0][0], 1.0 / projMat[1][1]) * (radius / -zEye * 0.5);\n"//z is negative in view frustum
        "}\n"
        "\n"
        "vec2 rand(vec2 co){\n"
        "   float n = 3.1415926 * 2.0 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
        "   return vec2(cos(n),sin(n));\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = texture(texPosition, texcoordOut).xyz;\n"
        "   if (P.z < 0.0) {\n"//any geometry render at the pixel?
        "       vec3 N = normalize(texture(texNormal, texcoordOut).rgb);\n"
        //"       vec2 rn = normalize(texture(texRandDir, texcoordOut * randSampleScale).xy * 2.0 - 1.0);\n"
        "       vec2 rn = rand(texcoordOut);\n"
        "       float ao = 0.0;\n"
        "       vec2 baseOffset = radiusToUVOffset(radius, P.z);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (reflect(sampleDirs[i] * baseOffset, rn));\n"
        "           ao += getAO(N, P, uv);\n"
        "       }\n"
        "       ao = 1.0 - ao * density / float(MAX_SAMPLES);\n"
        "       FragColor = vec4(ao, ao, ao, 1.0);\n"
        "   } else {\n"
        "       FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "   }\n"

        "}\n";

    const char* vsAo =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psAo =
        //"#version 300 es\n"
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texDepth;\n"
        "uniform sampler2D texRandDir;\n"
        "uniform mat4 projInv;\n"
        "uniform vec2 sizeInv;\n"
        "uniform float nearZ;\n"
        "uniform float farZ;\n"
        "uniform vec2 sampleDirs[MAX_SAMPLES];\n"
        "uniform vec2 randSampleScale;\n"//=randSize/screenSize
        "uniform float radius;\n"// in view space units
        "uniform float falloff;\n"// 1 to INF
        "uniform float bias;\n"//0 to 1
        "uniform float density;\n"//0 to INF
        "uniform mat4 projMat;\n"
        "in vec2 texcoordOut;\n"
        "vec3 GetPosition(vec2 uv){\n"
        "   vec4 P = projInv * vec4(vec3(uv, texture(texDepth, uv).r) * 2.0 - 1.0, 1.0);\n"
        "   P.xyz /= P.w;\n"
        "   return P.xyz;"
        "}\n"
        "vec3 GetNormal(vec3 P, vec2 uv){\n"
        "   vec2 dx = vec2(sizeInv.x * 1.0, 0.0);\n"
        "   vec2 dy = vec2(0.0, sizeInv.y * 1.0);\n"
        "   vec3 PX = GetPosition(uv + dx) - P;\n"
        "   vec3 PY = GetPosition(uv + dy) - P;\n"
        "   vec3 N = cross(PX, PY);\n"
        "   return normalize(N);\n"
        "}\n"
        "float LinearZ(vec2 uv){\n"
        "   float nearZ = 0.1;\n"
        "   float farZ = 10000.0;\n"
        "   float zn = texture(texDepth, uv).r * 2.0 - 1.0;\n"
        "   float ze = 2.0 * nearZ * farZ / (-(farZ + nearZ) + zn * (farZ - nearZ));\n"
        "   return (-ze - nearZ) / (farZ - nearZ);\n"
        "}\n"
        "\n"
        "float getAO(vec3 N, vec3 P, vec2 tc) {\n"
        "   vec3 pos = GetPosition(tc);\n"
        "   vec3 tp = pos.xyz - P.xyz;\n"
        "   float l = min(1.0, length(tp)/radius);\n"
        "   return max(0.0, dot(N,normalize(tp)) - bias) * max(0.0, 1.0 - pow(l, falloff));\n"
        "}\n"
        "\n"
        "vec2 radiusToUVOffset(float radius, float zEye){\n"
        "   return vec2(1.0 / projMat[0][0], 1.0 / projMat[1][1]) * (radius / -zEye * 0.5);\n"//z is negative in view frustum
        "}\n"
        "\n"
        "vec2 rand(vec2 co){\n"
        "   float n = 3.1415926 * 2.0 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
        "   return vec2(cos(n),sin(n));\n"
        "}\n"
        "void main() {\n"
        "   vec3 P = GetPosition(texcoordOut);\n"
        "   if (P.z < 0.0) {\n"//any geometry render at the pixel?
        "       vec3 N = GetNormal(P, texcoordOut);\n"
        "       vec2 rn = normalize(texture(texRandDir, texcoordOut * randSampleScale).xy * 2.0 - 1.0);\n"
        //"       vec2 rn = rand(texcoordOut);\n"
        "       float ao = 0.0;\n"
        "       vec2 baseOffset = radiusToUVOffset(radius, P.z);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (reflect(sampleDirs[i] * baseOffset, rn));\n"
        "           ao += getAO(N, P, uv);\n"
        "       }\n"
        "       ao = 1.0 - ao * density / float(MAX_SAMPLES);\n"
        "       FragColor = vec4(ao, ao, ao, 1.0);\n"
        "   } else {\n"
        "       FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "   }\n"
        "}\n";
#else
	const char* vsDSAo =
		"attribute vec2 position;\n"
		"varying vec2 texcoordOut;\n"
		"void main() {\n"
		"   texcoordOut = position.xy * 0.5 + 0.5;\n"
		"   gl_Position = vec4(position, 0.0, 1.0);\n"
		"}\n";

	const char* psDSAo =
		//"#define MAX_SAMPLES 8\n"//Should be added when create AO program
		"#ifdef GL_ES\n"
		"precision highp float;\n"
		"#endif\n"
		"uniform sampler2D texNormal;\n"
		"uniform sampler2D texPosition;\n"
		"uniform sampler2D texRandDir;\n"
		"uniform vec2 sampleDirs[MAX_SAMPLES];\n"
		"uniform vec2 randSampleScale;\n"//=randSize/screenSize
		"uniform float radius;\n"// in view space units
		"uniform float falloff;\n"// 1 to INF
		"uniform float bias;\n"//0 to 1
		"uniform float density;\n"//0 to INF
		"uniform mat4 projMat;\n"
		"varying vec2 texcoordOut;\n"
		"float getAO(vec3 N, vec3 P, vec2 tc) {\n"
		"   vec3 pos = texture2D(texPosition, tc).xyz;\n"
		"   vec3 tp = pos.xyz - P.xyz;\n"
		"   float l = min(1.0, length(tp)/radius);\n"
		"   return max(0.0, dot(N,normalize(tp)) - bias) * max(0.0, 1.0 - pow(l, falloff));\n"
		"}\n"
		"\n"
		"vec2 radiusToUVOffset(float radius, float zEye){\n"
		"   return vec2(1.0 / projMat[0][0], 1.0 / projMat[1][1]) * (radius / -zEye * 0.5);\n"//z is negative in view frustum
		"}\n"
		"\n"
		"vec2 rand(vec2 co){\n"
		"   float n = 3.1415926 * 2.0 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
		"   return vec2(cos(n),sin(n));\n"
		"}\n"
		"void main() {\n"
		"   vec3 P = texture2D(texPosition, texcoordOut).xyz;\n"
		"   if (P.z < 0.0) {\n"//any geometry render at the pixel?
		"       vec3 N = normalize(texture2D(texNormal, texcoordOut).rgb);\n"
		//"       vec2 rn = normalize(texture2D(texRandDir, texcoordOut * randSampleScale).xy * 2.0 - 1.0);\n"
		"       vec2 rn = rand(texcoordOut);\n"
		"       float ao = 0.0;\n"
		"       vec2 baseOffset = radiusToUVOffset(radius, P.z);\n"
		"       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
		"       {\n"
		"           vec2 uv = texcoordOut + (reflect(sampleDirs[i] * baseOffset, rn));\n"
		"           ao += getAO(N, P, uv);\n"
		"       }\n"
		"       ao = 1.0 - ao * density / float(MAX_SAMPLES);\n"
		"       gl_FragColor = vec4(ao, ao, ao, 1.0);\n"
		"   } else {\n"
		"       gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
		"   }\n"

		"}\n";

	const char* vsAo =
		"attribute vec2 position;\n"
		"varying vec2 texcoordOut;\n"
		"void main() {\n"
		"   texcoordOut = position.xy * 0.5 + 0.5;\n"
		"   gl_Position = vec4(position, 0.0, 1.0);\n"
		"}\n";

	const char* psAo =
		//"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
		"uniform sampler2D texDepth;\n"
		"uniform sampler2D texRandDir;\n"
		"uniform mat4 projInv;\n"
		"uniform vec2 sizeInv;\n"
		"uniform float nearZ;\n"
		"uniform float farZ;\n"
		"uniform vec2 sampleDirs[MAX_SAMPLES];\n"
		"uniform vec2 randSampleScale;\n"//=randSize/screenSize
		"uniform float radius;\n"// in view space units
		"uniform float falloff;\n"// 1 to INF
		"uniform float bias;\n"//0 to 1
		"uniform float density;\n"//0 to INF
		"uniform mat4 projMat;\n"
		"varying vec2 texcoordOut;\n"
		"vec3 GetPosition(vec2 uv){\n"
		"   vec4 P = projInv * vec4(vec3(uv, texture2D(texDepth, uv).r) * 2.0 - 1.0, 1.0);\n"
		"   P.xyz /= P.w;\n"
		"   return P.xyz;"
		"}\n"
		"vec3 GetNormal(vec3 P, vec2 uv){\n"
		"   vec2 dx = vec2(sizeInv.x * 1.0, 0.0);\n"
		"   vec2 dy = vec2(0.0, sizeInv.y * 1.0);\n"
		"   vec3 PX = GetPosition(uv + dx) - P;\n"
		"   vec3 PY = GetPosition(uv + dy) - P;\n"
		"   vec3 N = cross(PX, PY);\n"
		"   return normalize(N);\n"
		"}\n"
		"float LinearZ(vec2 uv){\n"
		"   float nearZ = 0.1;\n"
		"   float farZ = 10000.0;\n"
		"   float zn = texture2D(texDepth, uv).r * 2.0 - 1.0;\n"
		"   float ze = 2.0 * nearZ * farZ / (-(farZ + nearZ) + zn * (farZ - nearZ));\n"
		"   return (-ze - nearZ) / (farZ - nearZ);\n"
		"}\n"
		"\n"
		"float getAO(vec3 N, vec3 P, vec2 tc) {\n"
		"   vec3 pos = GetPosition(tc);\n"
		"   vec3 tp = pos.xyz - P.xyz;\n"
		"   float l = min(1.0, length(tp)/radius);\n"
		"   return max(0.0, dot(N,normalize(tp)) - bias) * max(0.0, 1.0 - pow(l, falloff));\n"
		"}\n"
		"\n"
		"vec2 radiusToUVOffset(float radius, float zEye){\n"
		"   return vec2(1.0 / projMat[0][0], 1.0 / projMat[1][1]) * (radius / -zEye * 0.5);\n"//z is negative in view frustum
		"}\n"
		"\n"
		"vec2 rand(vec2 co){\n"
		"   float n = 3.1415926 * 2.0 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
		"   return vec2(cos(n),sin(n));\n"
		"}\n"
		"void main() {\n"
		"   vec3 P = GetPosition(texcoordOut);\n"
		"   if (P.z < 0.0) {\n"//any geometry render at the pixel?
		"       vec3 N = GetNormal(P, texcoordOut);\n"
		"       vec2 rn = normalize(texture2D(texRandDir, texcoordOut * randSampleScale).xy * 2.0 - 1.0);\n"
		//"       vec2 rn = rand(texcoordOut);\n"
		"       float ao = 0.0;\n"
		"       vec2 baseOffset = radiusToUVOffset(radius, P.z);\n"
		"       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
		"       {\n"
		"           vec2 uv = texcoordOut + (reflect(sampleDirs[i] * baseOffset, rn));\n"
		"           ao += getAO(N, P, uv);\n"
		"       }\n"
		"       ao = 1.0 - ao * density / float(MAX_SAMPLES);\n"
		"       gl_FragColor = vec4(ao, ao, ao, 1.0);\n"
		"   } else {\n"
		"       gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
		"   }\n"
		"}\n";
#endif
	if (mAOProgram)
	{
		delete mAOProgram;
		delete mRandTexture;
		mRandDirs.clear();
	}

	std::map<std::string, int> attrLocs;
	attrLocs["position"] = 0;

	char header[50];
#if defined(USE_OPENGL3)
    sprintf(header, "#version 330\n#define MAX_SAMPLES %d\n", mMaxSamples);
#elif defined(USE_OPENGL3_ES)
    sprintf(header, "#version 300 es\n#define MAX_SAMPLES %d\n", mMaxSamples);
#else
    sprintf(header, "#define MAX_SAMPLES %d\n", mMaxSamples);
#endif
	std::string buf(header);
	if (mDepthOnly)
	{
		buf += psAo;
        mAOProgram = new GLShaderProgram(vsAo, buf.c_str(), attrLocs, __LINE__, __FUNCTION__);
	}
	else
	{
		buf += psDSAo;
        mAOProgram = new GLShaderProgram(vsDSAo, buf.c_str(), attrLocs, __LINE__, __FUNCTION__);
	}


	int samplesPerRing = 4;
	int rings = mMaxSamples / samplesPerRing;
	for (int j = 0; j < rings; ++j)
	{
		float r = (j + 1) / (float)rings;
		float offset = j % 2 ? (0.5f / samplesPerRing) : 0.0f;
		for (int i = 0; i < samplesPerRing; ++i)
		{
			float a = PI * 2.0f * (i / (float)samplesPerRing + offset);
			float x = r * (float)cos(a);
			float y = r * (float)sin(a);
			mRandDirs.push_back(x);
			mRandDirs.push_back(y);
		}
	}

	unsigned char* rd = new unsigned char[mRandTextureSize * mRandTextureSize * 4];
	for (int i = 0; i < mRandTextureSize * mRandTextureSize; ++i)
	{
		unsigned char* p = rd + i * 4;
		float a = PI * 2.0f * (rand() / (float)RAND_MAX);
		float x = (float)cos(a);
		float y = (float)sin(a);
		p[0] = (unsigned char)(255 * (x * 0.5f + 0.5f));
		p[1] = (unsigned char)(255 * (y * 0.5f + 0.5f));
		p[2] = 255;
		p[3] = 255;
	}
	mRandTexture = mRenderer->CreateTexture(mRandTextureSize, mRandTextureSize, rd, 4, false, false, true, false);
	delete[] rd;
}

void Effect3dSSAO::SetSamples(int maxSample)
{
	if (maxSample != mMaxSamples)
	{
		mMaxSamples = maxSample;
		Build();
	}
}

void Effect3dSSAO::SetRadius(float radius)
{
	mRadius = radius;
}

void Effect3dSSAO::SetFalloff(float falloff)
{
	mFalloff = falloff;
}

void Effect3dSSAO::SetBias(float bias)
{
	mBias = bias;
}

void Effect3dSSAO::SetDensity(float density)
{
	mDensity = density;
}

void Effect3dSSAO::SetProjectionMatrix(const Matrix4& mat)
{
	mProjectMatrix = mat;
}

void Effect3dSSAO::SetTextures(GLTexture** textures)
{
	if (mDepthOnly)
	{
		mDepthTexture = textures[0];
	}
	else
	{
		mNormalTexture = textures[0];
		mPositionTexture = textures[1];
	}
}

void Effect3dSSAO::Bind()
{
	if (!mAOProgram)
	{
		return;
	}

	mAOProgram->Bind();
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	mAOProgram->SetFloat2Array("sampleDirs", &mRandDirs[0], mMaxSamples);
	mAOProgram->SetFloat("radius", mRadius);
	mAOProgram->SetFloat("falloff", mFalloff);
	mAOProgram->SetFloat("bias", mBias);
	mAOProgram->SetFloat("density", mDensity);
	mAOProgram->SetMatrix("projMat", &mProjectMatrix.m00);
	mAOProgram->SetTexture("texRandDir", 2, mRandTexture);

	if (mDepthOnly)
	{
		Matrix4 projMatrixInv = mProjectMatrix.Inverse();
		mAOProgram->SetTexture("texDepth", 0, mDepthTexture);
		mAOProgram->SetMatrix("projInv", &projMatrixInv.m00);
		mAOProgram->SetFloat2("sizeInv", 1.0f / mDepthTexture->GetWidth(), 1.0f / mDepthTexture->GetHeight());
		mAOProgram->SetFloat2("randSampleScale",
			mDepthTexture->GetWidth() / (float)mRandTexture->GetWidth(),
			mDepthTexture->GetHeight() / (float)mRandTexture->GetHeight());
	}
	else
	{
		mAOProgram->SetTexture("texNormal", 0, mNormalTexture);
		mAOProgram->SetTexture("texPosition", 1, mPositionTexture);
		
		mAOProgram->SetFloat2("randSampleScale",
			mPositionTexture->GetWidth() / (float)mRandTexture->GetWidth(),
			mPositionTexture->GetHeight() / (float)mRandTexture->GetHeight());
	}
}


Effect3dAOBlur::Effect3dAOBlur(GLRenderer* renderer, int maxSamples, float radius)
    : mRenderer(renderer)
    , mMaxSamples(maxSamples)
    , mRadius(radius)
    , mProgram(NULL)
{
    Build();
}

Effect3dAOBlur::~Effect3dAOBlur()
{
    delete mProgram;
}

void Effect3dAOBlur::Build()
{
#if defined(USE_OPENGL3)
    const char* vsDSAo =
        "#version 330\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psDSAo =
        //"#version 330\n"
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "out vec4 FragColor;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texColor;\n"
        "uniform vec2 radius;\n"
        "uniform vec2 offsets[MAX_SAMPLES];\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "       vec4 ao = vec4(texture(texColor, texcoordOut).xyz, 1.0);\n"
        "       vec4 N = texture(texNormal, texcoordOut);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (offsets[i] * radius);\n"
        "           vec4 sn = texture(texNormal, uv);\n"
        "           float f = max(dot(sn, N), 0.0);\n"
        "           vec4 c = texture(texColor, uv);\n"
        "           ao += vec4(c.rgb * f, f);\n"
        "       }\n"
        "       ao.rgb /= ao.a;\n"
        "       FragColor = vec4(ao.rgb, 1.0);\n"
        "}\n";
#elif defined(USE_OPENGL3_ES)
    const char* vsDSAo =
        "#version 300 es\n"
        "in vec2 position;\n"
        "out vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psDSAo =
        //"#version 300 es\n"
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "precision highp float;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texColor;\n"
        "uniform vec2 radius;\n"
        "uniform vec2 offsets[MAX_SAMPLES];\n"
        "in vec2 texcoordOut;\n"
        "void main() {\n"
        "       vec4 ao = vec4(texture(texColor, texcoordOut).xyz, 1.0);\n"
        "       vec4 N = texture(texNormal, texcoordOut);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (offsets[i] * radius);\n"
        "           vec4 sn = texture(texNormal, uv);\n"
        "           float f = max(dot(sn, N), 0.0);\n"
        "           vec4 c = texture(texColor, uv);\n"
        "           ao += vec4(c.rgb * f, f);\n"
        "       }\n"
        "       ao.rgb /= ao.a;\n"
        "       FragColor = vec4(ao.rgb, 1.0);\n"
        "}\n";
#else
    const char* vsDSAo =
        "attribute vec2 position;\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "   texcoordOut = position.xy * 0.5 + 0.5;\n"
        "   gl_Position = vec4(position, 0.0, 1.0);\n"
        "}\n";

    const char* psDSAo =
        //"#define MAX_SAMPLES 8\n"//Should be added when create AO program
        "#ifdef GL_ES\n"
        "precision highp float;\n"
        "#endif\n"
        "uniform sampler2D texNormal;\n"
        "uniform sampler2D texColor;\n"
        "uniform vec2 radius;\n"
        "uniform vec2 offsets[MAX_SAMPLES];\n"
        "varying vec2 texcoordOut;\n"
        "void main() {\n"
        "       vec4 ao = vec4(texture2D(texColor, texcoordOut).xyz, 1.0);\n"
        "       vec4 N = texture2D(texNormal, texcoordOut);\n"
        "       for (int i = 0; i < MAX_SAMPLES; ++i)\n"
        "       {\n"
        "           vec2 uv = texcoordOut + (offsets[i] * radius);\n"
        "           vec4 sn = texture2D(texNormal, uv);\n"
        "           float f = max(dot(sn, N), 0.0);\n"
        "           vec4 c = texture2D(texColor, uv);\n"
        "           ao += vec4(c.rgb * f, f);\n"
        "       }\n"
        "       ao.rgb /= ao.a;\n"
        "       gl_FragColor = vec4(ao.rgb, 1.0);\n"
        "}\n";
#endif

    if (mProgram)
    {
        delete mProgram;
    }

    std::map<std::string, int> attrLocs;
    attrLocs["position"] = 0;

    char header[50];

#if defined(USE_OPENGL3)
    sprintf(header, "#version 330\n#define MAX_SAMPLES %d\n", mMaxSamples);
#elif defined(USE_OPENGL3_ES)
    sprintf(header, "#version 300 es\n#define MAX_SAMPLES %d\n", mMaxSamples);
#else
    sprintf(header, "#define MAX_SAMPLES %d\n", mMaxSamples);
#endif

    std::string buf(header);
    buf += psDSAo;
    mProgram = new GLShaderProgram(vsDSAo, buf.c_str(), attrLocs, __LINE__, __FUNCTION__);

    for (int i = 0; i < mMaxSamples; ++i)
    {
        float r = rand() / (float)RAND_MAX;
        float a = PI * 2.0f * (rand() / (float)RAND_MAX);
        float x = r * (float)cos(a);
        float y = r * (float)sin(a);
        mRandDirs.push_back(x);
        mRandDirs.push_back(y);
    }
}

void Effect3dAOBlur::SetSamples(int maxSample)
{
    if (maxSample != mMaxSamples)
    {
        mMaxSamples = maxSample;
        Build();
    }
}

void Effect3dAOBlur::SetRadius(float radius)
{
    mRadius = radius;
}

void Effect3dAOBlur::SetTextures(GLTexture** textures)
{
    mColorTexture = textures[0];
    mNormalTexture = textures[1];
}

void Effect3dAOBlur::Bind()
{
    if (!mProgram)
    {
        return;
    }

    mProgram->Bind();
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    mProgram->SetFloat2Array("offsets", &mRandDirs[0], mMaxSamples);
    mProgram->SetFloat2("radius", mRadius / (float)mColorTexture->GetWidth(), mRadius / (float)mColorTexture->GetHeight());
    mProgram->SetTexture("texColor", 0, mColorTexture);
    mProgram->SetTexture("texNormal", 1, mNormalTexture);
}

Canvas3dState::Canvas3dState()
    :target(GLRenderer::GetInstance()->GetDefaultTarget())
    ,effect(GLRenderer::GetInstance()->GetCanvas3d()->GetForwardShadingEffect())
    ,camera(GLRenderer::GetInstance()->GetCanvas3d()->GetDefaultCamera())
{
    for (int i = 0; i < 8; ++i)
    {
        textures[i] = GLRenderer::GetInstance()->GetDefaultTexture();
    }
}

Canvas3d::Canvas3d(GLRenderer* renderer)
    :mRenderer(renderer)
{
    mUnlightedEffect = new Effect3dUnlighted();
    mForwardShadingEffect = new Effect3dForwardShading();

    mAOBlurEffect = new Effect3dAOBlur(renderer, 4, 4);
#if defined(USE_OPENGL3) || defined(USE_OPENGL3_ES)
    mDeferredSSAOEffect = new Effect3dSSAO(renderer, false, 16, 128, 10, 4.0f, 0.0f, 2.0f);
    mGPassEffect = new Effect3dGPass();
	mDSDirectionalLightingEffect = new Effect3dDSDirectionalLighting();
    mDSPointLightingEffect = new Effect3dDSPointLighting();
    mDSHemisphereLightingEffect = new Effect3dDSDiHemisphereLighting();
    mShadowPassEffect = new Effect3dShadowPass();
#else
    mDeferredSSAOEffect = NULL;
    mGPassEffect = NULL;
    mDSDirectionalLightingEffect = NULL;
    mDSPointLightingEffect = NULL;
    mDSHemisphereLightingEffect = NULL;
    mShadowPassEffect = NULL;
#endif
	mQuad = renderer->CreateScreenAlignedQuad();
}

Canvas3d::~Canvas3d()
{
    delete mDSDirectionalLightingEffect;
    delete mDSPointLightingEffect;
    delete mDSHemisphereLightingEffect;
    delete mGPassEffect;
    delete mForwardShadingEffect;
    delete mDeferredSSAOEffect;
    delete mAOBlurEffect;
    delete mShadowPassEffect;
    delete mUnlightedEffect;
	delete mQuad;
}

void Canvas3d::SetState(const Canvas3dState& state)
{
    state.effect->SetWorldMatrix(state.transform);
    state.effect->SetProjectionMatrix(state.camera->GetProjectionMatrix());
    state.effect->SetViewMatrix(state.camera->GetViewMatrix());
    state.effect->SetTextures((GLTexture**)state.textures);
    state.effect->Bind();
    state.target->Bind();
}

void Canvas3d::Draw(GLMesh* mesh)
{
    mesh->Draw();
}

void Canvas3d::DrawFullScreenQuad()
{
	mQuad->Draw();
}

//------------------------------------------------
class RectTreeNode
{
public:
    struct Rect2
    {
        Rect2(int x, int y, int w, int h) :x(x), y(y), width(w), height(h) {}

        int x;
        int y;
        int width;
        int height;
    };
    RectTreeNode(const Rect2& rc);
    ~RectTreeNode();
public:
    RectTreeNode* Insert(int w, int h);
    const Rect2& GetRect() { return bounds; }
private:
    Rect2 bounds;
    bool isEmpty;
    RectTreeNode* child[2];
};

RectTreeNode::RectTreeNode(const Rect2& rc)
    :bounds(rc)
    , isEmpty(true)
{
    child[0] = child[1] = NULL;
}

RectTreeNode::~RectTreeNode()
{
    delete child[0];
    delete child[1];
}

RectTreeNode* RectTreeNode::Insert(int width, int height)
{
    if (child[0])
    {
        if (child[0]->isEmpty)
        {
            RectTreeNode* n = child[0]->Insert(width, height);
            if (n != NULL)
            {
                return n;
            }
        }
        return child[1]->Insert(width, height);
    }
    else
    {
        if (!isEmpty)
        {
            return NULL;
        }

        int w = width + 2;
        int h = height + 2;

        int dx = bounds.width - w;
        int dy = bounds.height - h;

        if (dx == 0 && dy == 0)
        {
            isEmpty = false;
            return this;
        }

        if (dx < 0 || dy < 0)
        {
            return NULL;
        }

        if (dx >= dy)
        {
            child[0] = new RectTreeNode(Rect2(bounds.x, bounds.y, w, bounds.height));
            child[1] = new RectTreeNode(Rect2(bounds.x + w, bounds.y, bounds.width - w, bounds.height));
        }
        else
        {
            child[0] = new RectTreeNode(Rect2(bounds.x, bounds.y, bounds.width, h));
            child[1] = new RectTreeNode(Rect2(bounds.x, bounds.y + h, bounds.width, bounds.height - h));
        }

        return child[0]->Insert(width, height);
    }
}


//------------------------------------------------
class FontImpl
{
public:
    FontImpl(int fontHeight, int atlasSize, const char* path);
    ~FontImpl();
    Glyph* GetGlyph(int glyphId);
    int GetXAdvance(int glyphId);
    void SetFontHeight(int fontHeight);
    int GetFontHeight() { return mFontHeight; }
    int GetAscent() { return mAscent; }
    int GetDescent() { return mDescent; }
    void* GetAtlasBitmap();
    int GetAtlasSize() { return mAtlasSize; }
    void RenderText(const char* text, float xPos, float yPos, float fontHeight, std::vector<float>& vertices);
    void BakeText(const char* text, void** bitmap, int* width, int* height);
    void GetAtlasUpdateRects(std::vector<Font::AtlasRect>& updatedRects);
    void ExportAtlas(const char* path);
private:
    stbtt_fontinfo mFontInfo;
    unsigned char* mFontBuffer;
    int mAtlasSize;
    int mAscent;
    int mDescent;
    float mScale;
    typedef std::map<int, Glyph*> GlyphList;
    int mFontHeight;
    GlyphList mGlyphs;
    RectTreeNode* mRectTree;
    unsigned char* mAtlasBitmap;
    std::vector<Font::AtlasRect> mUpdatedRects;
};

FontImpl::FontImpl(int fontHeight, int atlasSize, const char* path)
    :mFontBuffer(NULL)
    , mAtlasSize(atlasSize)
    , mFontHeight(fontHeight)
{
    if (!path)
    {
        path = ":/font/arial.ttf";
    }

    int size = 0;
    mFontBuffer = (unsigned char*)LoadFile(path, size);

    stbtt_InitFont(&mFontInfo, mFontBuffer, 0);
    SetFontHeight(fontHeight);

    RectTreeNode::Rect2 rc(0, 0, atlasSize, atlasSize);
    mRectTree = new RectTreeNode(rc);
    mAtlasBitmap = new unsigned char[atlasSize * atlasSize * 4];
    memset(mAtlasBitmap, 0xFF, atlasSize * atlasSize * 4);
    for (int i = 0; i < atlasSize * atlasSize; ++i)
    {
        mAtlasBitmap[i * 4 + 3] = 0;
    }

    // pre-fetch
    GetGlyph(' ');
    for (int i = 40; i <= 176; ++i)
    {
        GetGlyph(i);
    }
}

FontImpl::~FontImpl()
{
    for (GlyphList::iterator it = mGlyphs.begin(); it != mGlyphs.end(); ++it)
    {
        delete it->second;
    }
    delete[] mFontBuffer;
    delete mRectTree;
    delete[] mAtlasBitmap;
}

Glyph* FontImpl::GetGlyph(int ch)
{
    GlyphList::iterator it = mGlyphs.find(ch);
    if (it != mGlyphs.end())
    {
        return it->second;
    }
    int width;
    int height;
    int xOffset;
    int yOffset;
    unsigned char* img;
    int glyphIndex = stbtt_FindGlyphIndex(&mFontInfo, ch);
    if (glyphIndex == 0)
    {
        return mGlyphs[' '];
    }
    img = stbtt_GetCodepointBitmap(&mFontInfo, mScale, mScale, ch, &width, &height, &xOffset, &yOffset);
    RectTreeNode* node = mRectTree->Insert(width, height);
    assert(node);//@TODO: if atlas is full, remove old one and try again
    if (img)
    {
        for (int y = 0; y < height; ++y)
        {
            unsigned char* dst = mAtlasBitmap + ((node->GetRect().y + y) * mAtlasSize + node->GetRect().x) * 4;
            unsigned char* src = img + (height - 1 - y)* width;
            for (int x = 0; x < width; ++x)
            {
                dst[x * 4 + 0] = 0xFF;
                dst[x * 4 + 1] = 0xFF;
                dst[x * 4 + 2] = 0xFF;
                dst[x * 4 + 3] = src[x];
            }
        }
        stbtt_FreeBitmap(img, 0);
    }

    int xAdvance;
    int bearing;
    stbtt_GetCodepointHMetrics(&mFontInfo, ch, &xAdvance, &bearing);
    xAdvance = (int)ceil(xAdvance * mScale);
    Glyph* g = new Glyph(width, height, xOffset, mDescent - yOffset - height, xAdvance, node->GetRect().x, node->GetRect().y);
    mGlyphs[ch] = g;

    Font::AtlasRect rc;
    rc.x = node->GetRect().x;
    rc.y = node->GetRect().y;
    rc.width = width;
    rc.height = height;
    mUpdatedRects.push_back(rc);
    return g;
}

void FontImpl::GetAtlasUpdateRects(std::vector<Font::AtlasRect>& updatedRects)
{
    updatedRects = mUpdatedRects;
    mUpdatedRects.clear();
}

void FontImpl::SetFontHeight(int fontHeight)
{
    if (fontHeight < 1)
    {
        fontHeight = 1;
    }
    float scale = stbtt_ScaleForPixelHeight(&mFontInfo, (float)fontHeight);
    mScale = scale;
    int ascent;
    int descent;
    int lineGap;
    stbtt_GetFontVMetrics(&mFontInfo, &ascent, &descent, &lineGap);
    mAscent = (int)(ascent * scale);
    mDescent = -(int)(descent * scale);
}

int FontImpl::GetXAdvance(int ch)
{
    GlyphList::iterator it = mGlyphs.find(ch);
    if (it != mGlyphs.end())
    {
        return it->second->GetXAdvance();
    }
    int xAdvance = 0;
    int bearing = 0;
    stbtt_GetCodepointHMetrics(&mFontInfo, ch, &xAdvance, &bearing);
    return xAdvance;
}

void* FontImpl::GetAtlasBitmap()
{
    return mAtlasBitmap;
}

static int GetUTF8Length(const char *s)
{
    int i = 0;
    int j = 0;
    while (s[i])
    {
        if ((s[i] & 0xc0) != 0x80)
        {
            j++;
        }
        i++;
    }
    return j;
}

static void UTF8ToUTF16(const char* utf8Str, int** out, int* outsize)
{
    unsigned char* p = (unsigned char*)utf8Str;
    int* result = NULL;

    int len = GetUTF8Length(utf8Str);

    result = new int[len + 1];
    memset(result, 0, sizeof(int) * (len + 1));
    int* pDst = result;

    while (*p)
    {
        unsigned char ch = *p;
        if ((ch & 0xFE) == 0xFC)//1111110x
        {
            *pDst++ = ' ';
            p += 6;
        }
        else if ((ch & 0xFC) == 0xF8)//111110xx
        {
            *pDst++ = ' ';
            p += 5;
        }
        else if ((ch & 0xF8) == 0xF0)//11110xxx
        {
            *pDst++ = ' ';
            p += 4;
        }
        else if ((ch & 0xF0) == 0xE0)//1110xxxx
        {
            *pDst++ =
                ((((p[0] << 4) & 0xF0) | ((p[1] >> 2) & 0x0F)) << 8)
                | (((p[1] << 6) & 0xC0) | ((p[2]) & 0x3F));
            p += 3;
        }
        else if ((ch & 0xE0) == 0xC0)//110xxxxx
        {
            *pDst++ =
                (((p[0] >> 2) & 0x07) << 8)
                | (((p[0] << 6) & 0xC0) | (p[1] & 0x3F));
            p += 2;
        }
        else//0xxxxxxx
        {
            *pDst++ = *p++;
        }
    }

    *out = result;
    *outsize = len;
}

void FontImpl::RenderText(const char* text, float xPos, float yPos, float fontHeight, std::vector<float>& vertices)
{
    int* codepoints = NULL;
    int n = 0;
    UTF8ToUTF16(text, &codepoints, &n);
    int xAdvance = 0;
    float scale = fontHeight / mFontHeight;
    for (int i = 0; i < n; ++i)
    {
        int ch = codepoints[i];
        if (ch < 40 && ch != ' ')
        {
            continue;
        }
        Glyph* g = GetGlyph(ch);
        if (g == NULL)
        {
            continue;
        }
        int w = g->GetWidth();
        int h = g->GetHeight();

        float x = xPos + xAdvance + g->GetXOffset();
        float xw = x + w * scale;
        float y = yPos + g->GetYOffset();
        float yh = y + h * scale;
        float tx = g->GetAtlasX() / (float)mAtlasSize;
        float ty = g->GetAtlasY() / (float)mAtlasSize;
        float txw = (g->GetAtlasX() + w) / (float)mAtlasSize;
        float tyh = (g->GetAtlasY() + h) / (float)mAtlasSize;

        vertices.push_back(x); vertices.push_back(y);           vertices.push_back(tx); vertices.push_back(ty);
        vertices.push_back(xw); vertices.push_back(y);       vertices.push_back(txw); vertices.push_back(ty);
        vertices.push_back(xw); vertices.push_back(yh);   vertices.push_back(txw); vertices.push_back(tyh);
        vertices.push_back(x); vertices.push_back(yh);       vertices.push_back(tx); vertices.push_back(tyh);
        xAdvance += (int)(g->GetXAdvance() * scale);
    }

    delete[] codepoints;
}

void FontImpl::BakeText(const char* text, void** bitmap, int* width, int* height)
{
    std::vector<float> vs;
    RenderText(text, 0, 0, (float)mFontHeight, vs);
    size_t n = vs.size() / 16;
    if (n == 0)
    {
        *bitmap = NULL;
        *width = 0;
        *height = 0;
        return;
    }

    float minX = vs[0];
    float minY = vs[1];
    float maxX = vs[0];
    float maxY = vs[1];
    for (int i = 1; i < (int)vs.size() / 4; ++i)
    {
        float* v = &vs[i * 4];
        float x = v[0];
        float y = v[1];
        if (x < minX)
        {
            minX = x;
        }
        if (x > maxX)
        {
            maxX = x;
        }
        if (y < minY)
        {
            minY = y;
        }
        if (y > maxY)
        {
            maxY = y;
        }
    }
    int w = (int)ceil(maxX - minX) + 1;
    int h = (int)ceil(maxY - minY) + 1;

    unsigned char* img = new unsigned char[w * h];
    memset(img, 0, w * h);
    for (size_t i = 0; i < n; ++i)
    {
        float* v = &vs[i * 16];
        int px = (int)(v[0] - minX);
        int py = (int)(v[1] - minY);
        int tx = (int)(v[2] * mAtlasSize);
        int ty = (int)(v[3] * mAtlasSize);
        int tw = (int)(v[4 * 2] - minX - px);
        int th = (int)(v[4 * 2 + 1] - minY - py);

        if (!(px >= 0 && py >= 0 && px + tw < w && py + th < h && tx + tw < mAtlasSize && ty + th < mAtlasSize))
        {
            assert(px >= 0 && py >= 0 && px + tw < w && py + th < h && tx + tw < mAtlasSize && ty + th < mAtlasSize);
        }
        else
        {
            for (int y = 0; y < th; ++y)
            {
                unsigned char* dst = img + ((py + y) * w + px);
                unsigned char* src = mAtlasBitmap + ((ty + y) * mAtlasSize + tx) * 4;
                for (int x = 0; x < tw; ++x)
                {
                    dst[x] = src[x * 4 + 3];
                }
            }
        }
    }

    *bitmap = img;
    *width = w;
    *height = h;
}

void FontImpl::ExportAtlas(const char* path)
{
    unsigned char* flipImg = new unsigned char[mAtlasSize * mAtlasSize * 4];
    for (int i = 0; i < mAtlasSize; ++i)
    {
        memcpy(flipImg + i * mAtlasSize * 4, mAtlasBitmap + (mAtlasSize - 1 - i) * mAtlasSize * 4, mAtlasSize * 4);
    }
    
    SaveImg(path, mAtlasSize, mAtlasSize, 4, flipImg);
    delete[] flipImg;
}
//------------------------------------------------
Font::Font(int fontHeight, int atlasSize, const char* defaultFontPath)
{
    impl = new FontImpl(fontHeight, atlasSize, defaultFontPath);
}

Font::~Font()
{
    delete impl;
}

Glyph* Font::GetGlyph(int ch)
{
    return impl->GetGlyph(ch);
}

int Font::GetXAdvance(int ch)
{
    return impl->GetXAdvance(ch);
}

int Font::GetFontHeight()
{
    return impl->GetFontHeight();
}

int Font::GetAscent()
{
    return impl->GetAscent();
}

int Font::GetDescent()
{
    return impl->GetDescent();
}

void* Font::GetAtlasBitmap()
{
    return impl->GetAtlasBitmap();
}

int Font::GetAtlasSize()
{
    return impl->GetAtlasSize();
}

void Font::RenderText(const char* text, float xPos, float yPos, float fontHeight, std::vector<float>& vertices)
{
    return impl->RenderText(text, xPos, yPos, fontHeight, vertices);
}

void Font::BakeText(const char* text, void** bitmap, int* width, int* height)
{
    return impl->BakeText(text, bitmap, width, height);
}

void Font::GetAtlasUpdateRects(std::vector<Font::AtlasRect>& updatedRects)
{
    impl->GetAtlasUpdateRects(updatedRects);
}

void Font::ExportAtlas(const char* path)
{
    impl->ExportAtlas(path);
}


UIView::UIView(UIApplication* app, float x, float y, float w, float h)
    :mApplication(app)
    ,mPosition(x, y)
    ,mWidth(w)
    ,mHeight(h)
    ,mParentView(NULL)
    ,mLayout(NULL)
{
}

UIView::~UIView()
{
    if (mParentView)
    {
        mParentView->RemoveChildView(this);
    }
    for (size_t i = 0; i < mChildViews.size(); ++i)
    {
        delete mChildViews[i];
    }
}

void UIView::AddChildView(UIView* v)
{
    mChildViews.push_back(v);
    v->mParentView = this;
}

void UIView::RemoveChildView(UIView* v)
{
    for (size_t i = 0; i < mChildViews.size(); ++i)
    {
        if (mChildViews[i] == v)
        {
            std::vector<UIView*>::iterator it = mChildViews.begin();
            it += i;
            mChildViews.erase(it);
            break;
        }
    }
}

int UIView::GetChildViewCount()
{
    return (int)mChildViews.size();

}

UIView* UIView::GetChildView(int index)
{
    if (index < 0 || index >= (int)mChildViews.size())
    {
        return NULL;
    }
    return mChildViews[index];
}

void  UIView::GetRenderables(std::vector<UIRenderable*>& objects)
{
    Render(objects);
    for (size_t i = 0; i < mChildViews.size(); ++i)
    {
        mChildViews[i]->GetRenderables(objects);
    }
}

void UIView::Render(std::vector<UIRenderable*>& /*objects*/)
{
}

void UIView::OnEvent(UIEvent* e)
{
    if (e->GetType() == UIEventTypeDestroy)
    {
        delete this;
        return;
    }
    for (size_t i = 0; i < mListeners.size(); ++i)
    {
        mListeners[i]->OnEvent(e, this);
    }
}

bool UIView::HitTest(float x, float y)
{
    return x >= 0 && x <= mWidth &&
        y >= 0 && y <= mHeight;
}

void UIView::SetPosition(float x, float y)
{
    mPosition.Set(x, y);
    Update();
}

void UIView::SetPosition(const Vector2& position)
{
    mPosition = position;
    Update();
}

void UIView::ModPosition(const Vector2& deltaPosition)
{
    mPosition += deltaPosition;
    Update();
}

void UIView::Resize(float w, float h)
{
    if (w == mWidth && h == mHeight)
    {
        return;
    }
    mWidth = w;
    mHeight = h;
    Update();
}

void UIView::AddListener(UIEventListener* listener)
{
    mListeners.push_back(listener);
}

void UIView::RemoveListener(UIEventListener* listener)
{
    for (size_t i = 0; i < mListeners.size(); ++i)
    {
        if (mListeners[i] == listener)
        {
            std::vector<UIEventListener*>::iterator it = mListeners.begin();
            it += i;
            mListeners.erase(it);
            break;
        }
    }
}

void UIView::SetLayout(UILayout* layout)
{
    if (layout == mLayout)
    {
        return;
    }
    mLayout = layout;
    layout->Update();
}

void UIView::UpdateLayout()
{
    std::vector<Vector2> childSizes;
    float totalMinSize = 0;
    bool verticalLayout = false;
    for (size_t i = 0; i < mChildViews.size(); ++i)
    {
        UIView* child = mChildViews[i];
        Vector2 childSize = child->GetMinSize();
        childSizes.push_back(childSize);
        totalMinSize += verticalLayout ? childSize.y : childSize.x;
    }

    Vector2 maxSize = GetMaxSize();
    float maxS = verticalLayout ? maxSize.y : maxSize.x;
    if (totalMinSize > maxS)
    {
        // error

    }
    else
    {
        // distribute space
        float restSpace = (verticalLayout ? mHeight : mWidth) - totalMinSize;
    }
}

Vector2 UIView::GetMinSize()
{
    return Vector2(0,0);
}

Vector2 UIView::GetMaxSize()
{
    return Vector2(600000,600000);
}

UIView* UIView::GetTopViewAt(float x, float y)
{
    int n = (int)mChildViews.size();
    for (int i = n - 1; i >= 0; --i)
    {
        UIView* v = mChildViews[i];
        float xLocal = x - v->mPosition.x;
        float yLocal = y - v->mPosition.y;
        if (v->HitTest(xLocal, yLocal))
        {
            return v->GetTopViewAt(xLocal, yLocal);
        }
    }

    return this;
}

Vector2 UIView::WorldToLocal(const Vector2& worldPos)
{
    Vector2 offset = mPosition;
    UIView* p = mParentView;
    while (p != NULL)
    {
        offset += p->mPosition;
        p = p->mParentView;
    }
    return worldPos - offset;
}

Vector2 UIView::LocalToWorld(const Vector2& localPos)
{
    Vector2 offset = mPosition;
    UIView* p = mParentView;
    while (p != NULL)
    {
        offset += p->mPosition;
        p = p->mParentView;
    }
    return localPos + offset;
}

void UIView::PostEvent(UIEvent* ev, UIView* sender)
{
    mApplication->PostEvent(ev, sender);
}

void UIView::Destroy()
{
    PostEvent(new UIDestroyEvent(), this);
}

void UIView::Update()
{
    mApplication->Update();
}

UIRootView::UIRootView(UIApplication* app, float x, float y, float w, float h)
    :UIView(app,x,y,w,h)
{
    mUiRect = new UIRect(w, h, Color(0.5f, 0.5f, 0.5f, 1.0f));
    mUiRect->SetPosition(Vector2(x, y));
}

UIRootView::~UIRootView()
{
    delete mUiRect;
}

void UIRootView::Render(std::vector<UIRenderable*>& objects)
{
    mUiRect->SetPosition(LocalToWorld(Vector2(0,0)));
    mUiRect->SetSize(mWidth, mHeight);
    objects.push_back(mUiRect);
}

UIApplication::UIApplication(UIRenderer* renderer, int w, int h, UIApplicationListener* listener)
    :mListener(listener)
    ,mFocusView(NULL)
    ,mRenderer(renderer)
    ,mNeedRepaint(true)
{
    mRoot = new UIRootView(this, 0, 0, (float)w, (float)h);
    mFocusView = mRoot;
    renderer->SetSize(w, h);
    InitUIResources();
}

UIApplication::~UIApplication()
{
    delete mRoot;
}

void UIApplication::Render()
{
    std::vector<UIRenderable*> objects;
    mRoot->GetRenderables(objects);
    mRenderer->Render(objects);
}

void UIApplication::OnEvent(UIEvent* e)
{
    switch(e->GetType())
    {
    case UIEventTypeTouchDown:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            mFocusView = mRoot->GetTopViewAt(ev->GetX(), ev->GetY());
            if (mFocusView)
            {
                UITouchEvent* localEvent = GetLocalTouchEvent(ev);
                mFocusView->OnEvent(localEvent);
                delete localEvent;
            }
        }
        break;
    case UIEventTypeTouchMove:
    case UIEventTypeTouchUp:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            if (mFocusView)
            {
                UITouchEvent* localEvent = GetLocalTouchEvent(ev);
                mFocusView->OnEvent(localEvent);
                delete localEvent;
            }
        }
        break;
    case UIEventTypeResize:
        {
            UIResizeEvent* ev = (UIResizeEvent*)e;
            mRenderer->SetSize((int)ev->GetWidth(), (int)ev->GetHeight());
            mRoot->Resize(ev->GetWidth(), ev->GetHeight());
        }
        break;
    default:
        if (mFocusView)
        {
            mFocusView->OnEvent(e);
        }
        break;
    }

    while (!mEventQueue.empty())
    {
        UIMessage& msg = mEventQueue.front();
        msg.sender->OnEvent(msg.event);
        mEventQueue.pop_front();
        delete msg.event;
    }
    if (mNeedRepaint)
    {
        mListener->OnRepaint();
        mNeedRepaint = false;
    }
}

UITouchEvent* UIApplication::GetLocalTouchEvent(UITouchEvent* e)
{
    Vector2 localPos(e->GetX(), e->GetY());
    Vector2 worldPos = mFocusView->WorldToLocal(localPos);
    return new UITouchEvent(e->GetType(), worldPos.x, worldPos.y, e->GetPressure());
}

void UIApplication::InitUIResources()
{
    mRenderer->AddTexture("window/close", "ui/window/close.png");
}

void UIApplication::PostEvent(UIEvent* ev, UIView* sender)
{
    UIMessage msg(ev, sender);
    mEventQueue.push_back(msg);
}

void UIApplication::Quit()
{
    mListener->OnQuit();
}

void UIApplication::Update()
{
    mNeedRepaint = true;
}

//---------------------------------------------
UIRect::UIRect(float width, float height, const Color& color)
    :mPosition(0,0)
    ,mSize(width, height)
    ,mColor(color)
{
}

UIRect::~UIRect()
{
}

void UIRect::SetPosition(const Vector2& pos)
{
    mPosition = pos;
}

void UIRect::SetSize(float width, float height)
{
    mSize.Set(width, height);
}

void UIRect::SetColor(const Color& color)
{
    mColor = color;
}

//-------------------------------------------------
UIPolygon::UIPolygon(const std::vector<Vector2>& vertices, const Color& color)
    :mPosition(0,0)
    ,mVertices(vertices)
    ,mColor(color)
{
}

UIPolygon::~UIPolygon()
{
}

void UIPolygon::SetPosition(const Vector2& pos)
{
    if (pos == mPosition)
    {
        return;
    }
    Vector2 delta = pos - mPosition;
    for (size_t i = 0; i < mVertices.size(); ++i)
    {
        mVertices[i] += delta;
    }
    mPosition = pos;
}

void UIPolygon::SetColor(const Color& color)
{
    mColor = color;
}

//-------------------------------------------------
UIBitmap::UIBitmap(float w, float h, const UITextureRect& texture)
    :mPosition(0,0)
    ,mSize(w,h)
    ,mOpacity(1.0f)
    ,mTexture(texture)
{

}

void UIBitmap::SetTexture(const UITextureRect& texture)
{
    mTexture = texture;
}

void UIBitmap::SetPosition(const Vector2& pos)
{
    mPosition = pos;
}

void UIBitmap::SetSize(float width, float height)
{
    mSize.Set(width, height);
}

void UIBitmap::SetOpacity(float opacity)
{
    mOpacity = opacity;
}
//--------------------------------------
UIText::UIText(const std::string& text, float fontHeight)
    :mText(text)
    , mPosition(0,0)
    , mColor(0,0,0,1)
    , mFontHeight(fontHeight)
{
    const AABB2& box = GLRenderer::GetInstance()->GetTextSize(mText.c_str(), mFontHeight);
    mWidth = box.Width();
    mHeight = box.Height();
}

void UIText::SetText(const std::string& text)
{
    mText = text;
}

void UIText::SetPosition(const Vector2& pos)
{
    mPosition = pos;
}

void UIText::SetColor(const Color& color)
{
    mColor = color;
}

void UIText::SetFontHeight(float fontHeight)
{
    mFontHeight = fontHeight;
}
//--------------------------------------
UIWindow::UIWindow(UIApplication* app, float x, float y, float w, float h)
    :UIView(app, x, y, w, h)
    ,mTitleListener(this)
{
    mBgRect = new UIRect(mWidth, mHeight, Color(1.0f,1.0f,1.0f,1));

    UIWindowTitleBar* titleBar = new UIWindowTitleBar(app, this);
    AddChildView(titleBar);

    mContainerView = new UIView(app, 0, titleBar->GetHeight(), w, h - titleBar->GetHeight());
    AddChildView(mContainerView);
}

UIWindow::~UIWindow()
{
    delete mBgRect;
}

void UIWindow::Render(std::vector<UIRenderable*>& objects)
{
    mBgRect->SetPosition(LocalToWorld(Vector2(0,0)));
    objects.push_back(mBgRect);
}

void UIWindow::AddContentView(UIView* view)
{
    mContainerView->AddChildView(view);
}

UIWindow::WinTitleListener::WinTitleListener(UIWindow* win)
    :mWindow(win)
{

}

bool UIWindow::WinTitleListener::OnEvent(UIEvent* event, UIView* /*sender*/)
{
    switch (event->GetType())
    {
    case UIEventTypeClick:
        {
            UICloseEvent* closeEvent = new UICloseEvent();
            mWindow->PostEvent(closeEvent, mWindow);
        }
        break;
    default:
        break;
    }
    return true;
}


UIWindowTitleBar::UIWindowTitleBar(UIApplication* app, UIWindow* window)
    :UIView(app, 0, 0, window->GetWidth(), 30)
{
    mBgRect = new UIRect(mWidth, mHeight, Color(0.7f,0.7f,0.7f,1));
    mTitleLabel = new UIText("Untitled", 12);

    int w = 17;
    int h = 17;
    UITextureRect tex;
    tex.name = "window/close";
    tex.x = 0;
    tex.y = 0;
    tex.width = w;
    tex.height = h;
    UIButton* closeButton = new UIButton(app, (float)w, (float)h, tex, tex);
    closeButton->SetPosition(mWidth - w - (mHeight - h) / 2, (mHeight - h) / 2);
    closeButton->AddListener(&window->mTitleListener);
    AddChildView(closeButton);
}

UIWindowTitleBar::~UIWindowTitleBar()
{
    delete mTitleLabel;
    delete mBgRect;
}

void UIWindowTitleBar::OnEvent(UIEvent* e)
{
    switch(e->GetType())
    {
    case UIEventTypeTouchDown:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            mDragStartPosition = LocalToWorld(Vector2(ev->GetX(), ev->GetY()));
            mWindowStartPosition = mParentView->GetPosition();
            //LOGE("Start %f,%f  %f,%f", mDragStartPosition.x, mDragStartPosition.y, mParentView->GetPosition().x, mParentView->GetPosition().y);
        }
        break;
    case UIEventTypeTouchMove:
    case UIEventTypeTouchUp:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            Vector2 pos = LocalToWorld(Vector2(ev->GetX(), ev->GetY()));
            mParentView->SetPosition(mWindowStartPosition + pos - mDragStartPosition);
            //LOGE("Move %f,%f  %f,%f", pos.x, pos.y, mParentView->GetPosition().x, mParentView->GetPosition().y);
            Update();
        }
        break;
    default:
        break;
    }
}

void UIWindowTitleBar::Render(std::vector<UIRenderable*>& objects)
{
    mBgRect->SetPosition(LocalToWorld(Vector2(0,0)));
    objects.push_back(mBgRect);
    mTitleLabel->SetPosition(LocalToWorld(Vector2(10, 0)));
    objects.push_back(mTitleLabel);
}

//--------------------------------------
UIButton::UIButton(UIApplication* app, float w, float h, const UITextureRect& normalTexture, const UITextureRect& pressedTexture)
    :UIView(app,0,0,w,h)
    ,mNormalTexture(normalTexture)
    ,mPressedTexture(pressedTexture)
    ,mPressed(false)
{
    mBitmap = new UIBitmap(w, h, normalTexture);
    mText = NULL;
    mRect = NULL;
}

UIButton::UIButton(UIApplication* app, const char* text)
    :UIView(app, 0, 0, 1, 1)
    ,mPressed(false)
    ,mPadding(10)
{
    mText = new UIText(text, 24);
    float w = mText->GetWidth() + mPadding * 2;
    float h = 24 + mPadding * 2;
    Resize(w, h);
    mRect = new UIRect(w, h, Color(0.7f, 0.7f, 0.7f, 1.0f));
    mBitmap = NULL;
}

UIButton::~UIButton()
{
    delete mBitmap;
    delete mText;
    delete mRect;
}

void UIButton::OnEvent(UIEvent* e)
{
    switch(e->GetType())
    {
    case UIEventTypeTouchDown:
        {
            mPressed = true;
            Update();
        }
        break;
    case UIEventTypeTouchMove:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            mPressed = HitTest(ev->GetX(), ev->GetY());
            Update();
        }
        break;
    case UIEventTypeTouchUp:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            //if (HitTest(ev->GetX(), ev->GetY()))
            {
                PostEvent(new UIClickEvent(ev->GetX(), ev->GetY()), this);
            }
            mPressed = false;
            Update();
        }
        break;
    case UIEventTypeClick:
        {
            UIView::OnEvent(e);
        }
        break;
    default:
        break;
    }
}

void UIButton::Render(std::vector<UIRenderable*>& objects)
{
    if (mBitmap)
    {
        mBitmap->SetTexture(mPressed ? mPressedTexture : mNormalTexture);
        mBitmap->SetPosition(LocalToWorld(Vector2(0, 0)));
        objects.push_back(mBitmap);
    }
    if (mText)
    {
        mRect->SetPosition(LocalToWorld(Vector2(0, 0)));
        objects.push_back(mRect);
        mText->SetPosition(LocalToWorld(Vector2(mPadding, mPadding)));
        objects.push_back(mText);
    }
}

//--------------------------------------
UISlider::UISlider(UIApplication* app, float maxValue, float w, float h)
    :UIView(app,0,0,w,h)
    ,mMaxValue(maxValue)
    ,mPadding(2.0f)
{
    mText = new UIText("     ", 24);
    mRect = new UIRect(w, h, Color(0.7f, 0.7f, 0.7f, 1.0f));
    mCursor = new UIRect(12, h - mPadding * 2.0f, Color(0.5f, 0.5f, 0.5f, 1.0f));
}

UISlider::~UISlider()
{
    delete mText;
    delete mRect;
    delete mCursor;
}

void UISlider::OnEvent(UIEvent* e)
{
    switch(e->GetType())
    {
    case UIEventTypeTouchDown:
    case UIEventTypeTouchMove:
    case UIEventTypeTouchUp:
        {
            UITouchEvent* ev = (UITouchEvent*)e;
            UpdateValue(ev->GetX(), ev->GetY());
        }
        break;
    case UIEventTypeValueChange:
        {
            UIView::OnEvent(e);
        }
        break;
    default:
        break;
    }
}

void UISlider::Render(std::vector<UIRenderable*>& objects)
{
    mRect->SetPosition(LocalToWorld(Vector2(0, 0)));
    objects.push_back(mRect);
    Vector2 cursorPos(mPadding + (mValue / mMaxValue) * (mWidth - mPadding * 2) - mCursor->GetSize().x * 0.5f, mPadding);
    mCursor->SetPosition(LocalToWorld(cursorPos));
    objects.push_back(mCursor);
    mText->SetPosition(LocalToWorld(Vector2(0, 0)));
    char buf[50];
    sprintf(buf, "%5.f", mValue);
    mText->SetText(buf);
    objects.push_back(mText);
}

float UISlider::GetValue()
{
    return mValue;
}

void UISlider::SetValue(float v)
{
    if (v < 0)
    {
        v = 0;
    }
    if (v > mMaxValue)
    {
        v = mMaxValue;
    }
    if (v == mValue)
    {
        return;
    }
    mValue = v;
    Update();
    PostEvent(new UIValueChangeEvent(mValue), this);
}

void UISlider::UpdateValue(float x, float /*y*/)
{
    float s = (x - mPadding) / (mWidth - mPadding);
    SetValue(s * mMaxValue);
}

//--------------------------------------
UIImageView::UIImageView(UIApplication* app, UIRenderTarget* target)
    :UIView(app, 0, 0, (float)target->GetWidth(), (float)target->GetHeight())
{
    UITextureRect tex;
    tex.x = 0;
    tex.y = 0;
    tex.width = target->GetWidth();
    tex.height = target->GetHeight();
    tex.name = target->GetName();
    mBitmap = new UIBitmap((float)tex.width, (float)tex.height, tex);
}

UIImageView::UIImageView(UIApplication* app, float w, float h, const UITextureRect& texture)
    :UIView(app, 0, 0, w, h)
{
    mBitmap = new UIBitmap(w, h, texture);
}

UIImageView::~UIImageView()
{
    delete mBitmap;
}

void UIImageView::Render(std::vector<UIRenderable*>& objects)
{
    mBitmap->SetPosition(LocalToWorld(Vector2(0, 0)));
    objects.push_back(mBitmap);
}
//--------------------------------------
UIGLRenderTarget::UIGLRenderTarget(const std::string& name, int width, int height)
    :mName(name)
    ,mTarget(NULL)
{
    mTarget = GLRenderer::GetInstance()->CreateTarget(width, height, false, false);
}

UIGLRenderTarget::~UIGLRenderTarget()
{
    delete mTarget;
}

void UIGLRenderTarget::SetTarget(GLRenderTarget* target)
{
    if (target == mTarget)
    {
        return;
    }
    if (mTarget)
    {
        delete mTarget;
        mTarget = NULL;
    }
    mTarget = target;
}
//--------------------------------------

UIGLRenderer::UIGLRenderer()
{

}

UIGLRenderer::~UIGLRenderer()
{
    for (std::map<std::string, GLTexture*>::iterator it = mTextures.begin(); it != mTextures.end(); ++it)
    {
        delete it->second;
    }
    for (std::map<std::string, UIGLRenderTarget*>::iterator it = mTargets.begin(); it != mTargets.end(); ++it)
    {
        delete it->second;
    }
}

void UIGLRenderer::Render(std::vector<UIRenderable*>& objects)
{
    GLRenderer* renderer = GLRenderer::GetInstance();
    GLRenderTarget* target = renderer->GetDefaultTarget();
    Canvas2d* canvas = renderer->GetCanvas2d();
    Canvas2dState state;
    state.target = target;
    state.blendMode = BlendModeMix;
    canvas->SetState(state);

    for (size_t i = 0; i < objects.size(); ++i)
    {
        UIRenderable* o = objects[i];
        switch (o->GetType())
        {
        case UIRenderableTypeRect:
            {
                UIRect* rc = (UIRect*)o;
                state.texture = renderer->GetDefaultTexture();
                canvas->SetState(state);
                canvas->DrawRect(rc->GetPosition().x, mHeight - rc->GetPosition().y - rc->GetSize().y, rc->GetSize().x, rc->GetSize().y, rc->GetColor());
                AABB2 rect;
                rect.xMin = rc->GetPosition().x;
                rect.yMin = mHeight - rc->GetPosition().y - rc->GetSize().y;
                rect.xMax = rect.xMin + rc->GetSize().x;
                rect.yMax = rect.yMin + rc->GetSize().y;
                canvas->DrawRectOutline(rect, Color(0.2f,0.2f,0.2f,1.0f),1.0f,Canvas2d::OutlineDirectionInner);
            }
            break;
        case UIRenderableTypePolygon:
            {
                UIPolygon* rc = (UIPolygon*)o;
                state.texture = renderer->GetDefaultTexture();
                canvas->SetState(state);
                canvas->DrawPolygon(&rc->GetVertices()[0], (int)rc->GetVertices().size(), rc->GetColor());
            }
            break;
        case UIRenderableTypeBitmap:
            {
                UIBitmap* bmp = (UIBitmap*)o;
                state.texture = GetTexture(bmp->GetTexture().name);
                canvas->SetState(state);
                AABB2 dst;
                dst.xMin = bmp->GetPosition().x;
                dst.yMin = mHeight - bmp->GetPosition().y - bmp->GetSize().y;
                dst.xMax = dst.xMin + bmp->GetSize().x;
                dst.yMax = dst.yMin + bmp->GetSize().y;
                AABB2 src;
                src.xMin = (float)bmp->GetTexture().x;
                src.yMin = (float)bmp->GetTexture().y;
                src.xMax = (float)src.xMin + bmp->GetTexture().width;
                src.yMax = (float)src.yMin + bmp->GetTexture().height;
                canvas->DrawImage(dst, src, Color(1,1,1,bmp->GetOpacity()));
            }
            break;
        case UIRenderableTypeLabel:
        {
            UIText* label = (UIText*)o;
            canvas->SetState(state);
            canvas->DrawString(label->GetText().c_str(), label->GetPosition().x, mHeight - label->GetFontHeight() - label->GetPosition().y, label->GetFontHeight(), label->GetColor());
        }
        break;
        default:
            break;
        }
    }
}

void UIGLRenderer::AddTexture(const std::string& name, const std::string& path)
{
    GLTexture* texture = GLRenderer::GetInstance()->CreateTexture(path.c_str());
    if (texture)
    {
        mTextures[name] = texture;
    }
}

GLTexture* UIGLRenderer::GetTexture(const std::string& name)
{
    std::map<std::string, GLTexture*>::iterator it = mTextures.find(name);
    if (it != mTextures.end())
    {
        return it->second;
    }
    std::map<std::string, UIGLRenderTarget*>::iterator it2 = mTargets.find(name);
    if (it2 != mTargets.end())
    {
        return it2->second->GetTarget()->GetTexture();
    }
    return GLRenderer::GetInstance()->GetDefaultTexture();
}

UIRenderTarget* UIGLRenderer::AddRenderTarget(const std::string& name, int width, int height)
{
    UIGLRenderTarget* target = new UIGLRenderTarget(name, width, height);
    if (target)
    {
        mTargets[name] = target;
    }
    return target;
}

void UIGLRenderer::AddFont(const std::string& /*name*/, const std::string& /*path*/)
{

}

void UIGLRenderer::SetSize(int w, int h)
{
    mWidth = w;
    mHeight = h;
    GLRenderer::GetInstance()->GetDefaultTarget()->Replace(0, w, h);
}


GLSceneRenderer::GLSceneRenderer()
{
    mTarget = GLRenderer::GetInstance()->GetDefaultTarget();
}

GLSceneRenderer::~GLSceneRenderer()
{

}

GLRenderTarget* GLSceneRenderer::GetRenderTarget()
{
    return mTarget;
}

void GLSceneRenderer::ResizeRenderTarget(int /*width*/, int /*height*/)
{
    
}

void GLSceneRenderer::SetCamera(Camera3dComponent* camera)
{
    mCamera = camera;
}

void GLSceneRenderer::SetScene(Scene* scene)
{
    mScene = scene;
}

void GLSceneRenderer::Render()
{
    if (!mTarget || !mScene || !mCamera)
    {
        return;
    }
    
    // render shadow maps
    
    // render gbuffer

    // render ao map

    // shading opaque objects

    // render transparent objects

    // post processing
}
