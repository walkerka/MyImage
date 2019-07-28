#include "glwidget.h"
#include <QPainter>
#include <QOpenGLPaintDevice>
#include <math.h>

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , mSurface(NULL)
    , mThread(NULL)
{

}

GLWidget::~GLWidget()
{
    mThread->stop();
    mSurface->destroy();
}


//! [3]
static const char *vertexShaderSource =
    "attribute vec4 posAttr;\n"
    "uniform mat4 matrix;\n"
    "varying vec2 texcoord;\n"
    "void main() {\n"
    "   texcoord = posAttr.zw;\n"
    "   gl_Position = matrix * vec4(posAttr.xy,0.0,1.0);\n"
    "}\n";

static const char *fragmentShaderSource =
#ifdef __android__
    "precision highp float;"
#endif
    "uniform sampler2D tex;"
    "varying vec2 texcoord;\n"
    "void main() {\n"
    "   gl_FragColor = texture2D(tex, texcoord);\n"
    "}\n";

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    m_program = new QOpenGLShaderProgram(this);
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
    m_program->link();
    m_posAttr = m_program->attributeLocation("posAttr");
    m_matrixUniform = m_program->uniformLocation("matrix");
    m_program->bind();


    createRenderThread();
}

void GLWidget::resizeGL(int w, int h)
{
}

void GLWidget::paintGL()
{
    if (!mThread)
    {
        return;
    }

    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int textureId = mThread->GetTextureId();
    if (textureId)
    {
        //qDebug("foreground rendering textureId=%d", textureId);


        glViewport(0, 0, width(), height());

        glClear(GL_COLOR_BUFFER_BIT);

        QMatrix4x4 matrix;
        m_program->setUniformValue(m_matrixUniform, matrix);

        glBindTexture(GL_TEXTURE_2D, textureId);
        m_program->setUniformValue("tex", 0);

        GLfloat vertices[] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f
        };
        glVertexAttribPointer(m_posAttr, 4, GL_FLOAT, GL_FALSE, 0, vertices);
        glEnableVertexAttribArray(0);

        GLushort indices[] = {
            0, 1, 2,
            0, 2, 3
        };


        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);
    }
    update();
}

void GLWidget::createRenderThread()
{
    QOpenGLContext* ctx = new QOpenGLContext();
    ctx->setShareContext(context());
    ctx->setFormat(QSurfaceFormat::defaultFormat());
    ctx->create();

    mSurface = new QOffscreenSurface();
    mSurface->setFormat(QSurfaceFormat::defaultFormat());
    mSurface->create();

    mThread = new RenderThread(ctx, mSurface);
    ctx->moveToThread(mThread);
    mThread->start();
}

void GLWidget::Load(int w, int h, void* data)
{
//    if (mTexture && mTexture->GetWidth() == w && mTexture->GetHeight() == h)
//    {
//        mTexture->WriteTexture(data, w, h, true);
//    }
//    else
//    {
//        delete mTexture;
//        mTexture = mRenderer->CreateTexture(w, h, data);
//    }
    update();
}


RenderThread::RenderThread(QOpenGLContext* ctx, QSurface* surface)
    : mContext(ctx)
    , mSurface(surface)
    , mDone(false)
    , mTextureId(0)
{

}

RenderThread::~RenderThread()
{
    delete mContext;
}

void RenderThread::stop()
{
    mDone = true;
}

void RenderThread::run()
{
    mContext->makeCurrent(mSurface);
    initializeOpenGLFunctions();

    QOpenGLFramebufferObject* fbo = new QOpenGLFramebufferObject(1024,1024);
    mTextureId = fbo->texture();
    qDebug("create fbo. textureId=%d", mTextureId);

    double t = 0;
    while (!mDone)
    {
        //qDebug("background rendering.");
        fbo->bind();

        QOpenGLPaintDevice device(fbo->width(), fbo->height());
        QPainter painter;
        painter.begin(&device);
        painter.setBrush(QColor(255,255,255));
        painter.drawRect(0, 0, 1024, 1024);
        painter.setBrush(QColor((cos(t) + 1) * 0.5 * 255,0,(sin(t) + 1) * 0.5 * 255));
        painter.drawEllipse(500, 500, 100, 100);
        QFont font;
        font.setFamily("Arial");
        font.setPixelSize(50);
        painter.setFont(font);
        painter.drawText(100, 100, "OpenGL in background thread");
        painter.end();

//        fbo->bind();
//        glClearColor(cos(t),0,sin(t),1);
//        glClear(GL_COLOR_BUFFER_BIT);
        glFinish();
        t += 0.01f;
    }
    mContext->doneCurrent();
}
