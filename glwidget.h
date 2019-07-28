#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QOpenGLFunctions>
#include <QOpenGLFramebufferObject>
#include <QOpenGLShaderProgram>

#include <QThread>

class RenderThread: public QThread, QOpenGLFunctions
{
    Q_OBJECT
public:
    RenderThread(QOpenGLContext* ctx, QSurface* surface);
    ~RenderThread();
    void run();
    void stop();
    int GetTextureId() { return mTextureId; }
private:
    QOpenGLContext* mContext;
    QSurface* mSurface;
    bool mDone;
    int mTextureId;
};


class GLWidget : public QOpenGLWidget, QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit GLWidget(QWidget *parent = 0);
    ~GLWidget();

    void createRenderThread();

    void Load(int w, int h, void* data);

signals:

public slots:

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

private:
    QOffscreenSurface* mSurface;
    RenderThread* mThread;
    QOpenGLShaderProgram* m_program;
    int m_posAttr;
    int m_matrixUniform;
};

#endif // GLWIDGET_H
