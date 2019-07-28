#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QOpenGLWidget>
#include <QTimer>

class GestureDetector;
class GLRenderer;
class GLRenderTarget;
class GLTexture;
class Camera2;
class Transform2;

class MainWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
    MainWidget(QWidget *parent = 0);
    ~MainWidget();

    void BeginLoad();
    void Load(int w, int h, int pixelSize, void* data, int frameIndex = 0, int frameDelay = 0);
    void EndLoad();
    void SetTransform(const Transform2& t);
    void PlayOrStop();
    int GetFrameIndex();

signals:
    void wheelUp();
    void wheelDown();
    void tap(QPointF pos);
    void doubleTap(QPointF pos);

public slots:
    void onTimer();
    void onPan(QPointF position);
    void onPanStart(QPointF position);
    void onTwoPan(QPointF position);
    void onTwoPanStart(QPointF position);
    void onTap(QPointF pos);
    void onDoubleTap(QPointF pos);
    void onZoomStart(QPointF pos);
    void onZoom(QPointF pos, float scale);

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    bool event(QEvent *event);

private:
    GLRenderer* mRenderer;
    GLRenderTarget* mDefaultTarget;
    Camera2* mCamera;
    QTimer mTimer;
    int mFrameIndex;
    QList<int> mDelays;
    QList<GLTexture*> mTextures;
    GestureDetector* mGestureDetector;
    bool mZoom;
};

#endif // MAINWIDGET_H
