#include "mainwidget.h"
#include <QMouseEvent>
#include <QSurfaceFormat>
#include <QDebug>
#include "glrenderer.h"
#include "gesturedetector.h"

MainWidget::MainWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , mRenderer(NULL)
    , mCamera(new Camera2())
    , mFrameIndex(0)
{
    connect(&mTimer, SIGNAL(timeout()), this, SLOT(onTimer()));

    setAttribute(Qt::WA_AcceptTouchEvents);
#ifdef _WIN32
    setFocusPolicy(Qt::StrongFocus);
#endif
    mGestureDetector = new GestureDetector();

    // Added by priority desc
    PinchRecognizer* pinch = new PinchRecognizer(this);
    mGestureDetector->addGestureRecognizer(pinch);
    connect(pinch, &PinchRecognizer::zoom, this, &MainWidget::onZoom);
    connect(pinch, &PinchRecognizer::zoomStart, this, &MainWidget::onZoomStart);
    connect(pinch, &PinchRecognizer::pan, this, &MainWidget::onTwoPan);
    connect(pinch, &PinchRecognizer::panStart, this, &MainWidget::onTwoPanStart);

    PanRecognizer* pan = new PanRecognizer(this);
    mGestureDetector->addGestureRecognizer(pan);
    connect(pan, &PanRecognizer::panStart, this, &MainWidget::onPanStart);
    connect(pan, &PanRecognizer::pan, this, &MainWidget::onPan);

    TapRecognizer* tap = new TapRecognizer(this);
    mGestureDetector->addGestureRecognizer(tap);
    connect(tap, &TapRecognizer::tap, this, &MainWidget::onTap);
    connect(tap, &TapRecognizer::doubleTap, this, &MainWidget::onDoubleTap);
}

MainWidget::~MainWidget()
{
    delete mGestureDetector;
    for (int i = 0; i < mTextures.size(); ++i)
    {
        delete mTextures[i];
    }
    delete mCamera;
    delete mRenderer;
}

void MainWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton)
    {
        mZoom = true;
        mCamera->OnZoomBegin(event->pos().x(), height() - event->pos().y());
    }
    else
    {
        mZoom = false;
        mCamera->OnPanBegin(event->pos().x(), height() - event->pos().y());
    }
}

void MainWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (mZoom)
    {
        mCamera->OnZoomEnd(event->pos().x(), height() - event->pos().y());
    }
    else
    {
        mCamera->OnPanEnd(event->pos().x(), height() - event->pos().y());
    }
    update();
}

void MainWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (mZoom)
    {
        mCamera->OnZoomMove(event->pos().x(), height() - event->pos().y());
    }
    else
    {
        mCamera->OnPanMove(event->pos().x(), height() - event->pos().y());
    }
    update();
}

void MainWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    emit doubleTap(event->pos());
}

void MainWidget::wheelEvent(QWheelEvent *event)
{
    if (event->modifiers() & Qt::ControlModifier)
    {
        int y = height() - event->pos().y();
        mCamera->OnZoomBegin(event->pos().x(), y);
        float delta = event->angleDelta().y() > 0 ? 40.0f: -40.0f;
        mCamera->OnZoomMove(event->pos().x() + delta, y);
        update();
    }
    else
    {
        if (event->angleDelta().y() > 0)
        {
            emit wheelUp();
        }
        else
        {
            emit wheelDown();
        }
    }
}

void MainWidget::initializeGL()
{
    mRenderer = new GLRenderer();
    mDefaultTarget = mRenderer->GetDefaultTarget();
    mDefaultTarget->Replace(this->defaultFramebufferObject(), devicePixelRatio() * width(), devicePixelRatio() * height());
}

void MainWidget::resizeGL(int w, int h)
{
    mDefaultTarget->Replace(this->defaultFramebufferObject(), devicePixelRatio() * w, devicePixelRatio() * h);
}

void MainWidget::paintGL()
{
    mRenderer->GetDefaultTarget()->Clear(Color(0.5f,0.5f,0.5f,1));

    Canvas2d* canvas = mRenderer->GetCanvas2d();
    Canvas2dState state;
    state.transform = mCamera->GetTransform();

    if (mFrameIndex >= 0 && mFrameIndex < mTextures.size())
    {
        GLTexture* texture = mTextures[mFrameIndex];
        state.texture = texture;
        if (texture->GetChannels() == 1)
        {
            state.blendMode = BlendModeGreyscaleMix;
        }
        canvas->SetState(state);
        canvas->DrawImage(0, 0, AABB2(0,0,state.texture->GetWidth(),state.texture->GetHeight()), Color(1,1,1,1));
    }
}

bool MainWidget::event(QEvent *event)
{
    if (event->type() == QEvent::TouchBegin ||
        event->type() == QEvent::TouchUpdate ||
        event->type() == QEvent::TouchEnd ||
        event->type() == QEvent::TouchCancel)
    {
        event->accept();
#ifdef __APPLE__
        QTouchEvent* e = (QTouchEvent*)event;
        QList<QTouchEvent::TouchPoint> tps = e->touchPoints();
        for (int i = 0; i < tps.size(); ++i)
        {
            QTouchEvent::TouchPoint& p = tps[i];
            p.setPos(p.pos() * devicePixelRatio());
            p.setStartPos(p.startPos() * devicePixelRatio());
            p.setLastPos(p.lastPos() * devicePixelRatio());
        }
        QTouchEvent* te = new QTouchEvent(e->type(), e->device(), e->modifiers(), e->touchPointStates(), tps);
        mGestureDetector->touchEvent(te);
#else
        mGestureDetector->touchEvent((QTouchEvent*)event);
#endif
        return true;
    }

    return QOpenGLWidget::event(event);
}

void MainWidget::onTimer()
{
    if (mDelays.size() > 0)
    {
        update();
        mFrameIndex = (mFrameIndex + 1) % mDelays.size();
        mTimer.setSingleShot(true);
        mTimer.setInterval(mDelays[mFrameIndex]);
        mTimer.start();
    }
}

void MainWidget::onPan(QPointF position)
{
    mCamera->OnPanMove(position.x(), height() - position.y());
    update();
}

void MainWidget::onPanStart(QPointF position)
{
    mCamera->OnPanBegin(position.x(), height() - position.y());
}

void MainWidget::onTwoPan(QPointF position)
{

}

void MainWidget::onTwoPanStart(QPointF position)
{

}

void MainWidget::onTap(QPointF pos)
{
    emit tap(pos);
}

void MainWidget::onDoubleTap(QPointF pos)
{
    emit doubleTap(pos);
}

void MainWidget::onZoomStart(QPointF pos)
{
    mCamera->OnPinchZoomBegin(pos.x(), height() * devicePixelRatio() - 1 - pos.y());
}

void MainWidget::onZoom(QPointF pos, float scale)
{
    mCamera->OnPinchZoomMove(scale);
	update();
}

void MainWidget::Load(int w, int h, int pixelSize, void* data, int frameIndex, int frameDelay)
{
    QTime time;
    time.start();
	makeCurrent();
    if (frameIndex < mTextures.size())
    {
        GLTexture* texture = mTextures[frameIndex];
        if (texture && texture->GetWidth() == w && texture->GetHeight() == h && texture->GetChannels() == pixelSize)
        {
            texture->WriteTexture(data, w, h, true);
        }
        else
        {
            delete texture;
            mTextures[frameIndex] = mRenderer->CreateTexture(w, h, data, pixelSize, true, true, false, true);
        }
    }
    else
    {
        GLTexture* texture = mRenderer->CreateTexture(w, h, data, pixelSize, true, true, false, true);
        mTextures.push_back(texture);
    }
    qDebug() << "gpu=" << time.elapsed();
    mDelays.push_back(frameDelay);
}


void MainWidget::BeginLoad()
{
    mTimer.stop();
    mFrameIndex = 0;
    mDelays.clear();
}

void MainWidget::EndLoad()
{
    if (mDelays.size() == 0)
    {
        mFrameIndex = -1;
    }
    else if (mDelays.size() > 1)
    {
		mFrameIndex = (mFrameIndex + 1) % mDelays.size();
        mTimer.setSingleShot(true);
        mTimer.setInterval(mDelays[mFrameIndex]);
        mTimer.start();
    }
    update();
}

void MainWidget::SetTransform(const Transform2& t)
{
    mCamera->SetTransform(t);
    update();
}

void MainWidget::PlayOrStop()
{
    if (mDelays.size() > 1)
    {
        if (mTimer.isActive())
        {
            mTimer.stop();
        }
        else
        {
            mTimer.start();
        }
    }
}

int MainWidget::GetFrameIndex()
{
    return mFrameIndex;
}
