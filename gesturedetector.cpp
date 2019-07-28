#include "gesturedetector.h"
#include <QDebug>
#include <QLineF>
#include <math.h>

static float PAN_TOLERANCE = 25;
static float LONG_PRESS_TOLERANCE = 25;
static float LONG_PRESS_INTERVAL = 1000;
static float TAP_TOLERANCE = 25;
static float TAP_INTERVAL = 200;
static float DOUBLE_TAP_INTERVAL = 200;
static float PINCH_TOLERANCE = 70;

GestureDetector::GestureDetector()
    :mCurrent(NULL)
{
}

void GestureDetector::touchEvent(QTouchEvent *event)
{
    switch (event->type())
    {
    case QEvent::TouchBegin:
    {
        mTime.restart();
        break;
    }
    case QEvent::TouchEnd:
    {
        break;
    }
    case QEvent::TouchCancel:
    {
        break;
    }
    case QEvent::TouchUpdate:
    {
        break;
    }
    default:
        break;
    }

    for (int i = 0; i < mRecognizers.size(); ++i)
    {
        GestureRecognizer* r = mRecognizers[i];
        if (r == mCurrent)
        {
            break;
        }        
        if (r->onTouch(event, mHistory))
        {
            if (mCurrent)
            {
                stopCurrentGesture();
            }
            startGesture(r);
            break;
        }
    }
    if (mCurrent)
    {
        if (!mCurrent->onTouch(event, mHistory))
        {
            stopCurrentGesture();
        }
    }
    saveHistory(event);
}

void GestureDetector::saveHistory(QTouchEvent* event)
{
    GestureTouch touch;
    foreach(const QTouchEvent::TouchPoint& tp, event->touchPoints())
    {
        touch.points.push_back(tp.pos());
    }
    touch.t = mTime.elapsed();
    mHistory.push_back(touch);
}

void GestureDetector::stopCurrentGesture()
{
    mCurrent->stopGesture();
    mCurrent = NULL;
}

void GestureDetector::startGesture(GestureRecognizer* r)
{
    r->startGesture();
    mCurrent = r;
}

void GestureDetector::addGestureRecognizer(GestureRecognizer* r)
{
    mRecognizers.push_back(r);
}
//****************************************************
GestureRecognizer::GestureRecognizer(QObject* parent)
    :QObject(parent)
    ,mGestureStarted(false)
{
}

GestureRecognizer::~GestureRecognizer()
{
}

void GestureRecognizer::startGesture()
{
    mGestureStarted = true;
}

void GestureRecognizer::stopGesture()
{
    mGestureStarted = false;
}
//****************************************************
static float distance(const QPointF& p0, const QPointF& p1)
{
    return QVector2D(p0).distanceToPoint(QVector2D(p1));
}

static float getRotateAngle(const QTouchEvent::TouchPoint& p0, const QTouchEvent::TouchPoint& p1)
{
    return -QLineF(p0.pos(), p1.pos()).angle() + QLineF(p0.startPos(), p1.startPos()).angle();
}

static float getScale(const QTouchEvent::TouchPoint& p0, const QTouchEvent::TouchPoint& p1)
{
    return QLineF(p0.pos(), p1.pos()).length() / QLineF(p0.startPos(), p1.startPos()).length();
}

static float getTilt(const QTouchEvent::TouchPoint& point0, const QTouchEvent::TouchPoint& point1)
{
    float startY = (point0.startPos().y() + point1.startPos().y()) * 0.5f;
    float endY = (point0.pos().y() + point1.pos().y()) * 0.5f;
    return endY - startY;
}
//****************************************************
PinchRecognizer::PinchRecognizer(QObject* parent)
    :GestureRecognizer(parent)
    ,mState(StateNone)
    ,mZoomSpeed(0)
    ,mRotateSpeed(0)
    ,mStartLength(1)
    ,mStartTilt(0)
{
}

bool PinchRecognizer::onTouch(QTouchEvent* event, QList<GestureTouch>& history)
{
    if (event->touchPoints().size() != 2)
    {
        return false;
    }

    QTouchEvent::TouchPoint tp0 = event->touchPoints()[0];
    QTouchEvent::TouchPoint tp1 = event->touchPoints()[1];

    if (tp0.state() == Qt::TouchPointReleased || tp1.state() == Qt::TouchPointReleased)
    {
        return false;
    }

    if (mGestureStarted)
    {
        switch (mState)
        {
        case StateZoom:
        {
            float length = distance(tp0.pos(), tp1.pos());
            emit zoom(mCenter, length / mStartLength);
            break;
        }
        case StateRotate:
        {
            float angle = getRotateAngle(tp0, tp1);
            emit rotate(mCenter, angle - mStartAngle);
            break;
        }
        case StatePan:
        {
            emit pan((tp0.pos() + tp1.pos()) * 0.5f);
            break;
        }
        default:
            break;
        }

        return true;
    }
    else
    {
        if (distance(tp0.pos(), tp0.startPos()) >= PINCH_TOLERANCE ||
            distance(tp1.pos(), tp1.startPos()) >= PINCH_TOLERANCE)
        {
            float lastLength = distance(tp0.startPos(), tp1.startPos());
            float length = distance(tp0.pos(), tp1.pos());
            float deltaLength = length - lastLength;
            if (deltaLength < 0)
            {
                deltaLength = -deltaLength;
            }

            float deltaRotate = M_PI * ((lastLength + length) * 0.5f) * getRotateAngle(tp0, tp1) / 180.0f;
            if (deltaRotate < 0)
            {
                deltaRotate = -deltaRotate;
            }
            //qDebug("detect pinch: scale=%f, rotate=%f", deltaLength, deltaRotate);

            mCenter = (tp0.pos() + tp1.pos()) * 0.5f;
            if (deltaLength > deltaRotate && deltaLength > PINCH_TOLERANCE)
            {
                mState = StateZoom;
                mStartLength = length;
            }
//            else if (deltaRotate > deltaLength && deltaRotate > PINCH_TOLERANCE)
//            {
//                mState = StateRotate;
//                mStartAngle = getRotateAngle(tp0, tp1);
//            }
            else
            {
                mState = StatePan;
                mStartTilt = getTilt(tp0, tp1);
            }
            return true;
        }
        mState = StateNone;
        return false;
    }
}

void PinchRecognizer::startGesture()
{
    if (mState == StateZoom)
    {
        emit zoomStart(mCenter);
    }
    else if (mState == StateRotate)
    {
        emit rotateStart(mCenter);
    }
    else if (mState == StatePan)
    {
        emit panStart(mCenter);
    }
    mGestureStarted = true;
}

void PinchRecognizer::stopGesture()
{
    if (mGestureStarted)
    {
        if (mState == StateZoom)
        {
            emit zoomEnd(mCenter, mZoomSpeed);
        }
        else if (mState == StateRotate)
        {
            emit rotateEnd(mCenter, mRotateSpeed);
        }
        else if (mState == StatePan)
        {
            emit panEnd(mCenter);
        }
    }
    mGestureStarted = false;
    mState = StateNone;
}

//****************************************************
PanRecognizer::PanRecognizer(QObject* parent)
    :GestureRecognizer(parent)
    ,mDone(false)
{
}

bool PanRecognizer::onTouch(QTouchEvent* event, QList<GestureTouch>& history)
{
    if (event->type() == QEvent::TouchBegin)
    {
        mDone = false;
    }

    if (mDone)
    {
        return false;
    }

    if (event->touchPoints().size() != 1)
    {
        mDone = true;
        return false;
    }

    QTouchEvent::TouchPoint pt = event->touchPoints().first();
    if (mGestureStarted)
    {
        mPosition = pt.pos();
        if (event->type() == QEvent::TouchEnd || event->type() == QEvent::TouchCancel)
        {
            return false;
        }
        emit pan(mPosition);
        return true;
    }
    else
    {
        if (distance(pt.pos(),pt.startPos()) >= PAN_TOLERANCE)
        {
            mPosition = pt.startPos();
            return true;
        }
        return false;
    }
}

void PanRecognizer::startGesture()
{
    emit panStart(mPosition);
    mGestureStarted = true;
}

void PanRecognizer::stopGesture()
{
    if (mGestureStarted)
    {
        emit panEnd(mPosition);
    }
    mGestureStarted = false;
}

//****************************************************
TwoFingerTapRecognizer::TwoFingerTapRecognizer(QObject* parent)
    :GestureRecognizer(parent)
{
}

bool TwoFingerTapRecognizer::onTouch(QTouchEvent* event, QList<GestureTouch>& history)
{
    if (event->type() == QEvent::TouchBegin)
    {
        mStopped = false;
        mMaxTouchPoints = 1;
        mPressTime.restart();
    }
    else if (event->type() == QEvent::TouchEnd)
    {
        QTouchEvent::TouchPoint tp = event->touchPoints().first();
        if (!mStopped && mMaxTouchPoints == 2 &&
            mPressTime.elapsed() < TAP_INTERVAL &&
            distance(tp.startPos(), tp.pos()) <= TAP_TOLERANCE)
        {
            mPositions[1] = tp.pos();
            emit twoFingerTap(((mPositions[0] + mPositions[1])) * 0.5f);
        }
    }
    else if (event->type() == QEvent::TouchUpdate)
    {
        if (!mStopped)
        {
            if (event->touchPoints().size() > 2)
            {
                mStopped = true;
            }
            else if (event->touchPoints().size() == 2)
            {
                QTouchEvent::TouchPoint tp0 = event->touchPoints()[0];
                QTouchEvent::TouchPoint tp1 = event->touchPoints()[1];
                if (distance(tp0.startPos(), tp0.pos()) > TAP_TOLERANCE ||
                    distance(tp1.startPos(), tp1.pos()) > TAP_TOLERANCE)
                {
                    mStopped = true;
                    return false;
                }

                if (tp0.state() == Qt::TouchPointReleased)
                {
                    mPositions[0] = tp0.pos();
                }
                else if (tp1.state() == Qt::TouchPointReleased)
                {
                    mPositions[0] = tp1.pos();
                }
            }

            if (event->touchPoints().size() > mMaxTouchPoints)
            {
                mMaxTouchPoints = event->touchPoints().size();
            }
        }
    }
    else
    {
        mStopped = true;
    }
    return false;
}
//****************************************************
TapRecognizer::TapRecognizer(QObject* parent)
    :GestureRecognizer(parent)
    ,mStopped(true)
{
    mLastTapTime.start();
}

bool TapRecognizer::onTouch(QTouchEvent* event, QList<GestureTouch>& history)
{
    QTouchEvent::TouchPoint tp = event->touchPoints().first();
    if (event->type() == QEvent::TouchBegin)
    {
        if (mLastTapTime.elapsed() < DOUBLE_TAP_INTERVAL &&
            distance(mLastTapPosition, tp.pos()) < TAP_TOLERANCE)
        {
            emit doubleTap(tp.pos());
            mStopped = true;
            return false;
        }
        mStopped = false;
        mPressTime.restart();
    }
    else if (event->type() == QEvent::TouchEnd)
    {
        if (!mStopped && mPressTime.elapsed() < TAP_INTERVAL)
        {
            mLastTapTime.restart();
            mLastTapPosition = tp.pos();
            emit tap(tp.pos());
        }
    }
    else
    {
        if (event->touchPoints().size() != 1
           || distance(tp.pos(), tp.startPos()) > TAP_TOLERANCE)
        {
            mStopped = true;
        }
    }
    return false;
}

void TapRecognizer::stopGesture()
{
    mStopped = true;
}

//****************************************************
LongPressRecognizer::LongPressRecognizer(QObject* parent)
    :GestureRecognizer(parent)
    ,mTimeout(LONG_PRESS_INTERVAL)
{
    mTimer = new QTimer(this);
    connect(mTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
}

bool LongPressRecognizer::onTouch(QTouchEvent* event, QList<GestureTouch>& history)
{
    QTouchEvent::TouchPoint tp = event->touchPoints().first();
    if (event->type() == QEvent::TouchBegin)
    {
        mPosition = tp.pos();
        mTimer->start(mTimeout);
        return true;
    }
    else if (event->type() == QEvent::TouchEnd)
    {
        mTimer->stop();
        return false;
    }
    else
    {
        if (event->touchPoints().size() != 1
           || distance(tp.pos(), tp.startPos()) > LONG_PRESS_TOLERANCE)
        {
            mTimer->stop();
            return false;
        }
        else
        {
            return true;
        }
    }
}

void LongPressRecognizer::onTimeout()
{
    mTimer->stop();
    emit longPress(mPosition);
}

void LongPressRecognizer::stopGesture()
{
    mTimer->stop();
}
