#ifndef GESTUREDETECTOR_H
#define GESTUREDETECTOR_H
#include <QTouchEvent>
#include <QList>
#include <QTime>
#include <QTimer>

struct GestureTouch
{
    QList<QPointF> points;
    float t;
};

class GestureRecognizer: public QObject
{
    Q_OBJECT
public:
    GestureRecognizer(QObject* parent = NULL);
    virtual ~GestureRecognizer();
    virtual bool onTouch(QTouchEvent* event, QList<GestureTouch>& history) = 0;
    virtual void startGesture();
    virtual void stopGesture();
protected:
    bool mGestureStarted;
};

class GestureDetector
{
public:
    GestureDetector();

    void touchEvent(QTouchEvent *event);
    void startGestureRecognizer(GestureRecognizer* r);
    void stopCurrentGesture();
    void addGestureRecognizer(GestureRecognizer* r);

private:
    void saveHistory(QTouchEvent* event);
    void startGesture(GestureRecognizer* r);

signals:

private:
    QList<GestureTouch> mHistory;
    QTime mTime;
    GestureRecognizer* mCurrent;
    QList<GestureRecognizer*> mRecognizers;
};

//gestures
class PinchRecognizer: public GestureRecognizer
{
    Q_OBJECT
public:
    PinchRecognizer(QObject* parent = 0);
    bool onTouch(QTouchEvent* event, QList<GestureTouch>& history);
    void startGesture();
    void stopGesture();

signals:
    void zoomStart(QPointF pos);
    void zoom(QPointF pos, float scale);
    void zoomEnd(QPointF pos, float speed);
    void rotateStart(QPointF pos);
    void rotate(QPointF pos, float angle);
    void rotateEnd(QPointF pos, float speed);
    void panStart(QPointF pos);
    void pan(QPointF pos);
    void panEnd(QPointF pos);    

private:
    enum State
    {
        StateNone,
        StateZoom,
        StateRotate,
        StatePan
    };

    float getDistanceSpeed();
    float getAngularSpeed();
    float getCenterSpeed();
private:
    float mZoomSpeed;
    float mRotateSpeed;
    QPointF mTranslateVelocity;
    State mState;
    QPointF mCenter;
    float mStartLength;
    float mStartAngle;
    float mStartTilt;
};

class TwoFingerTapRecognizer: public GestureRecognizer
{
    Q_OBJECT
public:
    TwoFingerTapRecognizer(QObject* parent = 0);
    bool onTouch(QTouchEvent* event, QList<GestureTouch>& history);

signals:
    void twoFingerTap(QPointF center);

private:
    bool mStopped;
    QTime mPressTime;
    int mMaxTouchPoints;
    QPointF mPositions[2];
};

class TapRecognizer: public GestureRecognizer
{
    Q_OBJECT
public:
    TapRecognizer(QObject* parent = 0);
    bool onTouch(QTouchEvent* event, QList<GestureTouch>& history);
    void stopGesture();

signals:
    void tap(QPointF center);
    void doubleTap(QPointF center);
private:
    QTime mPressTime;
    QTime mLastTapTime;
    bool mStopped;
    QPointF mLastTapPosition;
};


class PanRecognizer: public GestureRecognizer
{
    Q_OBJECT
public:
    PanRecognizer(QObject* parent = NULL);
    bool onTouch(QTouchEvent* event, QList<GestureTouch>& history);
    void startGesture();
    void stopGesture();

signals:
    void panStart(QPointF position);
    void pan(QPointF delta);
    void panEnd(QPointF position);

private:
    QPointF mPosition;
    QPointF mTranslateVelocity;
    bool mDone;
};

class LongPressRecognizer: public GestureRecognizer
{
    Q_OBJECT
public:
    LongPressRecognizer(QObject* parent = NULL);
    bool onTouch(QTouchEvent* event, QList<GestureTouch>& history);
    void stopGesture();

public slots:
    void onTimeout();
signals:
    void longPress(QPointF center);
private:
    int mTimeout;
    QTimer* mTimer;
    QPointF mPosition;
    QPointF mTotal;
    int mCount;
};


#endif // GESTUREDETECTOR_H
