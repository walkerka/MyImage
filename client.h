#ifndef CLIENT_H
#define CLIENT_H
#include <QWidget>
#include <functional>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QByteArray>
#include <QFile>

class QLineEdit;
class QListWidget;

class HttpManager: public QObject
{
    Q_OBJECT
public:
    HttpManager();
    ~HttpManager();
    void Request(QString url, std::function<void(QByteArray&,QString)> onReply, QString savePath = "");

public slots:
    void httpFinished();
    void readyRead();
    void onError(QNetworkReply::NetworkError);

signals:

private:
    QNetworkAccessManager* mAccessManager;
    QNetworkReply* reply;
    std::function<void(QByteArray&,QString)> mReplyCallback;
    QString mSavePath;
    QFile* mFile;
    QString mError;
};

class Client : public QWidget
{
    Q_OBJECT
public:
    explicit Client(QWidget *parent = 0);

signals:

public slots:
    void refresh();
    void download();

private:
    QLineEdit* mAddrField;
    QListWidget* mArchiveList;
};

#endif // CLIENT_H
