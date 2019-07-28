#include "client.h"
#include <QtWidgets>
#include <QJsonDocument>
#include <QJsonObject>

HttpManager::HttpManager()
    :reply(Q_NULLPTR)
    ,mFile(NULL)
{
    mAccessManager = new QNetworkAccessManager();
}

HttpManager::~HttpManager()
{
    delete mAccessManager;
}

void HttpManager::Request(QString url, std::function<void(QByteArray&,QString)> onReply, QString savePath)
{
    if (reply)
    {
        return;
    }
    mReplyCallback = onReply;
    mSavePath = savePath;
    if (!mSavePath.isEmpty())
    {
        mFile = new QFile(savePath);
        if (!mFile->open(QFile::WriteOnly))
        {
            return;
        }
    }
    QUrl u = QUrl::fromUserInput((url));
    if (!u.isValid())
    {
        return;
    }
    reply = mAccessManager->get(QNetworkRequest(u));
    connect(reply, &QNetworkReply::finished, this, &HttpManager::httpFinished);
    connect(reply, &QIODevice::readyRead, this, &HttpManager::readyRead);
    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(onError(QNetworkReply::NetworkError)));
}

void HttpManager::httpFinished()
{
    if (mSavePath.isEmpty())
    {
        QByteArray ba = reply->readAll();
        mReplyCallback(ba, mError);
    }
    else
    {
        mFile->write(reply->readAll());
        mFile->close();
        delete mFile;
        mFile = NULL;
        QByteArray ba;
        mReplyCallback(ba, mError);
    }

    reply->deleteLater();
    reply = Q_NULLPTR;
    this->deleteLater();
}

void HttpManager::readyRead()
{
    if (mFile)
    {
        mFile->write(reply->readAll());
    }
}

void HttpManager::onError(QNetworkReply::NetworkError)
{
    mError = reply->errorString();
}

Client::Client(QWidget *parent)
    :QWidget(parent)
{
    QVBoxLayout* box = new QVBoxLayout();
    QFormLayout* form = new QFormLayout();

    QString addr = "localhost:9090";
    QSettings s("w", "my_image");
    QString recentAddr = s.value("recent/serverAddr").toString();
    if (!recentAddr.isEmpty())
    {
        addr = recentAddr;
    }
    mAddrField = new QLineEdit();
    mAddrField->setText(addr);
    mAddrField->setFocusPolicy(Qt::StrongFocus);
    form->addRow("Server", mAddrField);

    mArchiveList = new QListWidget();
    form->addRow("Archives", mArchiveList);

    box->addLayout(form);

    QHBoxLayout* buttonLayout = new QHBoxLayout();

    QPushButton* refreshButton = new QPushButton("Refresh");
    refreshButton->setAutoDefault(true);
    connect(refreshButton, SIGNAL(clicked(bool)), this, SLOT(refresh()));
    buttonLayout->addWidget(refreshButton);

    QPushButton* downloadButton = new QPushButton("Download");
    connect(downloadButton, SIGNAL(clicked(bool)), this, SLOT(download()));
    buttonLayout->addWidget(downloadButton);

    QPushButton* closeButton = new QPushButton("Close");
    connect(closeButton, SIGNAL(clicked(bool)), this, SLOT(hide()));
    buttonLayout->addWidget(closeButton);

    box->addLayout(buttonLayout);

    setLayout(box);
}

void Client::refresh()
{
    QString url = mAddrField->text();
    if (!url.isEmpty())
    {
        QSettings s("w", "my_image");
        s.setValue("recent/serverAddr", url);
    }

    HttpManager* http = new HttpManager();

    url += "/list";
    http->Request(url, [&](QByteArray& ba, QString err) {
        if (!err.isEmpty())
        {
            QString msg;
            msg.sprintf("Refresh failed. %s", err.toStdString().c_str());
            QMessageBox::information(this, tr("MyImage"),
                                  msg);
            return;
        }
        QJsonDocument doc = QJsonDocument::fromJson(ba);
        if (doc.isArray())
        {
            mArchiveList->clear();
            QJsonArray arr = doc.array();
            for (int i = 0; i < arr.size(); ++i)
            {
                QJsonObject obj = arr[i].toObject();
                int id = obj["id"].toInt();
                QString name = obj["name"].toString();
                QListWidgetItem* item = new QListWidgetItem(name);
                item->setData(Qt::UserRole, QVariant(id));
                mArchiveList->addItem(item);
            }
        }
    });
}

void Client::download()
{
    if (mArchiveList->selectedItems().size() == 0)
    {
        return;
    }

    QListWidgetItem* item = mArchiveList->selectedItems().first();
    int id = item->data(Qt::UserRole).toInt();

    QString path = QFileDialog::getSaveFileName(this, tr("Save as"), QString(), tr("*.ma"));
    if(path.isEmpty())
    {
        return;
    }
    if (!path.endsWith(".ma", Qt::CaseInsensitive))
    {
        path += ".ma";
    }

    HttpManager* http = new HttpManager();
    QString url;
    url.sprintf("%s/archive/%d", mAddrField->text().toStdString().c_str(), id);
    http->Request(url, [&](QByteArray& /*ba*/, QString err) {
        QString msg = tr("Download finished");
        if (!err.isEmpty())
        {
            msg.sprintf("Download failed. %s", err.toStdString().c_str());
        }
        QMessageBox::information(this, tr("MyImage"),
                              msg);
    }, path);
}
