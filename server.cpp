/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
#include "server.h"
#include <QtWidgets>
#include <QtNetwork>
#include <QThread>
#include <stdlib.h>
#include <QJsonDocument>
#include <QJsonObject>
#include "book.h"
#include "appcontext.h"

FortuneThread::FortuneThread(int socketDescriptor, QObject *parent)
    : QThread(parent), socketDescriptor(socketDescriptor)
{
}

void FortuneThread::run()
{
    QTcpSocket tcpSocket;
    if (!tcpSocket.setSocketDescriptor(socketDescriptor)) {
        emit error(tcpSocket.error());
        return;
    }

    tcpSocket.waitForReadyRead(-1);
    char rawUrl[256];
    tcpSocket.readLine(rawUrl, 256);
    QString url(rawUrl);
    url = url.split(" ")[1];
    QStringList parts = url.split("/");
    QString type = parts[1];
    qDebug() << "Accepted request:\n" << tcpSocket.readAll().toStdString().c_str();

    QString resp;
    resp.append("HTTP/1.1 200 OK\r\n");
    resp.append("Server: img/1.0\r\n");

    if (type == "list")
    {
        QList<int> ids;
        QStringList names;
        AppContext::Instance()->GetArchives(ids, names);
        QJsonArray root;
        for (int i = 0; i < ids.size(); ++i)
        {
            QJsonObject obj;
            obj["id"] = ids[i];
            obj["name"] = names[i];
            root.append(obj);
        }
        QJsonDocument doc;
        doc.setArray(root);
        QByteArray data = doc.toJson();

        resp.append("Content-Type: text/plain\r\n");
        resp.append("Content-Encoding: utf-8\r\n");
        resp.append(QString().sprintf("Content-Length: %d\r\n", data.size()));
        resp.append("\r\n");
        tcpSocket.write(resp.toStdString().c_str(), resp.length());
        if (data.size() > 0)
        {
            tcpSocket.write(data);
        }
    }
    else if (type == "archive")
    {
        QString id = parts[2];
        QString path = AppContext::Instance()->GetArchivePath(id.toInt());
        if (!path.isEmpty())
        {
            QFile file(path);
            if (file.open(QFile::ReadOnly))
            {
                qint64 size = file.size();
                resp.append("Content-Type: application/octet-stream\r\n");
                resp.append(QString().sprintf("Content-Length: %lld\r\n", size));
                resp.append("\r\n");
                tcpSocket.write(resp.toStdString().c_str(), resp.length());

                const qint64 BUFFER_SIZE = 1024 * 4;
                char* buf = new char[BUFFER_SIZE];
                while (size > 0)
                {   
                    qint64 readBytes = file.read(buf, BUFFER_SIZE);
                    if (readBytes > 0)
                    {
                        qint64 n = 0;
                        while (n < readBytes)
                        {
                            n += tcpSocket.write(buf + n, readBytes - n);
                        }
                        size -= readBytes;
                    }
                }
                delete[] buf;
                file.close();
            }
        }
    }

    tcpSocket.disconnectFromHost();
    tcpSocket.waitForDisconnected(480 * 1000);
}

FortuneServer::FortuneServer(QObject *parent)
    : QTcpServer(parent)
{
}

void FortuneServer::incomingConnection(qintptr socketDescriptor)
{
    FortuneThread *thread = new FortuneThread(socketDescriptor, this);
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
    thread->start();
}

Server::Server(QWidget *parent)
    : QWidget(parent)
    , statusLabel(new QLabel)
    , tcpServer(Q_NULLPTR)
    , networkSession(0)
{
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    statusLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);

    QNetworkConfigurationManager manager;
    if (manager.capabilities() & QNetworkConfigurationManager::NetworkSessionRequired) {
        // Get saved network configuration
        QSettings settings(QSettings::UserScope, QLatin1String("QtProject"));
        settings.beginGroup(QLatin1String("QtNetwork"));
        const QString id = settings.value(QLatin1String("DefaultNetworkConfiguration")).toString();
        settings.endGroup();

        // If the saved network configuration is not currently discovered use the system default
        QNetworkConfiguration config = manager.configurationFromIdentifier(id);
        if ((config.state() & QNetworkConfiguration::Discovered) !=
            QNetworkConfiguration::Discovered) {
            config = manager.defaultConfiguration();
        }

        networkSession = new QNetworkSession(config, this);
        connect(networkSession, &QNetworkSession::opened, this, &Server::sessionOpened);

        statusLabel->setText(tr("Opening network session."));
        networkSession->open();
    } else {
        sessionOpened();
    }

    QPushButton* addButton = new QPushButton(tr("Add"));
    connect(addButton, SIGNAL(clicked(bool)), this, SLOT(addSharedArchives()));

    QPushButton *quitButton = new QPushButton(tr("Quit"));
    quitButton->setAutoDefault(false);
    connect(quitButton, &QAbstractButton::clicked, this, &QWidget::hide);

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(addButton);
    buttonLayout->addWidget(quitButton);
    buttonLayout->addStretch(1);

    shareListView = new QListWidget();

    QVBoxLayout *mainLayout = Q_NULLPTR;
    mainLayout = new QVBoxLayout(this);

    mainLayout->addWidget(statusLabel);
    mainLayout->addWidget(shareListView);
    mainLayout->addLayout(buttonLayout);

    setWindowTitle(QGuiApplication::applicationDisplayName());
    updateShareList();
}

void Server::sessionOpened()
{
    // Save the used configuration
    if (networkSession) {
        QNetworkConfiguration config = networkSession->configuration();
        QString id;
        if (config.type() == QNetworkConfiguration::UserChoice)
            id = networkSession->sessionProperty(QLatin1String("UserChoiceConfiguration")).toString();
        else
            id = config.identifier();

        QSettings settings(QSettings::UserScope, QLatin1String("QtProject"));
        settings.beginGroup(QLatin1String("QtNetwork"));
        settings.setValue(QLatin1String("DefaultNetworkConfiguration"), id);
        settings.endGroup();
    }

    tcpServer = new FortuneServer(this);
    QHostAddress addr;
    if (!tcpServer->listen(addr, 9090)) {
        QMessageBox::critical(this, tr("Server"),
                              tr("Unable to start the server: %1.")
                              .arg(tcpServer->errorString()));
        close();
        return;
    }

    QString ipAddress;
    QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();
    // use the first non-localhost IPv4 address
    for (int i = 0; i < ipAddressesList.size(); ++i) {
        if (ipAddressesList.at(i) != QHostAddress::LocalHost &&
            ipAddressesList.at(i).toIPv4Address()) {
            ipAddress = ipAddressesList.at(i).toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();
    statusLabel->setText(tr("The server is running on\n\nIP: %1\nport: %2")
                         .arg(ipAddress).arg(tcpServer->serverPort()));
}

void Server::addSharedArchives()
{
    QStringList paths = QFileDialog::getOpenFileNames(this, tr("Open"), QString(), tr("*.ma"));
    if(paths.size() == 0)
    {
        return;
    }

    AppContext* app = AppContext::Instance();
    for (int i = 0; i < paths.size(); ++i)
    {
        const QString& path = paths[i];
        QFileInfo fi(path);
        app->AddArchive(fi.baseName(), path);
    }
}

void Server::updateShareList()
{
    shareListView->clear();
    AppContext* app = AppContext::Instance();
    QList<int> ids;
    QStringList names;
    app->GetArchives(ids, names);
    for (int i = 0; i < ids.size(); ++i)
    {
        int id = ids[i];
        const QString& name = names[i];
        QListWidgetItem* item = new QListWidgetItem(name);
        item->setData(Qt::UserRole, QVariant(id));
        shareListView->addItem(item);
    }
}
