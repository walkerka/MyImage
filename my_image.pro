#-------------------------------------------------
#
# Project created by QtCreator 2016-07-25T20:26:06
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MyImage
TEMPLATE = app

SOURCES += main.cpp\
    mainwindow.cpp \
    mainwidget.cpp \
    glrenderer.cpp \
    sqlitecontext.cpp \
    sqlite3.c \
    book.cpp \
    stb.cpp \
    booklistdialog.cpp \
    gesturedetector.cpp \
    server.cpp \
    appcontext.cpp \
    client.cpp

HEADERS  += mainwindow.h \
    mainwidget.h \
    sqlite3.h \
    sqlite3ext.h \
    sqlitecontext.h \
    book.h \
    glrenderer.h \
    booklistdialog.h \
    gesturedetector.h \
    server.h \
    appcontext.h \
    client.h

FORMS    += mainwindow.ui

RESOURCES += \
    resource.qrc

win32 {
SOURCES += gl3w.c
DEFINES += USE_OPENGL3 _USE_MATH_DEFINES
LIBS += -lglu32 -lopengl32
}

DISTFILES +=
