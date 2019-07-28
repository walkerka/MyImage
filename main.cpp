#include "mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QFileInfo>

int main(int argc, char *argv[])
{
    QApplication::setAttribute(Qt::AA_ShareOpenGLContexts, true);
    QApplication a(argc, argv);

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if(!TARGET_OS_IPHONE)
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);
#endif
#endif

    MainWindow w;
    w.show();

    if (argc > 1)
    {
        QFileInfo fi(argv[1]);
        if (fi.exists() && fi.isFile() && fi.suffix().toLower() == "ma")
        {
            w.openArchive(fi.absoluteFilePath());
        }
    }

    return a.exec();
}
