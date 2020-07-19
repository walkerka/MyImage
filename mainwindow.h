#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMap>
#include <QSet>
#include <QMutex>
#include <QThread>

namespace Ui {
class MainWindow;
}

class AppContext;
class Archive;
class Book;
class Image;
class BookListDialog;
class MainWidget;
class Server;
class Client;

class ImageLoader: public QObject
{
    Q_OBJECT
public:
    ImageLoader();
    ~ImageLoader();

    void load(int id);
    void prefetch(QList<int> ids);
    void clear();

signals:
    void imageLoaded(Image* image);
public slots:
    void onLoad();

private:
    QList<int> mPendingImages;
    QMutex mLock;
};

enum StretchMode
{
    StretchModeNone,
    StretchModeBestFit,
    StretchModeFitWidth,
    StretchModeFitHeight
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);
    void resizeEvent(QResizeEvent *event);

public slots:
    void newArchive();
    void openArchive();
    void openArchive(const QString& path);
    void shareArchives();
    void downloadArchives();
    void addBook();
    void addBook(const QString& name);
    void openBook(int bookId);
    void deleteBook();
    void exportPdf();
    void exportDir();
    void exportPage();
    void deletePage();
    void addImage();
    void addFolder();
    void addZip(const QString& path);
    void addDir(const QString& path);
    void loadPage();
    void prevPage();
    void nextPage();
    void firstPage();
    void lastPage();
    void modPage(int delta);
    void onTap(QPointF p);
    void onDoubleTap(QPointF p);
    void onImageLoaded(Image* img);
    void prefetch();
    void resetTransform();
    void setStretchFitWidth();
    void setStretchFitHeight();
    void setStretchBestFit();
    void setStretchNone();
    void copyImage();

private:
    Ui::MainWindow *ui;
    AppContext* mApp;
    Archive* mArchive;
    Book* mBook;
    ImageLoader* mLoader;
    QThread mLoadThread;
    QMap<int, Image*> mImages;
    QSet<int> mPendingImages;
    int mCacheSize;
    StretchMode mStretchMode;
    BookListDialog* mBooksView;
    Server* mServerView;
    Client* mClientView;
    MainWidget* mImageView;
};

#endif // MAINWINDOW_H
