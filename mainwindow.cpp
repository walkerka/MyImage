#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "appcontext.h"
#include "sqlitecontext.h"
#include "glrenderer.h"
#include "book.h"
#include "booklistdialog.h"
#include "server.h"
#include "client.h"
#include <QtWidgets>
#include <QSettings>
#include <qglobal.h>

ImageLoader::ImageLoader()
{
}

ImageLoader::~ImageLoader()
{
}

void ImageLoader::load(int id)
{
    mLock.lock();
    mPendingImages.push_front(id);
    for (int i = 1; i < mPendingImages.size(); ++i)
    {
        if (mPendingImages[i] == id)
        {
            mPendingImages.removeAt(i);
            --i;
        }
    }
    mLock.unlock();
    QMetaObject::invokeMethod(this, "onLoad", Qt::QueuedConnection);
}

void ImageLoader::prefetch(QList<int> ids)
{
    mLock.lock();
    for (int i = 0; i < ids.size(); ++i)
    {
        int id = ids[i];
        if (mPendingImages.indexOf(id) == -1)
        {
            mPendingImages.push_back(id);
        }
    }
    mLock.unlock();
    QMetaObject::invokeMethod(this, "onLoad", Qt::QueuedConnection);
}

void ImageLoader::clear()
{
    mLock.lock();
    mPendingImages.clear();
    mLock.unlock();
}

void ImageLoader::onLoad()
{
    mLock.lock();
    int n = mPendingImages.size();
    mLock.unlock();

    while (n > 0)
    {
        mLock.lock();
        int id = mPendingImages.front();
        mPendingImages.pop_front();
        mLock.unlock();
        
        Image* img = new Image(id);
        qDebug() << "load image" << img->GetName();
        emit imageLoaded(img);

        mLock.lock();
        n = mPendingImages.size();
        mLock.unlock();
    }
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    mArchive(NULL),
    mBook(NULL),
    mCacheSize(10),
    mStretchMode(StretchModeFitWidth),
    mServerView(NULL),
    mClientView(NULL)
{
    ui->setupUi(this);

#ifdef TARGET_OS_IPHONE
    mCacheSize = 1;
#endif
    QDir::setCurrent(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    mApp = new AppContext();

    mBooksView = new BookListDialog();
    mBooksView->setVisible(false);

    mClientView = new Client();
    mClientView->setVisible(false);
    
    mLoader = new ImageLoader();
    mLoader->moveToThread(&mLoadThread);
    connect(mLoader, SIGNAL(imageLoaded(Image*)), this, SLOT(onImageLoaded(Image*)), Qt::QueuedConnection);
    mLoadThread.start();

    connect(mBooksView, SIGNAL(openBook(int)), this, SLOT(openBook(int)));
    connect(mBooksView, SIGNAL(addBook()), this, SLOT(addBook()));
    connect(ui->actionExit, SIGNAL(triggered(bool)), this, SLOT(close()));
    connect(ui->actionNew, SIGNAL(triggered(bool)), this, SLOT(newArchive()));
    connect(ui->actionOpen, SIGNAL(triggered(bool)), this, SLOT(openArchive()));
    connect(ui->actionShare, SIGNAL(triggered(bool)), this, SLOT(shareArchives()));
    connect(ui->actionDownload, SIGNAL(triggered(bool)), this, SLOT(downloadArchives()));
    connect(ui->actionAddBook, SIGNAL(triggered(bool)), this, SLOT(addBook()));
    connect(ui->actionOpenBook, SIGNAL(triggered(bool)), mBooksView, SLOT(open()));
    connect(ui->actionDeleteBook, SIGNAL(triggered(bool)), this, SLOT(deleteBook()));
    connect(ui->actionExportPdf, SIGNAL(triggered(bool)), this, SLOT(exportPdf()));
    connect(ui->actionExportImageSequence, SIGNAL(triggered(bool)), this, SLOT(exportDir()));
    connect(ui->actionExportPage, SIGNAL(triggered(bool)), this, SLOT(exportPage()));
    connect(ui->actionDeletePage, SIGNAL(triggered(bool)), this, SLOT(deletePage()));
    connect(ui->actionAddImage, SIGNAL(triggered(bool)), this, SLOT(addImage()));
    connect(ui->actionAddFolder, SIGNAL(triggered(bool)), this, SLOT(addFolder()));
    connect(ui->actionPrevPage, SIGNAL(triggered(bool)), this, SLOT(prevPage()));
    connect(ui->actionNextPage, SIGNAL(triggered(bool)), this, SLOT(nextPage()));
    connect(ui->actionFirstPage, SIGNAL(triggered(bool)), this, SLOT(firstPage()));
    connect(ui->actionLastPage, SIGNAL(triggered(bool)), this, SLOT(lastPage()));
    connect(ui->actionFitWidth, SIGNAL(triggered(bool)), this, SLOT(setStretchFitWidth()));
    connect(ui->actionFitHeight, SIGNAL(triggered(bool)), this, SLOT(setStretchFitHeight()));
    connect(ui->actionBestFit, SIGNAL(triggered(bool)), this, SLOT(setStretchBestFit()));
    connect(ui->actionNoStretch, SIGNAL(triggered(bool)), this, SLOT(setStretchNone()));
    connect(ui->actionCopyImage, SIGNAL(triggered(bool)), this, SLOT(copyImage()));
    connect(ui->centralWidget, SIGNAL(wheelUp()), this, SLOT(prevPage()));
    connect(ui->centralWidget, SIGNAL(wheelDown()), this, SLOT(nextPage()));
    connect(ui->centralWidget, SIGNAL(tap(QPointF)), this, SLOT(onTap(QPointF)));
    connect(ui->centralWidget, SIGNAL(doubleTap(QPointF)), this, SLOT(onDoubleTap(QPointF)));
    setAcceptDrops(true);

    QSettings s("w", "my_image");
    QString recentArchive = s.value("recent/archive").toString();
    if (!recentArchive.isEmpty())
    {
        QFileInfo fi(recentArchive);
        if (fi.exists())
        {
            mArchive = new Archive(recentArchive);
            QMetaObject::invokeMethod(mBooksView, "open", Qt::QueuedConnection);
        }
    }
    int stretchMode = s.value("view/stretch", QVariant((int)StretchModeFitWidth)).toInt();
    mStretchMode = (StretchMode)stretchMode;
}

MainWindow::~MainWindow()
{
    QSettings s("w", "my_image");
    s.setValue("view/stretch", QVariant((int)mStretchMode));

    mLoadThread.exit();
    delete mBook;
    delete mArchive;
    delete mBooksView;
    delete mServerView;
    delete mApp;
    delete ui;
}

void MainWindow::newArchive()
{
    QString path = QFileDialog::getSaveFileName(this, tr("New"), QString(), tr("*.ma"));
    if(path.isEmpty())
    {
        return;
    }
    if (!path.endsWith(".ma", Qt::CaseInsensitive))
    {
        path += ".ma";
    }

    if (mArchive)
    {
        delete mArchive;
        mArchive = NULL;
    }
    mArchive = new Archive(path);
    int len = 0;
    char* sql = LoadFile(":/doc.sql", len);
    mArchive->GetContext()->Execute(sql);

    QSettings s("w", "my_image");
    s.setValue("recent/archive", path);

    addBook();
}

void MainWindow::openArchive()
{
    QString path = QFileDialog::getOpenFileName(this, tr("Open"), QString(), tr("*.ma"));
    if(path.isEmpty())
    {
        return;
    }
    openArchive(path);
}

void MainWindow::openArchive(const QString& path)
{
    if (mArchive)
    {
        delete mArchive;
        mArchive = NULL;
    }
    mArchive = new Archive(path);
    if (mArchive->GetContext() == NULL)
    {
        delete mArchive;
        mArchive = NULL;
        return;
    }
    QSettings s("w", "my_image");
    s.setValue("recent/archive", path);

    mBooksView->open();
}

void MainWindow::shareArchives()
{
    if (!mServerView)
    {
        mServerView = new Server();
    }
    mServerView->show();
}

void MainWindow::downloadArchives()
{
    mClientView->show();
}

void MainWindow::addBook()
{
    if (!mArchive)
    {
        return;
    }
    QString name = QInputDialog::getText(this, tr("Add Book"), tr("Name"));
    if (name.isEmpty())
    {
        return;
    }
    addBook(name);
}

void MainWindow::addBook(const QString& name)
{
    if (!mArchive)
    {
        return;
    }
    if (mBook)
    {
        delete mBook;
        mBook = NULL;
    }
    mBook = Book::AddBook(name);
    mBooksView->updateBooks();
}

void MainWindow::openBook(int bookId)
{
    if (!mArchive)
    {
        return;
    }

    if (mBook && Book::Exist(mBook->GetId()))
    {
        delete mBook;
        mBook = NULL;
    }

    if (bookId)
    {
        if (mBook)
        {
            delete mBook;
            mBook = NULL;
        }
        mBook = Book::LoadBook(bookId);
        modPage(0);
    }
}

void MainWindow::deleteBook()
{
    if (mBook)
    {
        if (QMessageBox::warning(this, tr("Delete"), tr("Delete current book?"), QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
        {
            Book::DeleteBook(mBook->GetId());
            delete mBook;
            mBook = NULL;
            mBooksView->updateBooks();
        }
    }
}

void MainWindow::exportPdf()
{
    if (!mBook)
    {
        return;
    }
    QString path = QFileDialog::getSaveFileName(this, tr("Export"), QString(), tr("*.pdf"));
    if(path.isEmpty())
    {
        return;
    }
    mBook->ExportPdf(path);
}

void MainWindow::exportDir()
{
    if (!mBook)
    {
        return;
    }
    QString path = QFileDialog::getExistingDirectory(this, tr("Export"), QString());
    if(path.isEmpty())
    {
        return;
    }
    mBook->ExportDir(path);
}

void MainWindow::exportPage()
{
    if (!mBook)
    {
        return;
    }
    int id = mBook->GetCurrentImage();
    if (!mImages.contains(id))
    {
        return;
    }
    Image* img = mImages[id];
    QString name = QDir::current().absolutePath() + "/" + img->GetName();
    QFileInfo fi(name);
    QString suffix = fi.suffix();
    if (suffix.isEmpty())
    {
        suffix = "*";
    }
    suffix.sprintf("*.%s", fi.suffix().toStdString().c_str());
    QString path = QFileDialog::getSaveFileName(this, tr("Export"), name, suffix);
    if(path.isEmpty())
    {
        return;
    }
    img->Save(path);
}

void MainWindow::deletePage()
{
    if (mBook)
    {
        if (QMessageBox::warning(this, tr("Delete"), tr("Delete current page?"), QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
        {
            mBook->DeletePage(mBook->GetCurrentPage());
            loadPage();
        }
    }
}

void MainWindow::addImage()
{
    if (!mBook)
    {
        return;
    }
    QStringList files = QFileDialog::getOpenFileNames(this, "Add Image", QString(), "*.png *.jpg *.jpeg *.bmp *.tga *.gif *.zip");
    foreach (QString f, files)
    {
        if (f.toLower().endsWith(".zip"))
        {
            addZip(f);
        }
        else
        {
            mBook->AddPage(f);
        }
    }

    loadPage();
}

void MainWindow::addFolder()
{
    if (!mBook)
    {
        return;
    }
    QString folder = QFileDialog::getExistingDirectory(this, "Add Folder", QString());
    if (!folder.isEmpty())
    {
        addDir(folder);
    }
    loadPage();
}

void MainWindow::loadPage()
{
    if (mBook && mBook->GetImages().size() > 0)
    {
        int imgId = mBook->GetCurrentImage();
        if (mImages.find(imgId) != mImages.end())
        {
            Image& image = *mImages[imgId];
            image.UpdateAccessTime();
            ui->centralWidget->BeginLoad();
            for (int i = 0; i < image.GetNumFrames(); ++i)
            {
                ui->centralWidget->Load(image.GetWidth(), image.GetHeight(), image.GetPixelSize(), image.GetFrameData(i), i, image.GetFrameDelay(i));
            }

            ui->centralWidget->EndLoad();
            resetTransform();
            setWindowTitle(image.GetName());

            prefetch();
        }
        else
        {
            mLoader->load(imgId);
        }
    }
    else
    {
        ui->centralWidget->BeginLoad();
        ui->centralWidget->EndLoad();
    }
}

void MainWindow::prefetch()
{
    int from = mBook->GetCurrentPage() + 1;
    int to = from + mCacheSize / 2;
    if (from < 0)
    {
        from = 0;
    }
    if (to >= mBook->GetImages().size())
    {
        to = mBook->GetImages().size() - 1;
    }
    QList<int> ids;
    for (int i = from; i <= to; ++i)
    {
        if (i == mBook->GetCurrentPage())
        {
            continue;
        }
        int id = mBook->GetImages()[i];
        if (mImages.find(id) != mImages.end() || mPendingImages.contains(id))
        {
            continue;
        }
        ids.push_back(id);
        mPendingImages.insert(id);
    }
    if (ids.size() > 0)
    {
        mLoader->prefetch(ids);
    }
}

void MainWindow::onImageLoaded(Image* img)
{
    mPendingImages.remove(img->GetId());
    mImages[img->GetId()] = img;
    while (mImages.size() >= mCacheSize)
    {
        int id = 0;
        int accessTime = 1000000000;
        Image* oldest = NULL;
        for (QMap<int, Image*>::iterator it = mImages.begin(); it != mImages.end(); ++it)
        {
            Image* img = it.value();
            int t = img->GetAccessTime();
            if (t < accessTime)
            {
                accessTime = t;
                id = it.key();
                oldest = img;
            }
        }
        if (oldest)
        {
            mImages.remove(id);
            delete oldest;
        }
    }
    if (mBook->GetCurrentImage() == img->GetId())
    {
        loadPage();
    }
}

void MainWindow::modPage(int delta)
{
    if (mBook)
    {
        mBook->ModCurrentPage(delta);
        loadPage();
    }
}

void MainWindow::prevPage()
{
    modPage(-1);
}

void MainWindow::nextPage()
{
    modPage(1);
}

void MainWindow::firstPage()
{
    modPage(-mBook->GetImages().size());
}

void MainWindow::lastPage()
{
    modPage(mBook->GetImages().size());
}

void MainWindow::onTap(QPointF p)
{
    if (p.x() > ui->centralWidget->width() * devicePixelRatio() * 0.75f)
    {
        modPage(1);
    }
    else if (p.x() < ui->centralWidget->width() * devicePixelRatio() * 0.25f)
    {
        modPage(-1);
    }
    else
    {
        ui->centralWidget->PlayOrStop();
    }
}

void MainWindow::onDoubleTap(QPointF p)
{
    mBooksView->open();
}

static void findImages(const QString& root, QStringList& imagePaths)
{
    QDir dir(root);
    QFileInfoList fi = dir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot);
    foreach (QFileInfo f, fi)
    {
        if (f.isDir())
        {
            findImages(f.absoluteFilePath(), imagePaths);
        }
        else
        {
            QString s = f.suffix().toLower();
            if (s == "png" || s == "jpg" || s == "jpeg" || s == "bmp" || s == "gif" || s == "tga")
            {
                imagePaths.push_back(f.absoluteFilePath());
            }
        }
    }
}

void MainWindow::addZip(const QString& path)
{
#if defined(_WIN32)
    QProcess process;
    QString cmd;
    cmd.sprintf("7z x \"%s\" -aoa -r -y -o\"./tmp/\"", path.toStdString().c_str());
    process.execute(cmd);

    addDir("./tmp");
    QDir tmp("./tmp");
    tmp.removeRecursively();
#elif defined(Q_OS_MACOS)
    QProcess process;
    QString cmd;
    cmd.sprintf("unzip -o '%s' -d './tmp/'", path.toStdString().c_str());
    process.execute(cmd);

    addDir("./tmp");
    QDir tmp("./tmp");
    tmp.removeRecursively();
#endif
}

void MainWindow::addDir(const QString& path)
{
    QStringList files;
    findImages(path, files);
    foreach (QString f, files)
    {
        mBook->AddPage(f);
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent *e)
{
    if (e->mimeData()->hasUrls())
    {
        e->acceptProposedAction();
    }
    else
    {
        e->ignore();
    }
}

void MainWindow::dropEvent(QDropEvent *e)
{
    if (e->mimeData()->hasUrls())
    {
        foreach (QUrl url, e->mimeData()->urls())
        {
            QFileInfo fi(url.toLocalFile());
            if (fi.isDir())
            {
                addBook(fi.baseName());
                addDir(fi.absoluteFilePath());
            }
            else if (fi.suffix().compare("png", Qt::CaseInsensitive) == 0
             || fi.suffix().compare("jpg", Qt::CaseInsensitive) == 0
             || fi.suffix().compare("jpeg", Qt::CaseInsensitive) == 0
             || fi.suffix().compare("gif", Qt::CaseInsensitive) == 0
             || fi.suffix().compare("tga", Qt::CaseInsensitive) == 0
             || fi.suffix().compare("bmp", Qt::CaseInsensitive) == 0)
            {
                if (mBook)
                {
                    mBook->AddPage(fi.absoluteFilePath());
                }
            }
            else if (fi.suffix().compare("zip", Qt::CaseInsensitive) == 0
                || fi.suffix().compare("rar", Qt::CaseInsensitive) == 0
                || fi.suffix().compare("7z", Qt::CaseInsensitive) == 0
                || fi.suffix().compare("gz", Qt::CaseInsensitive) == 0
                || fi.suffix().compare("tar", Qt::CaseInsensitive) == 0
                )
            {
                addBook(fi.baseName());
                addZip(fi.absoluteFilePath());
            }
            else if (fi.suffix().compare("ma", Qt::CaseInsensitive) == 0)
            {
                openArchive(fi.absoluteFilePath());
            }
        }
        e->accept();
        loadPage();
    }
}

void MainWindow::resetTransform()
{
    if (!mBook || !mImages.contains(mBook->GetCurrentImage()))
    {
        return;
    }
    Image* img = mImages[mBook->GetCurrentImage()];
    int w = img->GetWidth();
    int h = img->GetHeight();
    float sw = ui->centralWidget->width() * devicePixelRatio();
    float sh = ui->centralWidget->height() * devicePixelRatio();
    Transform2 tran;
    switch (mStretchMode) {
    case StretchModeFitWidth:
        tran.SetScale(sw/w, sw/w);
        tran.SetTranslate(0, sh - h*sw / w);
        break;
    case StretchModeFitHeight:
        tran.SetScale(sh/h, sh/h);
        tran.SetTranslate(0, 0);
        break;
    case StretchModeBestFit:
        if (sw/w > sh/h)
        {
            float scale = sw/w;
            tran.SetScale(scale, scale);
            tran.SetTranslate(0, sh - h*scale);
        }
        else
        {
            float scale = sh/h;
            tran.SetScale(scale, scale);
            tran.SetTranslate(0, 0);
        }
        break;
    case StretchModeNone:
    default:
        tran.SetTranslate(0, sh - h);
        break;
    }
    ui->centralWidget->SetTransform(tran);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    if (mBook)
    {
        int id = mBook->GetCurrentImage();
        if (id && mImages.contains(id))
        {
            resetTransform();
        }
    }
}

void MainWindow::setStretchFitWidth()
{
    mStretchMode = StretchModeFitWidth;
    resetTransform();
}

void MainWindow::setStretchFitHeight()
{
    mStretchMode = StretchModeFitHeight;
    resetTransform();
}

void MainWindow::setStretchBestFit()
{
    mStretchMode = StretchModeBestFit;
    resetTransform();
}

void MainWindow::setStretchNone()
{
    mStretchMode = StretchModeNone;
    resetTransform();
}

void MainWindow::copyImage()
{
    if (!mBook)
    {
        return;
    }
    int id = mBook->GetCurrentImage();
    if (!mImages.contains(id))
    {
        return;
    }
    Image* img = mImages[id];
    int frameIndex = ui->centralWidget->GetFrameIndex();
    unsigned char* data = img->GetFrameData(frameIndex);
    QImage::Format format;
    switch(img->GetPixelSize())
    {
    case 3:
        format = QImage::Format_RGB888;
        break;
    case 4:
        format = QImage::Format_RGBA8888;
        break;
    case 1:
        format = QImage::Format_Grayscale8;
        break;
    default:
        qDebug("invalid format");
        return;
    }
    QImage qimg(data, img->GetWidth(), img->GetHeight(), img->GetWidth() * img->GetPixelSize(), format);
    QClipboard *clipboard = QGuiApplication::clipboard();
    clipboard->setImage(qimg.mirrored());
}
