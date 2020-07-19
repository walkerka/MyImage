#include "book.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sqlitecontext.h"
#include "glrenderer.h"
#include <QImage>
#include <QFileInfo>
#include <QTime>
#include <QDebug>
#include <QDir>
#include <QPdfWriter>
#include <QPainter>
#include "glrenderer.h"
#include <future>

inline SqliteContext* context()
{
    return Archive::GetInstance()->GetContext();
}

Archive* Archive::sInstance = NULL;

Archive::Archive(const QString& path)
    :mPath(path.toStdString())
{
    mContext = new SqliteContext();
    if (!mContext->Open(path.toStdString().c_str()))
    {
        delete mContext;
        mContext = NULL;
    }

    sInstance = this;
}

Archive::~Archive()
{
    delete mContext;
    sInstance = NULL;
}

SqliteContext* Archive::GetContext(bool forceNewInstance)
{
    if (forceNewInstance)
    {
        auto context = new SqliteContext();
        if (!context->Open(mPath.c_str()))
        {
            delete context;
            context = NULL;
        }
        return context;
    }
    return mContext;
}

Image::Image(int id)
    :mId(id)
	,mData(NULL)
    ,mAccessTime(QTime::currentTime().msecsSinceStartOfDay())
{
    QTime time;
    time.start();
    SqliteParams ps;
    ps.AddInt(id);
    context()->QueryWithParams("select width,height,name,imageData from image where id=?", ps, [&](SqliteParams& row) {
        mWidth = row.GetInt(0);
        mHeight = row.GetInt(1);
        mName = row.GetString(2).c_str();
        size_t len = 0;
        unsigned char* buf = (unsigned char*)row.GetBlob(3, &len);
        qDebug() << "db=" << time.elapsed();
        time.restart();
        if (mName.endsWith(".gif", Qt::CaseInsensitive))
        {
            char tempPath[100];
            sprintf(tempPath, "./temp_%d.gif", id);
            SaveFile(tempPath, buf, len);
            mData = LoadAnim(tempPath, mWidth, mHeight, mPixelSize, mFrames);
            QFile file(tempPath);
            file.remove();
        }
        else
        {
            mData = LoadImgFromMemory(buf, len, mWidth, mHeight, mPixelSize);
            if (mData)
            {
                mFrames = 1;
            }
        }
        qDebug() << "decode=" << time.elapsed();
    });
}

Image::Image(const QString& path)
    :mId(0)
    ,mAccessTime(QTime::currentTime().msecsSinceStartOfDay())
{
    QFileInfo fi(path);
    unsigned char* buf = NULL;
    int len = 0;
    QString suffix = fi.suffix().toLower();
    if (suffix == "gif" || IsGif(path.toStdString().c_str()))
    {
        mData = LoadAnim(path.toStdString().c_str(), mWidth, mHeight, mPixelSize, mFrames);
        buf = (unsigned char*)LoadFile(path.toStdString().c_str(), len);
    }
    else
    {
        mData = LoadImg(path.toStdString().c_str(), mWidth, mHeight, mPixelSize);
        if (mData)
        {
            mFrames = 1;
        }
        if (suffix == "jpeg" || suffix == "jpg" || suffix == "png")
        {
            buf = (unsigned char*)LoadFile(path.toStdString().c_str(), len);
        }
        else
        {
            buf = SaveImgToMemory(mWidth, mHeight, mPixelSize, mData, len);
        }
    }

    if (mData)
    {
        mName = fi.fileName();
        if (mFrames > 1 && fi.suffix().toLower() != "gif")
        {
            mName.append(".gif");
        }

        auto context = Archive::GetInstance()->GetContext();

        context->BeginTransaction();
        SqliteParams ps;
        ps.AddInt(mWidth);
        ps.AddInt(mHeight);
        ps.AddString(mName.toStdString());
        ps.AddBlob(buf, len);
        context->ExecuteWithParams("insert into image (width,height,name,imageData) values (?,?,?,?)", ps);
        mId = context->QueryInt("select last_insert_rowid();");
        context->CommitTransaction();
        delete[] buf;
    }
}

Image::Image()
    :mId(0)
    ,mWidth(0)
    ,mHeight(0)
    ,mAccessTime(QTime::currentTime().msecsSinceStartOfDay())
{
    auto context = Archive::GetInstance()->GetContext();

    SqliteParams ps;
    ps.AddInt(mWidth);
    ps.AddInt(mHeight);
    ps.AddString("");

    context->ExecuteWithParams("insert into image (width,height,name) values (?,?,?)", ps);
    mId = context->QueryInt("select last_insert_rowid();");
}

Image::~Image()
{
    ReleaseImg(mData);
}

unsigned char* Image::GetFrameData(int frameIndex)
{
	int frameSize = mWidth * mHeight * mPixelSize + 2;
	return mData + frameSize * frameIndex;
}

int Image::GetFrameDelay(int frameIndex)
{
    if (mFrames <= 1)
    {
        return 0;
    }
	int frameSize = mWidth * mHeight * mPixelSize + 2;
	unsigned short* delay = (unsigned short*)(mData + (frameSize * (frameIndex + 1) - 2));
	return (int)*delay;
}

void Image::UpdateAccessTime()
{
    mAccessTime = QTime::currentTime().msecsSinceStartOfDay();
}

void Image::Load(const QString& path)
{
    mAccessTime = QTime::currentTime().msecsSinceStartOfDay();
    QFileInfo fi(path);
    unsigned char* buf = NULL;
    int len = 0;
    QString suffix = fi.suffix().toLower();
    if (suffix == "gif" || IsGif(path.toStdString().c_str()))
    {
        mData = LoadAnim(path.toStdString().c_str(), mWidth, mHeight, mPixelSize, mFrames);
        buf = (unsigned char*)LoadFile(path.toStdString().c_str(), len);
    }
    else
    {
        mData = LoadImg(path.toStdString().c_str(), mWidth, mHeight, mPixelSize);
        if (mData)
        {
            mFrames = 1;
        }
        if (suffix == "jpeg" || suffix == "jpg" || suffix == "png")
        {
            buf = (unsigned char*)LoadFile(path.toStdString().c_str(), len);
        }
        else
        {
            buf = SaveImgToMemory(mWidth, mHeight, mPixelSize, mData, len);
        }
    }
    mName = fi.fileName();
    if (mFrames > 1 && fi.suffix().toLower() != "gif")
    {
        mName.append(".gif");
    }

    mEncodeBuf = buf;
    mEncodeLength = len;
}

void Image::Sync()
{
    if (mData)
    {
        auto context = Archive::GetInstance()->GetContext();

        SqliteParams ps;
        ps.AddInt(mWidth);
        ps.AddInt(mHeight);
        ps.AddString(mName.toStdString());
        ps.AddBlob(mEncodeBuf, mEncodeLength);
        ps.AddInt(mId);
        context->ExecuteWithParams("update image set width=?,height=?,name=?,imageData=? where id=?", ps);
        delete[] mEncodeBuf;
        mEncodeBuf = nullptr;
        mEncodeLength = 0;
    }
}

void Image::Save(const QString& path)
{
    SqliteParams ps;
    ps.AddInt(mId);
    context()->QueryWithParams("select width,height,name,imageData from image where id=?", ps, [&](SqliteParams& row) {
        mWidth = row.GetInt(0);
        mHeight = row.GetInt(1);
        mName = row.GetString(2).c_str();
        size_t len = 0;
        const char* buf = (const char*)row.GetBlob(3, &len);
        QFile file(path);
        if (file.open(QFile::WriteOnly))
        {
            file.write(buf, len);
            file.close();
        }
    });
}

Book::Book(int id)
    :mId(id)
    ,mCurrentPage(0)
{

}

Book::~Book()
{

}

int Book::AddPage(const QString& imagePath)
{
    Image img(imagePath);
    int imgId = img.GetId();
    if (imgId != 0)
    {
        SqliteParams ps;
        ps.AddInt(mId);
        ps.AddInt(imgId);
        ps.AddInt(mImageIds.size());
        context()->ExecuteWithParams("insert into bookPage (bookId,imageId,pageId) values (?,?,?)", ps);
        mImageIds.push_back(imgId);
    }
    return imgId;
}

void Book::AddPageList(const std::vector<std::string>& imagePaths)
{
    auto ctx = context();
    ctx->BeginTransaction();
    std::vector<std::future<Image*>> tasks;
    for (auto& path: imagePaths)
    {
        Image* img = new Image();
        int imgId = img->GetId();
        if (imgId != 0)
        {
            SqliteParams ps;
            ps.AddInt(mId);
            ps.AddInt(imgId);
            int pageId = mImageIds.size();
            ps.AddInt(pageId);
            ctx->ExecuteWithParams("insert into bookPage (bookId,imageId,pageId) values (?,?,?)", ps);
            mImageIds.push_back(imgId);
            qDebug() << "page: " << pageId << "," << img->GetName();

            tasks.emplace_back(std::async(std::launch::async, [img,path](){
                img->Load(path.c_str());
                return img;
            }));
        }
        else {
            qDebug() << "page: lost " << img->GetName();
        }
    }

    std::vector<Image*> imgs;
    for (auto& t: tasks)
    {
        auto img = t.get();
        img->Sync();
        imgs.push_back(img);
    }

    for(Image* img: imgs)
    {
        delete img;
    }
    ctx->CommitTransaction();
}

void Book::DeletePage(int pageId)
{
    if (pageId < 0 || pageId >= mImageIds.size())
    {
        return;
    }
    context()->BeginTransaction();
    {
        SqliteParams ps;
        ps.AddInt(mId);
        ps.AddInt(pageId);
        context()->ExecuteWithParams("delete from image where id in (select imageId from bookPage where bookId=? and pageId=?)", ps);
    }
    {
        SqliteParams ps;
        ps.AddInt(mId);
        ps.AddInt(pageId);
        context()->ExecuteWithParams("delete from bookPage where bookId=? and pageId=?", ps);
    }

    context()->CommitTransaction();
    mImageIds.removeAt(pageId);
    if (mCurrentPage >= mImageIds.size())
    {
        mCurrentPage = mImageIds.size() - 1;
    }
}

Book* Book::AddBook(const QString& name)
{
    Book* book = NULL;
    context()->BeginTransaction();
    SqliteParams ps;
    ps.AddString(name.toStdString());
    ps.AddString("");
    context()->ExecuteWithParams("insert into book (name,keywords,createDate,homePageId) values (?,?,datetime(),0)", ps);
    int id = context()->QueryInt("select last_insert_rowid()");
    book = new Book(id);
    book->mName = name;
    book->mKeys = "";
    context()->CommitTransaction();
    return book;
}

void Book::DeleteBook(int id)
{
    context()->BeginTransaction();
    {
        SqliteParams ps;
        ps.AddInt(id);
        context()->ExecuteWithParams("delete from image where id in (select imageId from bookPage where bookId=?)", ps);
    }
    {
        SqliteParams ps;
        ps.AddInt(id);
        context()->ExecuteWithParams("delete from bookPage where bookId=?", ps);
        context()->ExecuteWithParams("delete from book where id=?", ps);
    }

    context()->CommitTransaction();
}

Book* Book::LoadBook(int id)
{
    Book* book = NULL;
    SqliteParams ps;
    ps.AddInt(id);
    context()->QueryWithParams("select name,keywords from book where id=?", ps, [&](SqliteParams& row) {
        book = new Book(id);
        book->mName = row.GetString(0).c_str();
        book->mKeys = row.GetString(1).c_str();
    });
    if (book)
    {
        context()->QueryWithParams("select imageId from bookPage where bookId=? order by pageId", ps, [&](SqliteParams& row) {
            book->mImageIds.push_back(row.GetInt(0));
        });
    }
    return book;
}

void Book::SearchBooks(const QString& namePattern, const QString& key, QList<int>& resultIds, QStringList& names)
{
    if (namePattern.isEmpty() && key.isEmpty())
    {
        context()->Query("select id, name from book order by name", [&](SqliteParams& row) {
            resultIds.push_back(row.GetInt(0));
            names.push_back(row.GetString(1).c_str());
        });
    }
    else
    {
        SqliteParams ps;
        ps.AddString("%" + namePattern.toStdString() + "%");
        ps.AddString("%" + key.toStdString() + "%");
        context()->QueryWithParams("select id,name from book where name like ? and keywords like ? order by name", ps, [&](SqliteParams& row) {
            resultIds.push_back(row.GetInt(0));
            names.push_back(row.GetString(1).c_str());
        });
    }
}

int Book::GetBookHomePage(int bookId)
{
    int imgId = 0;
    SqliteParams ps;
    ps.AddInt(bookId);
    context()->QueryWithParams("select homePageId from book where id=?", ps, [&](SqliteParams& row) {
        imgId = row.GetInt(0);
    });
    if (imgId == 0)
    {
        context()->QueryWithParams("select imageId from bookPage where bookId=? and pageId=0", ps, [&](SqliteParams& row) {
            imgId = row.GetInt(0);
        });
    }
    return imgId;
}

int Book::GetCurrentImage()
{
    if (mImageIds.size() == 0)
    {
        return 0;
    }
    int idx = mCurrentPage;
    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx > mImageIds.size())
    {
        idx = mImageIds.size() - 1;
    }
    return mImageIds[idx];
}

void Book::ModCurrentPage(int delta)
{
    int idx = mCurrentPage + delta;
    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx >= mImageIds.size())
    {
        idx = mImageIds.size() - 1;
    }
    if (mImageIds.size() == 0)
    {
        idx = -1;
    }
    mCurrentPage = idx;
}

bool Book::Exist(int id)
{
    SqliteParams ps;
    ps.AddInt(id);
    return context()->QueryIntWithParams("select count(*) from book where id=?", ps) > 0;
}

bool Book::Export(const QString& path)
{
    QFileInfo fi(path);
    QString suffix = fi.suffix().toLower();
    if (suffix == "pdf")
    {
        return ExportPdf(path);
    }
    else if (suffix == "")
    {
        return ExportDir(path);
    }
    else
    {
        return false;
    }
}

bool Book::ExportDir(const QString& path)
{
    QDir dir(path);
    if (!dir.exists())
    {
        if (!dir.mkpath("."))
        {
            return false;
        }
    }
    SqliteParams ps;
    ps.AddInt(mId);
    const char* sql = "select name,imageData from image where id in (select imageId from bookPage where bookId=? order by pageId)";
    context()->QueryWithParams(sql, ps, [&](SqliteParams& row) {
        std::string name = row.GetString(0);
        size_t len = 0;
        const void* data = row.GetBlob(1, &len);
        QString imgPath = path + "/" + name.c_str();
        SaveFile(imgPath.toStdString().c_str(), data, (int)len);
    });
    return true;
}

bool Book::ExportPdf(const QString& path)
{
    QFile pdf_file(path);
    pdf_file.open(QIODevice::WriteOnly);
    QPdfWriter *pdf_writer = new QPdfWriter(&pdf_file);

    QPainter *pdf_painter= new QPainter();
    pdf_painter->begin(pdf_writer);

    int page = 0;
    SqliteParams ps;
    ps.AddInt(mId);
    const char* sql = "select name,imageData from image where id in (select imageId from bookPage where bookId=? order by pageId)";
    context()->QueryWithParams(sql, ps, [&](SqliteParams& row) {
//        std::string name = row.GetString(0);
        size_t len = 0;
        const void* data = row.GetBlob(1, &len);
//        std::string imgPath = (QDir::tempPath() + "/").toStdString();
//        imgPath += name;
//        SaveFile(imgPath.c_str(), data, (int)len);
//        QImage img(imgPath.c_str());
        QImage img = QImage::fromData((const uchar*)data, len);

        QRect rc = pdf_painter->viewport();
        QSize size = img.size();
        size.scale(rc.size(), Qt::KeepAspectRatio);
        pdf_painter->drawImage(QRect(0,0,size.width(),size.height()), img);

        if (page + 1 < mImageIds.size())
        {
            pdf_writer->newPage();
        }

//        QFile f(imgPath.c_str());
//        f.remove();
        ++page;
    });

    pdf_painter->end();
    delete pdf_painter;
    delete pdf_writer;
    pdf_file.close();
    return true;
}
