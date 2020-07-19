#ifndef BOOK_H
#define BOOK_H
#include <QString>
#include <QStringList>
#include <vector>

class SqliteContext;

class Archive
{
public:
    Archive(const QString& path);
    ~Archive();
    static Archive* GetInstance() { return sInstance; }
    SqliteContext* GetContext(bool forceNewInstance = false);

private:
    SqliteContext* mContext;
    static Archive* sInstance;
    std::string mPath;
};

class Image
{
public:
    Image(int id);
    Image(const QString& path);
    Image();
    ~Image();

    int GetId() { return mId; }
    int GetWidth() { return mWidth; }
    int GetHeight() { return mHeight; }
	const QString& GetName() { return mName; }
    int GetPixelSize() { return mPixelSize; }
    unsigned char* GetData() { return mData; }
	int GetNumFrames() { return mFrames; }
	unsigned char* GetFrameData(int frameIndex);
	int GetFrameDelay(int frameIndex);
    int GetAccessTime() const { return mAccessTime; }
    void UpdateAccessTime();
    void Load(const QString& path);
    void Sync();
    void Save(const QString& path);

private:
    int mId;
    int mWidth;
    int mHeight;
    QString mName;
    int mPixelSize;
    int mFrames;
    unsigned char* mData;
    int mAccessTime;

    unsigned char* mEncodeBuf = nullptr;
    int mEncodeLength = 0;
};

class Book
{
public:
    Book(int id = 0);
    ~Book();

    int GetId() const { return mId; }
    const QString& GetName() const { return mName; }
    int AddPage(const QString& imagePath);
    void AddPageList(const std::vector<std::string>& imagePaths);
    void DeletePage(int pageId);
    const QList<int>& GetImages() { return mImageIds; }
    int GetCurrentImage();
    int GetCurrentPage() { return mCurrentPage; }
    void ModCurrentPage(int delta);
    bool Export(const QString& path);
    bool ExportDir(const QString& path);
    bool ExportPdf(const QString& path);

    static Book* AddBook(const QString& name);
    static void DeleteBook(int id);
    static Book* LoadBook(int id);
    static void SearchBooks(const QString& namePattern, const QString& key, QList<int>& resultIds, QStringList& names);
    static bool Exist(int id);
    static int GetBookHomePage(int bookId);

private:
    int mId;
    QString mName;
    QString mKeys;
    QList<int> mImageIds;
    int mCurrentPage;
};

#endif // BOOK_H
