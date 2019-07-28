#ifndef APPCONTEXT_H
#define APPCONTEXT_H
#include <QList>
#include <QStringList>

class SqliteContext;

class AppContext
{
public:
    AppContext();
    ~AppContext();

    static AppContext* Instance() { return mInstance; }
    void GetArchives(QList<int>& ids, QStringList& names);
    void AddArchive(const QString& name, const QString& path);
    void DeleteArchive(int id);
    QString GetArchivePath(int id);

private:
    static AppContext* mInstance;
    SqliteContext* mDb;
};

#endif // APPCONTEXT_H
