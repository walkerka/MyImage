#include "appcontext.h"
#include "sqlitecontext.h"
#include "glrenderer.h"
#include <QFileInfo>

AppContext* AppContext::mInstance = NULL;

AppContext::AppContext()
{
    mInstance = this;
    mDb = new SqliteContext();
    QFileInfo fi("app.db");
    bool exists = fi.exists();
    mDb->Open("app.db");
    if (!exists)
    {
        int size = 0;
        char* sql = LoadFile(":/app.sql", size);
        if (sql)
        {
            mDb->Execute(sql);
        }
    }
}

AppContext::~AppContext()
{
    mInstance = NULL;
    delete mDb;
}

void AppContext::GetArchives(QList<int>& ids, QStringList& names)
{
    mDb->Query("select id,name from archive", [&](SqliteParams& row){
        ids.push_back(row.GetInt(0));
        names.push_back(row.GetString(1).c_str());
    });
}

void AppContext::AddArchive(const QString& name, const QString& path)
{
    SqliteParams ps;
    ps.AddString(path.toStdString());
    mDb->BeginTransaction();
    int count = mDb->QueryIntWithParams("select count(*) from archive where path=?", ps);
    if (count == 0)
    {
        ps.AddString(name.toStdString());
        mDb->ExecuteWithParams("insert into archive (path,name) values (?,?)", ps);
    }
    mDb->CommitTransaction();
}

void AppContext::DeleteArchive(int id)
{
    SqliteParams ps;
    ps.AddInt(id);
    mDb->ExecuteWithParams("delete from archive where id=?", ps);
}

QString AppContext::GetArchivePath(int id)
{
    QString path;
    SqliteParams ps;
    ps.AddInt(id);
    mDb->QueryWithParams("select path from archive where id=?", ps, [&](SqliteParams& row){
        path = row.GetString(0).c_str();
    });
    return path;
}
