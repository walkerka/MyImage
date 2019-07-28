#include "sqlitecontext.h"
#include "sqlite3.h"
#include <assert.h>
#include <QDebug>

SqliteParams::SqliteParams()
{

}

SqliteParams::~SqliteParams()
{
    for (size_t i = 0; i < mParams.size(); ++i)
    {
        delete mParams[i];
    }
}

SqliteParams& SqliteParams::AddInt(int value, const char* field)
{
    SqliteParamInt* p = new SqliteParamInt(value);
    p->SetName(field);
    mParams.push_back(p);
    return *this;
}

SqliteParams& SqliteParams::AddFloat(float value, const char* field)
{
    SqliteParamFloat* p = new SqliteParamFloat(value);
    p->SetName(field);
    mParams.push_back(p);
    return *this;
}

SqliteParams& SqliteParams::AddBlob(const void* data, size_t size, const char* field)
{
    SqliteParamBlob* p = new SqliteParamBlob(data, size);
    p->SetName(field);
    mParams.push_back(p);
    return *this;
}

SqliteParams& SqliteParams::AddString(const std::string& value, const char* field)
{
    SqliteParamString* p = new SqliteParamString(value);
    p->SetName(field);
    mParams.push_back(p);
    return *this;
}

SqliteParams& SqliteParams::AddNull(const char* field)
{
    SqliteParamNull* p = new SqliteParamNull();
    p->SetName(field);
    mParams.push_back(p);
    return *this;
}

int SqliteParams::GetParamCount()
{
    return (int)mParams.size();
}

SqliteParam* SqliteParams::GetParam(int index)
{
    return mParams[index];
}

int SqliteParams::GetInt(int index)
{
    SqliteParam* p = mParams[index];
    assert(mParams[index] && mParams[index]->GetType() == SqliteParam::TypeInt);
    return ((SqliteParamInt*)p)->GetValue();
}

float SqliteParams::GetFloat(int index)
{
    SqliteParam* p = mParams[index];
    assert(mParams[index] && mParams[index]->GetType() == SqliteParam::TypeFloat);
    return ((SqliteParamFloat*)p)->GetValue();
}

std::string SqliteParams::GetString(int index)
{
    SqliteParam* p = mParams[index];
    assert(mParams[index] && mParams[index]->GetType() == SqliteParam::TypeString);
    return ((SqliteParamString*)p)->GetValue();
}

const void* SqliteParams::GetBlob(int index, size_t* size)
{
    SqliteParam* p = mParams[index];
    assert(mParams[index] && mParams[index]->GetType() == SqliteParam::TypeBlob);
    SqliteParamBlob* blob = (SqliteParamBlob*)p;
    if (size)
    {
        *size = blob->GetSize();
    }
    return ((SqliteParamBlob*)p)->GetData();
}

SqliteContext::SqliteContext()
    :mDb(0)
{
}

SqliteContext::~SqliteContext()
{
    Close();
}

bool SqliteContext::Open(const char* path)
{
    sqlite3_config(SQLITE_CONFIG_SERIALIZED);
    int rs = sqlite3_open(path, &mDb);
    if (rs != SQLITE_OK)
    {
        const char* err = sqlite3_errmsg(mDb);
        qDebug(err);
        return false;
    }
    return true;
}

void SqliteContext::Close()
{
    if (mDb)
    {
        sqlite3_close(mDb);
        mDb = 0;
    }
}

void SqliteContext::BeginTransaction()
{
    sqlite3_exec(mDb, "BEGIN", 0, 0, 0);
}

void SqliteContext::CommitTransaction()
{
    sqlite3_exec(mDb, "COMMIT", 0, 0, 0);
}

bool SqliteContext::Execute(const char* sql)
{
    if (mDb == NULL)
    {
        return false;
    }
    char* err = NULL;
    bool result = sqlite3_exec(mDb, sql, NULL, NULL, &err) == SQLITE_OK;
    if (!result)
    {
        qDebug(err);
        sqlite3_free(err);
    }
    return result;
}

static void BindParams(sqlite3_stmt* stmtUpdate, SqliteParams& params)
{
    for (int i = 0; i < params.GetParamCount(); ++i)
    {
        SqliteParam* p = params.GetParam(i);
        SqliteParam::Type type = p->GetType();
        switch (type)
        {
        case SqliteParam::TypeInt:
            {
                SqliteParamInt* pm = (SqliteParamInt*)p;
                sqlite3_bind_int(stmtUpdate, i + 1, pm->GetValue());
            }
            break;
        case SqliteParam::TypeFloat:
            {
                SqliteParamFloat* pm = (SqliteParamFloat*)p;
                sqlite3_bind_double(stmtUpdate, i + 1, pm->GetValue());
            }
            break;
        case SqliteParam::TypeBlob:
            {
                SqliteParamBlob* pm = (SqliteParamBlob*)p;
                sqlite3_bind_blob(stmtUpdate, i + 1, pm->GetData(), pm->GetSize(), SQLITE_STATIC);
            }
            break;
        case SqliteParam::TypeString:
            {
                SqliteParamString* pm = (SqliteParamString*)p;
                sqlite3_bind_text(stmtUpdate, i + 1, pm->GetValue().c_str(), -1, SQLITE_STATIC);
            }
            break;
        default:
            break;
        }
    }
}

static void GetResults(sqlite3_stmt* stmtSelect, SqliteParams& params)
{
    int n = sqlite3_column_count(stmtSelect);
    for (int i = 0; i < n; ++i)
    {
        int type = sqlite3_column_type(stmtSelect, i);
        switch (type)
        {
        case SQLITE_INTEGER:
            {
                params.AddInt(sqlite3_column_int(stmtSelect, i));
            }
            break;
        case SQLITE_FLOAT:
            {
                params.AddFloat((float)sqlite3_column_double(stmtSelect, i));
            }
            break;
        case SQLITE_BLOB:
            {
                int size = sqlite3_column_bytes(stmtSelect, i);
                const void* data = sqlite3_column_blob(stmtSelect, i);
                params.AddBlob(data, size);
            }
            break;
        case SQLITE3_TEXT:
            {
                params.AddString((const char*)sqlite3_column_text(stmtSelect, i));
            }
            break;
        case SQLITE_NULL:
        default:
            {
                params.AddNull("");
            }
            break;
        }
    }
}

bool SqliteContext::ExecuteWithParams(const char* sql, SqliteParams& params)
{
    if (mDb == NULL)
    {
        return false;
    }
    bool result = false;
    sqlite3_stmt* stmtUpdate = 0;
    if (SQLITE_OK == sqlite3_prepare_v2(mDb, sql, -1, &stmtUpdate, 0))
    {
        BindParams(stmtUpdate, params);
        result = sqlite3_step(stmtUpdate) == SQLITE_DONE;
        sqlite3_finalize(stmtUpdate);
    }
    else
    {
        const char* err = sqlite3_errmsg(mDb);
        qDebug(err);
    }
    return result;
}

bool SqliteContext::QueryWithParams(const char* sql, SqliteParams& params, std::function<void (SqliteParams& row)> handler)
{
    if (mDb == NULL)
    {
        return false;
    }
    bool result = false;
    sqlite3_stmt* stmtSelect = 0;
    if (SQLITE_OK == sqlite3_prepare_v2(mDb, sql, -1, &stmtSelect, 0))
    {
        BindParams(stmtSelect, params);
        while (sqlite3_step(stmtSelect) == SQLITE_ROW)
        {
            SqliteParams row;
            GetResults(stmtSelect, row);
            handler(row);
            result = true;
        }
        sqlite3_finalize(stmtSelect);
    }
    else
    {
        const char* err = sqlite3_errmsg(mDb);
        qDebug(err);
    }
    return result;
}

bool SqliteContext::Query(const char* sql, std::function<void (SqliteParams& row)> handler)
{
    SqliteParams params;
    return QueryWithParams(sql, params, handler);
}

int SqliteContext::QueryIntWithParams(const char* sql, SqliteParams& params)
{
    if (mDb == NULL)
    {
        return false;
    }
    int result = 0;
    sqlite3_stmt* stmtSelect = 0;
    if (SQLITE_OK == sqlite3_prepare_v2(mDb, sql, -1, &stmtSelect, 0))
    {
        BindParams(stmtSelect, params);
        if (sqlite3_step(stmtSelect) == SQLITE_ROW)
        {
            SqliteParams row;
            GetResults(stmtSelect, row);
            if (row.GetParamCount() == 1)
            {
                SqliteParam* p = row.GetParam(0);
                if (p->GetType() == SqliteParam::TypeInt)
                {
                    result = static_cast<SqliteParamInt*>(p)->GetValue();
                }
            }
        }
        sqlite3_finalize(stmtSelect);
    }
    else
    {
        const char* err = sqlite3_errmsg(mDb);
        qDebug(err);
    }
    return result;
}

int SqliteContext::QueryInt(const char* sql)
{
    SqliteParams params;
    return QueryIntWithParams(sql, params);
}
