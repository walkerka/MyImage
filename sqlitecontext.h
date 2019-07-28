#ifndef SQLITECONTEXT_H
#define SQLITECONTEXT_H
#include <vector>
#include <string>
#include <functional>

struct sqlite3;

class SqliteParam
{
public:
    enum Type
    {
        TypeInt,
        TypeFloat,
        TypeBlob,
        TypeString,
        TypeNull
    };

    virtual ~SqliteParam() {}
    virtual Type GetType() = 0;
    const std::string& GetName() { return mName; }
    void SetName(const std::string& n) { mName = n; }

private:
    std::string mName;
};

class SqliteParamInt: public SqliteParam
{
public:
    SqliteParamInt(int value):mValue(value) {}
    Type GetType() { return TypeInt; }
    int GetValue() { return mValue; }
    void SetValue(int v) { mValue = v; }
private:
    int mValue;
};

class SqliteParamFloat: public SqliteParam
{
public:
    SqliteParamFloat(float value):mValue(value) {}
    Type GetType() { return TypeFloat; }
    float GetValue() { return mValue; }
    void SetValue(float v) { mValue = v; }
private:
    float mValue;
};

class SqliteParamBlob: public SqliteParam
{
public:
    SqliteParamBlob(const void* data, size_t size):mData(data), mSize(size) {}
    Type GetType() { return TypeBlob; }
    const void* GetData() { return mData; }
    size_t GetSize() { return mSize; }
    void SetValue(const void* data, size_t size)
    {
        mData = data;
        mSize = size;
    }
private:
    const void* mData;
    size_t mSize;
};

class SqliteParamString: public SqliteParam
{
public:
    SqliteParamString(const std::string& value):mValue(value) {}
    Type GetType() { return TypeString; }
    const std::string& GetValue() { return mValue; }
    void SetValue(const std::string& value) { mValue = value; }
private:
    std::string mValue;
};

class SqliteParamNull: public SqliteParam
{
public:
    SqliteParamNull() {}
    Type GetType() { return TypeNull; }
};

class SqliteParams
{
public:
    SqliteParams();
    ~SqliteParams();
    SqliteParams& AddInt(int value, const char* field = "");
    SqliteParams& AddFloat(float value, const char* field = "");
    SqliteParams& AddBlob(const void* data, size_t size, const char* field = "");
    SqliteParams& AddString(const std::string& value, const char* field = "");
    SqliteParams& AddNull(const char* field = "");
    int GetParamCount();
    SqliteParam* GetParam(int index);
    int GetInt(int index);
    float GetFloat(int index);
    std::string GetString(int index);
    const void* GetBlob(int index, size_t* size = 0);
private:
    std::vector<SqliteParam*> mParams;
};

class SqliteRowHandler
{
public:
    virtual void OnReadRow(SqliteParams& row) = 0;
};

class SqliteContext
{
public:
    SqliteContext();
    ~SqliteContext();
    bool Open(const char* path);
    void Close();
    void BeginTransaction();
    void CommitTransaction();
    bool Execute(const char* sql);
    bool ExecuteWithParams(const char* sql, SqliteParams& params);
    bool QueryWithParams(const char* sql, SqliteParams& params, std::function<void (SqliteParams& row)> handler);
    bool Query(const char* sql, std::function<void (SqliteParams& row)> handler);
    int QueryIntWithParams(const char* sql, SqliteParams& params);
    int QueryInt(const char* sql);

private:
    sqlite3* mDb;    
};

#endif // SQLITECONTEXT_H
