#ifndef BOOKLISTDIALOG_H
#define BOOKLISTDIALOG_H

#include <QWidget>

class QListWidget;
class QListWidgetItem;
class QLineEdit;
class QAction;
class QLabel;

class BookListDialog : public QWidget
{
    Q_OBJECT
public:
    explicit BookListDialog(QWidget *parent = 0);
    ~BookListDialog();

    int bookId() { return mId; }

signals:
    void openBook(int bookId);
    void addBook();

public slots:
    void open();
    void onBookOpen();
    void onBookAdd();
    void updateBooks();
    void deleteBooks();
    void onSelect(QListWidgetItem*,QListWidgetItem*);
    void onBookDoubleClick(QListWidgetItem*);
    void onContextMenu(const QPoint&);

private:
    QListWidget* mBooksField;
    QLineEdit* mNameLineEdit;
    QLineEdit* mKeyLineEdit;
    QAction* mActionDelete;
    QLabel* mPreviewField;
    int mId;
};

#endif // BOOKLISTDIALOG_H
